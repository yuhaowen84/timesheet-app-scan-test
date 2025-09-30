# pages/3_Scan_Timesheet.py
import re
from typing import List, Tuple, Dict, Optional
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

st.title("3) Scan Weekly Timesheet (14 lines -> 7 days)")

entries = st.session_state.get("entries")
start_date = st.session_state.get("start_date")
if not entries or not start_date:
    st.warning("Please set a start date and create entries in Enter Timesheet first.")
    st.stop()

try:
    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
except Exception:
    st.error(
        "Missing dependencies. Add to requirements.txt:\n"
        "streamlit torch torchvision torchaudio transformers pillow opencv-python-headless numpy pandas"
    )
    st.stop()

@st.cache_resource(show_spinner=False)
def load_trocr():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten").to(device)
    model.eval()
    return processor, model, device

processor, trocr_model, device = load_trocr()

st.markdown(
    "Assumptions:\n"
    "- Main body has 14 lines; every 2 lines = 1 day (top + bottom).\n"
    "- Top-left two cells show starting day (e.g., 07 over Sep).\n"
    "- Detect SL/OFF/ADO tokens.\n"
    "- Times left->right: R_on, A_on, (skip 3 cols), R_off, A_off.\n"
    "- Rightmost time-like value on the day = Worked.\n"
)

slot = st.selectbox("Apply these 7 days to:", ["Days 1-7 (first week)", "Days 8-14 (second week)"], index=0)
start_day_index = 0 if slot.startswith("Days 1") else 7
if "scan_target_day" in st.session_state:
    start_day_index = 0 if int(st.session_state["scan_target_day"]) < 7 else 7

col_cam, col_up = st.columns(2)
with col_cam:
    photo = st.camera_input("Capture the full 7-day page")
with col_up:
    upload = st.file_uploader("Or upload a photo (JPG/PNG)", type=["jpg", "jpeg", "png"])

img = None
if photo:
    img = Image.open(photo).convert("RGB")
elif upload:
    img = Image.open(upload).convert("RGB")

if not img:
    st.info("Take a photo or upload an image to continue.")
    st.stop()

st.image(img, caption="Selected image", use_container_width=True)

MONTH_MAP = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "SEPT": 9, "OCT": 10, "NOV": 11, "DEC": 12
}
TIME_COLON = re.compile(r"\b([0-2]?\d:[0-5]\d)\b")
TIME_BLOCK = re.compile(r"\b(\d{3,4})\b")
DAY_NUM = re.compile(r"^(?:0?[1-9]|[12]\d|3[01])$")

SICK_TOKEN = re.compile(r"^SL$", re.IGNORECASE)
OFF_TOKEN  = re.compile(r"^OFF$", re.IGNORECASE)
ADO_TOKEN  = re.compile(r"^ADO$", re.IGNORECASE)

def norm_hhmm(text: str) -> str:
    s = text.strip()
    m = TIME_COLON.fullmatch(s)
    if m:
        try:
            h, mm = s.split(":")
            h_i, m_i = int(h), int(mm)
            if 0 <= h_i <= 29 and 0 <= m_i <= 59:
                return f"{h_i:02d}:{m_i:02d}"
        except Exception:
            return ""
        return ""
    m2 = TIME_BLOCK.fullmatch(s)
    if m2:
        raw = m2.group(1)
        try:
            h_i, m_i = int(raw[:-2]), int(raw[-2:])
            if 0 <= h_i <= 29 and 0 <= m_i <= 59:
                return f"{h_i:02d}:{m_i:02d}"
        except Exception:
            return ""
    return ""

def extract_month(token: str) -> Optional[int]:
    t = token.strip().upper()
    t = re.sub(r"[^A-Z]", "", t)
    return MONTH_MAP.get(t)

def _preprocess_for_cc(pil_img: Image.Image) -> np.ndarray:
    import cv2
    img = np.array(pil_img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    bw = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)
    return bw

def _bbox_filter(w: int, h: int, img_w: int, img_h: int) -> bool:
    area = w * h
    if area < 60:
        return False
    if w < 6 or h < 6:
        return False
    if w > img_w * 0.95 and h > img_h * 0.8:
        return False
    return True

def trocr_ocr_dataframe(image: Image.Image) -> pd.DataFrame:
    import cv2
    rgb = np.array(image)
    bin_img = _preprocess_for_cc(image)
    H, W = bin_img.shape[:2]
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if _bbox_filter(w, h, W, H):
            boxes.append((x, y, w, h))
    if not boxes:
        return pd.DataFrame(columns=["left","top","width","height","conf","text"])
    boxes.sort(key=lambda b: (b[1], b[0]))

    rows, crops, metas = [], [], []
    for (x, y, w, h) in boxes:
        pad = max(2, int(0.08 * max(w, h)))
        x0 = max(0, x - pad); y0 = max(0, y - pad)
        x1 = min(W, x + w + pad); y1 = min(H, y + h + pad)
        crops.append(Image.fromarray(rgb[y0:y1, x0:x1, :]))
        metas.append((x0, y0, x1-x0, y1-y0))

    bs = 8
    with torch.no_grad():
        for i in range(0, len(crops), bs):
            batch = crops[i:i+bs]
            inputs = processor(images=batch, return_tensors="pt").to(device)
            generated_ids = trocr_model.generate(**inputs, max_length=48, num_beams=1, early_stopping=True)
            texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
            for j, txt in enumerate(texts):
                l, t, w, h = metas[i + j]
                clean = txt.strip()
                if clean:
                    rows.append({"left": int(l), "top": int(t), "width": int(w), "height": int(h), "conf": None, "text": clean})

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values(by=["top", "left"]).reset_index(drop=True)
    y_tol = max(10, int(df["height"].median() or 12))
    line_id = 0; prev_y = None; ids = []
    for _, r in df.iterrows():
        if prev_y is None or abs(r["top"] - prev_y) > y_tol:
            line_id += 1
            prev_y = r["top"]
        ids.append(line_id)
    df["line_num"] = ids
    return df

with st.spinner("OCR via TrOCR (local)..."):
    df = trocr_ocr_dataframe(img)

if df.empty:
    st.error("OCR returned no text. Try a clearer, flatter image.")
    st.stop()

lines: List[List[Dict]] = []
for ln, g in df.groupby(["line_num"], sort=True):
    tokens = g.sort_values(by="left").to_dict("records")
    lines.append([{"text": r["text"], "x": r["left"], "y": r["top"]} for r in tokens])

lines = sorted(lines, key=lambda row: np.median([t["y"] for t in row]))
if len(lines) > 14:
    lines = lines[:14]

pairs: List[Tuple[List[Dict], List[Dict]]] = []
for i in range(0, len(lines), 2):
    if i + 1 < len(lines):
        pairs.append((lines[i], lines[i+1]))
    else:
        pairs.append((lines[i], []))

def first_pair_anchor(pairs) -> Optional[datetime]:
    if not pairs:
        return None
    top, bot = pairs[0]
    top_tokens = sorted(top, key=lambda t: t["x"])
    bot_tokens = sorted(bot, key=lambda t: t["x"]) if bot else []

    day_num = None
    for t in top_tokens:
        digits = re.sub(r"\D", "", t["text"])
        if digits and DAY_NUM.fullmatch(digits):
            day_num = digits
            break

    month_num = None
    for t in bot_tokens:
        m = extract_month(t["text"])
        if m:
            month_num = m
            break

    if day_num is None or month_num is None:
        return None

    y = start_date.year
    try:
        anch = datetime(year=y, month=month_num, day=int(day_num))
        if abs((anch.date() - start_date).days) > 60:
            prev_y = datetime(year=y-1, month=month_num, day=int(day_num))
            next_y = datetime(year=y+1, month=month_num, day=int(day_num))
            anch = min([anch, prev_y, next_y], key=lambda d: abs((d.date() - start_date).days))
        return anch
    except Exception:
        return None

anchor_dt = first_pair_anchor(pairs)
if anchor_dt:
    st.info(f"Detected starting date from top-left cells: {anchor_dt.strftime('%Y-%m-%d')}")
else:
    st.warning("Could not detect the starting date. Falling back to your selected start date.")

def flags_from_pair(top_line: List[Dict], bottom_line: List[Dict]) -> Dict[str, bool]:
    toks = top_line + bottom_line
    return {
        "ado": any(ADO_TOKEN.fullmatch(t["text"]) for t in toks),
        "off": any(OFF_TOKEN.fullmatch(t["text"]) for t in toks),
        "sick": any(SICK_TOKEN.fullmatch(t["text"]) for t in toks),
    }

def times_from_pair(top_line: List[Dict], bottom_line: List[Dict]) -> Dict[str, str]:
    toks = sorted(top_line + bottom_line, key=lambda t: t["x"])
    cand: List[Tuple[str, int]] = []
    for t in toks:
        n = norm_hhmm(t["text"])
        if n:
            cand.append((n, t["x"]))
    if not cand:
        return {"rs_on": "", "as_on": "", "rs_off": "", "as_off": "", "worked": ""}

    worked_idx = max(range(len(cand)), key=lambda i: cand[i][1])
    worked = cand[worked_idx][0]
    remaining = [c for i, c in enumerate(cand) if i != worked_idx]
    remaining = sorted(remaining, key=lambda z: z[1])

    rs_on  = remaining[0][0] if len(remaining) >= 1 else ""
    as_on  = remaining[1][0] if len(remaining) >= 2 else ""
    rs_off = remaining[2][0] if len(remaining) >= 3 else ""
    as_off = remaining[3][0] if len(remaining) >= 4 else ""
    return {"rs_on": rs_on, "as_on": as_on, "rs_off": rs_off, "as_off": as_off, "worked": worked}

raw_days: List[Dict] = []
for i, (top, bot) in enumerate(pairs[:7]):
    f = flags_from_pair(top, bot)
    t = times_from_pair(top, bot)
    raw_days.append({**f, **t})

dates_for_preview = []
for i in range(len(raw_days)):
    if anchor_dt:
        dates_for_preview.append((anchor_dt + timedelta(days=i)).strftime("%Y-%m-%d"))
    else:
        dates_for_preview.append((start_date + timedelta(days=i)).strftime("%Y-%m-%d"))

st.subheader("Detected (editable) 7-day preview")
df_prev = pd.DataFrame([
    {
        "Date": dates_for_preview[i],
        "Sick (SL)": "Yes" if d["sick"] else "No",
        "Off": "Yes" if d["off"] else "No",
        "ADO": "Yes" if d["ado"] else "No",
        "R Sign-on": d["rs_on"],
        "A Sign-on": d["as_on"],
        "R Sign-off": d["rs_off"],
        "A Sign-off": d["as_off"],
        "Worked": d["worked"],
    } for i, d in enumerate(raw_days)
])
st.dataframe(df_prev, use_container_width=True)

st.markdown("Adjust any values before applying")
edited: List[Dict] = []
for i, d in enumerate(raw_days, start=1):
    st.markdown(f"Day {i}: {dates_for_preview[i-1]}")
    c1, c2, c3, c4, c5, c6, c7, c8, c9 = st.columns(9)
    date_str = c1.text_input("Date (YYYY-MM-DD)", value=dates_for_preview[i-1], key=f"date_{i}")
    sick     = c2.checkbox("Sick (SL)", value=d["sick"], key=f"sick_{i}")
    off      = c3.checkbox("Off", value=d["off"], key=f"off_{i}")
    ado      = c4.checkbox("ADO", value=d["ado"], key=f"ado_{i}")
    rs_on    = c5.text_input("R Sign-on", value=d["rs_on"], key=f"ron_{i}")
    as_on    = c6.text_input("A Sign-on", value=d["as_on"], key=f"aon_{i}")
    rs_off   = c7.text_input("R Sign-off", value=d["rs_off"], key=f"roff_{i}")
    as_off   = c8.text_input("A Sign-off", value=d["as_off"], key=f"aoff_{i}")
    worked   = c9.text_input("Worked", value=d["worked"], key=f"work_{i}")
    edited.append({
        "date_str": date_str.strip(),
        "sick": bool(sick),
        "off": bool(off),
        "ado": bool(ado),
        "rs_on": rs_on.strip(), "as_on": as_on.strip(),
        "rs_off": rs_off.strip(), "as_off": as_off.strip(),
        "worked": worked.strip()
    })

st.caption("Extra is not set by OCR. You can adjust Extra or any flags later on the Enter Timesheet page.")

def apply_week():
    for i in range(7):
        target = start_day_index + i
        if target >= len(entries) or i >= len(edited):
            break
        ed = edited[i]
        entries[target].update({
            "rs_on": ed["rs_on"],
            "as_on": ed["as_on"],
            "rs_off": ed["rs_off"],
            "as_off": ed["as_off"],
            "worked": ed["worked"],
            "sick": ed["sick"],
            "off": ed["off"],
            "ado": ed["ado"],
        })
        if ed["date_str"]:
            entries[target]["date_str"] = ed["date_str"]
    st.session_state["entries"] = entries

btn_label = f"Apply to {'Days 1-7' if start_day_index == 0 else 'Days 8-14'}"
if st.button(btn_label, type="primary"):
    apply_week()
    st.success("Saved 7 days into your 14-day entries. Open Enter Timesheet or Review Calculations to continue.")

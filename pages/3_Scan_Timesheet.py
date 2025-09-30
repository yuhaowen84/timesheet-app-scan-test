# pages/3_Scan_Timesheet.py
import re
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from datetime import datetime, timedelta

st.title("3) Scan Weekly Timesheet (14 lines → 7 days)")

# ---- Ensure session state exists (created on Enter Timesheet page) ----
entries = st.session_state.get("entries")
start_date = st.session_state.get("start_date")
if not entries or not start_date:
    st.warning("Please set a start date and create entries in **Enter Timesheet** first.")
    st.stop()

# ---- OCR backends ----
AVAILABLE = {}
PREFER_TESS = False
try:
    import pytesseract
    _ = pytesseract.get_tesseract_version()
    AVAILABLE["Tesseract (pytesseract)"] = "pytesseract"
    PREFER_TESS = True
except Exception:
    pass
try:
    import easyocr
    AVAILABLE["EasyOCR"] = "easyocr"
except Exception:
    pass

if not AVAILABLE:
    st.error(
        "No OCR backend available.\n\n"
        "Install one of:\n"
        "• **pytesseract** + Tesseract system binary (preferred for structured tables)\n"
        "• **easyocr** (tolerates handwriting better; larger install)\n"
    )
    st.stop()

backend_label = st.radio("OCR engine", list(AVAILABLE.keys()), index=0 if PREFER_TESS else 0, horizontal=True)
backend = AVAILABLE[backend_label]

st.markdown("""
**Assumptions (matching your sheet):**  
- Main body has **14 lines**; **every 2 lines = 1 day** (top + bottom).  
- **Top-left two cells** of the entire 14-line block show the starting day (e.g., `07` over `Sep`) → used as the **anchor date** for the 7-day preview (auto-increments 6 more days).  
- Next column = **Sick/Off/ADO**: we detect **SL** (Sick), **OFF**, and **ADO** (case-insensitive).  
- Times (left→right): **Rostered ON**, **Actual ON**, *(skip 3 cols)*, **Rostered OFF**, **Actual OFF**.  
- **Rightmost time-like value** on the day = **Worked**.
""")

# ---- Where to place the 7 detected days in your 14-day period ----
slot = st.selectbox("Apply these 7 days to:", ["Days 1–7 (first week)", "Days 8–14 (second week)"], index=0)
start_day_index = 0 if slot.startswith("Days 1") else 7
if "scan_target_day" in st.session_state:
    start_day_index = 0 if int(st.session_state["scan_target_day"]) < 7 else 7

# ---- Inputs: camera OR upload ----
col_cam, col_up = st.columns(2)
with col_cam:
    photo = st.camera_input("📸 Capture the full 7-day page")
with col_up:
    upload = st.file_uploader("Or upload a photo (JPG/PNG)", type=["jpg", "jpeg", "png"])

img = None
if photo:
    img = Image.open(photo)
elif upload:
    img = Image.open(upload)

if not img:
    st.info("Take a photo or upload an image to continue.")
    st.stop()

st.image(img, caption="Selected image", use_container_width=True)

# ---------- Regex + helpers ----------
MONTH_MAP = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "SEPT": 9, "OCT": 10, "NOV": 11, "DEC": 12
}
TIME_COLON = re.compile(r"\b([0-2]?\d:[0-5]\d)\b")
TIME_BLOCK = re.compile(r"\b(\d{3,4})\b")
DAY_NUM = re.compile(r"^(?:0?[1-9]|[12]\d|3[01])$")  # 1..31

SICK_TOKEN = re.compile(r"^SL$", re.IGNORECASE)
OFF_TOKEN  = re.compile(r"^OFF$", re.IGNORECASE)
ADO_TOKEN  = re.compile(r"^ADO$", re.IGNORECASE)

def norm_hhmm(text: str) -> str:
    """Normalize '7:30'→'07:30', '0730'→'07:30'; otherwise ''."""
    s = text.strip()
    m = TIME_COLON.fullmatch(s)
    if m:
        h, mm = s.split(":")
        if 0 <= int(h) <= 29 and 0 <= int(mm) <= 59:
            return f"{int(h):02d}:{int(mm):02d}"
        return ""
    m2 = TIME_BLOCK.fullmatch(s)
    if m2:
        raw = m2.group(1)
        h, mm = int(raw[:-2]), int(raw[-2:])
        if 0 <= h <= 29 and 0 <= mm <= 59:
            return f"{h):02d}:{mm:02d}"  # small typo guard below fixes it
    return ""

# fix a tiny typo in f-string above if it happens
def safe_norm(text: str) -> str:
    out = norm_hhmm(text)
    if out and ")" in out:
        h, mm = out.replace(")", "").split(":")
        return f"{int(h):02d}:{int(mm):02d}"
    return out

def extract_month(token: str) -> Optional[int]:
    t = token.strip().upper()
    t = re.sub(r"[^A-Z]", "", t)
    return MONTH_MAP.get(t)

def ocr_tesseract_dataframe(image: Image.Image) -> pd.DataFrame:
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DATAFRAME)
    data = data.dropna(subset=["text"]).copy()
    data["text"] = data["text"].astype(str).str.strip()
    data = data[data["text"] != ""]
    if data.empty:
        return data
    cols = ["left","top","width","height","conf","text","block_num","par_num","line_num"]
    data = data[cols].sort_values(by=["block_num","par_num","line_num","left"]).reset_index(drop=True)
    return data

def ocr_easyocr_dataframe(image: Image.Image) -> pd.DataFrame:
    reader = easyocr.Reader(['en'], gpu=False)
    arr = np.array(image)
    results = reader.readtext(arr, detail=1, paragraph=False)  # [(box, text, conf), ...]
    rows = []
    for (box, txt, conf) in results:
        xs = [p[0] for p in box]; ys = [p[1] for p in box]
        left, top = min(xs), min(ys)
        width, height = max(xs) - left, max(ys) - top
        rows.append({
            "left": int(left), "top": int(top), "width": int(width), "height": int(height),
            "conf": float(conf) * 100, "text": str(txt).strip(),
            "block_num": 0, "par_num": 0, "line_num": 0
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Build rough line numbers by y-clustering
    df = df.sort_values(by=["top","left"]).reset_index(drop=True)
    y_tol = max(8, int(df["height"].median() or 10))
    line_id = 0; prev_y = None; ids = []
    for _, r in df.iterrows():
        if prev_y is None or abs(r["top"] - prev_y) > y_tol:
            line_id += 1; prev_y = r["top"]
        ids.append(line_id)
    df["line_num"] = ids
    return df

# ---- Run OCR to dataframe of words with positions ----
with st.spinner(f"OCR via {backend_label}…"):
    if backend == "pytesseract":
        df = ocr_tesseract_dataframe(img.convert("L"))
    else:
        df = ocr_easyocr_dataframe(img)

if df.empty:
    st.error("OCR returned no text. Try a clearer, flatter image.")
    st.stop()

# Group by visual lines
lines = []
for (b, p, ln), g in df.groupby(["block_num","par_num","line_num"], sort=True):
    tokens = g.sort_values(by="left").to_dict("records")
    lines.append([{"text": r["text"], "x": r["left"], "y": r["top"]} for r in tokens])

# Sort lines top→bottom
lines = sorted(lines, key=lambda row: np.median([t["y"] for t in row]))
# Keep 14 lines if more
if len(lines) > 14:
    lines = lines[:14]

# ---- Pair every 2 lines → 1 day ----
pairs: List[Tuple[List[Dict], List[Dict]]] = []
for i in range(0, len(lines), 2):
    if i+1 < len(lines):
        pairs.append((lines[i], lines[i+1]))
    else:
        pairs.append((lines[i], []))

# ---------- FIRST-PAIR DATE ANCHOR ----------
def first_pair_anchor(pairs) -> Optional[datetime]:
    """Use the very first pair's top-left two cells (e.g., '07' over 'Sep') to form the starting date."""
    if not pairs:
        return None
    top, bot = pairs[0]
    top_tokens = sorted(top, key=lambda t: t["x"])
    bot_tokens = sorted(bot, key=lambda t: t["x"]) if bot else []

    # Find first day number on top line
    day_num = None
    for t in top_tokens:
        if DAY_NUM.fullmatch(re.sub(r"\D","", t["text"])):
            day_num = re.sub(r"\D","", t["text"])
            break
    # Find first month token on bottom line
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
        # if the anchor is far (> 60 days) from the user-selected start_date, try previous/next year
        if abs((anch.date() - start_date).days) > 60:
            try_alt = datetime(year=y-1, month=month_num, day=int(day_num))
            if abs((try_alt.date() - start_date).days) < abs((anch.date() - start_date).days):
                anch = try_alt
            else:
                try_alt2 = datetime(year=y+1, month=month_num, day=int(day_num))
                if abs((try_alt2.date() - start_date).days) < abs((anch.date() - start_date).days):
                    anch = try_alt2
        return anch
    except Exception:
        return None

anchor_dt = first_pair_anchor(pairs)
if anchor_dt:
    st.info(f"Detected starting date from top-left cells: **{anchor_dt.strftime('%Y-%m-%d')}**")
else:
    st.warning("Could not confidently detect the starting date from the top-left cells. "
               "Dates in the preview will fall back to your app's start date.")

# ---------- PER-DAY EXTRACTION ----------
def flags_from_pair(top_line: List[Dict], bottom_line: List[Dict]) -> Dict[str, bool]:
    """Detect ADO / OFF / SICK (SL); precedence in your app later is ADO > Sick/Off."""
    toks = top_line + bottom_line
    has_ado = any(ADO_TOKEN.fullmatch(t["text"]) for t in toks)
    has_off = any(OFF_TOKEN.fullmatch(t["text"]) for t in toks)
    has_sl  = any(SICK_TOKEN.fullmatch(t["text"]) for t in toks)
    return {"ado": has_ado, "off": has_off, "sick": has_sl}

def times_from_pair(top_line: List[Dict], bottom_line: List[Dict]) -> Dict[str, str]:
    """Collect time-like tokens across both lines; rightmost = Worked; first two = R_on/A_on; next two = R_off/A_off."""
    toks = sorted(top_line + bottom_line, key=lambda t: t["x"])
    cand = []
    for t in toks:
        n = safe_norm(t["text"])
        if n:
            cand.append((n, t["x"]))
    if not cand:
        return {"rs_on":"","as_on":"","rs_off":"","as_off":"","worked":""}

    # Worked = rightmost
    worked_idx = max(range(len(cand)), key=lambda i: cand[i][1])
    worked = cand[worked_idx][0]

    # Remaining in left-to-right order
    remaining = [c for i, c in enumerate(cand) if i != worked_idx]
    remaining = sorted(remaining, key=lambda z: z[1])

    rs_on = remaining[0][0] if len(remaining) >= 1 else ""
    as_on = remaining[1][0] if len(remaining) >= 2 else ""
    rs_off = remaining[2][0] if len(remaining) >= 3 else ""
    as_off = remaining[3][0] if len(remaining) >= 4 else ""

    return {"rs_on": rs_on, "as_on": as_on, "rs_off": rs_off, "as_off": as_off, "worked": worked}

# Build 7 days from 14 lines
raw_days: List[Dict] = []
for i, (top, bot) in enumerate(pairs[:7]):
    f = flags_from_pair(top, bot)
    t = times_from_pair(top, bot)
    raw_days.append({**f, **t})

# Build preview dates from anchor (if available) or fall back to start_date + i
dates_for_preview = []
for i in range(len(raw_days)):
    if anchor_dt:
        dates_for_preview.append((anchor_dt + timedelta(days=i)).strftime("%Y-%m-%d"))
    else:
        dates_for_preview.append((start_date + timedelta(days=i)).strftime("%Y-%m-%d"))

# ---- Preview & edit table ----
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

st.markdown("### Adjust any values before applying")
edited: List[Dict] = []
for i, d in enumerate(raw_days, start=1):
    st.markdown(f"**Day {i}: {dates_for_preview[i-1]}**")
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

st.markdown("---")
st.caption("Extra is not set by OCR. You can adjust Extra or any flags later on the **Enter Timesheet** page.")

def apply_week():
    # Write into entries[start_day_index .. start_day_index+6]
    for i in range(7):
        target = start_day_index + i
        if target >= len(entries): break
        if i >= len(edited): break
        ed = edited[i]
        # Update fields; keep existing 'extra' unless you want to clear it
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

if st.button(f"Apply to { 'Days 1–7' if start_day_index == 0 else 'Days 8–14' } ✅", type="primary"):
    apply_week()
    st.success("Saved 7 days into your 14-day entries. Open **Enter Timesheet** or **Review Calculations** to continue.")

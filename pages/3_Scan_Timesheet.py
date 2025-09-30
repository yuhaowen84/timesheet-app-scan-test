# pages/3_Scan_Timesheet.py
import re
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from datetime import datetime

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
**What this expects (based on your sheet photo):**  
- The main body has **14 lines**; **every 2 lines = 1 day**.  
- Leftmost column: two stacked cells → **top = day (e.g., 07)**, **bottom = month (e.g., Sep)**.  
- Next column: **Sick/Off/ADO**, we only treat **`SL`** as **Sick**.  
- Then: **Rostered ON**, **Actual ON**, *(skip 3 columns)*, **Rostered OFF**, **Actual OFF**.  
- **Rightmost column** is **Worked** time.  
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
            return f"{h:02d}:{mm:02d}"
    return ""

def extract_month(token: str) -> Optional[int]:
    t = token.strip().upper()
    t = re.sub(r"[^A-Z]", "", t)  # keep letters only
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
    line_id = 0
    prev_y = None
    ids = []
    for _, r in df.iterrows():
        if prev_y is None or abs(r["top"] - prev_y) > y_tol:
            line_id += 1
            prev_y = r["top"]
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

# Group by visual lines; keep **14 lines** (main body) if possible
lines = []
for (b, p, ln), g in df.groupby(["block_num","par_num","line_num"], sort=True):
    tokens = g.sort_values(by="left").to_dict("records")
    # we only need text + x for ordering
    lines.append([{"text": r["text"], "x": r["left"], "y": r["top"]} for r in tokens])

# Keep the 14 most central lines by y (heuristic: main body). If more, slice to first 14.
lines = sorted(lines, key=lambda row: np.median([t["y"] for t in row]))  # top→bottom
if len(lines) > 14:
    lines = lines[:14]

if len(lines) < 14:
    st.warning(f"Detected only {len(lines)} lines. The sheet body should have 14. I’ll proceed by pairing what I have, but review carefully.")

# ---- Pair every 2 lines → 1 day ----
pairs: List[Tuple[List[Dict], List[Dict]]] = []
for i in range(0, len(lines), 2):
    if i+1 < len(lines):
        pairs.append((lines[i], lines[i+1]))
    else:
        pairs.append((lines[i], []))  # last odd line, best effort

def extract_day_from_pair(top_line: List[Dict], bottom_line: List[Dict]) -> Dict:
    """
    For a pair:
      - Date = (day from top line) + (month from bottom line), inferred year = start_date.year
      - Sick = 'SL' anywhere in the pair
      - Times: pick from combined tokens (left-to-right)
          * Worked = rightmost time-like token
          * Remaining time-like tokens (left-to-right): first two = R_on, A_on; next two = R_off, A_off
    """
    # Left-to-right tokens (text, x) for both lines
    top_tokens = sorted(top_line, key=lambda t: t["x"])
    bot_tokens = sorted(bottom_line, key=lambda t: t["x"]) if bottom_line else []

    # 1) Date parts
    day_num: Optional[str] = None
    for t in top_tokens:
        if DAY_NUM.fullmatch(t["text"]):
            day_num = t["text"]
            break
    month_num: Optional[int] = None
    for t in bot_tokens:
        m = extract_month(t["text"])
        if m:
            month_num = m
            break

    # Fallbacks: if month not found on bottom, try top; if still none, use start_date.month
    if month_num is None:
        for t in top_tokens:
            m = extract_month(t["text"])
            if m:
                month_num = m
                break
    if month_num is None:
        month_num = start_date.month
    # Fallback day
    if day_num is None:
        # take the leftmost 1–2 digit number on either line as day
        for t in top_tokens + bot_tokens:
            s = re.sub(r"\D", "", t["text"])
            if s and len(s) <= 2 and DAY_NUM.fullmatch(s):
                day_num = s
                break

    # Build date string DD/MM/YYYY
    try:
        d = int(day_num) if day_num else start_date.day
        m = int(month_num)
        y = start_date.year
        date_obj = datetime(year=y, month=m, day=max(1, min(31, d)))
        date_str = date_obj.strftime("%Y-%m-%d")
    except Exception:
        date_str = ""  # if we truly can't form it, leave blank

    # 2) Sick detection: any 'SL' tokens in either line
    sick = any(SICK_TOKEN.fullmatch(t["text"]) for t in (top_tokens + bot_tokens))

    # 3) Times
    # Collect candidate times with positions
    all_tokens = top_tokens + bot_tokens
    time_candidates = []
    for t in all_tokens:
        n = norm_hhmm(t["text"])
        if n:
            time_candidates.append((n, t["x"]))

    # If none, leave blanks
    ron = aon = roff = aoff = worked = ""

    if time_candidates:
        # Worked = rightmost by x
        worked_idx = max(range(len(time_candidates)), key=lambda i: time_candidates[i][1])
        worked = time_candidates[worked_idx][0]

        # Remaining (left-to-right)
        remaining = [tc for i, tc in enumerate(time_candidates) if i != worked_idx]
        remaining = sorted(remaining, key=lambda t: t[1])

        # First two = ON times, next two = OFF times
        if remaining:
            ron = remaining[0][0]
        if len(remaining) >= 2:
            aon = remaining[1][0]
        if len(remaining) >= 3:
            roff = remaining[2][0]
        if len(remaining) >= 4:
            aoff = remaining[3][0]

    return {
        "date_str": date_str,
        "sick": sick,
        "rs_on": ron, "as_on": aon,
        "rs_off": roff, "as_off": aoff,
        "worked": worked
    }

days: List[Dict] = []
for i, (top, bot) in enumerate(pairs):
    d = extract_day_from_pair(top, bot)
    days.append(d)

# Keep only 7 days
days = days[:7]

# ---- Preview & edit table ----
st.subheader("Detected (editable) 7-day preview")
df_prev = pd.DataFrame([
    {
        "Date": d["date_str"] or "(unknown)",
        "Sick (SL)": "Yes" if d["sick"] else "No",
        "R Sign-on": d["rs_on"],
        "A Sign-on": d["as_on"],
        "R Sign-off": d["rs_off"],
        "A Sign-off": d["as_off"],
        "Worked": d["worked"],
    } for d in days
])
st.dataframe(df_prev, use_container_width=True)

st.markdown("### Adjust any values before applying")
edited: List[Dict] = []
for i, d in enumerate(days, start=1):
    st.markdown(f"**Day {i}: {d['date_str'] or '(unknown date)'}**")
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    date_str = c1.text_input("Date (YYYY-MM-DD)", value=d["date_str"], key=f"date_{i}")
    sick     = c2.checkbox("Sick (SL)", value=d["sick"], key=f"sick_{i}")
    rs_on    = c3.text_input("R Sign-on", value=d["rs_on"], key=f"ron_{i}")
    as_on    = c4.text_input("A Sign-on", value=d["as_on"], key=f"aon_{i}")
    rs_off   = c5.text_input("R Sign-off", value=d["rs_off"], key=f"roff_{i}")
    as_off   = c6.text_input("A Sign-off", value=d["as_off"], key=f"aoff_{i}")
    worked   = c7.text_input("Worked", value=d["worked"], key=f"work_{i}")
    edited.append({
        "date_str": date_str.strip(),
        "sick": bool(sick),
        "rs_on": rs_on.strip(), "as_on": as_on.strip(),
        "rs_off": rs_off.strip(), "as_off": as_off.strip(),
        "worked": worked.strip()
    })

st.markdown("---")
st.caption("Extra / ADO / Off are not set by OCR. You can adjust them later on the **Enter Timesheet** page.")

def apply_week():
    # Write into entries[start_day_index .. start_day_index+6]
    for i in range(7):
        target = start_day_index + i
        if target >= len(entries): break
        if i >= len(edited): break
        ed = edited[i]
        # Keep existing 'extra', 'off', 'ado' untouched; set sick from OCR
        entries[target].update({
            "rs_on": ed["rs_on"],
            "as_on": ed["as_on"],
            "rs_off": ed["rs_off"],
            "as_off": ed["as_off"],
            "worked": ed["worked"],
            "sick": ed["sick"],
            # Preserve existing flags unless you want OCR to override them:
            "off": entries[target].get("off", False),
            "ado": entries[target].get("ado", False),
        })
        # If date_str present, keep for display (does not change app's start_date logic)
        if ed["date_str"]:
            entries[target]["date_str"] = ed["date_str"]
    st.session_state["entries"] = entries

if st.button(f"Apply to { 'Days 1–7' if start_day_index == 0 else 'Days 8–14' } ✅", type="primary"):
    apply_week()
    st.success("Saved 7 days into your 14-day entries. Open **Enter Timesheet** or **Review Calculations** to continue.")

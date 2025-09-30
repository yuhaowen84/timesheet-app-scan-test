# pages/3_Scan_Timesheet.py
import re
from typing import List, Tuple, Dict, Optional
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

st.title("3) Scan Weekly Timesheet (14 lines → 7 days)")

# ---- Ensure session state exists (created on Enter Timesheet page) ----
entries = st.session_state.get("entries")
start_date = st.session_state.get("start_date")
if not entries or not start_date:
    st.warning("Please set a start date and create entries in **Enter Timesheet** first.")
    st.stop()

# ---- EasyOCR only ----
try:
    import easyocr
except Exception:
    st.error(
        "EasyOCR is not installed. Install it (and its deps) and rerun:\n\n"
        "```bash\npip install easyocr\n```"
    )
    st.stop()

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
        try:
            h_i, m_i = int(h), int(mm)
            if 0 <= h_i <= 29 and 0 <= m_i <= 59:
                return f"{h_i:02d}:{m_i:02d}"
        except:
            pass
        return ""
    m2 = TIME_BLOCK.fullmatch(s)
    if m2:
        raw = m2.group(1)
        try:
            h_i, m_i = int(raw[:-2]), int(raw[-2:])
            if 0 <= h_i <= 29 and 0 <= m_i <= 59:
                return f"{h_i:02d}:{m_i:02d}"
        except:
            return ""
    return ""

def extract_month(token: str) -> Optional[int]:
    t = token.strip().upper()
    t = re.sub(r"[^A-Z]", "", t)
    return MONTH_MAP.get(t)

def ocr_easyocr_dataframe(image: Image.Image) -> pd.DataFrame:
    """Run EasyOCR and convert to a dataframe with rough line IDs via y-clustering."""
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
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values(by=["top","left"]).reset_index(drop=True)
    # Assign rough line numbers by grouping close y values
    y_tol = max(8, int(df["height"].median() or 10))
    line_id = 0; prev_y = None; ids = []
    for _, r in df.iterrows():
        if prev_y is None or abs(r["top"] - prev_y) > y_tol:
            line_id += 1
            prev_y = r["top"]
        ids.append(line_id)
    df["line_num"] = ids
    return df

# ---- Run OCR via EasyOCR ----
with st.spinner("OCR via EasyOCR…"):
    df = ocr_easyocr_dataframe(img)

if df.empty:
    st.error("OCR returned no text. Try a clearer, flatter image.")
    st.stop()

# Group words by visual lines
lines: List[List[Dict]] = []
for ln, g in df.groupby(["line_num"], sort=True):
    tokens = g.sort_values(by="left").to_dict("records")
    lines.append([{"text": r["text"], "x": r["left"], "y": r["top"]} for r in tokens])

# Sort lines top→bottom
lines = sorted(lines, key=lambda row: np.median([t["y"] for t in row]))
# Keep first 14 lines if more
if len(lines) > 14:
    lines = lines[:14]

# Pair every 2 lines → 1 day
pairs: List[Tuple[List[Dict], List[Dict]]] = []
for i in range(0, len(lines), 2):
    if i + 1 < len(lines):
        pairs.append((lines[i], lines[i+1]))
    else:
        pairs.append((lines[i], []))  # last odd line: best effort

# ---------- FIRST-PAIR DATE ANCHOR ----------
def first_pair_anchor(pairs) -> Optional[datetime]:
    """Use first pair's top-left two cells ('07' over 'Sep') as the starting date."""
    if not pairs:
        return None
    top, bot = pairs[0]
    top_tokens = sorted(top, key=lambda t: t["x"])
    bot_tokens = sorted(bot, key=lambda t: t["x"]) if bot else []

    # Day number from top line (first small integer token)
    day_num = None
    for t in top_tokens:
        digits = re.sub(r"\D", "", t["text"])
        if digits and DAY_NUM.fullmatch(digits):
            day_num = digits
            break

    # Month from bottom line (first month token)
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
        # If anchor far from selected start_date, try ±1 year
        if abs((anch.date() - start_date).days) > 60:
            prev_y = datetime(year=y-1, month=month_num, day=int(day_num))
            next_y = datetime(year=y+1, month=month_num, day=int(day_num))
            anch = min([anch, prev_y, next_y], key=lambda d: abs((d.date() - start_date).days))
        return anch
    except Exception:
        return None

anchor_dt = first_pair_anchor(pairs)
if anchor_dt:
    st.info(f"Detected starting date from top-left cells: **{anchor_dt.strftime('%Y-%m-%d')}**")
else:
    st.warning("Could not confidently detect the starting date from the top-left cells. "
               "Preview dates will fall back to your app’s start date + i.")

# ---------- PER-DAY EXTRACTION ----------
def flags_from_pair(top_line: List[Dict], bottom_line: List[Dict]) -> Dict[str, bool]:
    """Detect ADO / OFF / SICK (SL)."""
    toks = top_line + bottom_line
    has_ado = any(ADO_TOKEN.fullmatch(t["text"]) for t in toks)
    has_off = any(OFF_TOKEN.fullmatch(t["text"]) for t in toks)
    has_sl  = any(SICK_TOKEN.fullmatch(t["text"]) for t in toks)
    return {"ado": has_ado, "off": has_off, "sick": has_sl}

def times_from_pair(top_line: List[Dict], bottom_line: List[Dict]) -> Dict[str, str]:
    """
    Collect time-like tokens across both lines:
      - rightmost time = Worked
      - remaining (left→right): first two = R_on/A_on; next two = R_off/A_off
    """
    toks = sorted(top_line + bottom_line, key=lambda t: t["x"])
    cand: List[Tuple[str, int]] = []
    for t in toks:
        n = norm_hhmm(t["text"])
        if n:
            cand.append((n, t["x"]))

    if not cand:
        return {"rs_on": "", "as_on": "", "rs_off": "", "as_off": "", "worked": ""}

    # Worked = rightmost by x
    worked_idx = max(range(len(cand)), key=lambda i: cand[i][1])
    worked = cand[worked_idx][0]

    # Remaining in left-to-right order
    remaining = [c for i, c in enumerate(cand) if i != worked_idx]
    remaining = sorted(remaining, key=lambda z: z[1])

    rs_on  = remaining[0][0] if len(remaining) >= 1 else ""
    as_on  = remaining[1][0] if len(remaining) >= 2 else ""
    rs_off = remaining[2][0] if len(remaining) >= 3 else ""
    as_off = remaining[3][0] if len(remaining) >= 4 else ""

    return {"rs_on": rs_on, "as_on": as_on, "rs_off": rs_off, "as_off": as_off, "worked": worked}

# Build 7 days from 14 lines
raw_days: List[Dict] = []
for i, (top, bot) in enumerate(pairs[:7]):
    f = flags_from_pair(top, bot)
    t = times_from_pair(top, bot)
    raw_days.append({**f, **t})

# Preview dates from anchor or fallback
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

btn_label = f"Apply to {'Days 1–7' if start_day_index == 0 else 'Days 8–14'} ✅"
if st.button(btn_label, type="primary"):
    apply_week()
    st.success("Saved 7 days into your 14-day entries. Open **Enter Timesheet** or **Review Calculations** to continue.")

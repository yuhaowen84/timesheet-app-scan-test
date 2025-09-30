# pages/3_Scan_Timesheet.py
import re
from typing import List, Dict, Tuple
import numpy as np
from PIL import Image
import streamlit as st
import pandas as pd

st.title("3) Scan Weekly Timesheet (7 rows) — Camera / Upload + OCR")

# ---- Ensure session (created by Enter Timesheet page) ----
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
        "• **pytesseract** + Tesseract system binary (preferred for layout)\n"
        "• **easyocr** (better handwriting tolerance, heavier install)\n"
    )
    st.stop()

backend_label = st.radio(
    "OCR engine",
    list(AVAILABLE.keys()),
    index=0 if PREFER_TESS else 0,
    horizontal=True
)
backend = AVAILABLE[backend_label]

st.markdown("""
**Tips for best results**
- Capture or upload the **whole 7-day page** (flat, well lit).
- The **leftmost column** should show the date/day number for each row.
- You can edit the detected values before saving.
""")

# ---- Target week placement in your 14-day period ----
st.markdown("### Where to put these 7 rows in your 14-day entry")
colw1, colw2 = st.columns(2)
week_slot = colw1.selectbox(
    "Apply to which days?",
    ["Days 1–7 (first week)", "Days 8–14 (second week)"],
    index=0
)
start_day_index = 0 if week_slot.startswith("Days 1") else 7

# (Optional) default day if you jumped from Enter page
if "scan_target_day" in st.session_state:
    # if user came from a specific day, align to that day’s week
    idx = int(st.session_state["scan_target_day"])
    start_day_index = 0 if idx < 7 else 7

# ---- Inputs: camera OR upload ----
col_cam, col_up = st.columns(2)
with col_cam:
    photo = st.camera_input("📸 Take a photo of the 7-day sheet")
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

st.image(img, caption="Selected image", use_column_width=True)

# ---------- Helpers ----------
TIME_RE_COLON = re.compile(r"\b([0-2]?\d:[0-5]\d)\b")
TIME_RE_BLOCK = re.compile(r"\b(\d{3,4})\b")
DATE_RE = re.compile(r"\b(?:(?:[0-3]?\d(?:/|-)[01]?\d)|(?:[0-3]?\d))\b")  # 15, 15/06, 15-6

def normalize_hhmm(s: str) -> str:
    s = s.strip()
    # already H:MM or HH:MM
    m = TIME_RE_COLON.fullmatch(s)
    if m:
        h, m = s.split(":")
        return f"{int(h):02d}:{int(m):02d}"
    # 3–4 digit block like 732 or 1332
    m2 = TIME_RE_BLOCK.fullmatch(s)
    if m2:
        raw = m2.group(1)
        h = int(raw[:-2]); mm = int(raw[-2:])
        if 0 <= h <= 29 and 0 <= mm <= 59:
            return f"{h:02d}:{mm:02d}"
    return ""

def pick_four_times(tokens_ordered: List[str]) -> List[str]:
    """Return up to 4 normalized times found left-to-right on a row."""
    out = []
    for t in tokens_ordered:
        n = normalize_hhmm(t)
        if n:
            out.append(n)
        if len(out) == 4:
            break
    # pad to always length 4
    while len(out) < 4:
        out.append("")
    return out

# ---------- OCR pipelines ----------
def ocr_with_tesseract(image: Image.Image) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
    ['left','top','width','height','conf','text','block_num','par_num','line_num']
    """
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DATAFRAME)
    data = data.dropna(subset=["text"]).copy()
    # Clean text
    data["text"] = data["text"].astype(str).str.strip()
    data = data[data["text"] != ""]
    # Keep important cols
    cols = ["left","top","width","height","conf","text","block_num","par_num","line_num"]
    data = data[cols].reset_index(drop=True)
    return data

def ocr_with_easyocr(image: Image.Image) -> pd.DataFrame:
    """
    EasyOCR returns boxes + text. Convert into a tesseract-like dataframe with line estimation using y center.
    """
    reader = easyocr.Reader(['en'], gpu=False)
    arr = np.array(image)
    results = reader.readtext(arr, detail=1, paragraph=False)  # [(box, text, conf), ...]
    rows = []
    for (box, txt, conf) in results:
        # box = [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        xs = [p[0] for p in box]; ys = [p[1] for p in box]
        left, top = min(xs), min(ys)
        width, height = max(xs) - left, max(ys) - top
        rows.append({
            "left": int(left), "top": int(top), "width": int(width), "height": int(height),
            "conf": float(conf)*100, "text": str(txt).strip()
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Rough line grouping via y bins
    df = df.sort_values(by=["top","left"]).reset_index(drop=True)
    # Create line ids by merging close y values
    y_tol = max(8, int(df["height"].median() or 10))  # tolerance by median height
    line_id = 0
    prev_y = None
    line_ids = []
    for _, r in df.iterrows():
        if prev_y is None or abs(r["top"] - prev_y) > y_tol:
            line_id += 1
            prev_y = r["top"]
        line_ids.append(line_id)
    df["block_num"] = 0; df["par_num"] = 0; df["line_num"] = line_ids
    return df

# ---------- Run OCR ----------
with st.spinner(f"OCR via {backend_label}…"):
    if backend == "pytesseract":
        df = ocr_with_tesseract(img.convert("L"))
    else:
        df = ocr_with_easyocr(img)

if df.empty:
    st.error("OCR returned no text. Try a clearer image.")
    st.stop()

# ---------- Group by lines & detect date anchors ----------
rows_detected: List[Dict] = []

if "block_num" not in df.columns:  # safety
    df["block_num"] = 0
if "par_num" not in df.columns:
    df["par_num"] = 0
if "line_num" not in df.columns:
    df["line_num"] = 0

# Sort by top,left for stable left-to-right
df = df.sort_values(by=["line_num","top","left"]).reset_index(drop=True)

for (b, p, ln), g in df.groupby(["block_num","par_num","line_num"], sort=True):
    tokens = g.sort_values(by="left")
    # Identify a date-like anchor (leftmost numeric token matching DATE_RE)
    anchor = None
    for _, w in tokens.iterrows():
        txt = w["text"]
        if DATE_RE.fullmatch(txt):
            anchor = txt
            break
    if not anchor:
        continue

    # Collect token texts left-to-right for that line (to the right of the anchor too)
    token_texts = tokens["text"].tolist()
    # Option: only times to the right of the anchor
    # Find index of anchor and use following tokens; fall back to all
    try:
        start_idx = token_texts.index(anchor) + 1
        row_tokens = token_texts[start_idx:]
        if len(row_tokens) < 1:
            row_tokens = token_texts
    except ValueError:
        row_tokens = token_texts

    times4 = pick_four_times(row_tokens)

    rows_detected.append({
        "anchor": anchor,
        "tokens": row_tokens,
        "times": times4
    })

# Keep first 7 anchored rows
rows_detected = rows_detected[:7]

st.subheader("Detected rows (preview)")
if not rows_detected:
    st.warning("No date-anchored rows were detected. Try cropping/straightening the image or increasing contrast.")
    st.stop()

# Build preview/edit table
preview_data = []
for i, rdet in enumerate(rows_detected, start=1):
    ron, aon, roff, aoff = rdet["times"]
    preview_data.append({
        "Row": i,
        "Date Anchor": rdet["anchor"],
        "Rostered On": ron,
        "Actual On": aon,
        "Rostered Off": roff,
        "Actual Off": aoff,
    })

df_prev = pd.DataFrame(preview_data)
st.dataframe(df_prev, use_container_width=True)

st.markdown("### Adjust any values before applying")
# Editable inputs per row
edited = []
for i, rdet in enumerate(rows_detected, start=1):
    st.markdown(f"**Row {i} — Anchor: {rdet['anchor']}**")
    c1, c2, c3, c4 = st.columns(4)
    ron  = c1.text_input("Rostered On",  value=rdet["times"][0], key=f"ron_{i}")
    aon  = c2.text_input("Actual On",    value=rdet["times"][1], key=f"aon_{i}")
    roff = c3.text_input("Rostered Off", value=rdet["times"][2], key=f"roff_{i}")
    aoff = c4.text_input("Actual Off",   value=rdet["times"][3], key=f"aoff_{i}")
    edited.append((ron.strip(), aon.strip(), roff.strip(), aoff.strip()))

st.markdown("---")
st.caption("Worked/Extra will be left blank (your logic defaults Worked to 8h if blank). Flags (Sick/Off/ADO) are not set by OCR.")

def apply_week():
    # Apply 7 rows to entries[start_day_index ... start_day_index+6]
    for i in range(7):
        target = start_day_index + i
        if target >= len(entries):
            break
        if i >= len(edited):
            break
        ron, aon, roff, aoff = edited[i]
        # Do not overwrite if all fields are empty
        if not any([ron, aon, roff, aoff]):
            continue
        entries[target].update({
            "rs_on": ron, "as_on": aon,
            "rs_off": roff, "as_off": aoff,
            "worked": entries[target].get("worked",""),
            "extra": entries[target].get("extra",""),
            "sick": entries[target].get("sick", False),
            "off": entries[target].get("off", False),
            "ado": entries[target].get("ado", False),
        })
    st.session_state["entries"] = entries

if st.button(f"Apply to { 'Days 1–7' if start_day_index == 0 else 'Days 8–14' } ✅", type="primary"):
    apply_week()
    st.success("Saved 7 rows into your 14-day entries. Open **Enter Timesheet** or **Review Calculations** to continue.")

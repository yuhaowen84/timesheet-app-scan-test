# pages/3_Scan_Timesheet.py
import io
import re
import numpy as np
from PIL import Image
import streamlit as st

st.title("3) Scan Timesheet Row (Camera / Upload + OCR)")

# ---- Ensure session state exists (created on Enter Timesheet page) ----
entries = st.session_state.get("entries")
start_date = st.session_state.get("start_date")
if not entries or not start_date:
    st.warning("Please set a start date and create entries in **Enter Timesheet** first.")
    st.stop()

# ---- Try available OCR backends (EasyOCR first, then pytesseract) ----
AVAILABLE = {}
try:
    import easyocr
    AVAILABLE["EasyOCR"] = "easyocr"
except Exception:
    pass

try:
    import pytesseract
    _ = pytesseract.get_tesseract_version()  # requires Tesseract binary installed
    AVAILABLE["Tesseract (pytesseract)"] = "pytesseract"
except Exception:
    pass

if not AVAILABLE:
    st.error(
        "No OCR backend is available.\n\n"
        "Install one of the following locally / on your server:\n"
        "• **EasyOCR** (better for handwriting): `pip install easyocr` (downloads PyTorch)\n"
        "• **pytesseract** + **Tesseract** system binary\n"
        "Then rerun this page."
    )
    st.stop()

backend_label = st.radio("OCR engine", list(AVAILABLE.keys()), horizontal=True)
backend = AVAILABLE[backend_label]

st.markdown("""
**How to scan (best results):**
- Capture or upload **one row** at a time; keep the image **flat**, **close**, and **well-lit**.
- Ensure the four times are visible: *Rostered Sign-on*, *Actual Sign-on*, *Rostered Sign-off*, *Actual Sign-off*.
- You can edit anything after OCR before saving.
""")

# Which day to fill
day_index = st.number_input("Which day are you scanning?", min_value=1, max_value=14, value=1) - 1
row = entries[day_index]
st.caption(f"Target day: **{row['weekday']} — {row['date_str']}**")

# ---- Inputs: camera OR upload (both supported) ----
col_cam, col_up = st.columns(2)
with col_cam:
    photo = st.camera_input("📸 Take a photo of this day's row")
with col_up:
    upload = st.file_uploader("Or upload a photo (JPG/PNG)", type=["jpg", "jpeg", "png"])

image = None
if photo:
    image = Image.open(photo)
elif upload:
    image = Image.open(upload)

if not image:
    st.info("Take a photo or upload an image to continue.")
    st.stop()

st.image(image, caption="Selected image", use_column_width=True)

# ---- Run OCR ----
with st.spinner(f"Running OCR via {backend_label}..."):
    text = ""
    if backend == "easyocr":
        reader = easyocr.Reader(['en'], gpu=False)
        arr = np.array(image)
        pieces = reader.readtext(arr, detail=0, paragraph=True)
        text = "\n".join(pieces)
    else:  # pytesseract
        gray = Image.fromarray(np.array(image).astype("uint8")).convert("L")
        try:
            text = pytesseract.image_to_string(gray)
        except Exception as e:
            st.error(f"OCR failed: {e}")
            st.stop()

st.subheader("OCR raw text")
st.text(text if text.strip() else "(no text recognized)")

# ---- Extract time candidates from OCR text ----
def normalize_hhmm_block(s: str) -> str:
    """Accept '1332' -> '13:32', '732' -> '07:32', or keep '13:32' normalized."""
    s = s.strip()
    if re.fullmatch(r"[0-2]?\d:[0-5]\d", s):
        h, m = s.split(":"); return f"{int(h):02d}:{int(m):02d}"
    if re.fullmatch(r"\d{3,4}", s):
        h = int(s[:-2]); m = int(s[-2:])
        if 0 <= h <= 29 and 0 <= m <= 59:
            return f"{h:02d}:{m:02d}"
    return ""

def find_time_candidates(blob: str):
    with_colon = re.findall(r"\b([0-2]?\d:[0-5]\d)\b", blob)
    blocks = re.findall(r"\b(\d{3,4})\b", blob)
    norm = []
    for s in with_colon + blocks:
        t = normalize_hhmm_block(s)
        if t and t not in norm:
            norm.append(t)
    return norm

candidates = find_time_candidates(text)
if not candidates:
    st.warning("I couldn't find obvious time strings. You can type them below.")
else:
    st.success(f"Found {len(candidates)} time candidate(s): {', '.join(candidates)}")

# ---- Mapping UI ----
def pick_time(label, default_guess=""):
    opts = ["— choose —"] + candidates
    idx = opts.index(default_guess) if default_guess in opts else 0
    choice = st.selectbox(label, opts, index=idx)
    return "" if choice == "— choose —" else choice

st.markdown("### Map detected times to fields")
guess = (candidates + ["", "", "", ""])[:4]  # naive default ordering
ron  = pick_time("Rostered Sign-on",  guess[0])
aon  = pick_time("Actual Sign-on",    guess[1])
roff = pick_time("Rostered Sign-off", guess[2])
aoff = pick_time("Actual Sign-off",   guess[3])

# Manual fallbacks if user picked "— choose —"
ron  = st.text_input("Rostered Sign-on (manual if needed)",  value=ron)
aon  = st.text_input("Actual Sign-on (manual if needed)",    value=aon)
roff = st.text_input("Rostered Sign-off (manual if needed)", value=roff)
aoff = st.text_input("Actual Sign-off (manual if needed)",   value=aoff)

# Optional worked/extra
c1, c2 = st.columns(2)
worked = c1.text_input("Worked (HH:MM or HHMM, blank → 8h)", value="")
extra  = c2.text_input("Extra (HH:MM or HHMM)", value="")

# Flags
t1, t2, t3 = st.columns(3)
sick = t1.checkbox("Sick", value=False)
off  = t2.checkbox("Off",  value=False)
ado  = t3.checkbox("ADO",  value=False)

# ---- Save into session state ----
def apply_to_day():
    entries[day_index].update({
        "rs_on": ron.strip(), "as_on": aon.strip(),
        "rs_off": roff.strip(), "as_off": aoff.strip(),
        "worked": worked.strip(), "extra": extra.strip(),
        "sick": bool(sick), "off": bool(off), "ado": bool(ado),
    })
    st.session_state["entries"] = entries

if st.button("Apply to this day ✅", type="primary"):
    apply_to_day()
    st.success(f"Saved OCR values to {row['weekday']} ({row['date_str']}).\nOpen **Enter Timesheet** or **Review Calculations** to continue.")

# Home.py
import streamlit as st
from utils import build_rate_constants, DEFAULT_BASE_RATES

st.set_page_config(page_title="Timesheet Calculator", page_icon="🗓️", layout="wide")

st.title("🗓️ Timesheet Calculator")

# ---------- Quick Guide ----------
st.markdown("""
### How to use
1. **Set your rates** (optional): enter your *Ordinary*, *Afternoon penalty*, and *Night penalty* below.  
   - 💡 *Hint:* Open your latest payslip and look for **Ordinary rate** and **Afternoon penalty** (and **Night** if shown).
2. Go to **Enter Timesheet** (left sidebar). Fill each day (Sick/Off/ADO toggles available).
3. Open **Review Calculations** to see totals and a downloadable breakdown.
""")

# ---------- Base-rate form ----------
st.subheader("Set Base Rates (optional)")
st.caption("Only three inputs are needed. All other rates are auto-derived using the same multipliers you already use.")

# Load prior values from session or use defaults
base = st.session_state.get("base_rates", DEFAULT_BASE_RATES.copy())

col1, col2, col3 = st.columns(3)
with col1:
    ordinary = col1.number_input("Ordinary rate (per hour, AUD)", min_value=0.0, value=float(base["ordinary"]), step=0.01, format="%.2f")
with col2:
    afternoon = col2.number_input("Afternoon penalty (per hour, AUD)", min_value=0.0, value=float(base["afternoon_penalty"]), step=0.01, format="%.2f")
with col3:
    night = col3.number_input("Night penalty (per hour, AUD)", min_value=0.0, value=float(base["night_penalty"]), step=0.01, format="%.2f")

if st.button("Save Rates"):
    new_base = {
        "ordinary": float(ordinary),
        "afternoon_penalty": float(afternoon),
        "night_penalty": float(night),
    }
    st.session_state["base_rates"] = new_base
    st.session_state["rate_constants"] = build_rate_constants(new_base)
    st.success("Rates saved. Calculations will use your custom rates.")

# Show current effective rates (read-only preview)
effective = st.session_state.get("rate_constants", build_rate_constants(base))
with st.expander("Show derived rates (read-only preview)"):
    st.json(effective)

# Links to pages (optional: handy on desktop)
st.page_link("pages/1_Enter_Timesheet.py", label="➡️ Enter Timesheet")
st.page_link("pages/2_Review_Calculations.py", label="➡️ Review Calculations")

# pages/2_Review_Calculations.py
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta, time
from utils import (
    parse_time, parse_duration, calculate_row,
    NSW_PUBLIC_HOLIDAYS, rate_constants as DEFAULT_RATES
)

st.title("2) Review Calculations")

entries = st.session_state.get("entries")
start_date = st.session_state.get("start_date")

if not entries or not start_date:
    st.warning("No saved entries found. Please fill **Enter Timesheet** first.")
    st.stop()

# Use custom rates if set on Home, otherwise defaults
rates = st.session_state.get("rate_constants", DEFAULT_RATES)

rows = []
any_ado = False

for r in entries:
    weekday   = r["weekday"]
    date_str  = r["date_str"]
    date      = datetime.strptime(date_str, "%Y-%m-%d").date()

    values = [r["rs_on"], r["as_on"], r["rs_off"], r["as_off"], r["worked"], r["extra"]]
    sick = bool(r["sick"])
    off  = bool(r["off"])
    ado  = bool(r["ado"])

    # Precedence: ADO > Sick/Off > none
    effective_values = values.copy()
    chosen_flag = None
    if ado:
        effective_values[0] = "ADO"; chosen_flag = "ADO"
    elif sick or off:
        effective_values[0] = "OFF"; chosen_flag = "OFF"

    # -------- Unit (original logic) --------
    unit = 0.0
    if any(v.upper() in ["OFF", "ADO"] for v in effective_values) or sick:
        unit = 0.0
    else:
        RS_ON  = parse_time(effective_values[0])
        AS_ON  = parse_time(effective_values[1])
        RS_OFF = parse_time(effective_values[2])
        AS_OFF = parse_time(effective_values[3])
        worked_f = parse_duration(effective_values[4])
        extra_f  = parse_duration(effective_values[5])

        if RS_ON and RS_OFF and AS_ON and AS_OFF:
            rs_start = datetime.combine(date, RS_ON)
            rs_end   = datetime.combine(date, RS_OFF)
            if RS_OFF < RS_ON:
                rs_end += timedelta(days=1)

            as_start = datetime.combine(date, AS_ON)
            as_end   = datetime.combine(date, AS_OFF)
            if AS_OFF < AS_ON:
                as_end += timedelta(days=1)

            built_up = 0
            if as_start < rs_start:        # lift-up
                delta = (rs_end - as_end).total_seconds() / 3600
            elif as_end > rs_end:          # lay-back
                delta = abs((as_start - rs_start).total_seconds() / 3600)
            elif (as_start >= rs_start and as_end <= rs_end
                  and (as_end - as_start) < (rs_end - rs_start)):  # built-up
                delta = abs((rs_end - rs_start).total_seconds() / 3600) - 8
                built_up = 1
            else:
                delta = 0.0

            worked_use = worked_f if worked_f and built_up == 0 else 8
            unit = delta + (worked_use - 8) + (extra_f or 0)
        else:
            unit = 0.0

    unit = round(unit, 2)

    # -------- Penalty --------
    penalty = "No"
    AS_ON  = parse_time(effective_values[1])
    AS_OFF = parse_time(effective_values[3])
    if (not any(v.upper() in ["OFF","ADO"] for v in effective_values)
        and not sick and AS_ON and AS_OFF and weekday not in ["Saturday","Sunday"]):
        m1 = AS_ON.hour*60 + AS_ON.minute
        m2 = AS_OFF.hour*60 + AS_OFF.minute
        if m2 < m1: m2 += 1440
        if 1080 <= m1 % 1440 <= 1439 or 0 <= m1 % 1440 <= 239:
            penalty = "Night"
        elif 240 <= m1 % 1440 <= 330:
            penalty = "Morning"
        elif m1 <= 1080 <= m2:
            penalty = "Afternoon"

    # -------- Special --------
    special = "No"
    if (not any(v.upper() in ["OFF","ADO"] for v in effective_values)
        and not sick and weekday not in ["Saturday","Sunday"]):
        if (AS_ON and time(1,1) <= AS_ON <= time(3,59)) or (AS_OFF and time(1,1) <= AS_OFF <= time(3,59)):
            special = "Yes"

    # -------- Rates (pass effective rates) --------
    ot, prate, sload, srate, drate, lrate, dcount = calculate_row(
        weekday, effective_values, sick, penalty, special, unit, rates=rates
    )

    if any(v.upper() == "ADO" for v in effective_values):
        any_ado = True

    is_holiday = "Yes" if date_str in NSW_PUBLIC_HOLIDAYS else "No"
    display_rs_on = chosen_flag if chosen_flag else values[0]

    rows.append([
        weekday, date_str, display_rs_on, values[1], values[2], values[3], values[4], values[5],
        "Yes" if sick else "No", f"{unit:.2f}", penalty, special, is_holiday,
        f"{ot:.2f}", f"{prate:.2f}", f"{sload:.2f}", f"{srate:.2f}",
        f"{lrate:.2f}", f"{drate:.2f}", f"{dcount:.2f}"
    ])

# Build table
cols = [
    "Weekday","Date","R Sign-on","A Sign-on","R Sign-off","A Sign-off","Worked","Extra","Sick",
    "Unit","Penalty","Special","Holiday",
    "OT Rate","Penalty Rate","Special Ldg","Sick Rate","Loading","Daily Rate","Daily Count"
]
df = pd.DataFrame(rows, columns=cols)

# Totals as floats
totals_float = [pd.to_numeric(df[c], errors="coerce").fillna(0).sum() for c in cols[13:]]

# Long-fortnight deduction if no ADO anywhere (use effective rates)
if not any_ado:
    deduction = 0.5 * rates["Ordinary Hours"] * 8  # half a day of ordinary
    totals_float[-1] -= deduction
    st.warning(f"Applied long-fortnight deduction: -{deduction:.2f}")

# TOTAL row (formatted)
totals_fmt = [f"{t:.2f}" for t in totals_float]
df.loc[len(df)] = ["TOTAL","","","","","","","","","", "", "", ""] + totals_fmt

# -------- Summary header --------
start_str = start_date.strftime("%Y-%m-%d")
end_str = (start_date + timedelta(days=13)).strftime("%Y-%m-%d")
total_amount = totals_float[-1]

st.success(
    f"🎉 Congrats! For this fortnight ({start_str} → {end_str}), you have earned:\n\n"
    f"💰 **${total_amount:,.2f} AUD** before tax!"
)

st.markdown("---")
st.subheader("Here is the detailed breakdown of earnings")

# Style: highlight TOTAL row
def highlight_total(row):
    return ['background-color: #d0ffd0' if row.name == len(df)-1 else '' for _ in row]

st.dataframe(df.style.apply(highlight_total, axis=1), use_container_width=True)

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta, time
import math

# ---------- Constants ----------
NSW_PUBLIC_HOLIDAYS = {
    "2025-01-01", "2025-01-27", "2025-04-18", "2025-04-19", "2025-04-20", "2025-04-21",
    "2025-04-25", "2025-06-09", "2025-10-06", "2025-12-25", "2025-12-26"
}
rate_constants = {
    "Afternoon Shift": 4.84, "Night Shift": 5.69, "Early Morning": 4.84,
    "Special Loading": 5.69, "OT 150%": 74.72763, "OT 200%": 99.63684,
    "ADO Adjustment": 49.81842, "Sat Loading 50%": 24.90921, "Sun Loading 100%": 49.81842,
    "Public Holiday": 49.81842, "PH Loading 50%": 24.90921, "PH Loading 100%": 49.81842,
    "Sick With MC": 49.81842, "Ordinary Hours": 49.81842
}

# ---------- Helpers ----------
def parse_time(text: str):
    text = (text or "").strip()
    if not text:
        return None
    if ":" in text:
        try:
            return datetime.strptime(text, "%H:%M").time()
        except:
            return None
    if text.isdigit() and len(text) in [3, 4]:
        try:
            h = int(text[:-2]); m = int(text[-2:])
            return time(hour=h, minute=m)
        except:
            return None
    return None

def parse_duration(text: str) -> float:
    text = (text or "").strip()
    if not text:
        return 0
    if ":" in text:
        try:
            h, m = map(int, text.split(":"))
            return h + m / 60
        except:
            return 0
    if text.isdigit():
        try:
            h = int(text[:-2]); m = int(text[-2:])
            return h + m / 60
        except:
            return 0
    return 0

def calculate_row(day, values, sick, penalty_value, special_value, unit_val):
    # values: [rs_on, as_on, rs_off, as_off, worked, extra]
    ot_rate = 0
    if values[0].upper() == "ADO" and unit_val >= 0:
        ot_rate = round(unit_val * rate_constants["ADO Adjustment"], 2)
    elif values[0].upper() not in ["OFF", "ADO"] and unit_val >= 0:
        if day in ["Saturday", "Sunday"]:
            ot_rate = round(unit_val * rate_constants["OT 200%"], 2)
        else:
            ot_rate = round(unit_val * rate_constants["OT 150%"], 2)
    else:
        # negative or OFF/ADO -> ordinary + applicable loading
        if day == "Saturday":
            ot_rate = round(unit_val * rate_constants["Sat Loading 50%"] + unit_val * rate_constants["Ordinary Hours"], 2)
        elif day == "Sunday":
            ot_rate = round(unit_val * rate_constants["Sun Loading 100%"] + unit_val * rate_constants["Ordinary Hours"], 2)
        else:
            if penalty_value in ["Afternoon", "Morning"]:
                ot_rate = round(unit_val * rate_constants["Afternoon Shift"] + unit_val * rate_constants["Ordinary Hours"], 2)
            elif penalty_value == "Night":
                ot_rate = round(unit_val * rate_constants["Night Shift"] + unit_val * rate_constants["Ordinary Hours"], 2)
            else:
                ot_rate = round(unit_val * rate_constants["Ordinary Hours"], 2)

    # Penalty hours: floor(worked), default 8 if blank/invalid
    worked_hours = parse_duration(values[4])
    if worked_hours == 0:
        worked_hours = 8
    penalty_hours = math.floor(worked_hours)

    penalty_rate = 0
    if penalty_value == "Afternoon":
        penalty_rate = round(penalty_hours * rate_constants["Afternoon Shift"], 2)
    elif penalty_value == "Night":
        penalty_rate = round(penalty_hours * rate_constants["Night Shift"], 2)
    elif penalty_value == "Morning":
        penalty_rate = round(penalty_hours * rate_constants["Early Morning"], 2)

    special_loading = round(rate_constants["Special Loading"], 2) if special_value == "Yes" else 0
    sick_rate = round(8 * rate_constants["Sick With MC"], 2) if sick else 0
    daily_rate = 0 if values[0].upper() in ["OFF", "ADO"] else round(8 * rate_constants["Ordinary Hours"], 2)

    if any(v.upper() == "ADO" for v in values):
        daily_rate += round(4 * rate_constants["Ordinary Hours"], 2)

    loading = 0
    if values[0].upper() not in ["OFF", "ADO"]:
        if day == "Saturday":
            loading = round(8 * rate_constants["Sat Loading 50%"], 2)
        elif day == "Sunday":
            loading = round(8 * rate_constants["Sun Loading 100%"], 2)

    daily_count = ot_rate + penalty_rate + special_loading + sick_rate + daily_rate + loading
    return ot_rate, penalty_rate, special_loading, sick_rate, daily_rate, loading, daily_count

# ---------- App ----------
st.title("📊 Timesheet Calculator (14-day Fortnight • Web)")

start_date = st.date_input("Select Start Date")

if start_date:
    rows = []
    any_ado = False

    with st.form("timesheet_form"):
        st.caption("Enter HH:MM (e.g., 07:30) or HHMM (e.g., 0730). Worked/Extra accept HH:MM or HHMM. Worked defaults to 8h if blank.")

        for i in range(14):
            date = start_date + timedelta(days=i)
            weekday = date.strftime("%A")
            date_str = date.strftime("%Y-%m-%d")

            st.markdown(f"**{weekday} {date_str}**")
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            rs_on = c1.text_input("R Sign-on", key=f"rs_on_{i}")
            as_on = c2.text_input("A Sign-on", key=f"as_on_{i}")
            rs_off = c3.text_input("R Sign-off", key=f"rs_off_{i}")
            as_off = c4.text_input("A Sign-off", key=f"as_off_{i}")
            worked = c5.text_input("Worked", key=f"worked_{i}")
            extra  = c6.text_input("Extra",  key=f"extra_{i}")
            sick   = st.checkbox("Sick", key=f"sick_{i}")

            values = [rs_on.strip(), as_on.strip(), rs_off.strip(), as_off.strip(), worked.strip(), extra.strip()]

            # Holiday flag
            is_holiday = "Yes" if date_str in NSW_PUBLIC_HOLIDAYS else "No"

            # ---------- Unit  ----------
            unit = 0.0
            if any(v.upper() in ["OFF", "ADO"] for v in values) or sick:
                unit = 0.0
            else:
                RS_ON  = parse_time(values[0])  # R Sign-on
                AS_ON  = parse_time(values[1])  # A Sign-on
                RS_OFF = parse_time(values[2])  # R Sign-off
                AS_OFF = parse_time(values[3])  # A Sign-off
                worked_f = parse_duration(values[4])  # Worked
                extra_f  = parse_duration(values[5])  # Extra

                if RS_ON and RS_OFF and AS_ON and AS_OFF:
                    # Create datetime objects with midnight rollover
                    rs_start = datetime.combine(date, RS_ON)
                    rs_end   = datetime.combine(date, RS_OFF)
                    if RS_OFF < RS_ON:
                        rs_end += timedelta(days=1)

                    as_start = datetime.combine(date, AS_ON)
                    as_end   = datetime.combine(date, AS_OFF)
                    if AS_OFF < AS_ON:
                        as_end += timedelta(days=1)

                    built_up = 0
                    if as_start < rs_start:  # early sign-on (lift-up)
                        delta_dt = rs_end - as_end
                        delta = delta_dt.total_seconds() / 3600
                        # print("lift-up")
                    elif as_end > rs_end:    # late sign-off (lay-back)
                        delta_dt = as_start - rs_start
                        delta = abs(delta_dt.total_seconds() / 3600)
                        # print("lay-back")
                    elif as_start >= rs_start and as_end <= rs_end and (as_end - as_start) < (rs_end - rs_start):  # built up
                        delta_dt = rs_end - rs_start
                        delta = abs(delta_dt.total_seconds() / 3600) - 8
                        built_up = 1
                        # print("built-up")
                    else:
                        delta = 0.0

                    if worked_f and built_up == 0:
                        worked_use = worked_f
                    elif worked_f and built_up == 1:
                        worked_use = 8
                    else:
                        worked_use = 8

                    unit = delta + (worked_use - 8) + (extra_f or 0)
                else:
                    unit = 0.0

            # ---------- Penalty  ----------
            penalty = "No"
            AS_ON  = parse_time(values[1])
            AS_OFF = parse_time(values[3])
            if not any(v.upper() in ["OFF", "ADO"] for v in values) and not sick and AS_ON and AS_OFF and weekday not in ["Saturday", "Sunday"]:
                m1 = AS_ON.hour * 60 + AS_ON.minute
                m2 = AS_OFF.hour * 60 + AS_OFF.minute
                if m2 < m1:
                    m2 += 1440
                if 1080 <= m1 % 1440 <= 1439 or 0 <= m1 % 1440 <= 239:
                    penalty = "Night"
                elif 240 <= m1 % 1440 <= 330:
                    penalty = "Morning"
                elif m1 <= 1080 <= m2:
                    penalty = "Afternoon"

            # ---------- Special  ----------
            special = "No"
            if not any(v.upper() in ["OFF", "ADO"] for v in values) and not sick and weekday not in ["Saturday", "Sunday"]:
                if (AS_ON and time(1, 1) <= AS_ON <= time(3, 59)) or (AS_OFF and time(1, 1) <= AS_OFF <= time(3, 59)):
                    special = "Yes"

            # ---------- Rates ----------
            ot, prate, sload, srate, drate, lrate, dcount = calculate_row(
                weekday, values, sick, penalty, special, round(unit, 2)
            )

            if any(v.upper() == "ADO" for v in values):
                any_ado = True

            rows.append([
                weekday, date_str, values[0], values[1], values[2], values[3], values[4], values[5],
                "Yes" if sick else "No", f"{unit:.2f}", penalty, special, is_holiday,
                f"{ot:.2f}", f"{prate:.2f}", f"{sload:.2f}", f"{srate:.2f}",
                f"{lrate:.2f}", f"{drate:.2f}", f"{dcount:.2f}"
            ])

        submitted = st.form_submit_button("Calculate")

    if submitted:
        cols = [
            "Weekday","Date","R Sign-on","A Sign-on","R Sign-off","A Sign-off","Worked","Extra","Sick",
            "Unit","Penalty","Special","Holiday",
            "OT Rate","Penalty Rate","Special Ldg","Sick Rate","Loading","Daily Rate","Daily Count"
        ]
        df = pd.DataFrame(rows, columns=cols)

        # Totals for numeric columns 13..19
        totals = [pd.to_numeric(df[c], errors="coerce").fillna(0).sum() for c in cols[13:]]

        # Long fortnight deduction if no ADO anywhere
        if not any_ado:
            deduction = 0.5 * rate_constants["Ordinary Hours"] * 8  # half of daily rate
            totals[-1] -= deduction
            st.warning(f"Applied long-fortnight deduction: -{deduction:.2f}")

        # Format totals to 2 decimals
        totals_fmt = [f"{t:.2f}" for t in totals]
        total_row = ["TOTAL", "", "", "", "", "", "", "", "", "", "", "", ""] + totals_fmt
        df.loc[len(df)] = total_row


        def highlight_total(row):
            return ['background-color: #d0ffd0' if row.name == len(df)-1 else '' for _ in row]

        st.dataframe(df.style.apply(highlight_total, axis=1), use_container_width=True)

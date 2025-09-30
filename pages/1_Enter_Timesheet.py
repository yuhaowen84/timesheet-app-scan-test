# pages/1_Enter_Timesheet.py
import streamlit as st
from datetime import datetime, timedelta

st.title("1) Enter Timesheet")

# keep last chosen date
start_date = st.date_input("Select Start Date", value=st.session_state.get("start_date"))
st.caption("Enter HH:MM or HHMM. Worked/Extra accept HH:MM or HHMM. Worked defaults to 8h.")

if start_date:
    st.session_state["start_date"] = start_date
    # prepare container for rows
    all_rows = []

    with st.form("timesheet_form"):
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

            t1, t2, t3 = st.columns(3)
            sick_chk = t1.checkbox("Sick", key=f"sick_{i}")
            off_chk  = t2.checkbox("Off",  key=f"off_{i}")
            ado_chk  = t3.checkbox("ADO",  key=f"ado_{i}")

            # store raw entries + toggles
            all_rows.append({
                "weekday": weekday, "date_str": date_str,
                "rs_on": rs_on.strip(), "as_on": as_on.strip(),
                "rs_off": rs_off.strip(), "as_off": as_off.strip(),
                "worked": worked.strip(), "extra": extra.strip(),
                "sick": sick_chk, "off": off_chk, "ado": ado_chk,
            })

        saved = st.form_submit_button("Save Entries")

    if saved:
        st.session_state["entries"] = all_rows
        st.success("Saved! Go to **Review Calculations** from the sidebar.")
else:
    st.info("Choose a start date to begin.")

# pages/1_Enter_Timesheet.py
import streamlit as st
from datetime import timedelta

st.title("1) Enter Timesheet (One Day Per Page)")

# ---------- session helpers ----------
def ensure_entries(start_date):
    """Create a 14-day empty structure if not present or if start_date changed."""
    start_str = start_date.strftime("%Y-%m-%d")
    if (
        "entries" not in st.session_state
        or "entries_start" not in st.session_state
        or st.session_state["entries_start"] != start_str
        or len(st.session_state["entries"]) != 14
    ):
        entries = []
        for i in range(14):
            d = start_date + timedelta(days=i)
            entries.append({
                "weekday": d.strftime("%A"),
                "date_str": d.strftime("%Y-%m-%d"),
                "rs_on": "", "as_on": "", "rs_off": "", "as_off": "",
                "worked": "", "extra": "",
                "sick": False, "off": False, "ado": False,
            })
        st.session_state["entries"] = entries
        st.session_state["entries_start"] = start_str
        st.session_state["day_index"] = 0

def get_day(i):
    return st.session_state["entries"][i]

def set_day(i, data):
    st.session_state["entries"][i] = data

def progress_count():
    """Rough completion: count rows with any time/flag filled."""
    cnt = 0
    for r in st.session_state.get("entries", []):
        if any([r["rs_on"], r["as_on"], r["rs_off"], r["as_off"],
                r["worked"], r["extra"], r["sick"], r["off"], r["ado"]]):
            cnt += 1
    return cnt

# ---------- UI: pick start date ----------
start_date = st.date_input("Select Start Date", value=st.session_state.get("start_date"))
if not start_date:
    st.info("Choose a start date to begin.")
    st.stop()

# set/refresh structures
st.session_state["start_date"] = start_date
ensure_entries(start_date)

# ---------- navigation header ----------
total_days = 14
day_index = st.session_state.get("day_index", 0)
day_index = max(0, min(total_days - 1, day_index))
st.session_state["day_index"] = day_index

left, mid, right = st.columns([1, 2, 1])
with left:
    if st.button("⬅️ Previous", use_container_width=True, disabled=(day_index == 0)):
        st.session_state["day_index"] = max(0, day_index - 1)
        st.rerun()

with mid:
    # day selector (0-based internal, 1-based display)
    idx = st.number_input("Day", min_value=1, max_value=14,
                          value=day_index + 1, step=1,
                          label_visibility="collapsed")
    if (idx - 1) != day_index:
        st.session_state["day_index"] = idx - 1
        st.rerun()

with right:
    if st.button("Next ➡️", use_container_width=True, disabled=(day_index == total_days - 1)):
        st.session_state["day_index"] = min(total_days - 1, day_index + 1)
        st.rerun()

# ---------- progress ----------
done = progress_count()
st.progress(done / total_days, text=f"Progress: {done}/{total_days} days have entries")

# ---------- day form ----------
row = get_day(st.session_state["day_index"])
st.subheader(f'{row["weekday"]} — {row["date_str"]}')

with st.form("day_form", clear_on_submit=False):
    c1, c2, c3, c4 = st.columns(4)
    rs_on = c1.text_input("R Sign-on", value=row["rs_on"], key=f'rs_on_{day_index}')
    as_on = c2.text_input("A Sign-on", value=row["as_on"], key=f'as_on_{day_index}')
    rs_off = c3.text_input("R Sign-off", value=row["rs_off"], key=f'rs_off_{day_index}')
    as_off = c4.text_input("A Sign-off", value=row["as_off"], key=f'as_off_{day_index}')

    c5, c6 = st.columns(2)
    worked = c5.text_input("Worked", value=row["worked"], key=f'worked_{day_index}',
                           help="HH:MM or HHMM (blank → 8h)")
    extra  = c6.text_input("Extra",  value=row["extra"],  key=f'extra_{day_index}',
                           help="HH:MM or HHMM")

    t1, t2, t3 = st.columns(3)
    sick_chk = t1.checkbox("Sick", value=row["sick"], key=f'sick_{day_index}',
                           help="Behaves like OFF for unit; also adds Sick rate")
    off_chk  = t2.checkbox("Off",  value=row["off"],  key=f'off_{day_index}',
                           help='Acts like entering "OFF"')
    ado_chk  = t3.checkbox("ADO",  value=row["ado"],  key=f'ado_{day_index}',
                           help='Acts like entering "ADO"')

    # convenience: copy previous day's inputs
    copy_col, _, _ = st.columns([1,2,1])
    copy_prev = copy_col.checkbox("Copy previous day", key=f'copy_prev_{day_index}',
                                  disabled=(day_index == 0),
                                  help="Prefill with yesterday's values (time fields + flags).")

    # Save actions
    save = st.form_submit_button("Save Day ✅")
    save_next = st.form_submit_button("Save & Next ➡️")

# If copy previous toggled, prefill fields immediately
if "copy_prev_state" not in st.session_state:
    st.session_state["copy_prev_state"] = [False] * total_days

if copy_prev and not st.session_state["copy_prev_state"][day_index]:
    prev = get_day(day_index - 1)
    rs_on, as_on, rs_off, as_off = prev["rs_on"], prev["as_on"], prev["rs_off"], prev["as_off"]
    worked, extra = prev["worked"], prev["extra"]
    sick_chk, off_chk, ado_chk = prev["sick"], prev["off"], prev["ado"]
    st.session_state["copy_prev_state"][day_index] = True
elif not copy_prev and st.session_state["copy_prev_state"][day_index]:
    st.session_state["copy_prev_state"][day_index] = False

# Persist this day's values
row_update = {
    "weekday": row["weekday"], "date_str": row["date_str"],
    "rs_on": rs_on.strip(), "as_on": as_on.strip(),
    "rs_off": rs_off.strip(), "as_off": as_off.strip(),
    "worked": worked.strip(), "extra": extra.strip(),
    "sick": bool(sick_chk), "off": bool(off_chk), "ado": bool(ado_chk),
}
set_day(day_index, row_update)

if save or save_next:
    st.success(f"Saved {row['weekday']} ({row['date_str']}).")
    if save_next and day_index < total_days - 1:
        st.session_state["day_index"] = day_index + 1
        st.rerun()

# footer nav
nav_left, _, nav_right = st.columns([1,2,1])
with nav_left:
    st.button("⬅️ Previous Day", use_container_width=True,
              disabled=(day_index == 0),
              on_click=lambda: st.session_state.update(day_index=day_index-1))
with nav_right:
    st.button("Next Day ➡️", use_container_width=True,
              disabled=(day_index == total_days-1),
              on_click=lambda: st.session_state.update(day_index=day_index+1))

st.markdown("---")
st.info("When you finish all days, open **Review Calculations** to see totals and breakdown.\n\n"
        "⚙️ Reminder: Your **Ordinary, Afternoon, and Night base rates** can be set from the **Home page**. "
        "They’ll flow into the calculations here.")

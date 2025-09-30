# utils.py
import math
from datetime import datetime, timedelta, time

# -------- Defaults (you can keep your existing numbers here) --------
DEFAULT_BASE_RATES = {
    "ordinary": 49.81842,
    "afternoon_penalty": 4.84,
    "night_penalty": 5.69,
}

def build_rate_constants(base: dict):
    """
    Build the full rate table from three base inputs.
    Multipliers match your existing constants.
    """
    ordinary = base["ordinary"]
    aft = base["afternoon_penalty"]
    night = base["night_penalty"]

    rates = {
        # penalties (per-hour adders)
        "Afternoon Shift": aft,
        "Early Morning": aft,          # same as Afternoon in your app
        "Night Shift": night,
        "Special Loading": night,      # same as Night in your app

        # OT / loadings (multiples of ordinary)
        "OT 150%": ordinary * 1.5,
        "OT 200%": ordinary * 2.0,
        "ADO Adjustment": ordinary,
        "Sat Loading 50%": ordinary * 0.5,
        "Sun Loading 100%": ordinary * 1.0,
        "Public Holiday": ordinary,
        "PH Loading 50%": ordinary * 0.5,
        "PH Loading 100%": ordinary * 1.0,

        # other ordinary-based
        "Sick With MC": ordinary,
        "Ordinary Hours": ordinary,
    }
    # round to 5 decimals for parity with your previous constants
    return {k: round(v, 5) for k, v in rates.items()}

# Keep a default set available (used if user doesn't customize)
rate_constants = build_rate_constants(DEFAULT_BASE_RATES)

NSW_PUBLIC_HOLIDAYS = {
    "2025-01-01", "2025-01-27", "2025-04-18", "2025-04-19", "2025-04-20", "2025-04-21",
    "2025-04-25", "2025-06-09", "2025-10-06", "2025-12-25", "2025-12-26"
}

def parse_time(text: str):
    text = (text or "").strip()
    if not text: return None
    try:
        if ":" in text:
            return datetime.strptime(text, "%H:%M").time()
        if text.isdigit() and len(text) in [3,4]:
            h, m = int(text[:-2]), int(text[-2:])
            return time(h, m)
    except:
        return None
    return None

def parse_duration(text: str) -> float:
    text = (text or "").strip()
    if not text: return 0
    try:
        if ":" in text:
            h, m = map(int, text.split(":"))
            return h + m/60
        if text.isdigit():
            h, m = int(text[:-2]), int(text[-2:])
            return h + m/60
    except:
        return 0
    return 0

def calculate_row(day, values, sick, penalty_value, special_value, unit_val, rates=None):
    """
    values: [rs_on, as_on, rs_off, as_off, worked, extra]
    rates: dict of rate constants; if None, use module default
    """
    R = rates or rate_constants

    ot_rate = 0
    if values[0].upper() == "ADO" and unit_val >= 0:
        ot_rate = round(unit_val * R["ADO Adjustment"], 2)
    elif values[0].upper() not in ["OFF", "ADO"] and unit_val >= 0:
        if day in ["Saturday", "Sunday"]:
            ot_rate = round(unit_val * R["OT 200%"], 2)
        else:
            ot_rate = round(unit_val * R["OT 150%"], 2)
    else:
        # negative or OFF/ADO -> ordinary + applicable loading
        if day == "Saturday":
            ot_rate = round(unit_val * (R["Sat Loading 50%"] + R["Ordinary Hours"]), 2)
        elif day == "Sunday":
            ot_rate = round(unit_val * (R["Sun Loading 100%"] + R["Ordinary Hours"]), 2)
        else:
            if penalty_value in ["Afternoon", "Morning"]:
                ot_rate = round(unit_val * (R["Afternoon Shift"] + R["Ordinary Hours"]), 2)
            elif penalty_value == "Night":
                ot_rate = round(unit_val * (R["Night Shift"] + R["Ordinary Hours"]), 2)
            else:
                ot_rate = round(unit_val * R["Ordinary Hours"], 2)

    # Penalty hours: floor(worked), default 8 if blank/invalid
    worked_hours = parse_duration(values[4]) or 8
    penalty_hours = math.floor(worked_hours)

    penalty_rate = 0
    if penalty_value == "Afternoon":
        penalty_rate = round(penalty_hours * R["Afternoon Shift"], 2)
    elif penalty_value == "Night":
        penalty_rate = round(penalty_hours * R["Night Shift"], 2)
    elif penalty_value == "Morning":
        penalty_rate = round(penalty_hours * R["Early Morning"], 2)

    special_loading = round(R["Special Loading"], 2) if special_value == "Yes" else 0
    sick_rate = round(8 * R["Sick With MC"], 2) if sick else 0
    daily_rate = 0 if values[0].upper() in ["OFF", "ADO"] else round(8 * R["Ordinary Hours"], 2)

    if any(v.upper() == "ADO" for v in values):
        daily_rate += round(4 * R["Ordinary Hours"], 2)

    loading = 0
    if values[0].upper() not in ["OFF", "ADO"]:
        if day == "Saturday":
            loading = round(8 * R["Sat Loading 50%"], 2)
        elif day == "Sunday":
            loading = round(8 * R["Sun Loading 100%"], 2)

    daily_count = ot_rate + penalty_rate + special_loading + sick_rate + daily_rate + loading
    return ot_rate, penalty_rate, special_loading, sick_rate, daily_rate, loading, daily_count

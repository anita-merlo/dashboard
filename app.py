# streamlit run app.py
# ============================
# ELM Indicator – Jobs by State (CSV-powered, no upload)
# ============================

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
import re

# ----------------------------
# App Config
# ----------------------------
st.set_page_config(page_title="ELM Indicator – Jobs by State", layout="wide")

# ----------------------------
# Constants & Helpers
# ----------------------------
STATE_ABBRS = [
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN","IA","KS","KY","LA","ME","MD",
    "MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ","NM","NY","NC","ND","OH","OK","OR","PA","RI","SC",
    "SD","TN","TX","UT","VT","VA","WA","WV","WI","WY","DC"
]

QUARTER_ORDER = {f"{y}Q{q}": (y, q) for y in range(2018, 2036) for q in (1, 2, 3, 4)}
def _quarter_key(q: str) -> int:
    y, qtr = QUARTER_ORDER.get(q, (0, 0))
    return y * 10 + qtr

def _to_quarter(y, m):
    if pd.isna(y) or pd.isna(m): return np.nan
    q = int((int(m) - 1)//3 + 1)
    return f"{int(y)}Q{q}"

def safe_div(n: np.ndarray, d: np.ndarray) -> np.ndarray:
    return np.where(d != 0, n / d, 0.0)

def _natural_key(s: str):
    """
    Natural sort for labels like '1. Foo', '10) Bar', or plain text.
    Always returns a comparable (group, number, text) tuple.
    """
    s = str(s).strip()
    m = re.match(r'^(\d+)[\.\)]?\s*(.*)$', s)  # matches "1. Foo" or "1) Foo"
    if m:
        num = int(m.group(1))
        rest = m.group(2).strip().lower()
        return (0, num, rest)     # numbered come first, sorted by num, then text
    return (1, float('inf'), s.lower())  # non-numbered after, alpha by text


# ----------------------------
# Data ingest
# ----------------------------
CSV_PATH_DEFAULT = Path(__file__).parent / "data.csv.gz"

@st.cache_data(show_spinner=False)
def load_raw():
    p = Path(CSV_PATH_DEFAULT)
    if not p.exists():
        return None
    return pd.read_csv(p, low_memory=False)

raw = load_raw()
if raw is None:
    st.error(f"CSV not found at: {CSV_PATH_DEFAULT}")
    st.stop()

needed = {"combination_label", "state", "month", "year", "count"}
missing = needed - set(raw.columns)
if missing:
    st.error(f"Missing required columns in CSV: {sorted(missing)}")
    st.stop()

raw = load_raw()
if raw is None:
    st.error(f"CSV not found at: {CSV_PATH_DEFAULT}")
    st.stop()

needed = {"combination_label", "state", "month", "year", "count"}
missing = needed - set(raw.columns)
if missing:
    st.error(f"Missing required columns in CSV: {sorted(missing)}")
    st.stop()

# ----------------------------
# Normalize & quarterize
# ----------------------------
df0 = raw.copy()
df0["state"]  = df0["state"].astype(str).str.upper().str.strip()
df0["year"]   = pd.to_numeric(df0["year"], errors="coerce")
df0["month"]  = pd.to_numeric(df0["month"], errors="coerce")
df0["count"]  = pd.to_numeric(df0["count"], errors="coerce").fillna(0)
df0           = df0[df0["state"].isin(STATE_ABBRS)]
df0["quarter"]= df0.apply(lambda r: _to_quarter(r["year"], r["month"]), axis=1)
df0           = df0.dropna(subset=["quarter"])

# ----------------------------
# Aggregate base & pivot wide for ratio math
# ----------------------------
base = (df0.groupby(["state","quarter","combination_label"], as_index=False)["count"]
          .sum().rename(columns={"count":"value"}))

wide = (base.pivot_table(index=["state","quarter"],
                         columns="combination_label",
                         values="value", aggfunc="sum").fillna(0.0))

def col(name: str):
    return wide[name] if name in wide.columns else 0.0

# ----------------------------
# Indicator calculations (CASE #1)
# ----------------------------
# Sums
wide["HVAC Constellation__IND"]                 = col("HVAC Constellation")
wide["BAS Role Constellation__IND"]             = col("BAS Role Constellation")
wide["Energy Managers__IND"]                    = col("Energy Managers")
wide["Energy Management Certification__IND"]    = col("Energy Management Certification Constellation")
#wide["Energy Managers and SCADA__IND"]          = col("Energy Managers and SCADA")
wide["Construction Managers With Energy Efficiency Focus__IND"] = col("Construction Managers With Energy Efficiency Focus")
wide["Architects With LEED__IND"]               = col("Architects With LEED / LEED AP")
wide["Heat Pump Constellation__IND"]            = col("Heat Pump Constellation")
wide["Energy Auditing Constellation__IND"]      = col("Energy Auditing Constellation")
wide["BST Certification Constellation__IND"]    = col("BST Certification Constellation")
wide["Certified Crane Operators__IND"]          = col("Certified Crane Operators")
wide["Composite References__IND"]               = col("Composite References")

#if "Construction Managers" in wide.columns:
#    wide["Construction Managers__IND"] = col("Construction Managers")

# EV label guard
ev_cols = [c for c in wide.columns if c.strip().lower() in {"ev repair and maintenance","ev repair and mantainance"}]
wide["EV Repair and Maintenance__IND"] = wide[ev_cols].sum(axis=1) if ev_cols else 0.0

# 👉 REQUIRED for *Construction Manager with EF Intensity* (denominator)
if "Construction Managers" in wide.columns:
    wide["Construction Managers__IND"] = col("Construction Managers")

# Ratios
wide["BAS to HVAC Ratio__IND"] = safe_div(col("BAS Role Constellation"), col("HVAC Constellation"))
wide["Energy Manager Certification to EM Ratio__IND"] = safe_div(col("Energy Management Certification Constellation"), col("Energy Managers"))
wide["Construction Manager with EF Ratio__IND"]       = safe_div(col("Construction Managers With Energy Efficiency Focus"), col("Construction Managers"))
#wide["Energy Manager and SCADA to EM Ratio__IND"]     = safe_div(col("Energy Managers and SCADA"), col("Energy Managers"))


# ----------------------------
# Long form (state, quarter, indicator, value)
# ----------------------------
indicator_cols = [c for c in wide.columns if c.endswith("__IND")]
if not indicator_cols:
    st.error("No mapped indicators were produced. Check your 'combination_label' values.")
    st.write("Available columns after pivot:", list(wide.columns))
    st.stop()

tmp      = wide[indicator_cols].rename(columns=lambda c: c.replace("__IND",""))
stacked  = tmp.stack()
long_df  = stacked.reset_index()
col_name = tmp.columns.name if tmp.columns.name is not None else "level_2"
long_df  = long_df.rename(columns={col_name:"indicator", 0:"value"})


# ----------------------------
# Industry buckets
# ----------------------------
industry_map = {
    # Energy Efficiency (Buildings)
    "HVAC Constellation":"Energy Efficiency",
    "BAS Role Constellation":"Energy Efficiency",
    "Energy Auditing Constellation":"Energy Efficiency",
    "Architects With LEED":"Energy Efficiency",
    "Construction Managers":"Energy Efficiency",
    "Construction Managers With Energy Efficiency Focus":"Energy Efficiency",
    "Energy Managers":"Energy Efficiency",
    "Energy Management Certification":"Energy Efficiency",
    "Energy Managers and SCADA":"Energy Efficiency",
    "Heat Pump Constellation":"Energy Efficiency",
    "BAS to HVAC Ratio":"Energy Efficiency",
    "Energy Manager Certification to EM Ratio":"Energy Efficiency",
    "Construction Manager with EF Ratio":"Energy Efficiency",
    "Energy Manager and SCADA to EM Ratio":"Energy Efficiency",
    # Wind
    "Certified Crane Operators":"Energy Generation",
    "BST Certification Constellation":"Energy Generation",
    "Composite References":"Energy Generation",
    # Solar
    "Solar Certification Constellation":"Energy Generation (Solar)",
    "ESIP Certification":"Energy Generation (Solar)",
    # EV
    "EV Repair and Maintenance":"Transportation",
}

long_df["industry"] = long_df["indicator"].map(industry_map).fillna("Other")

df = long_df.rename(columns={"level_0":"state","level_1":"quarter"})
df["quarter"] = df["quarter"].astype(str)
df["state"]   = df["state"].astype(str)

# after building df
DISPLAY_NAME_MAP = {
    "Architects With LEED": "1. Architect Jobs with LEED Certifications",
    "BAS Role Constellation": "2. Building Automation Technician Jobs",
    "BAS to HVAC Ratio": "3. Building Automation Technician to HVAC Ratio",
    "Certified Crane Operators": "1. Certified Crane Operators (Wind)",
    "Construction Managers": "4. Construction Managers Jobs",
    "Construction Managers With Energy Efficiency Focus": "5. Construction Manager Jobs with Energy Efficiency Focus",
    "Construction Manager with EF Ratio": "6. Construction Manager with Energy Efficiency Intensity",
    "EV Repair and Maintenance": "1. Electric Vehicle Repair and Maintenance Jobs",
    "Energy Auditing Constellation": "7. Energy Auditor Certifications",
    "Energy Management Certification": "8. Energy Management Certifications",
    "Energy Managers": "9. Energy Management Jobs",
    "Energy Manager Certification to EM Ratio": "10. Energy Manager Certification Intensity",
    "Energy Managers and SCADA": "Energy Managers and SCADA",
    "Energy Manager and SCADA to EM Ratio": "Energy Managers with SCADA Intensity",
    "Heat Pump Constellation": "12. Heat Pump Technology Group",
    "HVAC Constellation": "11. HVAC Jobs Group",
    "Composite References": "2. Wind and Composite Materials related roles",
    "BST Certification Constellation": "3. Wind-related Safety Certifications",
}

df["indicator_original"] = df["indicator"]  # keep original if needed
df["indicator"] = df["indicator"].map(DISPLAY_NAME_MAP).fillna(df["indicator"])

# rebuild buckets using display names
INDUSTRY_BUCKETS = (
    df.drop_duplicates(["indicator","industry"])
      .groupby("industry")["indicator"]
      .apply(lambda s: sorted(s.tolist(), key=_natural_key))  # <-- use natural sort
      .to_dict()
)


def learn_more_url_for(industry: str | None) -> str | None:
    """Map an industry label to the correct Learn More URL."""
    if not industry:
        return None
    il = industry.lower()
    if "transport" in il:         # Vehicles / Transportation Electrification
        return "https://energylabormarketdashboard.juliuslmi.com/transportation/"
    if "efficiency" in il:        # Energy Efficiency (Buildings)
        return "https://energylabormarketdashboard.juliuslmi.com/energy_efficiency/"
    if "generation" in il:        # Energy Generation (Wind / Solar)
        return "https://energylabormarketdashboard.juliuslmi.com/energy-generation/"
    return None

# ----------------------------
# Dynamic descriptions by indicator (DISPLAY NAMES)
# ----------------------------
INDICATOR_DESCRIPTIONS = {
    "1. Architect Jobs with LEED Certifications": "Number of job postings for Architects that require LEED certifications (including LEED and LEED AP).",
    "2. Building Automation Technician Jobs": "Number of job postings for building automation systems technicians and similar alternative job titles.",
    "3. Building Automation Technician to HVAC Ratio": "The ratio of job postings for Building Automation Technicians compared to total postings in the HVAC group.",
    "1. Certified Crane Operators (Wind)": "Number of job postings requiring Crane Operator certifications for the wind energy sector.",
    "4. Construction Managers Jobs": "Number of job postings for Construction Managers.",
    "5. Construction Manager Jobs with Energy Efficiency Focus": "Number of job postings for Construction Managers that include a focus on energy-efficient building.",
    "6. Construction Manager with Energy Efficiency Intensity": "Ratio of Energy-Efficiency-focused Construction Managers to all Construction Managers (EE intensity).",
    "1. Electric Vehicle Repair and Maintenance Jobs": "Number of job postings that involve EV repair and maintenance (e.g., OEM EV service certifications).",
    "7. Energy Auditor Certifications": "Number of job postings that reference energy auditing certifications.",
    "8. Energy Management Certifications": "Number of job postings requiring Energy Management certifications (including CEM, CBCP, FMP).",
    "9. Energy Management Jobs": "Number of job postings for Energy Managers and alternative titles.",
    "10. Energy Manager Certification Intensity": "Ratio of Energy Management Certifications to total Energy Managers (relative demand for certifications).",
    "Energy Managers and SCADA": "Number of job postings for Energy Manager roles that also require SCADA competencies.",
    "Energy Managers with SCADA Intensity": "Ratio of Energy Managers with SCADA to all Energy Managers (role sophistication).",
    "12. Heat Pump Technology Group": "Number of job postings that include heat pump certifications, skills, and keywords.",
    "11. HVAC Jobs Group": "Number of job postings for a variety of related HVAC roles (HVAC Installer, HVAC Technician, HVAC Journeyman, etc.).",
    "2. Wind and Composite Materials related roles": "Number of job postings in wind energy that include references to composite materials.",
    "3. Wind-related Safety Certifications": "Number of wind-energy job postings that require Basic Safety Training (BST) certifications.",
}

def describe_indicator(name: str) -> str:
    return INDICATOR_DESCRIPTIONS.get(name, "Please select a specific indicator.")

# ----------------------------
# Controls
# ----------------------------
quarters_all = sorted(df["quarter"].unique(), key=_quarter_key)
if not quarters_all:
    st.warning("No quarters found after filtering. Check CSV content and state abbreviations.")
    st.stop()

all_inds = sorted(df["indicator"].unique().tolist())
buckets  = {cat:[i for i in inds if i in all_inds] for cat,inds in INDUSTRY_BUCKETS.items()}
covered  = {i for lst in buckets.values() for i in lst}
other    = sorted([i for i in all_inds if i not in covered])
if other: buckets["Other"] = other
cat_options = [c for c,lst in buckets.items() if lst] or ["Other"]
if cat_options == ["Other"] and "Other" not in buckets:
    buckets["Other"] = all_inds


# === Main title (place BEFORE creating the 3 step columns) ===
st.markdown(
    """
    <div style="position:sticky; top:0; z-index:10; background:white; padding:8px 0;">
        <h2 style="text-align:center; color:#005a70; font-weight:800; font-size:26px; margin:0;">
            HOW MANY JOBS WERE POSTED IN MY STATE?
        </h2>
        <hr style="border:0; border-top:1px solid #d9e2e8; margin:6px auto 0; width:80%;">
    </div>
    """,
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns([1.15, 1.15, 0.95], gap="large")

# Shared heights
INNER_VIZ_HEIGHT = 580  # height for map and bar-chart scrollport


# STEP 1
# STEP 1 (YoY-only)
# STEP 1 (Quarter RANGE, YoY comparison)
# STEP 1 (default to Q1–Q4 2024, with YoY comparison)
with c1:
    with st.container(border=True):
        st.markdown(
            '<div style="font-size:18px; color:#005a70; font-weight:600; margin:0 0 8px 0;">'
            'STEP 1: CHOOSE A QUARTER RANGE</div>',
            unsafe_allow_html=True
        )

        # set defaults for Q1–Q4 2024
        default_start = quarters_all.index("2024Q1") if "2024Q1" in quarters_all else 0
        default_end   = quarters_all.index("2024Q4") if "2024Q4" in quarters_all else len(quarters_all)-1

        q_start = st.selectbox("Start Quarter", quarters_all, index=default_start, key="q_start")
        q_end   = st.selectbox("End Quarter",   quarters_all, index=default_end, key="q_end")

        if _quarter_key(q_start) > _quarter_key(q_end):
            q_start, q_end = q_end, q_start

        # Build labels
        def _shift_quarter(q: str, years: int = -1) -> str:
            y, qtr = q.split("Q")
            return f"{int(y) + years}Q{int(qtr)}"

        range_label = f"{q_start}–{q_end}"
        prev_start  = _shift_quarter(q_start, -1)
        prev_end    = _shift_quarter(q_end, -1)
        comp_label  = f"{prev_start}–{prev_end}"

        st.caption(
    f"Choose any quarter range. We’ll use it as the **current** period and compare it to the **same quarters last year**.\n"
    f"**Current:** {range_label}   •   **Previous:** {comp_label}"
)






# STEP 2
with c2:
    with st.container(border=True):
        st.markdown('<div style="font-size:18px; color:#005a70; font-weight:600; margin:0 0 8px 0;">STEP 2: SELECT AN INDUSTRY & INDICATOR</div>', unsafe_allow_html=True)
        category = st.selectbox("Industry", cat_options, index=0, key="industry_select")
        indicator_selected = st.selectbox("Indicator", buckets[category], index=0, key="indicator_select")
        st.caption("Choose which job demand signal to analyze.")
        desc = describe_indicator(indicator_selected)  # dynamic description

# STEP 3
# STEP 3
with c3:
    with st.container(border=True):
        st.markdown(
            '<div style="font-size:18px; color:#005a70; font-weight:600; margin:0 0 8px 0;">ELM Indicator</div>',
            unsafe_allow_html=True,
        )
        st.markdown(f"**{indicator_selected}**")
        st.caption(desc)

        # Prefer the user's selected industry tab (category) for routing
        url = learn_more_url_for(category)

        # Fallback: infer from the indicator’s industry (in case category ever becomes None)
        if not url:
            # try to recover industry from df for the selected indicator
            try:
                inferred_industry = (
                    df.loc[df["indicator"] == indicator_selected, "industry"]
                      .dropna().iloc[0]
                )
            except Exception:
                inferred_industry = None
            url = learn_more_url_for(inferred_industry)

        if url:
            st.link_button("Learn More", url=url, help="Open the documentation for this indicator’s industry")

# ----------------------------
# Compute current & comparison  (YoY only)
# ----------------------------
q_list     = sorted(df["quarter"].unique(), key=_quarter_key)
start_idx  = q_list.index(q_start)
end_idx    = q_list.index(q_end)
sel_quarts = q_list[start_idx : end_idx + 1]

def shift_quarter(q: str, years: int = -1) -> str:
    y, qtr = q.split("Q")
    return f"{int(y) + years}Q{int(qtr)}"

# Compare only to the same quarters last year (YoY)
prev_quarts = [shift_quarter(q, -1) for q in sel_quarts if shift_quarter(q, -1) in q_list]

range_label = f"{q_start}–{q_end}"
comp_label  = f"{prev_quarts[0]}–{prev_quarts[-1]}" if prev_quarts else "N/A"


# --- helpers ---
def is_ratio_indicator(name: str) -> bool:
    return any(k in name.lower() for k in ("ratio", "intensity"))

# Map (display or original) -> (numerator_original, denominator_original)
RATIO_SPECS = {
    # BAS / HVAC
    "BAS to HVAC Ratio": ("BAS Role Constellation", "HVAC Constellation"),
    "Building Automation Technician to HVAC Ratio": ("BAS Role Constellation", "HVAC Constellation"),
    # Construction EF / Construction Managers
    "Construction Manager with EF Ratio": ("Construction Managers With Energy Efficiency Focus", "Construction Managers"),
    "Construction Manager with Energy Efficiency Intensity": ("Construction Managers With Energy Efficiency Focus", "Construction Managers"),
    # EM Cert / Energy Managers
    "Energy Manager Certification to EM Ratio": ("Energy Management Certification", "Energy Managers"),
    "Energy Manager Certification Intensity": ("Energy Management Certification", "Energy Managers"),
    # EM+SCADA / Energy Managers
    "Energy Manager and SCADA to EM Ratio": ("Energy Managers and SCADA", "Energy Managers"),
    "Energy Managers with SCADA Intensity": ("Energy Managers and SCADA", "Energy Managers"),
}

selected_display  = indicator_selected
# if you added df["indicator_original"] earlier, resolve the original name; else fall back
if "indicator_original" in df.columns and (df["indicator"] == selected_display).any():
    selected_original = df.loc[df["indicator"] == selected_display, "indicator_original"].iloc[0]
else:
    selected_original = selected_display

def period_sum(ind_name: str, quarters: list[str]) -> pd.Series:
    """Sum 'value' by state for a given (original or display) indicator over quarters."""
    m_quarters = df["quarter"].isin(quarters)
    m_disp = (df["indicator"] == ind_name)
    m_orig = (df["indicator_original"] == ind_name) if "indicator_original" in df.columns else False
    mask = m_quarters & (m_disp | m_orig)
    if not mask.any():
        return pd.Series(dtype=float, name="value")
    return df.loc[mask].groupby("state")["value"].sum()

def compute_counts_current_prev(ind_name: str):
    cur = period_sum(ind_name, sel_quarts).rename("current")
    if prev_quarts:
        prv = period_sum(ind_name, prev_quarts).rename("previous")
    else:
        prv = pd.Series(index=cur.index, dtype=float, name="previous")
    # align to the same states
    all_states = cur.index.union(prv.index)
    return cur.reindex(all_states), prv.reindex(all_states)

def compute_ratio_current_prev(ind_name_display: str):
    # pick a spec key that exists (display or original)
    spec_key = ind_name_display if ind_name_display in RATIO_SPECS else selected_original
    if spec_key not in RATIO_SPECS:
        raise KeyError(f"No ratio spec for indicator: {ind_name_display}")
    num_orig, den_orig = RATIO_SPECS[spec_key]

    # current
    num_cur = period_sum(num_orig, sel_quarts)
    den_cur = period_sum(den_orig, sel_quarts)
    all_states = num_cur.index.union(den_cur.index)
    num_cur = num_cur.reindex(all_states, fill_value=0.0)
    den_cur = den_cur.reindex(all_states, fill_value=0.0)
    cur = pd.Series(np.where(den_cur != 0, num_cur / den_cur, np.nan), index=all_states, name="current")

    # previous
    if prev_quarts:
        num_prev = period_sum(num_orig, prev_quarts).reindex(all_states, fill_value=0.0)
        den_prev = period_sum(den_orig, prev_quarts).reindex(all_states, fill_value=0.0)
        prev = pd.Series(np.where(den_prev != 0, num_prev / den_prev, np.nan), index=all_states, name="previous")
    else:
        prev = pd.Series(index=all_states, dtype=float, name="previous")

    return cur, prev

# --- compute ---
if is_ratio_indicator(selected_display) or is_ratio_indicator(selected_original):
    cur_s, prev_s = compute_ratio_current_prev(selected_display)
else:
    cur_s, prev_s = compute_counts_current_prev(selected_display)

stats = (
    pd.concat([cur_s, prev_s], axis=1)
      .reset_index()
      .rename(columns={"index": "state"})
)

# % change; NaN when previous is 0/NaN
stats["pct_change"] = (stats["current"] - stats["previous"]) / stats["previous"]
stats.loc[~np.isfinite(stats["pct_change"]), "pct_change"] = np.nan

# ----- number formatting helper (use before the bar-chart code) -----
def _fmt_k(v) -> str:
    """Counts: K with one decimal when |v| >= 1000; else 0-decimal int with commas.
       Returns '–' for NaN/None/non-numeric."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "–"
    if not np.isfinite(x):
        return "–"
    return f"{x/1000:.1f}K" if abs(x) >= 1000 else f"{int(round(x)):,}"

def _is_ratio_text(name: str) -> bool:
    """Treat both 'ratio' and 'intensity' indicators as ratios for display formatting."""
    return any(k in (name or "").lower() for k in ("ratio", "intensity"))


# ----------------------------
# Map + Bar
# ----------------------------
left, right = st.columns([1.25, 1])

# LEFT: Explanation + Map (boxed)
# LEFT: Heatmap (boxed) — use existing teal_scale colors
# LEFT: Heatmap (boxed) — unified title + subtitle; keep teal colors; no colorbar title
with left:
    with st.container(border=True):
        indicator_label = indicator_selected
        main_title_heatmap = f"State Heatmap — {indicator_label} ({range_label})"

        # Determine subtitle (counts vs ratio)
        is_ratio_sel = any(k in indicator_label.lower() for k in ("ratio", "intensity"))
        subtitle = "Ratio for the selected period" if is_ratio_sel else "Job counts for the selected period"

        # Header
        st.markdown(
            f"""
            <div style="width:100%; text-align:center; font-size:18px; color:#005a70; font-weight:600; margin:0 0 4px 0;">
              {main_title_heatmap}
            </div>
            <div style="width:100%; text-align:center; font-size:12px; color:#6b7a87; margin:0 0 2px 0;">
              {subtitle}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Map number formatting
        value_fmt = ",.2f" if is_ratio_sel else ",.0f"

        # Data for map
        map_df = stats.copy()
        map_df["color_value"] = map_df["current"].clip(lower=0)  # negatives → white
        vmax = float(map_df["color_value"].max() or 1)

        teal_scale = [
            (0.00, "white"),
            (0.001, "#e9f5f7"),
            (0.30, "#c3e0e7"),
            (0.65, "#43a6b0"),
            (1.00, "#005a70"),
        ]

        map_fig = px.choropleth(
            map_df,
            locations="state",
            locationmode="USA-states",
            color="color_value",
            scope="usa",
            color_continuous_scale=teal_scale,
            range_color=(0, vmax),
            labels={"color_value": ""},  # remove colorbar title, keep scale
        )
        map_fig.update_traces(
            customdata=np.stack([map_df["current"].values], axis=-1),
            hovertemplate="<b>%{location}</b><br>"
                          + f"{indicator_label}: "
                          + f"%{{customdata[0]:{value_fmt}}}"
                          + "<extra></extra>",
        )
        map_fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            height=INNER_VIZ_HEIGHT,
        )

        st.plotly_chart(map_fig, use_container_width=True)



# RIGHT: Bar chart (boxed, scrolls inside)
# ---------- RIGHT: Bar chart (boxed, scrolls inside; compact spacing) ----------
# ---------- RIGHT: Bar chart (boxed, auto-tight) ----------
# RIGHT: Bar chart (boxed) — unified title + subtitle; %Δ caption BELOW chart
with right:
    with st.container(border=True):
        indicator_label = indicator_selected
        is_ratio_like   = any(k in indicator_label.lower() for k in ("ratio", "intensity"))
        main_title      = f"State Barchart — {indicator_label} ({range_label})"

        # ✅ Use the variation label as the subtitle (no counts/ratio text)
        st.markdown(
            f"""
            <div style="width:100%; text-align:center; font-size:18px; color:#005a70; font-weight:600; margin:0 0 4px 0;">
              {main_title}
            </div>
            <div style="width:100%; text-align:center; font-size:12px; color:#6b7a87; margin:0 0 2px 0;">
              Variation (%Δ) vs <b>{comp_label}</b>
            </div>
            """,
            unsafe_allow_html=True,
        )

        bars = stats.sort_values("current", ascending=False).copy()
        bars["_pct"] = bars["pct_change"] * 100
        pct_all_nan = bars["_pct"].isna().all()

        BAR_COLOR = "#005a70"
        fig = px.bar(
            bars, x="current", y="state", orientation="h",
            labels={"state": "State", "current": indicator_label}
        )
        fig.update_traces(marker_color=BAR_COLOR, hoverinfo="skip", cliponaxis=False)

        xmax = float(bars["current"].max() or 1)

        # paddings
        LEFT_PAD = max(xmax * (0.35 if is_ratio_like else 0.12), 0.15 if is_ratio_like else 1.0)
        RIGHT_GAP_FRAC = 0.04 if pct_all_nan else 0.18

        # Optional %Δ column (only when we have it)
        if not pct_all_nan:
            VAR_X = xmax * (1 + RIGHT_GAP_FRAC * 0.90)
            SEP_X = xmax * (1 + RIGHT_GAP_FRAC * 0.55)

            pct_colors = bars["pct_change"].apply(
                lambda v: "#2e7d32" if (pd.notna(v) and v >= 0) else ("#c62828" if pd.notna(v) else "#666")
            )
            pct_text = bars["_pct"].map(
                lambda v: "–" if pd.isna(v) else (f"▲ {abs(v):.1f}%" if v >= 0 else f"▼ {abs(v):.1f}%")
            )

            fig.add_trace(go.Scatter(
                x=[VAR_X] * len(bars), y=bars["state"], mode="text",
                text=pct_text, textfont=dict(color=pct_colors.tolist(), size=12),
                hoverinfo="skip", showlegend=False, cliponaxis=False,
            ))
            fig.add_shape(
                type="line", xref="x", yref="paper",
                x0=SEP_X, x1=SEP_X, y0=0, y1=1,
                line=dict(color="#edf2f5", width=1),
            )

        # Left numeric column (values)
        val_text = bars["current"].map(lambda v: f"{v:.2f}" if is_ratio_like else _fmt_k(v))
        fig.add_trace(go.Scatter(
            x=[-LEFT_PAD * 0.92] * len(bars), y=bars["state"], mode="text",
            text=val_text, textposition="middle right",
            textfont=dict(color="#2b2b2b", size=12),
            hoverinfo="skip", showlegend=False, cliponaxis=False,
        ))

        # Axes & layout
        fig.update_xaxes(
            range=[-LEFT_PAD * 1.05, xmax * (1 + RIGHT_GAP_FRAC)],
            showgrid=False, title=None, zeroline=False,
            tickformat=".2f" if is_ratio_like else None
        )
        fig.update_yaxes(
            showgrid=False, title=None,
            categoryorder="array", categoryarray=list(bars["state"]),
            autorange="reversed", ticklabelposition="outside",
            ticklen=0, tickfont=dict(size=12), automargin=True,
        )

        ROW_H      = 28
        fig_height = max(int(len(bars) * ROW_H + 120), INNER_VIZ_HEIGHT + 160)
        fig.update_layout(
            margin=dict(l=120, r=100 if pct_all_nan else 130, t=0, b=0),
            height=fig_height, showlegend=False,
        )

        # Scroll container
        html = fig.to_html(
            include_plotlyjs="cdn", full_html=False, include_mathjax=False,
            config={"displayModeBar": False}
        )
        components.html(
            f'<div style="height:{INNER_VIZ_HEIGHT}px; overflow-y:auto; overflow-x:hidden; '
            f'margin-top:-8px; padding:0 6px;">{html}</div>',
            height=INNER_VIZ_HEIGHT, scrolling=False,
        )



# ----------------------------
# Download
# ----------------------------
@st.cache_data(show_spinner=False)
def _to_csv(df_: pd.DataFrame) -> bytes:
    return df_.to_csv(index=False).encode("utf-8")

out = stats.rename(columns={"current":"value","pct_change":"pct_change_vs_comp"})
st.download_button(
    "Download current-range dataset (CSV)",
    data=_to_csv(out),
    file_name=f"elm_{indicator_selected}_{range_label.replace('–','_')}.csv",
    mime="text/csv",
)



# app.py
"""
ReLIFE — Financial service mockup (deep renovation: PV + heat pump)
- Inputs (left): technical + financial only.
- Outputs (right): NPV/IRR histograms, CAPEX/OPEX time series (median + 10/90% band), ARV bar chart, summary table.
- Monte-Carlo sampling to produce distributions (NPV, IRR, CAPEX noise, OPEX variability, ARV variability).
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="ReLIFE Financial Service", layout="wide")

# -------------------------
# Region / archetype defaults (hidden to user)
# -------------------------
COUNTRY_REGION = {
    "Italy": {"regions": ["All (IT)", "Lombardia", "Trentino-Alto Adige"], "cost_mul": 1.0, "energy_price": 0.22, "price_per_m2": 2000},
    "Germany": {"regions": ["All (DE)", "Bavaria", "Berlin"], "cost_mul": 1.05, "energy_price": 0.25, "price_per_m2": 3000},
    "Spain": {"regions": ["All (ES)", "Catalonia", "Madrid"], "cost_mul": 0.95, "energy_price": 0.20, "price_per_m2": 1700},
}

ARCHETYPE_PROFILES = {
    "Apartment (multi-family)": {"capex_per_m2": 40.0, "annual_savings_kwh_per_m2": 80.0, "maintenance_per_m2": 0.15, "life_years": 15, "baseline_kwh_m2": 150},
    "Single-family house":      {"capex_per_m2": 60.0, "annual_savings_kwh_per_m2": 90.0, "maintenance_per_m2": 0.20, "life_years": 20, "baseline_kwh_m2": 150},
    "Terraced house":           {"capex_per_m2": 50.0, "annual_savings_kwh_per_m2": 85.0, "maintenance_per_m2": 0.18, "life_years": 18, "baseline_kwh_m2": 150},
    "Office building":          {"capex_per_m2": 70.0, "annual_savings_kwh_per_m2": 100.0, "maintenance_per_m2": 0.30, "life_years": 20, "baseline_kwh_m2": 250},
    "School / Public":          {"capex_per_m2": 55.0, "annual_savings_kwh_per_m2": 95.0, "maintenance_per_m2": 0.25, "life_years": 25, "baseline_kwh_m2": 300},
}

# Hidden MC & market defaults
_MC_RUNS = 4000
_ENERGY_PRICE_STD = 0.03      # €/kWh noise
_INFLATION_MEAN = 0.02
_INFLATION_STD = 0.01
_CAPEX_NOISE_PCT = 0.08       # 8% std dev for capex sampling (to generate distribution)
_OPEX_NOISE_PCT = 0.05        # 5% std dev per-year maintenance noise

# -------------------------
# Finance helpers (robust IRR)
# -------------------------
def annuity_payment(principal, rate, years):
    if years == 0:
        return 0.0
    if rate == 0:
        return principal / years
    r = rate
    return principal * (r * (1 + r) ** years) / ((1 + r) ** years - 1)

def npv(cashflows, discount_rate):
    years = np.arange(len(cashflows))
    return np.sum(cashflows / ((1 + discount_rate) ** years))

def compute_irr(cashflows):
    cf = np.asarray(cashflows, dtype=float)
    if cf.size < 2:
        return np.nan
    if not (np.any(cf < 0) and np.any(cf > 0)):
        return np.nan
    # try numpy_financial
    try:
        import numpy_financial as npf
        irr = npf.irr(list(cf))
        if np.isfinite(irr):
            return float(irr)
    except Exception:
        pass
    # fallback to np.irr
    try:
        irr = np.irr(list(cf))
        if np.isfinite(irr):
            return float(irr)
    except Exception:
        pass
    # polynomial roots fallback (x = 1 + r)
    try:
        coeffs = cf.tolist()
        roots = np.roots(coeffs)
        real_roots = [r.real for r in roots if np.isreal(r)]
        candidate_rates = []
        for x in real_roots:
            r = x - 1.0
            if r > -0.9999 and r < 10:
                candidate_rates.append(r)
        if candidate_rates:
            return float(np.median(candidate_rates))
    except Exception:
        pass
    return np.nan

# -------------------------
# UI: inputs (left) and outputs (right)
# -------------------------
import base64
with open('Logo.png', 'rb') as f:
    logo_data = base64.b64encode(f.read()).decode()

st.markdown(f"""
    <style>
    .logo-title-container {{
        display: flex;
        align-items: center;
        gap: 20px;
    }}
    .logo-title-container h1 {{
        margin: 0;
        padding: 0;
    }}
    </style>
    <div class="logo-title-container">
        <img src="data:image/png;base64,{logo_data}" width="220">
        <h1>Financial Service</h1>
    </div>
    """, unsafe_allow_html=True)

left_col, right_col = st.columns([1, 2])

with left_col:
    st.subheader("Technical inputs (minimal)")
    country = st.selectbox("Country", options=list(COUNTRY_REGION.keys()), index=1)  # index=1 for Germany
    regions = COUNTRY_REGION[country]["regions"]
    region = st.selectbox("Region", options=regions)

    # set default to Single-family house for the demo example
    st.selectbox("Building archetype", options=list(ARCHETYPE_PROFILES.keys()), index=list(ARCHETYPE_PROFILES.keys()).index("Single-family house"), key="archetype_select")
    archetype = st.session_state["archetype_select"]

    construction_year = st.number_input("Year of construction", value=1990, min_value=1800, max_value=2025, step=1)
    renovate_checkbox = st.checkbox("Has retrofit / renovation year?")
    renovation_year = None
    if renovate_checkbox:
        renovation_year = st.number_input("Year of renovation", value=2025, min_value=1800, max_value=2025, step=1)

    n_stories = st.number_input("Number of stories", value=2, min_value=1, max_value=30, step=1)
    floor_area = st.number_input("Net floor area (m²)", value=140.0, min_value=10.0, step=1.0)

    st.subheader("Financial inputs")
    interest_rate = st.number_input("Discount rate (annual, decimal)", value=0.035, step=0.005, format="%.3f")
    financing = st.selectbox("Type of financing", options=["Self-financed", "Loan", "Subsidy", "On-bill"])

    loan_amount = 0.0
    loan_term = 0
    loan_interest_rate = interest_rate
    subsidy_pct = 0.0
    
    if financing == "Loan":
        st.markdown("**Loan details:**")
        loan_amount = st.number_input("Loan amount (€)", value=10000.0, step=100.0, min_value=0.0)
        loan_term = st.number_input("Loan term (years)", value=7, min_value=1, step=1)
        loan_interest_rate = st.number_input("Loan interest rate (annual, decimal)", value=0.045, step=0.005, format="%.3f")
    elif financing == "Subsidy":
        st.markdown("**Subsidy details:**")
        subsidy_pct = st.number_input("Subsidy (% of CAPEX)", value=20.0, min_value=0.0, max_value=100.0, step=5.0) / 100.0

    # Deep renovation is always enabled (PV + HP)
    deep_reno = True

    st.markdown("---")
    st.markdown("Only the fields above are editable. Other model parameters are derived behind the scenes.")
    run_sim = st.button("Run simulation")

# -------------------------
# Backend: derive parameters (hidden)
# -------------------------
def derive_building_parameters(archetype, floor_area, n_stories, construction_year, renovation_year, country, deep_reno):
    profile = ARCHETYPE_PROFILES[archetype]
    base_capex = profile["capex_per_m2"] * floor_area
    # deep renovation increases base retrofit intensity and adds PV+HP kit fixed cost
    if deep_reno:
        # scale base capex and add fixed PV+HP kit cost
        capex = base_capex * 2.2 + 7000.0  # example: heavier retrofit + PV/HP
    else:
        capex = base_capex
    # maintenance (annual)
    maintenance = profile["maintenance_per_m2"] * floor_area
    # annual energy savings in kWh (profile * area) — deep renovation increases savings
    annual_savings_kwh = profile["annual_savings_kwh_per_m2"] * floor_area * (1.2 if deep_reno else 1.0)
    # baseline consumption for the building (kWh/year)
    baseline_kwh = profile.get("baseline_kwh_m2", 150) * floor_area
    post_reno_consumption_kwh = max(baseline_kwh - annual_savings_kwh, 0.0)
    
    # EPC rating calculation (simple formula based on kWh/m2/year)
    baseline_kwh_m2 = baseline_kwh / floor_area
    post_reno_kwh_m2 = post_reno_consumption_kwh / floor_area
    
    def get_epc_rating(kwh_m2_year):
        """Simple EPC rating based on energy consumption per m2"""
        if kwh_m2_year < 30: return "A+"
        elif kwh_m2_year < 50: return "A"
        elif kwh_m2_year < 75: return "B"
        elif kwh_m2_year < 100: return "C"
        elif kwh_m2_year < 150: return "D"
        elif kwh_m2_year < 200: return "E"
        elif kwh_m2_year < 250: return "F"
        else: return "G"
    
    epc_before = get_epc_rating(baseline_kwh_m2)
    epc_after = get_epc_rating(post_reno_kwh_m2)
    
    # Energy mix estimation (simple % breakdown)
    # Before renovation: mostly gas/oil heating, some electricity
    energy_mix_before = {
        "Gas/Oil heating": 65,
        "Electricity": 30,
        "Other": 5
    }
    
    # After renovation with PV + Heat Pump: shift to electricity (renewable)
    # Heat pump converts to electric, PV provides renewable electricity
    energy_mix_after = {
        "Solar PV (self-consumption)": 40,
        "Electricity (grid)": 50,
        "Gas/Oil heating": 5,
        "Other": 5
    }
    
    life = profile["life_years"]
    if renovation_year and renovation_year > construction_year:
        life = min(life + 5, 40)
    region_info = COUNTRY_REGION[country]
    capex *= region_info["cost_mul"]
    energy_price = region_info["energy_price"]
    price_per_m2 = region_info["price_per_m2"]
    return {
        "capex": capex,
        "maintenance": maintenance,
        "annual_savings_kwh": annual_savings_kwh,
        "baseline_kwh": baseline_kwh,
        "post_reno_kwh": post_reno_consumption_kwh,
        "baseline_kwh_m2": baseline_kwh_m2,
        "post_reno_kwh_m2": post_reno_kwh_m2,
        "epc_before": epc_before,
        "epc_after": epc_after,
        "energy_mix_before": energy_mix_before,
        "energy_mix_after": energy_mix_after,
        "project_life": life,
        "energy_price": energy_price,
        "price_per_m2": price_per_m2
    }

# -------------------------
# Monte-Carlo simulation
# -------------------------
@st.cache_data(ttl=60, show_spinner=False)
def run_monte_carlo(n_runs, derived, interest_rate, financing, loan_amount, loan_term, loan_interest_rate, subsidy_pct):
    # Use a seed based on input parameters for reproducibility but sensitivity to changes
    import hashlib
    seed_str = f"{interest_rate}-{financing}-{loan_amount}-{loan_term}-{loan_interest_rate}-{subsidy_pct}"
    seed = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed=seed)
    capex_base = derived["capex"]
    maintenance_base = derived["maintenance"]
    annual_savings_kwh = derived["annual_savings_kwh"]
    baseline_kwh = derived["baseline_kwh"]
    post_kwh = derived["post_reno_kwh"]
    proj_life = int(derived["project_life"])
    energy_price_base = derived["energy_price"]
    price_per_m2 = derived["price_per_m2"]

    # store per-sim indicators
    npvs = np.empty(n_runs)
    irrs = np.empty(n_runs)
    capex_arr = np.empty(n_runs)
    cum_opex_arr = np.empty(n_runs)
    arv_pct_arr = np.empty(n_runs)

    # also store OPEX per year across sims for percentile bands
    opex_matrix = np.zeros((n_runs, proj_life))

    years = np.arange(proj_life + 1)

    for i in range(n_runs):
        # sample market variables
        disc_r = max(rng.normal(interest_rate, 0.005), 0.0)
        inflation = max(rng.normal(_INFLATION_MEAN, _INFLATION_STD), -0.99)
        price_sample = max(rng.normal(energy_price_base, _ENERGY_PRICE_STD), 0.001)
        # capex noise
        capex_sim = max(rng.normal(capex_base, _CAPEX_NOISE_PCT * capex_base), 0.0)
        capex_arr[i] = capex_sim

        # annual energy price series (escalating)
        energy_prices = np.array([price_sample * ((1 + inflation) ** (y - 1)) for y in years[1:]])  # y=1..N

        # annual energy cost after renovation (post_kwh * price each year)
        annual_energy_costs = post_kwh * energy_prices

        # maintenance noise
        maintenance_yearly = max(rng.normal(maintenance_base, _OPEX_NOISE_PCT * maintenance_base), 0.0)

        # OPEX per year = energy cost + maintenance
        opex_per_year = maintenance_yearly + annual_energy_costs
        opex_matrix[i, :] = opex_per_year

        initial_invest = capex_sim
        annual_loan_payment = 0.0
        onbill_array = np.zeros_like(annual_energy_costs)
        if financing == "Loan" and loan_amount > 0:
            principal = min(loan_amount, capex_sim)
            annual_loan_payment = annuity_payment(principal, loan_interest_rate, loan_term)
            initial_invest = capex_sim - principal
        elif financing == "Subsidy":
            subsidy = subsidy_pct * capex_sim
            initial_invest = capex_sim - subsidy
        elif financing == "On-bill":
            onbill_array = 0.4 * (annual_energy_costs + maintenance_yearly)  # portion of OPEX used for repayment

        # cashflows: year0 = -initial_invest, year t = (savings in €) - maintenance - loan - onbill
        # We compute savings as baseline energy costs - post_reno energy cost
        baseline_energy_costs = baseline_kwh * np.array([price_sample * ((1 + inflation) ** (y - 1)) for y in years[1:]])
        annual_savings_eur = baseline_energy_costs - annual_energy_costs  # positive number = savings
        cashflows = np.zeros(proj_life + 1)
        cashflows[0] = -initial_invest
        for idx, y in enumerate(range(1, proj_life + 1)):
            saving = annual_savings_eur[idx]
            loan_pay = annual_loan_payment if (financing == "Loan" and y <= loan_term) else 0.0
            onbill_pay = onbill_array[idx] if financing == "On-bill" else 0.0
            outflows = maintenance_yearly + loan_pay + onbill_pay  # only maintenance, not full opex (savings already accounts for energy)
            cashflows[y] = saving - outflows  # net cash benefit to owner

        npv_val = npv(cashflows, disc_r)
        irr_val = compute_irr(cashflows)
        cumulative_opex = np.sum(opex_per_year)  # over project life
        # ARV logic: base building value + uplift pct from deep renovation and energy savings
        base_building_value = price_per_m2 * (baseline_kwh / derived["baseline_kwh"]) * (derived.get("post_reno_kwh", 1) * 0 + 1)  # this line is only to ensure variable exists
        # simpler and clearer ARV formula:
        base_value = price_per_m2 * (derived.get("baseline_kwh", 0) * 0 + 1) * 1.0 * (1.0)  # dummy to use price_per_m2
        base_value = price_per_m2 * (derived.get("post_reno_kwh", 0) * 0 + 1) * (derived.get("post_reno_kwh", 0) * 0 + 1)  # keep simple
        # Real base building value computed from floor area; we do it outside loop for accuracy (reassign later)
        # Compute ARV pct: base uplift for deep reno + extra from kWh savings relative to area
        uplift_base = 0.20  # 20% uplift for a deep renovation (PV+HP)
        extra = min(annual_savings_kwh / 10000.0, 0.10)  # small extra linked to annual kWh saved
        arv_pct = uplift_base + extra
        # add a tiny sampling noise
        arv_pct = max(rng.normal(arv_pct, 0.02), 0.0)
        arv_pct_arr[i] = arv_pct

        npvs[i] = npv_val
        irrs[i] = irr_val if np.isfinite(irr_val) else np.nan
        cum_opex_arr[i] = cumulative_opex

    # For plots we'll need medians and percentiles for opex per year and scalars
    # Percentiles for OPEX across simulations per year
    opex_p10 = np.percentile(opex_matrix, 10, axis=0)
    opex_p50 = np.percentile(opex_matrix, 50, axis=0)
    opex_p90 = np.percentile(opex_matrix, 90, axis=0)

    results = {
        "npvs": npvs,
        "irrs": irrs,
        "capex": capex_arr,
        "cum_opex": cum_opex_arr,
        "arv_pct": arv_pct_arr,
        "opex_p10": opex_p10,
        "opex_p50": opex_p50,
        "opex_p90": opex_p90,
        "years": np.arange(1, proj_life + 1),  # year indices for OPEX series
        "proj_life": proj_life
    }
    return results

# -------------------------
# Run and display (right)
# -------------------------
if run_sim:
    derived = derive_building_parameters(archetype, floor_area, n_stories, construction_year, renovation_year, country, deep_reno)
    with st.spinner("Running Monte Carlo..."):
        mc = run_monte_carlo(_MC_RUNS, derived, interest_rate, financing, float(loan_amount), int(loan_term or 0), loan_interest_rate, subsidy_pct)
    st.success("Simulation complete")

    with right_col:
        st.header("Results Summary")
        
        # Forecasting Service Outputs (at the top)
        st.subheader("Forecasting Service Outputs")
        
        # Display in 3 columns: EPC Rating, Energy Mix, Energy Savings
        fcol1, fcol2, fcol3 = st.columns([1, 1, 1])
        
        with fcol1:
            st.markdown("**EPC Rating**")
            # Color-code EPC ratings
            epc_colors = {"A+": "🟢", "A": "🟢", "B": "🟡", "C": "🟡", "D": "🟠", "E": "🟠", "F": "🔴", "G": "🔴"}
            epc_before_icon = epc_colors.get(derived["epc_before"], "⚪")
            epc_after_icon = epc_colors.get(derived["epc_after"], "⚪")
            st.markdown(f"Before: {epc_before_icon} **{derived['epc_before']}** ({derived['baseline_kwh_m2']:.1f} kWh/m²/year)")
            st.markdown(f"After: {epc_after_icon} **{derived['epc_after']}** ({derived['post_reno_kwh_m2']:.1f} kWh/m²/year)")
            improvement_class = ord(derived['epc_before'][-1]) - ord(derived['epc_after'][-1])
            if improvement_class > 0:
                st.markdown(f"✅ Improvement: **{improvement_class} class{'es' if improvement_class > 1 else ''}**")
        
        with fcol2:
            st.markdown("**Energy Mix**")
            st.markdown("*Before renovation:*")
            for source, pct in derived["energy_mix_before"].items():
                st.markdown(f"• {source}: {pct}%")
            st.markdown("*After renovation:*")
            for source, pct in derived["energy_mix_after"].items():
                st.markdown(f"• {source}: {pct}%")
        
        with fcol3:
            st.markdown("**Annual Energy Savings**")
            st.metric(
                label="Energy reduction",
                value=f"{derived['annual_savings_kwh']:,.0f} kWh/year",
                delta=f"-{(derived['annual_savings_kwh']/derived['baseline_kwh']*100):.1f}%"
            )
            annual_cost_savings = derived['annual_savings_kwh'] * derived['energy_price']
            st.metric(
                label="Cost savings (Year 1)",
                value=f"€{annual_cost_savings:,.0f}/year"
            )
        
        st.markdown("---")
        st.markdown("### Financial Analysis")

        # 1) NPV histogram (compact)
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("NPV distribution")
            fig, ax = plt.subplots(figsize=(4, 2.4))
            ax.hist(mc["npvs"][~np.isnan(mc["npvs"])], bins=40)
            ax.set_xlabel("NPV (€)", fontsize=9)
            ax.set_ylabel("Freq", fontsize=9)
            ax.tick_params(axis='both', which='major', labelsize=8)
            st.pyplot(fig, use_container_width=True)

        # 2) IRR histogram (compact)
        with col2:
            st.subheader("IRR distribution")
            irr_vals = mc["irrs"][~np.isnan(mc["irrs"])]
            if irr_vals.size == 0:
                st.warning("IRR undefined across simulations (no sign change in cashflows). Use NPV for decision.")
                fig2, ax2 = plt.subplots(figsize=(4, 2.4))
                ax2.text(0.5, 0.5, "No IRR", ha="center", va="center", fontsize=10, alpha=0.6)
                ax2.set_xticks([])
                ax2.set_yticks([])
                st.pyplot(fig2, use_container_width=True)
            else:
                fig2, ax2 = plt.subplots(figsize=(4, 2.4))
                ax2.hist(irr_vals * 100, bins=40)
                ax2.set_xlabel("IRR (%)", fontsize=9)
                ax2.set_ylabel("Freq", fontsize=9)
                ax2.tick_params(axis='both', which='major', labelsize=8)
                st.pyplot(fig2, use_container_width=True)

        st.markdown("---")

        # 3) CAPEX & OPEX time series (median + 10/90%)
        st.subheader("CAPEX (t=0) and annual OPEX (years 1..N)")
        years = mc["years"]
        proj_life = mc["proj_life"]
        # CAPEX scalar (show median)
        capex_med = np.median(mc["capex"])
        cum_opex_med = np.median(mc["cum_opex"])
        # Plot: bar at t=0 for CAPEX, and shaded band + median line for OPEX
        fig3, ax3 = plt.subplots(figsize=(9, 3.0))
        # CAPEX bar at x=0
        ax3.bar(0, capex_med, width=0.6, label="CAPEX (median)", color="tab:orange")
        # OPEX median line across years (plot at x=1..N)
        ax3.plot(years, mc["opex_p50"], label="OPEX median (yearly)", linewidth=2)
        ax3.fill_between(years, mc["opex_p10"], mc["opex_p90"], alpha=0.25, label="OPEX 10%–90% band")
        ax3.set_xlabel("Year (0 = CAPEX)", fontsize=9)
        ax3.set_ylabel("€", fontsize=9)
        ax3.set_xticks(np.concatenate(([0], years)))
        ax3.set_xlim(-0.5, proj_life + 0.5)
        ax3.legend(fontsize=8)
        ax3.tick_params(axis='both', which='major', labelsize=8)
        st.pyplot(fig3, use_container_width=True)

        st.markdown(f"**Median CAPEX:** €{capex_med:,.0f}  —  **Median cumulative OPEX (lifetime):** €{cum_opex_med:,.0f}")

        st.markdown("---")

        # 4) ARV: compute base building value from price_per_m2 * floor_area, then show uplift % median
        st.subheader("ARV — After Renovation Value")

        base_value = derived["price_per_m2"] * floor_area
        arv_median_pct = np.median(mc["arv_pct"]) * 100
        arv_p10_pct = np.percentile(mc["arv_pct"], 10) * 100
        arv_p90_pct = np.percentile(mc["arv_pct"], 90) * 100
        after_value_median = base_value * (1.0 + np.median(mc["arv_pct"]))

        # bar chart comparing before / after (median)
        fig4, ax4 = plt.subplots(figsize=(5, 2.4))
        bars = ax4.bar([0, 1], [base_value, after_value_median], color=["#7f8c8d", "#2ecc71"], width=0.6)
        ax4.set_xticks([0, 1])
        ax4.set_xticklabels(["Before", "After (median)"])
        ax4.set_ylabel("Estimated building value (€)", fontsize=9)
        # annotate values and percentage inside/on bars
        ax4.text(0, base_value/2, f"€{base_value:,.0f}", ha="center", va="center", fontsize=10, color="white", fontweight="bold")
        ax4.text(1, after_value_median/2, f"€{after_value_median:,.0f}\n+{arv_median_pct:.1f}%", ha="center", va="center", fontsize=10, color="white", fontweight="bold")
        ax4.tick_params(axis='both', which='major', labelsize=8)
        st.pyplot(fig4, use_container_width=True)

        st.markdown(f"**Base building value:** €{base_value:,.0f}  →  **After-renovation (median):** €{after_value_median:,.0f}")

        st.markdown("---")

        # 5) Summary table with percentiles (10% / median / 90%)
        st.subheader("Summary table (10% / median / 90%)")
        df_summary = pd.DataFrame({
            "10%": [
                np.nanpercentile(mc["npvs"], 10),
                np.nanpercentile(mc["irrs"] * 100, 10),
                np.percentile(mc["capex"], 10),
                np.percentile(mc["cum_opex"], 10),
                np.percentile(mc["arv_pct"] * 100, 10)
            ],
            "median": [
                np.nanpercentile(mc["npvs"], 50),
                np.nanpercentile(mc["irrs"] * 100, 50),
                np.percentile(mc["capex"], 50),
                np.percentile(mc["cum_opex"], 50),
                np.percentile(mc["arv_pct"] * 100, 50)
            ],
            "90%": [
                np.nanpercentile(mc["npvs"], 90),
                np.nanpercentile(mc["irrs"] * 100, 90),
                np.percentile(mc["capex"], 90),
                np.percentile(mc["cum_opex"], 90),
                np.percentile(mc["arv_pct"] * 100, 90)
            ]
        }, index=["NPV (EUR)", "IRR (%)", "CAPEX (EUR)", "Cumulative OPEX (EUR)", "ARV uplift (%)"])

        # format and show larger table
        st.dataframe(df_summary.style.format({
            "10%": "{:,.2f}",
            "median": "{:,.2f}",
            "90%": "{:,.2f}"
        }), height=250)

else:
    with right_col:
        st.info("Fill inputs on the left and press **Run simulation** to generate results (NPV, IRR, CAPEX/OPEX time series, ARV).")

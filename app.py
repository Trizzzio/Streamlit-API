import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Financial Service Analytics",
    page_icon="💰",
    layout="wide"
)

# Title and description
st.title("💰 Financial Service Analytics Dashboard")
st.markdown("Calculate key financial indicators for investment analysis")

# Sidebar for user inputs
st.sidebar.header("📊 Input Parameters")

# Discount rate input
discount_rate = st.sidebar.slider(
    "Discount Rate (%)",
    min_value=0.0,
    max_value=20.0,
    value=10.0,
    step=0.5,
    help="The discount rate used for NPV calculations"
) / 100

# Initial investment
initial_investment = st.sidebar.number_input(
    "Initial Investment ($)",
    min_value=0,
    value=500000,
    step=10000,
    help="Initial capital expenditure"
)

# Tax rate
tax_rate = st.sidebar.slider(
    "Tax Rate (%)",
    min_value=0.0,
    max_value=50.0,
    value=25.0,
    step=1.0,
    help="Corporate tax rate"
) / 100

# File upload or use sample data
st.sidebar.header("📁 Data Source")
use_sample = st.sidebar.checkbox("Use Sample Data", value=True)

if use_sample:
    # Load sample data
    data_path = Path(__file__).parent / "sample_data.csv"
    if data_path.exists():
        df = pd.read_csv(data_path)
    else:
        # Create sample data if file doesn't exist
        df = pd.DataFrame({
            'Year': range(0, 6),
            'Revenue': [0, 250000, 350000, 450000, 550000, 650000],
            'Operating_Costs': [0, 100000, 120000, 140000, 160000, 180000],
            'Capital_Expenditure': [500000, 0, 0, 0, 0, 0]
        })
else:
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV file",
        type=['csv'],
        help="Upload a CSV with columns: Year, Revenue, Operating_Costs, Capital_Expenditure"
    )
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        st.warning("Please upload a CSV file or use sample data")
        st.stop()

# Calculate financial metrics
def calculate_cash_flows(df, tax_rate):
    """Calculate net cash flows from the data"""
    df = df.copy()
    df['EBIT'] = df['Revenue'] - df['Operating_Costs']
    df['Tax'] = df['EBIT'] * tax_rate
    df['NOPAT'] = df['EBIT'] - df['Tax']
    df['Net_Cash_Flow'] = df['NOPAT'] - df['Capital_Expenditure']
    return df

def calculate_npv(cash_flows, discount_rate):
    """Calculate Net Present Value"""
    npv = sum(cf / (1 + discount_rate) ** i for i, cf in enumerate(cash_flows))
    return npv

def calculate_irr(cash_flows, max_iterations=1000):
    """Calculate Internal Rate of Return using Newton-Raphson method"""
    if len(cash_flows) == 0:
        return None
    
    # Initial guess
    irr = 0.1
    
    for _ in range(max_iterations):
        npv = sum(cf / (1 + irr) ** i for i, cf in enumerate(cash_flows))
        dnpv = sum(-i * cf / (1 + irr) ** (i + 1) for i, cf in enumerate(cash_flows))
        
        if abs(dnpv) < 1e-10:
            break
            
        new_irr = irr - npv / dnpv
        
        if abs(new_irr - irr) < 1e-6:
            return new_irr
            
        irr = new_irr
    
    return irr

def calculate_roi(total_gains, initial_investment):
    """Calculate Return on Investment"""
    if initial_investment == 0:
        return 0
    return (total_gains / initial_investment) * 100

def calculate_payback_period(cash_flows):
    """Calculate Payback Period"""
    cumulative = 0
    for i, cf in enumerate(cash_flows):
        cumulative += cf
        if cumulative >= 0:
            # Linear interpolation for fractional year
            if i == 0:
                return 0
            previous_cumulative = cumulative - cf
            fraction = -previous_cumulative / cf
            return i - 1 + fraction
    return None  # Investment not recovered

def calculate_profitability_index(npv, initial_investment):
    """Calculate Profitability Index"""
    if initial_investment == 0:
        return 0
    return (npv + abs(initial_investment)) / abs(initial_investment)

# Process the data
df_calc = calculate_cash_flows(df, tax_rate)
cash_flows = df_calc['Net_Cash_Flow'].values

# Calculate all metrics
npv = calculate_npv(cash_flows, discount_rate)
irr = calculate_irr(cash_flows)
total_gains = sum(cash_flows[1:])  # Exclude initial investment
roi = calculate_roi(total_gains, abs(cash_flows[0]) if cash_flows[0] < 0 else initial_investment)
payback = calculate_payback_period(cash_flows)
pi = calculate_profitability_index(npv, abs(cash_flows[0]) if cash_flows[0] < 0 else initial_investment)

# Display metrics in columns
st.header("📈 Key Financial Indicators")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Net Present Value (NPV)",
        f"${npv:,.2f}",
        delta="Positive" if npv > 0 else "Negative",
        delta_color="normal" if npv > 0 else "inverse"
    )

with col2:
    if irr is not None:
        st.metric(
            "Internal Rate of Return (IRR)",
            f"{irr * 100:.2f}%",
            delta=f"{(irr - discount_rate) * 100:.2f}% vs discount rate"
        )
    else:
        st.metric("Internal Rate of Return (IRR)", "N/A")

with col3:
    st.metric(
        "Return on Investment (ROI)",
        f"{roi:.2f}%"
    )

with col4:
    if payback is not None:
        st.metric(
            "Payback Period",
            f"{payback:.2f} years"
        )
    else:
        st.metric("Payback Period", "Not recovered")

col5, col6 = st.columns(2)

with col5:
    st.metric(
        "Profitability Index (PI)",
        f"{pi:.2f}",
        delta="Acceptable" if pi > 1 else "Not acceptable",
        delta_color="normal" if pi > 1 else "inverse"
    )

with col6:
    total_cash_flow = sum(cash_flows)
    st.metric(
        "Total Net Cash Flow",
        f"${total_cash_flow:,.2f}"
    )

# Interpretation
st.header("💡 Interpretation")
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Investment Decision")
    if npv > 0 and (irr is None or irr > discount_rate) and pi > 1:
        st.success("✅ **ACCEPT** - The project appears financially viable")
        st.write("- NPV is positive, creating value")
        if irr is not None:
            st.write(f"- IRR ({irr*100:.2f}%) exceeds the discount rate ({discount_rate*100:.2f}%)")
        st.write(f"- Profitability Index ({pi:.2f}) is greater than 1")
    elif npv < 0 or pi < 1:
        st.error("❌ **REJECT** - The project may not be financially viable")
        if npv < 0:
            st.write("- NPV is negative, destroying value")
        if irr is not None and irr < discount_rate:
            st.write(f"- IRR ({irr*100:.2f}%) is below the discount rate ({discount_rate*100:.2f}%)")
        if pi < 1:
            st.write(f"- Profitability Index ({pi:.2f}) is less than 1")
    else:
        st.warning("⚠️ **REVIEW** - Mixed signals, requires further analysis")

with col_b:
    st.subheader("Risk Considerations")
    if payback is not None:
        if payback < 3:
            st.info(f"Low risk: Quick payback in {payback:.2f} years")
        elif payback < 5:
            st.info(f"Moderate risk: Payback in {payback:.2f} years")
        else:
            st.warning(f"Higher risk: Extended payback of {payback:.2f} years")
    else:
        st.error("High risk: Investment not recovered within project lifetime")

# Data visualization
st.header("📊 Financial Data Visualization")

# Tabs for different visualizations
tab1, tab2, tab3, tab4 = st.tabs(["Cash Flow Analysis", "Cumulative Cash Flow", "Revenue vs Costs", "Data Table"])

with tab1:
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        x=df_calc['Year'],
        y=df_calc['Net_Cash_Flow'],
        name='Net Cash Flow',
        marker_color=['red' if x < 0 else 'green' for x in df_calc['Net_Cash_Flow']]
    ))
    fig1.update_layout(
        title='Annual Net Cash Flow',
        xaxis_title='Year',
        yaxis_title='Cash Flow ($)',
        hovermode='x unified'
    )
    st.plotly_chart(fig1, use_container_width=True)

with tab2:
    cumulative_cf = np.cumsum(cash_flows)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=df_calc['Year'],
        y=cumulative_cf,
        mode='lines+markers',
        name='Cumulative Cash Flow',
        line=dict(width=3),
        fill='tozeroy'
    ))
    fig2.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Break-even")
    fig2.update_layout(
        title='Cumulative Cash Flow Over Time',
        xaxis_title='Year',
        yaxis_title='Cumulative Cash Flow ($)',
        hovermode='x unified'
    )
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=df_calc['Year'],
        y=df_calc['Revenue'],
        mode='lines+markers',
        name='Revenue',
        line=dict(color='green', width=3)
    ))
    fig3.add_trace(go.Scatter(
        x=df_calc['Year'],
        y=df_calc['Operating_Costs'],
        mode='lines+markers',
        name='Operating Costs',
        line=dict(color='red', width=3)
    ))
    fig3.add_trace(go.Scatter(
        x=df_calc['Year'],
        y=df_calc['EBIT'],
        mode='lines+markers',
        name='EBIT',
        line=dict(color='blue', width=3)
    ))
    fig3.update_layout(
        title='Revenue, Costs, and EBIT Trends',
        xaxis_title='Year',
        yaxis_title='Amount ($)',
        hovermode='x unified'
    )
    st.plotly_chart(fig3, use_container_width=True)

with tab4:
    st.subheader("Detailed Financial Data")
    st.dataframe(
        df_calc.style.format({
            'Revenue': '${:,.2f}',
            'Operating_Costs': '${:,.2f}',
            'Capital_Expenditure': '${:,.2f}',
            'EBIT': '${:,.2f}',
            'Tax': '${:,.2f}',
            'NOPAT': '${:,.2f}',
            'Net_Cash_Flow': '${:,.2f}'
        }),
        use_container_width=True
    )
    
    # Download button
    csv = df_calc.to_csv(index=False)
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv,
        file_name="financial_analysis_results.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Financial Service Analytics Dashboard | Built with Streamlit</p>
    <p style='font-size: 0.8em; color: gray;'>
        Disclaimer: This tool is for educational and mockup purposes only. 
        Consult with financial professionals for actual investment decisions.
    </p>
</div>
""", unsafe_allow_html=True)

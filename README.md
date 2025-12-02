# Financial Service Analytics Dashboard

A Streamlit-based web application for calculating and visualizing key financial indicators for investment analysis.

## Features

- **Financial Metrics Calculation:**
  - Net Present Value (NPV)
  - Internal Rate of Return (IRR)
  - Return on Investment (ROI)
  - Payback Period
  - Profitability Index (PI)

- **Interactive Inputs:**
  - Adjustable discount rate
  - Custom initial investment
  - Configurable tax rate
  - CSV file upload or sample data

- **Visualizations:**
  - Annual net cash flow charts
  - Cumulative cash flow analysis
  - Revenue vs. costs trends
  - Interactive data tables

## Local Setup

### Prerequisites
- Python 3.8 or higher

### Installation

1. Clone or download this repository

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

4. Open your browser to `http://localhost:8501`

## Deployment to Streamlit Cloud

### Option 1: Deploy via GitHub (Recommended)

1. **Create a GitHub repository:**
   - Go to [GitHub](https://github.com) and create a new repository
   - Upload these files to the repository:
     - `app.py`
     - `requirements.txt`
     - `sample_data.csv`
     - `README.md`

2. **Deploy to Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account
   - Click "New app"
   - Select your repository, branch (main), and main file path (`app.py`)
   - Click "Deploy"
   - Your app will be live at: `https://[your-app-name].streamlit.app`

### Option 2: Direct Deployment

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Paste your repository URL
4. Set the main file path to `app.py`
5. Click "Deploy"

## CSV Data Format

If uploading your own data, use this format:

```csv
Year,Revenue,Operating_Costs,Capital_Expenditure
0,0,0,500000
1,250000,100000,0
2,350000,120000,0
3,450000,140000,0
4,550000,160000,0
5,650000,180000,0
```

**Columns:**
- `Year`: Project year (starting from 0)
- `Revenue`: Annual revenue
- `Operating_Costs`: Annual operating expenses
- `Capital_Expenditure`: Capital investments

## Usage Tips

1. **Discount Rate**: Adjust based on your required rate of return or cost of capital
2. **Tax Rate**: Set according to applicable corporate tax rates
3. **Investment Decision**: 
   - Accept if NPV > 0, IRR > discount rate, and PI > 1
   - Reject if NPV < 0 or PI < 1
   - Review carefully if signals are mixed

## Disclaimer

This tool is for educational and demonstration purposes only. Always consult with qualified financial professionals before making investment decisions.

## License

MIT License - Feel free to use and modify for your needs.

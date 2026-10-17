import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# Load saved files
model = joblib.load(
    "house_price_project/house_price_model.pkl"
)

results = pd.read_csv(
    "house_price_project/model_comparison.csv"
)

feature_names = joblib.load(
    "house_price_project/feature_names.pkl"
)

# Custom styling
st.markdown("""
<style>

/* Background */
.stApp {
    background-color: #F3F8F5;
}

/* All headings */
h1, h2, h3 {
    color: #145A3A !important;
}

/* Normal text */
p, label {
    color: #183B2B !important;
}

/* Metric values */
[data-testid="stMetricValue"] {
    color: #176B45 !important;
}

/* Metric labels */
[data-testid="stMetricLabel"] {
    color: #315B46 !important;
}

/* Input labels */
.stSelectbox label,
.stNumberInput label,
.stSlider label {
    color: #183B2B !important;
}

/* Dropdown text */
.stSelectbox div,
.stNumberInput input {
    color: #183B2B !important;
}

/* Cards */
[data-testid="stMetric"] {
    background-color: #FFFFFF;
    border: 1px solid #D1E8D9;
    padding: 15px;
    border-radius: 12px;
}

/* Buttons */
.stButton button,
.stFormSubmitButton button {
    background-color: #168653;
    color: #FFFFFF !important;
    border-radius: 8px;
}

/* Light theme for charts */
[data-testid="stAppViewContainer"] {
    background-color: #F3F8F5;
}

[data-testid="stVegaLiteChart"],
[data-testid="stVegaLiteChart"] > div {
    background-color: #FFFFFF !important;
    border-radius: 12px;
}

/* Light chart containers */
.stPlotlyChart,
[data-testid="stArrowVegaLiteChart"] {
    background-color: #FFFFFF !important;
}

/* Dropdown styling */
.stSelectbox div[data-baseweb="select"] {
    background-color: #FFFFFF;
    color: #183B2B;
}
</style>
""", unsafe_allow_html=True)

# Header
st.title("🏠 House Price Prediction Dashboard")
st.caption(
    "Machine Learning | Real Estate Analytics | Predictive Insights"
)

st.divider()

# Performance metrics
best = results.sort_values(
    "R2 Score", ascending=False
).iloc[0]

c1, c2, c3 = st.columns(3)

c1.metric("🏆 Best Model", best["Model"])
c2.metric("📊 R² Score", f'{best["R2 Score"]:.3f}')
c3.metric("💰 MAE", f'{best["MAE"]:,.2f}')

st.divider()

# Model comparison
st.subheader("📊 Model Performance Comparison")

metric = st.selectbox(
    "Select performance metric",
    ["R2 Score", "MAE", "RMSE"]
)

chart_data = results.sort_values(
    metric,
    ascending=(metric != "R2 Score")
)

st.bar_chart(
    chart_data.set_index("Model")[[metric]],
    color="#22A06B"
)

st.divider()

# House prediction inputs
st.subheader("🏡 Predict a House Price")
st.write("Enter house details below.")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        overall_qual = st.slider(
            "Overall Quality",
            min_value=1,
            max_value=10,
            value=5
        )

        living_area = st.number_input(
            "Living Area (sq ft)",
            min_value=100,
            max_value=10000,
            value=1500
        )

        garage_cars = st.selectbox(
            "Garage Capacity",
            [0, 1, 2, 3, 4]
        )

    with col2:
        bedrooms = st.number_input(
            "Bedrooms",
            min_value=0,
            max_value=10,
            value=3
        )

        full_bath = st.number_input(
            "Full Bathrooms",
            min_value=0,
            max_value=5,
            value=2
        )

        year_built = st.number_input(
            "Year Built",
            min_value=1800,
            max_value=2026,
            value=2000
        )

    submit = st.form_submit_button(
        "🔮 Predict House Price"
    )

if submit:
    # Standard House Prices dataset baseline values.
    # Replace selected fields with user inputs.
    row = pd.DataFrame(
        [{
            feature: 0 for feature in feature_names
        }]
    )

    defaults = {
        "MSSubClass": 20,
        "MSZoning": "RL",
        "LotFrontage": 60.0,
        "LotArea": 8000,
        "Street": "Pave",
        "LotShape": "Reg",
        "LandContour": "Lvl",
        "Utilities": "AllPub",
        "LotConfig": "Inside",
        "LandSlope": "Gtl",
        "Neighborhood": "NAmes",
        "Condition1": "Norm",
        "Condition2": "Norm",
        "BldgType": "1Fam",
        "HouseStyle": "1Story",
        "OverallQual": overall_qual,
        "OverallCond": 5,
        "YearBuilt": year_built,
        "YearRemodAdd": year_built,
        "RoofStyle": "Gable",
        "RoofMatl": "CompShg",
        "Exterior1st": "VinylSd",
        "Exterior2nd": "VinylSd",
        "MasVnrType": "None",
        "MasVnrArea": 0.0,
        "ExterQual": "TA",
        "ExterCond": "TA",
        "Foundation": "PConc",
        "BsmtQual": "TA",
        "BsmtCond": "TA",
        "BsmtExposure": "No",
        "BsmtFinType1": "Unf",
        "BsmtFinSF1": 0,
        "BsmtFinType2": "Unf",
        "BsmtFinSF2": 0,
        "BsmtUnfSF": 500,
        "TotalBsmtSF": 800,
        "Heating": "GasA",
        "HeatingQC": "Ex",
        "CentralAir": "Y",
        "Electrical": "SBrkr",
        "1stFlrSF": 1000,
        "2ndFlrSF": 0,
        "LowQualFinSF": 0,
        "GrLivArea": living_area,
        "BsmtFullBath": 0,
        "BsmtHalfBath": 0,
        "FullBath": full_bath,
        "HalfBath": 0,
        "BedroomAbvGr": bedrooms,
        "KitchenAbvGr": 1,
        "KitchenQual": "TA",
        "TotRmsAbvGrd": 6,
        "Functional": "Typ",
        "Fireplaces": 0,
        "FireplaceQu": np.nan,
        "GarageType": "Attchd",
        "GarageYrBlt": year_built,
        "GarageFinish": "Unf",
        "GarageCars": garage_cars,
        "GarageArea": garage_cars * 240,
        "GarageQual": "TA",
        "GarageCond": "TA",
        "PavedDrive": "Y",
        "WoodDeckSF": 0,
        "OpenPorchSF": 25,
        "EnclosedPorch": 0,
        "3SsnPorch": 0,
        "ScreenPorch": 0,
        "PoolArea": 0,
        "PoolQC": np.nan,
        "Fence": np.nan,
        "MiscFeature": np.nan,
        "MiscVal": 0,
        "MoSold": 6,
        "YrSold": 2010,
        "SaleType": "WD",
        "SaleCondition": "Normal"
    }

    for col, value in defaults.items():
        if col in row.columns:
            row.loc[0, col] = value

    try:
        prediction = model.predict(row)[0]

        st.success("Prediction completed!")
        st.metric(
            "Estimated House Price",
            f"${prediction:,.0f}"
        )

        st.info(
            "This is an educational ML estimate, not a professional valuation."
        )

    except Exception as error:
        st.error(f"Prediction error: {error}")
        st.write(
            "Please verify that the dataset columns match "
            "the standard Ames Housing dataset."
        )

st.divider()

# Results table
st.subheader("📋 Model Evaluation Results")
st.dataframe(
    results.sort_values("R2 Score", ascending=False),
    use_container_width=True
)

st.caption(
    "Developed using Python, Pandas, Scikit-learn and Streamlit."
)

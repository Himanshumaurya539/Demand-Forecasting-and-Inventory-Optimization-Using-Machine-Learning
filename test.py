# streamlit_demand_dashboard.py
# Single-file Streamlit app for SKU demand prediction, safety stock and interactive dashboard

import streamlit as st
import pandas as pd
import numpy as np
import math
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

st.set_page_config(page_title="SKU Demand Dashboard", layout="wide")

@st.cache_data(show_spinner=False)
def load_and_prepare(df: pd.DataFrame):
    """Prepare the 'daily' dataframe and return training artifacts: daily, encoder, cat_cols, feature_cols, rf, ss_map"""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    # Required columns
    cols_needed = ['Date','SKU','Category','Qty','Amount']
    for c in cols_needed:
        if c not in df.columns:
            raise ValueError(f"Uploaded file must contain column: {c}")

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=False)
    df = df.dropna(subset=['Date','SKU']).copy()
    df['Qty'] = pd.to_numeric(df['Qty'], errors='coerce').fillna(0).astype(int)
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').fillna(0.0)

    # Aggregate
    daily = df.groupby(['SKU','Date']).agg({
        'Qty':'sum',
        'Amount':'sum',
        'Category': lambda x: x.mode().iloc[0] if len(x.mode())>0 else x.iloc[0]
    }).reset_index().rename(columns={'Qty':'demand','Amount':'revenue'})

    # full date index per SKU
    all_skus = daily['SKU'].unique()
    min_date, max_date = daily['Date'].min(), daily['Date'].max()
    full_index = pd.MultiIndex.from_product([all_skus, pd.date_range(min_date, max_date)], names=['SKU','Date'])
    daily = daily.set_index(['SKU','Date']).reindex(full_index).reset_index()
    daily['demand'] = daily['demand'].fillna(0)
    daily['revenue'] = daily['revenue'].fillna(0)
    daily['Category'] = daily['Category'].fillna('Unknown')

    # features
    daily['dayofweek'] = daily['Date'].dt.dayofweek
    daily['month'] = daily['Date'].dt.month
    daily['day'] = daily['Date'].dt.day

    for lag in [1,7,14,28]:
        daily[f'lag_{lag}'] = daily.groupby('SKU')['demand'].shift(lag).fillna(0)

    daily['rolling7_mean'] = daily.groupby('SKU')['demand'].shift(1).rolling(window=7, min_periods=1).mean().reset_index(level=0, drop=True)
    daily['rolling7_std'] = daily.groupby('SKU')['demand'].shift(1).rolling(window=7, min_periods=1).std().reset_index(level=0, drop=True).fillna(0)

    daily['target_next_day'] = daily.groupby('SKU')['demand'].shift(-1)
    daily = daily[~daily['target_next_day'].isna()].copy()

    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    cat_features = encoder.fit_transform(daily[['Category']])
    cat_cols = encoder.get_feature_names_out(['Category']).tolist()

    num_features = daily[['dayofweek', 'month', 'day', 'lag_1', 'lag_7','lag_14', 'lag_28',
                          'rolling7_mean', 'rolling7_std']]

    X = np.hstack([num_features, cat_features])
    y = daily['demand'].values

    # Train a RandomForest (fast; adjust hyperparams as needed)
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    # feature_cols to use later when building DataFrames for predict
    feature_cols = num_features.columns.tolist() + cat_cols

    # SKU-level std map for safety stock
    ss_map = daily.groupby('SKU')['demand'].std().fillna(daily['demand'].std()).to_dict()

    return {
        'daily': daily,
        'encoder': encoder,
        'cat_cols': cat_cols,
        'feature_cols': feature_cols,
        'rf': rf,
        'ss_map': ss_map
    }


# A helper predict function that matches training pipeline exactly
def predict_next_day(sku, date, category, lead_time, artifacts):
    daily = artifacts['daily']
    encoder = artifacts['encoder']
    rf = artifacts['rf']
    ss_map = artifacts['ss_map']

    date = pd.to_datetime(date)
    dayofweek = date.dayofweek
    month = date.month
    day = date.day

    # compute lags and rolling values from historical daily
    if sku in daily['SKU'].unique():
        recent = daily[(daily['SKU']==sku) & (daily['Date']<date)].sort_values('Date').tail(28)
        lag_1 = int(recent['demand'].iloc[-1]) if len(recent)>=1 else 0
        lag_7 = int(recent['demand'].tail(7).sum()) if len(recent)>=7 else int(recent['demand'].sum())
        lag_14 = int(recent['demand'].tail(14).sum()) if len(recent)>=14 else lag_7
        lag_28 = int(recent['demand'].tail(28).sum()) if len(recent)>=28 else lag_7
        rolling7_mean = recent['demand'].tail(7).mean() if len(recent)>0 else 0
        rolling7_std = recent['demand'].tail(7).std() if len(recent)>0 else 0
    else:
        # cold-start
        cat_group = daily[daily['Category']==category]
        if len(cat_group)==0:
            lag_1 = int(daily['demand'].mean())
            lag_7 = int(daily.groupby('SKU')['demand'].sum().mean())
            lag_14 = lag_7
            lag_28 = lag_7
            rolling7_mean = daily['demand'].mean()
            rolling7_std = daily['demand'].std()
        else:
            lag_1 = int(cat_group['demand'].mean())
            lag_7 = int(cat_group.groupby('SKU')['demand'].sum().mean())
            lag_14 = lag_7
            lag_28 = lag_7
            rolling7_mean = cat_group['demand'].mean()
            rolling7_std = cat_group['demand'].std()

    # avoid nan
    lag_1 = 0 if pd.isna(lag_1) else lag_1
    lag_7 = 0 if pd.isna(lag_7) else lag_7
    lag_14 = 0 if pd.isna(lag_14) else lag_14
    lag_28 = 0 if pd.isna(lag_28) else lag_28
    rolling7_mean = 0 if pd.isna(rolling7_mean) else rolling7_mean
    rolling7_std = 0 if pd.isna(rolling7_std) else rolling7_std

    cat_vector = encoder.transform([[category]])[0]

    feat = [dayofweek, month, day, lag_1, lag_7, lag_14, lag_28, rolling7_mean, rolling7_std] + list(cat_vector)
    feat_arr = np.array(feat).reshape(1,-1)

    pred = float(max(0, rf.predict(feat_arr)[0]))

    z = 1.65
    ss = ss_map.get(sku, np.mean(list(ss_map.values())))
    safety_stock = math.ceil(z * ss * math.sqrt(max(1, int(lead_time))))
    optimal_stock = math.ceil(pred + safety_stock)

    return {
        'predicted_next_day_demand': round(pred,2),
        'safety_stock': int(safety_stock),
        'optimal_reorder_quantity': int(optimal_stock)
    }


# -------------------- Streamlit UI --------------------
st.title("📈 SKU Demand Forecast & Reorder Dashboard")
st.markdown("Interactive dashboard: upload CSV or use sample data, train model, get predictions and visualizations in real-time.")

with st.sidebar:
    st.header("Data & Model")
    uploaded_file = st.file_uploader("Upload CSV (must contain Date,SKU,Category,Qty,Amount)", type=['csv'])
    use_sample = st.checkbox("Use sample demo data (if no file)", value=False)
    if uploaded_file is None and not use_sample:
        st.info("Upload a CSV or enable 'Use sample demo data' to proceed.")

    st.markdown("---")
    st.header("Prediction input")

# Load data
if uploaded_file is not None:
    try:
        raw_df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        st.stop()
elif use_sample:
    # create a tiny synthetic sample dataset if user wants demo
    dates = pd.date_range(end=pd.Timestamp.today(), periods=200)
    skus = [f"SKU_{i:03d}" for i in range(1,6)]
    rows = []
    cats = ['kurta','Top','Blouse','Set','Western Dress']
    rng = np.random.RandomState(42)
    for sku in skus:
        cat = rng.choice(cats)
        base = rng.randint(0,10)
        for d in dates:
            demand = max(0, int(base + rng.poisson(1.5)))
            rows.append({'Date':d, 'SKU':sku, 'Category':cat, 'Qty':demand, 'Amount':demand*100})
    raw_df = pd.DataFrame(rows)
else:
    st.stop()

# Prepare and train (cached)
with st.spinner('Preparing data and training model...'):
    artifacts = load_and_prepare(raw_df)

daily = artifacts['daily']
encoder = artifacts['encoder']
cat_cols = artifacts['cat_cols']
feature_cols = artifacts['feature_cols']
rf = artifacts['rf']
ss_map = artifacts['ss_map']

# Show dataset summary
st.subheader('Dataset summary')
col1, col2, col3 = st.columns([1,1,1])
col1.metric("SKUs", str(daily['SKU'].nunique()))
col2.metric("Date range", f"{daily['Date'].min().date()} → {daily['Date'].max().date()}")
col3.metric("Rows (daily)", str(len(daily)))

# Prediction controls in sidebar
with st.sidebar.expander('Manual Prediction', expanded=True):
    sku_list = ['<NEW SKU>'] + sorted(daily['SKU'].unique().tolist())
    sel_sku = st.selectbox('Select SKU', sku_list)
    sel_date = st.date_input('Prediction date', value=(daily['Date'].max() + pd.Timedelta(days=1)).date())
    # category options include encoder categories
    cat_options = list(encoder.categories_[0])
    sel_cat = st.selectbox('Category (for cold-start)', cat_options)
    lead_time = st.number_input('Lead time (days)', min_value=1, max_value=60, value=7)
    predict_btn = st.button('Predict')

# When predict button clicked
if predict_btn:
    sku_input = None if sel_sku=='<NEW SKU>' else sel_sku
    res = predict_next_day(sku_input, pd.to_datetime(sel_date), sel_cat, lead_time, artifacts)
    st.subheader('Prediction result')
    st.metric('Predicted next-day demand', res['predicted_next_day_demand'])
    st.metric('Safety stock (≈95% service)', res['safety_stock'])
    st.metric('Recommended reorder qty', res['optimal_reorder_quantity'])

    # Show historical plot (last 60 days for selected SKU if exists)
    if sku_input is not None:
        s = daily[daily['SKU']==sku_input].sort_values('Date')
        mask60 = s['Date'] > (s['Date'].max() - pd.Timedelta(days=60))

        # Build X_s using same feature_cols
        num_X_s = s[mask60][['dayofweek', 'month', 'day', 'lag_1', 'lag_7', 'lag_14', 'lag_28', 'rolling7_mean', 'rolling7_std']]
        cat_X_s = encoder.transform(s[mask60][['Category']])
        cat_X_s = pd.DataFrame(cat_X_s, columns=cat_cols, index=num_X_s.index)
        X_s = pd.concat([num_X_s, cat_X_s], axis=1).fillna(0)
        preds = rf.predict(X_s)

        fig, ax = plt.subplots(figsize=(10,3))
        ax.plot(s[mask60]['Date'], s[mask60]['demand'], label='Actual')
        ax.plot(s[mask60]['Date'], preds, linestyle='--', label='Predicted')
        ax.set_title(f"SKU: {sku_input} - Actual vs Predicted (last 60 days)")
        ax.legend()
        st.pyplot(fig)
    else:
        st.info('No historical plot for new SKU (cold-start).')

# Global model performance quick view
st.subheader('Model performance (on full dataset training predictions)')
# Quick in-sample preds to show approximate fit
num_all = daily[['dayofweek', 'month', 'day', 'lag_1', 'lag_7', 'lag_14', 'lag_28', 'rolling7_mean', 'rolling7_std']]
cat_all = encoder.transform(daily[['Category']])
X_all = np.hstack([num_all, cat_all])
train_preds = rf.predict(X_all)
mape = mean_absolute_percentage_error(np.clip(daily['demand'].values,1,None), np.clip(train_preds,1,None))
rmse = mean_squared_error(daily['demand'].values, train_preds, squared=False)
st.write(f"MAPE (in-sample): {mape:.4f} | RMSE (in-sample): {rmse:.4f}")

# Allow model download
with st.expander('Download model & artifacts'):
    if st.button('Save model to joblib and download'):
        joblib.dump({'rf':rf, 'encoder':encoder, 'feature_cols':feature_cols}, 'demand_model.joblib')
        with open('demand_model.joblib','rb') as f:
            st.download_button('Download joblib', data=f, file_name='demand_model.joblib')

st.markdown('---')
st.caption('App built for a demand forecasting project. Modify hyperparameters, plotting or features as needed.')

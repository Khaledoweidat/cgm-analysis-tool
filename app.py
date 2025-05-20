# CGM Analysis Tool - Streamlit Version

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("📈 CGM Data Analysis Tool")

uploaded_file = st.file_uploader("Upload your CGM CSV or Excel file", type=["csv", "xls", "xlsx"])

if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # Standardize column names
    df.columns = df.columns.str.lower()
    df = df.rename(columns={'timestamp': 'datetime', 'glucose': 'glucose_mmol'})

    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.dropna(subset=['glucose_mmol'])
    df = df.sort_values('datetime')

    # Clinical metrics
    mean_glucose = df['glucose_mmol'].mean()
    std_glucose = df['glucose_mmol'].std()
    cv_percent = (std_glucose / mean_glucose) * 100
    mean_glucose_mgdl = mean_glucose * 18.01559
    gmi = 3.31 + 0.02392 * mean_glucose_mgdl
    gmi_hba1c_mmol_mol = (gmi - 2.15) * 10.929

    tir = df[(df['glucose_mmol'] >= 3.9) & (df['glucose_mmol'] <= 10.0)].shape[0] / df.shape[0] * 100
    titr = df[(df['glucose_mmol'] >= 3.9) & (df['glucose_mmol'] <= 7.8)].shape[0] / df.shape[0] * 100
    tar = df[df['glucose_mmol'] > 10.0].shape[0] / df.shape[0] * 100
    tbr = df[df['glucose_mmol'] < 3.9].shape[0] / df.shape[0] * 100

    df['gmi_rolling'] = 3.31 + 0.02392 * (df['glucose_mmol'].rolling(3).mean() * 18.01559)
    gmi_var = df['gmi_rolling'].std()

    df['time'] = df['datetime'].dt.time
    pivot = df.pivot_table(index='time', columns=df['datetime'].dt.date, values='glucose_mmol')
    modd = pivot.diff(axis=1).abs().mean(axis=1).mean()

    df['delta'] = df['glucose_mmol'].diff()
    df['delta_time'] = df['datetime'].diff().dt.total_seconds() / 3600
    df['rate'] = df['delta'].abs() / df['delta_time']
    mag = df['rate'].dropna().mean()

    j_index = 0.001 * (mean_glucose + std_glucose)**2

    high_excursions = df[df['glucose_mmol'] > 10.0].shape[0]
    low_excursions = df[df['glucose_mmol'] < 3.9].shape[0]

    df['time_delta_min'] = df['datetime'].diff().dt.total_seconds() / 60
    df['glucose_x_time'] = (df['glucose_mmol'] + df['glucose_mmol'].shift(1)) / 2 * df['time_delta_min']
    glucose_auc = df['glucose_x_time'].sum(skipna=True)

    def risk_score(glucose):
        if glucose < 3.9:
            return 10 * ((3.9 - glucose) ** 2)
        elif glucose > 10.0:
            return 10 * ((glucose - 10.0) ** 2)
        else:
            return 0

    df['risk'] = df['glucose_mmol'].apply(risk_score)
    gri = df['risk'].mean()

    full_duration_min = (df['datetime'].max() - df['datetime'].min()).total_seconds() / 60
    expected_readings = full_duration_min / 5
    sensor_use_percent = df.shape[0] / expected_readings * 100

    # Display Summary Table
    summary = pd.DataFrame({
        'Metric': [
            'Average Glucose (mmol/L)', 'Standard Deviation (mmol/L)', 'CV (%)',
            'GMI (US %)', 'GMI Variability (%)', 'GMI (UK mmol/mol)',
            'Time in Range (3.9–10.0 mmol/L) %',
            'Time in Tight Range (3.9–7.8 mmol/L) %',
            'Time Above Range (>10.0 mmol/L) %',
            'Time Below Range (<3.9 mmol/L) %',
            'MODD (mmol/L)', 'MAG (mmol/L/hr)', 'J-Index',
            'High Excursions (>10.0 mmol/L)', 'Low Excursions (<3.9 mmol/L)',
            'Glucose Exposure AUC (mmol·min/L)', 'GRI (Glycemic Risk Index)',
            'Sensor Wear Time (%)'
        ],
        'Value': [
            f"{mean_glucose:.2f}", f"{std_glucose:.2f}", f"{cv_percent:.2f}",
            f"{gmi:.2f}", f"{gmi_var:.2f}", f"{gmi_hba1c_mmol_mol:.0f}",
            f"{tir:.2f}", f"{titr:.2f}", f"{tar:.2f}", f"{tbr:.2f}",
            f"{modd:.2f}", f"{mag:.2f}", f"{j_index:.2f}",
            high_excursions, low_excursions,
            f"{glucose_auc:.2f}", f"{gri:.2f}", f"{sensor_use_percent:.2f}"
        ]
    })

    st.subheader("📊 CGM Metrics Summary")
    st.dataframe(summary)

    # Pie chart for TIR/TAR/TBR
    st.subheader("📈 Time in Range Distribution")
    fig1, ax1 = plt.subplots()
    ax1.pie([tir, tar, tbr], labels=['TIR', 'TAR', 'TBR'], autopct='%1.1f%%', startangle=140,
            colors=['#66b3ff', '#ff9999', '#99ff99'])
    ax1.axis('equal')
    st.pyplot(fig1)

    # Histogram of glucose values
    st.subheader("📉 Glucose Value Distribution")
    fig2, ax2 = plt.subplots()
    df['glucose_mmol'].hist(bins=20, color='#4CAF50', ax=ax2)
    ax2.set_title('Glucose Histogram')
    ax2.set_xlabel('Glucose (mmol/L)')
    ax2.set_ylabel('Frequency')
    ax2.grid(True)
    st.pyplot(fig2)

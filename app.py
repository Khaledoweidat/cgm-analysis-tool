# CGM Analysis Tool - Streamlit Version (Batch Mode + Download)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

st.title("📈 CGM Data Analysis Tool (Batch Version)")

uploaded_files = st.file_uploader("Upload one or more CGM CSV or Excel files", type=["csv", "xls", "xlsx"], accept_multiple_files=True)

all_results = []

if uploaded_files:
    for uploaded_file in uploaded_files:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        df.columns = df.columns.str.lower()
        df = df.rename(columns={'timestamp': 'datetime', 'glucose': 'glucose_mmol'})
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.dropna(subset=['glucose_mmol'])
        df = df.sort_values('datetime')

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
        df = df[df['delta_time'] != 0]  # avoid division by zero
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

        # Collect results per file
        results = {
            'File': uploaded_file.name,
            'Average Glucose (mmol/L)': round(mean_glucose, 2),
            'Standard Deviation (mmol/L)': round(std_glucose, 2),
            'CV (%)': round(cv_percent, 2),
            'GMI (US %)': round(gmi, 2),
            'GMI (UK mmol/mol)': round(gmi_hba1c_mmol_mol, 0),
            'GMI Variability (%)': round(gmi_var, 2),
            'Time in Range (3.9–10.0 mmol/L) %': round(tir, 2),
            'Time in Tight Range (3.9–7.8 mmol/L) %': round(titr, 2),
            'Time Above Range (>10.0 mmol/L) %': round(tar, 2),
            'Time Below Range (<3.9 mmol/L) %': round(tbr, 2),
            'MODD (mmol/L)': round(modd, 2),
            'MAG (mmol/L/hr)': round(mag, 2),
            'J-Index': round(j_index, 2),
            'High Excursions': high_excursions,
            'Low Excursions': low_excursions,
            'Glucose Exposure AUC (mmol·min/L)': round(glucose_auc, 2),
            'GRI': round(gri, 2),
            'Sensor Wear Time (%)': round(sensor_use_percent, 2)
        }
        all_results.append(results)

    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    st.subheader("📊 Summary of All Files")
    st.dataframe(results_df)

    # Download as CSV
    csv = results_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Summary as CSV", data=csv, file_name="CGM_summary.csv", mime='text/csv')

    # Download as Excel
    output = BytesIO()
    results_df.to_excel(output, index=False)
    st.download_button("Download Summary as Excel", data=output.getvalue(), file_name="CGM_summary.xlsx", mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

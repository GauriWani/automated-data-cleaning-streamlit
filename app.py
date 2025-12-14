import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

st.set_page_config(page_title="Automated Data Cleaning Tool", layout="wide")

st.title("🧹 Automated Data Cleaning & EDA Tool")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Raw Data Preview")
    st.dataframe(df)

    st.subheader("Missing Value Summary")
    st.write(df.isnull().sum())

    option = st.selectbox(
        "Missing Value Handling",
        ["None", "Mean", "Median", "Drop Rows"]
    )

    df_clean = df.copy()

    if option in ["Mean", "Median"]:
        strategy = option.lower()
        imputer = SimpleImputer(strategy=strategy)
        df_clean[df_clean.columns] = imputer.fit_transform(df_clean)

    elif option == "Drop Rows":
        df_clean = df_clean.dropna()

    st.subheader("Outlier Handling")
    remove_outliers = st.checkbox("Remove outliers using IQR")

    st.subheader("Outlier Handling")
remove_outliers = st.checkbox("Remove outliers using IQR")

if remove_outliers:
    numeric_cols = df_clean.select_dtypes(include=np.number).columns

    if len(numeric_cols) == 0:
        st.warning("No numeric columns available for outlier detection.")
    else:
        Q1 = df_clean[numeric_cols].quantile(0.25)
        Q3 = df_clean[numeric_cols].quantile(0.75)
        IQR = Q3 - Q1

        mask = ~(
            (df_clean[numeric_cols] < (Q1 - 1.5 * IQR)) |
            (df_clean[numeric_cols] > (Q3 + 1.5 * IQR))
        ).any(axis=1)

        df_clean = df_clean[mask]


    st.subheader("Cleaned Data")
    st.dataframe(df_clean)

    st.download_button(
        "⬇ Download Cleaned CSV",
        df_clean.to_csv(index=False),
        file_name="cleaned_data.csv",
        mime="text/csv"
    )
# =============================================================
# Obesity Classification Dashboard + Predictor
# =============================================================
import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import umap.umap_ as umap
from sklearn.preprocessing import StandardScaler

# Required imports for loading the pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

import joblib

@st.cache_resource
def load_model():
    return joblib.load("random_forest_obesity.pkl")


# -------------------------------------------------------------
# 1️⃣ Streamlit Config
# -------------------------------------------------------------
st.set_page_config(page_title="🍏 Obesity Classification Dashboard", layout="wide")
st.title("🍏 Obesity Classification Dashboard")
st.markdown("""
This app visualizes the **Obesity dataset**, performs **exploratory analysis** with 
publication-grade plots, and provides an **interactive Random Forest predictor**.
""")

# -------------------------------------------------------------
# 2️⃣ Load Data & Model
# -------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("random_forest_obesity.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("data/obesity_dataset.csv")

model = load_model()
df = load_data()

# -------------------------------------------------------------
# 3️⃣ Tabs: EDA | UMAP | Prediction
# -------------------------------------------------------------
tabs = st.tabs(["📊 Data Exploration", "🌀 UMAP Projection", "🧠 Prediction"])

# =============================================================
# 📊 TAB 1: DATA EXPLORATION
# =============================================================
with tabs[0]:
    st.header("📊 Data Exploration")

    col1, col2 = st.columns(2)

    # Interactive Scatter Plot
    with col1:
        x_feature = st.selectbox("Select X-axis", df.columns)
        y_feature = st.selectbox("Select Y-axis", df.columns)
        hue_feature = st.selectbox("Color by", ["Target"] + [c for c in df.columns if c != "Target"])

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.scatterplot(
            data=df, x=x_feature, y=y_feature, hue=hue_feature,
            palette="Set2", edgecolor="black", alpha=0.8
        )
        ax.set_title(f"{y_feature} vs {x_feature}", fontsize=14, fontweight="bold", pad=10)
        sns.despine()
        st.pyplot(fig)

    # Correlation Heatmap
    with col2:
        st.write("### 🔥 Correlation Heatmap (Numerical Features Only)")
        num_df = df.select_dtypes(include=["float64", "int64"])
        corr = num_df.corr()

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            corr, cmap="coolwarm", center=0, annot=False,
            square=True, cbar_kws={"shrink": 0.8}
        )
        ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold", pad=10)
        st.pyplot(fig)

# =============================================================
# 🌀 TAB 2: UMAP VISUALIZATION
# =============================================================
with tabs[1]:
    st.header("🌀 UMAP Projection")

    st.markdown("""
    UMAP (Uniform Manifold Approximation and Projection) reduces high-dimensional data 
    into 2D for visualization, preserving the structure of the dataset.
    """)

    # Select numerical columns for UMAP
    num_features = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[num_features])

    reducer = umap.UMAP(random_state=42, n_neighbors=15, min_dist=0.1)
    embedding = reducer.fit_transform(scaled_data)

    umap_df = pd.DataFrame(embedding, columns=["UMAP_1", "UMAP_2"])
    umap_df["Target"] = df["Target"]

    # UMAP Scatter Plot
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.scatterplot(
        data=umap_df, x="UMAP_1", y="UMAP_2", hue="Target",
        palette="Set2", s=50, edgecolor="black", alpha=0.8
    )
    ax.set_title("UMAP Projection by Target", fontsize=16, fontweight="bold", pad=10)
    sns.despine()
    st.pyplot(fig)

# =============================================================
# 🧠 TAB 3: PREDICTION
# =============================================================
with tabs[2]:
    st.header("🧠 Predict Obesity Category")

    st.markdown("Provide your details below to predict the obesity class:")

    # Separate numerical and categorical features
    num_features = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    cat_features = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if "Target" in cat_features:
        cat_features.remove("Target")

    # User input widgets
    user_input = {}

    with st.form("prediction_form"):
        st.subheader("Enter Input Features")

        cols = st.columns(2)
        for i, col_name in enumerate(num_features):
            with cols[i % 2]:
                user_input[col_name] = st.number_input(
                    f"{col_name}", 
                    float(df[col_name].min()), 
                    float(df[col_name].max()), 
                    float(df[col_name].mean())
                )

        for i, col_name in enumerate(cat_features):
            with cols[i % 2]:
                user_input[col_name] = st.selectbox(f"{col_name}", df[col_name].unique())

        submitted = st.form_submit_button("🔍 Predict")

    if submitted:
        input_df = pd.DataFrame([user_input])
        prediction = model.predict(input_df)[0]

        target_labels = [
            "Insufficient_Weight", "Normal_Weight", "Overweight_Level_I",
            "Overweight_Level_II", "Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III"
        ]

        predicted_label = target_labels[int(prediction)] if isinstance(prediction, (int, np.integer)) else prediction

        st.success(f"🎯 Predicted Category: **{predicted_label}**")

        st.balloons()

# =============================================================
# Footer
# =============================================================
st.markdown("---")
st.caption("Built with ❤️ using Streamlit, Random Forest, and UMAP for dimensionality reduction.")

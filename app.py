import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import umap
import numpy as np
from sklearn.preprocessing import StandardScaler

# =============================
# CONFIGURATION
# =============================
st.set_page_config(page_title="Obesity Classifier Dashboard", layout="wide")
sns.set_theme(style="whitegrid")

# =============================
# LOAD DATA
# =============================
@st.cache_data
def load_data():
    df = pd.read_csv("obesity.csv")
    df.columns = df.columns.str.strip()
    return df

# =============================
# LOAD MODEL
# =============================
@st.cache_resource
def load_model():
    return joblib.load("random_forest_obesity.pkl")

df = load_data()
model = load_model()

# =============================
# SIDEBAR
# =============================
st.sidebar.title("⚙️ Navigation")
page = st.sidebar.radio("Go to:", ["📊 Data Exploration", "🌈 Visualization", "🤖 Model Prediction"])

# =============================
# PAGE 1: DATA EXPLORATION
# =============================
if page == "📊 Data Exploration":
    st.title("📊 Data Overview")
    st.write("Explore the dataset used for obesity level classification.")
    st.dataframe(df.head())

    st.markdown("### Summary Statistics")
    st.write(df.describe())

# =============================
# PAGE 2: VISUALIZATION
# =============================
elif page == "🌈 Visualization":
    st.title("🌈 Data Visualization")

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=['int64', 'float64']).columns.tolist()

    tab1, tab2, tab3 = st.tabs(["Scatter Plot", "Correlation Heatmap", "UMAP Projection"])

    # ---- Scatter Plot ----
    with tab1:
        st.subheader("Scatter Plot")
        x_feature = st.selectbox("Select X-axis", numeric_cols, key="x_scatter")
        y_feature = st.selectbox("Select Y-axis", numeric_cols, key="y_scatter")
        hue_feature = st.selectbox("Color by (optional)", [None] + categorical_cols, key="hue_scatter")

        try:
            fig, ax = plt.subplots(figsize=(7,5))
            sns.scatterplot(
                data=df,
                x=x_feature,
                y=y_feature,
                hue=hue_feature if hue_feature else None,
                palette="Set2",
                edgecolor="black",
                alpha=0.8,
                ax=ax
            )
            ax.set_title(f"{y_feature} vs {x_feature}", fontsize=14, fontweight="bold", pad=10)
            sns.despine()
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"⚠️ Could not plot scatter: {e}")

    # ---- Correlation Heatmap ----
    with tab2:
        st.subheader("Correlation Heatmap")
        corr = df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={'label': 'Correlation'}, ax=ax)
        ax.set_title("Correlation Matrix", fontsize=14, fontweight="bold")
        st.pyplot(fig)

    # ---- UMAP Projection ----
    with tab3:
        st.subheader("UMAP Projection")
        if len(numeric_cols) > 1:
            reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
            scaled_data = StandardScaler().fit_transform(df[numeric_cols])
            embedding = reducer.fit_transform(scaled_data)
            df_umap = pd.DataFrame(embedding, columns=["UMAP1", "UMAP2"])
            df_umap["Label"] = df[categorical_cols[0]] if categorical_cols else "None"

            fig, ax = plt.subplots(figsize=(7,5))
            sns.scatterplot(data=df_umap, x="UMAP1", y="UMAP2", hue="Label", palette="Spectral", alpha=0.8, ax=ax)
            ax.set_title("UMAP Projection of Feature Space", fontsize=14, fontweight="bold")
            st.pyplot(fig)
        else:
            st.warning("Not enough numeric features for UMAP projection.")

# =============================
# PAGE 3: MODEL PREDICTION
# =============================
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve, auc


elif page == "🤖 Model Prediction":
    st.title("🤖 Obesity Level Prediction")
    st.write("Provide feature values to predict the obesity class.")

    # Create input form
    feature_inputs = {}
    for col in df.columns:
        if df[col].dtype == 'object':
            feature_inputs[col] = st.selectbox(f"{col}", sorted(df[col].unique()))
        else:
            feature_inputs[col] = st.number_input(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))

    if st.button("Predict"):
        input_df = pd.DataFrame([feature_inputs])
        try:
            # Prediction
            prediction = model.predict(input_df)[0]
            probabilities = model.predict_proba(input_df)[0]

            # Class labels
            if hasattr(model, "classes_"):
                class_labels = model.classes_
            else:
                class_labels = [f"Class {i}" for i in range(len(probabilities))]

            st.success(f"🎯 Predicted Obesity Class: **{prediction}**")

            # Show probabilities in a table
            prob_df = pd.DataFrame({
                "Class": class_labels,
                "Probability": np.round(probabilities, 3)
            }).sort_values("Probability", ascending=False)
            st.subheader("Class Probabilities")
            st.dataframe(prob_df)

            # Optionally, ROC curve (precomputed)
            if len(class_labels) <= 7 and "Target" in df.columns:
                # Prepare y_true for ROC
                lb = LabelBinarizer()
                y_true = lb.fit_transform(df["Target"])
                y_score = model.predict_proba(df.drop(columns=["Target"]))

                # Plot ROC per class
                fig, ax = plt.subplots(figsize=(6,5))
                for i, cls in enumerate(lb.classes_):
                    fpr, tpr, _ = roc_curve(y_true[:, i], y_score[:, i])
                    roc_auc = auc(fpr, tpr)
                    ax.plot(fpr, tpr, lw=2, label=f"{cls} (AUC = {roc_auc:.2f})")

                ax.plot([0,1], [0,1], linestyle='--', color='grey')
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.set_title("ROC Curve (per Class)")
                ax.legend(loc="lower right")
                sns.despine()
                st.pyplot(fig)

        except Exception as e:
            st.error(f"⚠️ Prediction failed: {e}")

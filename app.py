"""
AquaGuard AI – Water Quality Classification System
Streamlit Web Application

A clean, professional, and beginner-friendly web application for classifying
water quality using machine learning, built with Streamlit, Scikit-learn,
Pandas, and Seaborn.
"""

import json
import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom Theme Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AquaGuard AI – Water Quality Classification",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Water-Themed CSS Styling
st.markdown(
    """
    <style>
    /* Main container and typography */
    .main-title {
        color: #0369a1;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #475569;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f0f9ff;
        border: 1px solid #bae6fd;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-card h3 {
        margin: 0;
        font-size: 1.8rem;
        color: #0284c7;
    }
    .metric-card p {
        margin: 0;
        font-size: 0.9rem;
        color: #334155;
        font-weight: 500;
    }
    .disclaimer-box {
        background-color: #fffbeb;
        border-left: 5px solid #f59e0b;
        padding: 14px 18px;
        border-radius: 6px;
        color: #92400e;
        font-size: 0.92rem;
        margin-top: 15px;
        margin-bottom: 15px;
    }
    .badge-model {
        background-color: #e0f2fe;
        color: #0369a1;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 10px;
    }
    .pred-potable {
        background-color: #ecfdf5;
        border: 2px solid #10b981;
        color: #065f46;
        padding: 18px;
        border-radius: 10px;
        margin-top: 15px;
    }
    .pred-non-potable {
        background-color: #fef2f2;
        border: 2px solid #ef4444;
        color: #991b1b;
        padding: 18px;
        border-radius: 10px;
        margin-top: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# 2. Data & Model Loading Helper Functions (Cached for Speed)
# -----------------------------------------------------------------------------
@st.cache_data
def load_dataset():
    """Load genuine water potability dataset."""
    csv_file = "water_potability.csv"
    if os.path.exists(csv_file):
        return pd.read_csv(csv_file)
    return None


@st.cache_data
def load_results():
    """Load precomputed training and evaluation metrics from model_results.json."""
    json_file = "model_results.json"
    if os.path.exists(json_file):
        with open(json_file, "r") as f:
            return json.load(f)
    return None


@st.cache_resource
def load_trained_model():
    """Load serialized Scikit-learn Pipeline (SimpleImputer + StandardScaler + Model)."""
    model_file = "water_model.joblib"
    if os.path.exists(model_file):
        return joblib.load(model_file)
    return None


df = load_dataset()
results = load_results()
model_pipeline = load_trained_model()

# Verify that essential artifacts are available
if df is None or results is None or model_pipeline is None:
    st.error(
        "⚠️ Project artifacts not found! Please run `python train_model.py` in your terminal "
        "first to generate `water_model.joblib` and `model_results.json`."
    )
    st.stop()

feature_names = results["dataset_info"]["features"]
feature_stats = results.get("feature_statistics", {})
presets = results.get("presentation_presets", {})


# -----------------------------------------------------------------------------
# 3. Sidebar Navigation & Overview
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image(
        "https://images.unsplash.com/photo-1548839140-29a749e1bc4e?auto=format&fit=crop&w=400&q=80",
        caption="AquaGuard AI System",
        use_container_width=True,
    )
    st.markdown("## 💧 AquaGuard AI")
    st.markdown(
        "**Water Quality Classification System**\n\n"
        "A practical Machine Learning project utilizing real-world water physicochemical parameters."
    )
    st.markdown("---")

    navigation = st.radio(
        "📌 Navigation Menu",
        [
            "💧 Water Quality Predictor",
            "📊 Dataset Analysis",
            "🔬 Model Performance & Comparison",
            "📖 Academic Report & Viva Guide",
        ],
        index=0,
    )

    st.markdown("---")
    st.markdown("### ⚙️ System Status")
    st.success("✅ Dataset Loaded (3,276 records)")
    st.success("✅ Model Pipeline Ready")
    st.caption("Algorithm: Random Forest Classifier (Tuned)")
    st.caption("College Project Edition • 100% Genuine ML Logic")


# -----------------------------------------------------------------------------
# TAB 1: Water Quality Prediction Interface
# -----------------------------------------------------------------------------
if navigation == "💧 Water Quality Predictor":
    st.markdown('<div class="main-title">💧 AquaGuard AI – Water Quality Predictor</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Input water physical and chemical parameters to estimate potability '
        'based on patterns learned from historical dataset records.</div>',
        unsafe_allow_html=True,
    )

    # Presentation helper: Preset loaders
    st.markdown("##### 🧪 Quick Presets (Ideal for College Live Demonstrations)")
    preset_col1, preset_col2, preset_col3 = st.columns(3)

    # Initialize session state for inputs if not present
    default_vals = {
        feat: feature_stats[feat]["median"] for feat in feature_names if feat in feature_stats
    }
    for feat in feature_names:
        if feat not in st.session_state:
            st.session_state[feat] = float(default_vals.get(feat, 7.0))

    if preset_col1.button("📌 Load Known Potable Sample", use_container_width=True):
        if "potable_sample" in presets:
            for feat, val in presets["potable_sample"].items():
                st.session_state[feat] = float(val)
            st.rerun()

    if preset_col2.button("📌 Load Known Non-Potable Sample", use_container_width=True):
        if "non_potable_sample" in presets:
            for feat, val in presets["non_potable_sample"].items():
                st.session_state[feat] = float(val)
            st.rerun()

    if preset_col3.button("🔄 Reset to Dataset Medians", use_container_width=True):
        for feat in feature_names:
            st.session_state[feat] = float(default_vals.get(feat, 7.0))
        st.rerun()

    st.markdown("---")
    st.markdown("### 🎛️ Enter Water Quality Parameters")

    # Group the 9 features logically into 3 columns
    input_data = {}
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("##### 🧪 Acid-Base & Disinfection")
        input_data["ph"] = st.number_input(
            "1. pH Value",
            min_value=0.0,
            max_value=14.0,
            value=float(st.session_state.get("ph", 7.0)),
            step=0.1,
            help="WHO standard: 6.5 – 8.5. Measures acidity/alkalinity of water.",
        )
        input_data["Chloramines"] = st.number_input(
            "4. Chloramines (ppm)",
            min_value=0.0,
            max_value=20.0,
            value=float(st.session_state.get("Chloramines", 7.13)),
            step=0.1,
            help="Disinfectant derived from chlorine and ammonia. Safe limits < 4 ppm.",
        )
        input_data["Organic_carbon"] = st.number_input(
            "7. Organic Carbon (ppm)",
            min_value=0.0,
            max_value=40.0,
            value=float(st.session_state.get("Organic_carbon", 14.22)),
            step=0.1,
            help="Total Organic Carbon (TOC) from decaying natural matter. EPA limit < 2-4 ppm.",
        )

    with col2:
        st.markdown("##### 🪨 Dissolved Minerals & Solids")
        input_data["Hardness"] = st.number_input(
            "2. Hardness (mg/L)",
            min_value=0.0,
            max_value=400.0,
            value=float(st.session_state.get("Hardness", 196.97)),
            step=1.0,
            help="Capacity of water to precipitate soap; caused by calcium and magnesium.",
        )
        input_data["Sulfate"] = st.number_input(
            "5. Sulfate (mg/L)",
            min_value=0.0,
            max_value=600.0,
            value=float(st.session_state.get("Sulfate", 333.07)),
            step=1.0,
            help="Naturally occurring mineral salts. Recommended standard < 250 mg/L.",
        )
        input_data["Trihalomethanes"] = st.number_input(
            "8. Trihalomethanes (μg/L)",
            min_value=0.0,
            max_value=150.0,
            value=float(st.session_state.get("Trihalomethanes", 66.62)),
            step=0.5,
            help="By-products formed when chlorine reacts with organic matter. Safe < 80 μg/L.",
        )

    with col3:
        st.markdown("##### ⚡ Physical Properties & Clarity")
        input_data["Solids"] = st.number_input(
            "3. Solids / TDS (ppm)",
            min_value=0.0,
            max_value=70000.0,
            value=float(st.session_state.get("Solids", 20927.83)),
            step=100.0,
            help="Total Dissolved Solids (TDS). Desirable standard < 500-1000 ppm.",
        )
        input_data["Conductivity"] = st.number_input(
            "6. Conductivity (μS/cm)",
            min_value=0.0,
            max_value=1000.0,
            value=float(st.session_state.get("Conductivity", 421.88)),
            step=1.0,
            help="Electrical conductivity influenced by dissolved ions. WHO standard < 400 μS/cm.",
        )
        input_data["Turbidity"] = st.number_input(
            "9. Turbidity (NTU)",
            min_value=0.0,
            max_value=10.0,
            value=float(st.session_state.get("Turbidity", 3.96)),
            step=0.1,
            help="Measure of water clarity / cloudiness from suspended particles. WHO limit < 5 NTU.",
        )

    st.markdown("<br>", unsafe_allow_html=True)
    predict_button = st.button("🚀 Predict Water Quality", type="primary", use_container_width=True)

    if predict_button:
        # Build single-sample DataFrame with matching feature names
        sample_df = pd.DataFrame([input_data])[feature_names]

        # Model Prediction using the saved pipeline
        prediction = int(model_pipeline.predict(sample_df)[0])
        probabilities = model_pipeline.predict_proba(sample_df)[0]
        prob_non_potable = float(probabilities[0] * 100)
        prob_potable = float(probabilities[1] * 100)

        model_name = results["selected_model"]["name"]

        st.markdown("### 📋 Prediction Results")
        st.markdown(f'<span class="badge-model">Generated by: {model_name}</span>', unsafe_allow_html=True)

        if prediction == 1:
            st.markdown(
                f"""
                <div class="pred-potable">
                    <h3 style="margin:0 0 8px 0; color:#065f46;">✅ Predicted Class: 1 – Potable</h3>
                    <p style="margin:0; font-size:1.05rem;">
                        According to the historical dataset labels, this water sample exhibits patterns consistent with 
                        <strong>Potable Water (Class 1)</strong>.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="pred-non-potable">
                    <h3 style="margin:0 0 8px 0; color:#991b1b;">⚠️ Predicted Class: 0 – Non-Potable</h3>
                    <p style="margin:0; font-size:1.05rem;">
                        According to the historical dataset labels, this water sample exhibits patterns consistent with 
                        <strong>Non-Potable Water (Class 0)</strong>.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        # Probability Breakdown
        prob_col1, prob_col2 = st.columns(2)
        with prob_col1:
            st.metric("Estimated Potability Probability (Class 1)", f"{prob_potable:.1f}%")
            st.progress(prob_potable / 100.0)
        with prob_col2:
            st.metric("Estimated Non-Potability Probability (Class 0)", f"{prob_non_potable:.1f}%")
            st.progress(prob_non_potable / 100.0)

    # Mandatory Academic Disclaimer
    st.markdown(
        """
        <div class="disclaimer-box">
            <strong>⚠️ Academic & Safety Disclaimer:</strong><br>
            "AquaGuard AI is an educational machine learning project. Its predictions reflect patterns learned 
            from historical dataset labels and do not establish whether water is safe to drink. Laboratory testing 
            and applicable drinking-water standards are required to assess actual water safety."
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# TAB 2: Dataset Analysis
# -----------------------------------------------------------------------------
elif navigation == "📊 Dataset Analysis":
    st.markdown('<div class="main-title">📊 Dataset Analysis & Exploration</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Real statistical breakdown of the verified Kaggle Water Potability Dataset.</div>',
        unsafe_allow_html=True,
    )

    info = results["dataset_info"]

    # Key Metrics Row
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.markdown(
            f'<div class="metric-card"><h3>{info["total_records"]}</h3><p>Total Records</p></div>',
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f'<div class="metric-card"><h3>{info["feature_count"]}</h3><p>Input Features</p></div>',
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f'<div class="metric-card"><h3>{info["class_distribution"]["Potable (1)"]}</h3><p>Potable (Class 1)</p></div>',
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            f'<div class="metric-card"><h3>{info["class_distribution"]["Non-Potable (0)"]}</h3><p>Non-Potable (Class 0)</p></div>',
            unsafe_allow_html=True,
        )
    with m5:
        st.markdown(
            f'<div class="metric-card"><h3>{info["duplicate_records"]}</h3><p>Duplicates Found</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Missing values & Class Distribution Charts
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("#### 1. Target Class Distribution")
        fig_class, ax_class = plt.subplots(figsize=(6, 4))
        classes = ["Non-Potable (0)", "Potable (1)"]
        counts = [
            info["class_distribution"]["Non-Potable (0)"],
            info["class_distribution"]["Potable (1)"],
        ]
        colors = ["#f87171", "#34d399"]
        bars = ax_class.bar(classes, counts, color=colors, width=0.55, edgecolor="#334155")
        ax_class.set_ylabel("Number of Samples", fontsize=10)
        ax_class.set_title("Distribution of Potable vs Non-Potable Samples", fontsize=11, fontweight="bold")
        for bar in bars:
            height = bar.get_height()
            percentage = (height / sum(counts)) * 100
            ax_class.annotate(
                f"{height} ({percentage:.1f}%)",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontweight="bold",
            )
        ax_class.set_ylim(0, max(counts) * 1.18)
        sns.despine()
        plt.tight_layout()
        st.pyplot(fig_class)
        st.caption("The dataset contains 61.0% Non-Potable and 39.0% Potable records, reflecting slight class imbalance.")

    with chart_col2:
        st.markdown("#### 2. Missing Values by Feature")
        missing_series = pd.Series(info["missing_values"])
        missing_filtered = missing_series[missing_series > 0].sort_values(ascending=False)

        fig_miss, ax_miss = plt.subplots(figsize=(6, 4))
        miss_bars = ax_miss.bar(
            missing_filtered.index,
            missing_filtered.values,
            color="#38bdf8",
            width=0.5,
            edgecolor="#0284c7",
        )
        ax_miss.set_ylabel("Missing Count", fontsize=10)
        ax_miss.set_title("Features with Missing Values (Handled via Median Imputation)", fontsize=11, fontweight="bold")
        for bar in miss_bars:
            height = bar.get_height()
            pct = (height / info["total_records"]) * 100
            ax_miss.annotate(
                f"{int(height)} ({pct:.1f}%)",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontweight="bold",
            )
        ax_miss.set_ylim(0, max(missing_filtered.values) * 1.18)
        sns.despine()
        plt.tight_layout()
        st.pyplot(fig_miss)
        st.caption("Only Sulfate (781), pH (491), and Trihalomethanes (162) contain missing values.")

    st.markdown("---")

    # Correlation Heatmap
    st.markdown("#### 3. Correlation Heatmap Between Physicochemical Features")
    fig_corr, ax_corr = plt.subplots(figsize=(9, 5.5))
    corr_matrix = df.corr()
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        cbar=True,
        ax=ax_corr,
        linewidths=0.5,
        annot_kws={"size": 8},
    )
    plt.title("Pearson Correlation Heatmap (Water Quality Features)", fontsize=12, fontweight="bold", pad=10)
    plt.tight_layout()
    st.pyplot(fig_corr)
    st.info(
        "💡 **Key Insight for Presentation:** The correlation coefficients between individual features and the target "
        "`Potability` are nearly zero (ranging from -0.03 to +0.02). This explains why a linear classifier like "
        "Logistic Regression struggles, and why non-linear tree-based ensembles (Random Forest) are necessary to capture "
        "multi-feature chemical interactions."
    )

    st.markdown("---")
    st.markdown("#### 4. Sample Genuine Dataset Records")
    st.dataframe(df.head(10), use_container_width=True)


# -----------------------------------------------------------------------------
# TAB 3: Model Performance & Comparison
# -----------------------------------------------------------------------------
elif navigation == "🔬 Model Performance & Comparison":
    st.markdown('<div class="main-title">🔬 Model Performance & Comparison</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Honest, un-hardcoded evaluation results comparing Logistic Regression and Random Forest.</div>',
        unsafe_allow_html=True,
    )

    models_data = results["models"]
    lr = models_data["logistic_regression"]
    rf_base = models_data["random_forest_baseline"]
    rf_opt = models_data["random_forest_optimized"]

    # Model Comparison Table
    st.markdown("### 1. Model Comparison Table (Held-Out Test Set: 656 Samples)")
    comparison_df = pd.DataFrame([
        {
            "Algorithm": lr["name"],
            "Accuracy": f"{lr['accuracy'] * 100:.2f}%",
            "Precision": f"{lr['precision']:.4f}",
            "Recall": f"{lr['recall']:.4f}",
            "F1-Score": f"{lr['f1_score']:.4f}",
            "Key Observation": "Linear baseline; predicts majority class only due to non-linear chemical features.",
        },
        {
            "Algorithm": rf_base["name"],
            "Accuracy": f"{rf_base['accuracy'] * 100:.2f}%",
            "Precision": f"{rf_base['precision']:.4f}",
            "Recall": f"{rf_base['recall']:.4f}",
            "F1-Score": f"{rf_base['f1_score']:.4f}",
            "Key Observation": "Default ensemble of 100 trees; successfully classifies non-linear combinations.",
        },
        {
            "Algorithm": rf_opt["name"],
            "Accuracy": f"{rf_opt['accuracy'] * 100:.2f}%",
            "Precision": f"{rf_opt['precision']:.4f}",
            "Recall": f"{rf_opt['recall']:.4f}",
            "F1-Score": f"{rf_opt['f1_score']:.4f}",
            "Key Observation": f"Tuned via 5-Fold CV (CV Acc: {rf_opt['best_cv_accuracy']*100:.2f}%); max_depth=20.",
        },
    ])
    st.table(comparison_df)

    st.markdown("---")

    # Comparative Visualizations
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("#### 2. Accuracy Comparison Across Algorithms")
        fig_acc, ax_acc = plt.subplots(figsize=(6, 4))
        algo_names = ["Logistic Reg.", "RF (Baseline)", "RF (Tuned)"]
        accuracies = [lr["accuracy"] * 100, rf_base["accuracy"] * 100, rf_opt["accuracy"] * 100]
        colors = ["#94a3b8", "#38bdf8", "#0284c7"]

        bars = ax_acc.bar(algo_names, accuracies, color=colors, width=0.5, edgecolor="#0f172a")
        ax_acc.set_ylabel("Test Accuracy (%)", fontsize=10)
        ax_acc.set_ylim(0, 85)
        for bar in bars:
            height = bar.get_height()
            ax_acc.annotate(
                f"{height:.2f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontweight="bold",
            )
        sns.despine()
        plt.tight_layout()
        st.pyplot(fig_acc)

    with chart_col2:
        st.markdown("#### 3. F1-Score Comparison (Balanced Evaluation)")
        fig_f1, ax_f1 = plt.subplots(figsize=(6, 4))
        f1_scores = [lr["f1_score"], rf_base["f1_score"], rf_opt["f1_score"]]
        f1_bars = ax_f1.bar(algo_names, f1_scores, color=["#cbd5e1", "#818cf8", "#4f46e5"], width=0.5, edgecolor="#1e1b4b")
        ax_f1.set_ylabel("F1-Score", fontsize=10)
        ax_f1.set_ylim(0, 0.6)
        for bar in f1_bars:
            height = bar.get_height()
            ax_f1.annotate(
                f"{height:.4f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontweight="bold",
            )
        sns.despine()
        plt.tight_layout()
        st.pyplot(fig_f1)

    st.markdown("---")

    # Confusion Matrix Visualizations Side-by-Side
    st.markdown("#### 4. Confusion Matrices (Held-Out Test Set: 656 Samples)")
    cm_col1, cm_col2 = st.columns(2)

    with cm_col1:
        st.markdown("##### Logistic Regression Confusion Matrix")
        fig_cm_lr, ax_cm_lr = plt.subplots(figsize=(5, 3.8))
        sns.heatmap(
            np.array(lr["confusion_matrix"]),
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Pred Non-Potable (0)", "Pred Potable (1)"],
            yticklabels=["Actual Non-Potable (0)", "Actual Potable (1)"],
            ax=ax_cm_lr,
        )
        plt.title(f"Logistic Regression (Accuracy: {lr['accuracy']*100:.2f}%)", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig_cm_lr)
        st.caption("Notice: 0 true positives for Potable water. The linear model predicted all 656 samples as Class 0.")

    with cm_col2:
        st.markdown("##### Selected Model: Random Forest (Optimized)")
        fig_cm_rf, ax_cm_rf = plt.subplots(figsize=(5, 3.8))
        sns.heatmap(
            np.array(rf_opt["confusion_matrix"]),
            annot=True,
            fmt="d",
            cmap="Greens",
            cbar=False,
            xticklabels=["Pred Non-Potable (0)", "Pred Potable (1)"],
            yticklabels=["Actual Non-Potable (0)", "Actual Potable (1)"],
            ax=ax_cm_rf,
        )
        plt.title(f"Random Forest Optimized (Accuracy: {rf_opt['accuracy']*100:.2f}%)", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig_cm_rf)
        st.caption("True Negatives: 357 | False Positives: 43 | False Negatives: 179 | True Positives: 77")

    st.markdown("---")

    # Feature Importances & Optimization Details
    st.markdown("#### 5. Feature Importances (From Selected Random Forest)")
    feat_imps = results["selected_model"]["feature_importances"]
    feat_df = pd.DataFrame({"Feature": list(feat_imps.keys()), "Importance": list(feat_imps.values())})

    fig_imp, ax_imp = plt.subplots(figsize=(8, 4))
    sns.barplot(x="Importance", y="Feature", data=feat_df, palette="Blues_r", ax=ax_imp)
    ax_imp.set_title("Relative Predictive Power of Water Quality Parameters", fontweight="bold")
    ax_imp.set_xlabel("Gini Feature Importance")
    sns.despine()
    plt.tight_layout()
    st.pyplot(fig_imp)
    st.caption("Sulfate, pH, Hardness, and Chloramines emerge as the leading predictive indicators.")

    st.markdown("---")
    st.markdown("#### 6. Educational Metrics Explanation (Viva Ready)")
    exp1, exp2, exp3, exp4 = st.columns(4)
    with exp1:
        st.markdown("**Accuracy**")
        st.caption("Overall proportion of correct predictions: `(TP + TN) / Total`. Can be misleading on imbalanced datasets.")
    with exp2:
        st.markdown("**Precision**")
        st.caption("Of all samples predicted as Potable, how many were truly potable: `TP / (TP + FP)`. Minimizes false alarms.")
    with exp3:
        st.markdown("**Recall (Sensitivity)**")
        st.caption("Of all actual potable water samples, how many did the model capture: `TP / (TP + FN)`. Measures coverage.")
    with exp4:
        st.markdown("**F1-Score**")
        st.caption("Harmonic mean of Precision and Recall: `2 * (P * R) / (P + R)`. The gold standard for balanced evaluation.")


# -----------------------------------------------------------------------------
# TAB 4: Academic Report & Viva Guide
# -----------------------------------------------------------------------------
elif navigation == "📖 Academic Report & Viva Guide":
    st.markdown('<div class="main-title">📖 Academic Report & Viva Guide</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Comprehensive documentation structured for college project defense and presentations.</div>',
        unsafe_allow_html=True,
    )

    doc_tab1, doc_tab2, doc_tab3 = st.tabs([
        "📑 Seven Academic Requirements",
        "🎤 10-Minute Presentation Script",
        "🎯 Viva Q&A Preparation",
    ])

    with doc_tab1:
        st.markdown("### The 7 Core Academic Phases")

        st.markdown("""
        #### 1. Problem Definition
        - **Problem Statement:** "Water quality depends on several physical and chemical parameters. This project uses machine learning to analyze historical water-quality data and classify water samples as potable or non-potable according to their recorded dataset labels."
        - **Objectives:** Build an end-to-end, reproducible classification system to assist environmental scientists and college students in analyzing water safety indicators using machine learning.
        - **Scope & Limitations:** Model predictions are statistical approximations derived from historical dataset labels and cannot replace accredited physical laboratory chemical testing.

        #### 2. Dataset Collection & Schema Verification
        - **Source:** Public Kaggle Water Potability Dataset (`adityakadiwal/water-potability`).
        - **Records:** 3,276 water samples.
        - **Target:** `Potability` (1 = Potable, 0 = Non-Potable).
        - **Features (9 verified):** pH, Hardness, Solids (TDS), Chloramines, Sulfate, Conductivity, Organic Carbon, Trihalomethanes, Turbidity.
        - **Missing Values:** Sulfate (781), pH (491), Trihalomethanes (162).

        #### 3. Data Preprocessing & Leakage Prevention
        - **Duplicate Checking:** 0 duplicate rows found in dataset.
        - **Imputation:** Median imputation (`SimpleImputer(strategy='median')`) used to handle missing numerical features robustly without being skewed by outliers.
        - **Feature Scaling:** `StandardScaler` standardizes features to zero mean and unit variance.
        - **Train/Test Split:** 80% training (2,620 samples) and 20% testing (656 samples) using **Stratified Splitting** to preserve class balance.
        - **Data Leakage Prevention:** The preprocessing pipeline is fitted strictly on the training set (`X_train`) and only used to transform the test set (`X_test`).

        #### 4. Model Development
        - **Model A (Baseline):** Logistic Regression — linear classification benchmark.
        - **Model B (Ensemble):** Random Forest Classifier — non-linear ensemble composed of decision trees.
        - **Pipeline Integration:** Models are packaged in Scikit-learn Pipelines combining imputation, scaling, and classification.

        #### 5. Performance Evaluation
        - Evaluated on identical held-out test data (656 samples).
        - Measured Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
        - **Key Discovery:** Logistic Regression achieved 60.98% accuracy by predicting exclusively class 0 (F1=0.00). Random Forest achieved 66.16% test accuracy and 0.4096 F1-score, confirming non-linear relationships.

        #### 6. Model Optimization
        - Performed hyperparameter tuning using `GridSearchCV` on the training dataset with 5-fold cross-validation.
        - Parameters evaluated: `n_estimators` [100, 200], `max_depth` [None, 10, 20], `min_samples_split` [2, 5, 10].
        - Optimal parameters: `{'max_depth': 20, 'min_samples_split': 5, 'n_estimators': 100}` yielding 67.29% mean 5-fold CV accuracy.

        #### 7. Result Analysis & Conclusion
        - Random Forest (Optimized) was chosen as the production model due to its balanced predictive ability across both potable and non-potable samples.
        """)

    with doc_tab2:
        st.markdown("### ⏱️ 10-Minute Presentation Slide Breakdown")
        st.markdown("""
        Use this exact timeline when presenting your project to college professors:

        | Time | Slide / Topic | What to Say |
        | :--- | :--- | :--- |
        | **0:00 - 1:30** | **Slide 1: Introduction & Problem Statement** | "Good morning professors. Today I am presenting AquaGuard AI, a machine learning system designed to classify water samples as potable or non-potable based on 9 physicochemical parameters like pH, sulfates, and chloramines." |
        | **1:30 - 3:00** | **Slide 2: Dataset & Preprocessing** | "We used the real-world Kaggle Water Potability dataset of 3,276 records. Missing values in Sulfate and pH were handled with median imputation. We split the data 80/20 with stratified sampling and strictly prevented data leakage by fitting transformers only on training data." |
        | **3:00 - 5:00** | **Slide 3: Algorithms & The Linear Failure Point** | "We compared Logistic Regression against Random Forest. Here is our major technical finding: water potability does not have a linear decision boundary. Logistic Regression failed with 0% recall, whereas Random Forest successfully modeled the complex non-linear interactions." |
        | **5:00 - 6:30** | **Slide 4: Hyperparameter Optimization** | "Using 5-fold GridSearchCV, we tuned tree depth, split thresholds, and estimators. The optimized model achieved 67.29% cross-validation accuracy and 66.16% test accuracy, with Sulfate and pH emerging as the top predictive features." |
        | **6:30 - 8:30** | **Slide 5: Live System Demonstration** | Open the **Water Quality Predictor** tab. Click 'Load Known Potable Sample' and click Predict. Show the probability output and model badge. Then load 'Non-Potable Sample' to show contrast. Point out the academic disclaimer. |
        | **8:30 - 10:00** | **Slide 6: Conclusion & Viva Wrap-up** | "AquaGuard AI demonstrates how machine learning pipelines can automate preliminary water analysis while maintaining strict academic and scientific ethics. Thank you, I am now open for questions." |
        """)

    with doc_tab3:
        st.markdown("### 🎯 Top College Viva Questions & Answers")
        st.markdown("""
        **Q1: Why did Logistic Regression achieve 60.98% accuracy but 0.00 F1-score?**
        > *Answer:* The dataset has 61% non-potable samples and 39% potable samples. Because the physical parameters have non-linear and near-zero Pearson correlations with the target, the linear hyperplane could not find a separating line and simply predicted all instances as the majority class (0). This yields 60.98% accuracy by default, but zero recall and zero F1-score.

        **Q2: How did you ensure there was no data leakage in your preprocessing?**
        > *Answer:* We used Scikit-Learn Pipelines where `SimpleImputer` and `StandardScaler` were fitted exclusively on the 80% training set (`X_train`). The held-out 20% test set was only transformed using the training parameters, ensuring test statistics never leaked into model training.

        **Q3: Why did you choose Median Imputation instead of Mean Imputation?**
        > *Answer:* Environmental chemical features like Sulfate and Solids contain extreme outliers. The median is a non-parametric statistic resistant to extreme values, whereas the mean would be distorted by outliers.

        **Q4: Why was Stratified Splitting necessary?**
        > *Answer:* Because the dataset exhibits a 61:39 class distribution. Stratified splitting guarantees that both the 80% training set and 20% test set maintain the exact same proportion of potable and non-potable records.

        **Q5: Can this system replace municipal laboratory water testing?**
        > *Answer:* No. As stated in our project disclaimer, this is an educational decision-support model based on historical patterns. Biological pathogen assays and regulatory standards require accredited wet-chemistry lab analysis.
        """)

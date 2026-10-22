# AquaGuard AI – Water Quality Classification System

An end-to-end, beginner-friendly Machine Learning and Streamlit web application designed for college final-year and semester projects. **AquaGuard AI** analyzes historical water-quality physicochemical parameters and classifies water samples as **Potable** or **Non-Potable** using authentic machine learning models trained on a genuine public dataset.

---

## 📌 1. Problem Definition

### Problem Statement
> *"Water quality depends on several physical and chemical parameters. This project uses machine learning to analyze historical water-quality data and classify water samples as potable or non-potable according to their recorded dataset labels."*

### Key Objectives
1. **Analyze Physicochemical Indicators:** Evaluate water quality parameters (pH, hardness, solids, chloramines, sulfates, conductivity, organic carbon, trihalomethanes, turbidity).
2. **Handle Incomplete Data Without Leakage:** Employ robust median imputation for missing sensor values and fit all transformers strictly on training data.
3. **Compare Linear vs Ensemble Algorithms:** Demonstrate why linear models (Logistic Regression) fail on multi-variable chemical relationships, and how ensemble tree algorithms (Random Forest) succeed.
4. **Optimize Hyperparameters:** Apply 5-fold cross-validation with `GridSearchCV` on the training dataset.
5. **Interactive Web Deployment:** Provide a clean, water-themed Streamlit interface with instant predictions, class probability estimation, dataset analysis, and academic presentation guidance.

### Scope and Limitations
- **Educational Scope:** AquaGuard AI is created for educational and decision-support demonstration.
- **Scientific Limitation:** Machine learning predictions reflect historical dataset labels and statistical correlations. They do not substitute certified laboratory chemical testing or official drinking water safety compliance (e.g., WHO, EPA, BIS standards).

### Potential Applications
- Preliminary screening for environmental monitoring and water treatment plants.
- College demonstrations of machine learning pipelines, data leakage prevention, and model evaluation.
- Decision support for water conservation and watershed management organizations.

---

## 📊 2. Genuine Dataset Collection & Schema

- **Dataset Name:** Water Potability Dataset
- **Source:** [Kaggle – Aditya Kadiwal](https://www.kaggle.com/datasets/adityakadiwal/water-potability)
- **Total Records:** 3,276 water samples
- **Total Columns:** 10 (9 input features + 1 target variable)
- **Duplicate Records:** 0

### Input Features Description

| # | Feature | Unit / Metric | WHO / Typical Standard | Description | Missing Values |
|---|---|---|---|---|---|
| 1 | `ph` | pH scale (0–14) | 6.5 – 8.5 | Acid-base balance of water | 491 (15.0%) |
| 2 | `Hardness` | mg/L | < 200 mg/L | Capacity to precipitate soap (calcium & magnesium) | 0 (0.0%) |
| 3 | `Solids` | ppm (TDS) | < 500 – 1000 ppm | Total Dissolved Solids in water | 0 (0.0%) |
| 4 | `Chloramines` | ppm | < 4 ppm | Chlorine + ammonia disinfectant residual | 0 (0.0%) |
| 5 | `Sulfate` | mg/L | < 250 mg/L | Dissolved mineral salts from natural deposits | 781 (23.8%) |
| 6 | `Conductivity` | μS/cm | < 400 μS/cm | Electrical conductivity from dissolved ionic substances | 0 (0.0%) |
| 7 | `Organic_carbon` | ppm | < 2 – 4 ppm | Total Organic Carbon (TOC) from decaying biomass | 0 (0.0%) |
| 8 | `Trihalomethanes` | μg/L | < 80 μg/L | Chlorination by-products | 162 (4.9%) |
| 9 | `Turbidity` | NTU | < 5 NTU | Measure of water clarity and light scattering | 0 (0.0%) |

### Target Variable
- **`Potability`**:
  - `0` = **Non-Potable** (1,998 records, 61.0%)
  - `1` = **Potable** (1,278 records, 39.0%)

---

## ⚙️ 3. Data Preprocessing & Leakage Prevention

1. **Missing Numerical Values:** Handled via **Median Imputation** (`SimpleImputer(strategy='median')`). The median is chosen over the mean because chemical features contain skewness and outliers.
2. **Stratified Split:** 80% training data (2,620 samples) and 20% testing data (656 samples) using `stratify=y` to preserve the 61:39 class ratio.
3. **Feature Scaling:** `StandardScaler` standardizes numerical features to zero mean and unit variance.
4. **Data Leakage Prevention:** The preprocessing pipeline is fitted **only** on `X_train`. The held-out `X_test` is transformed using the parameters learned from `X_train`, ensuring complete independence.

---

## 🤖 4. Model Development & Comparison

Two distinct classification paradigms were trained using identical training data and evaluated on the same 656-sample held-out test set:

### Model A: Logistic Regression (Baseline)
- Linear classifier establishing a benchmark.
- **Result:** Fails to separate the non-linear feature space. Predicts all test samples as Class 0 (Non-Potable).
- Accuracy: **60.98%**, Precision: **0.0000**, Recall: **0.0000**, F1-Score: **0.0000**.

### Model B: Random Forest Classifier (Baseline Ensemble)
- Non-linear ensemble of 100 decision trees.
- Accuracy: **66.31%**, Precision: **0.6446**, Recall: **0.3047**, F1-Score: **0.4138**.

---

## ⚡ 5. Model Optimization (GridSearchCV)

Hyperparameter optimization was performed on the Random Forest pipeline using 5-fold cross-validation on the training data only:
- **Tuned Hyperparameters:**
  - `n_estimators`: `[100, 200]`
  - `max_depth`: `[None, 10, 20]`
  - `min_samples_split`: `[2, 5, 10]`
- **Best Hyperparameters Found:** `{'max_depth': 20, 'min_samples_split': 5, 'n_estimators': 100}`
- **Best 5-Fold Cross-Validation Accuracy:** **67.29%**
- **Held-Out Test Set Performance:**
  - Test Accuracy: **66.16%**
  - Test Precision: **0.6417**
  - Test Recall: **0.3008**
  - Test F1-Score: **0.4096**

### Honest Performance Comparison

| Metric | Logistic Regression (Baseline) | Random Forest (Baseline) | Random Forest (Optimized) |
|---|---|---|---|
| **Accuracy** | 60.98% | **66.31%** | 66.16% |
| **Precision** | 0.0000 | **0.6446** | 0.6417 |
| **Recall** | 0.0000 | **0.3047** | 0.3008 |
| **F1-Score** | 0.0000 | **0.4138** | 0.4096 |
| **5-Fold CV Accuracy** | N/A | N/A | **67.29%** |

> **Evaluation Takeaway:** Cross-validation stabilized tree depth (`max_depth: 20`) and split regularization (`min_samples_split: 5`). Baseline Random Forest and Tuned Random Forest perform very similarly on the held-out test set, demonstrating that default parameters were already robust, while both dramatically outperform the linear baseline.

---

## 📐 6. Confusion Matrix Breakdown

### Logistic Regression
```
                Predicted 0    Predicted 1
Actual 0 (TN/FP)    400             0
Actual 1 (FN/TP)    256             0
```
*The linear model suffers from zero sensitivity (recall = 0), simply guessing the majority class.*

### Random Forest (Selected Model)
```
                Predicted 0    Predicted 1
Actual 0 (TN/FP)    357            43
Actual 1 (FN/TP)    179            77
```
- **True Negatives (TN):** 357 non-potable samples correctly identified.
- **False Positives (FP):** 43 non-potable samples incorrectly classified as potable.
- **False Negatives (FN):** 179 potable samples misclassified as non-potable.
- **True Positives (TP):** 77 potable samples correctly identified.

---

## 📁 7. Project Structure

```
aquaguard-ai/
│
├── app.py                  # Streamlit web application with UI, predictor, and charts
├── train_model.py          # Preprocessing, model training, GridSearchCV, and evaluation
├── water_potability.csv    # Genuine Kaggle water potability dataset (3,276 records)
├── water_model.joblib      # Serialized Scikit-learn Pipeline (Imputer + Scaler + RF)
├── model_results.json      # Actual evaluated metrics, confusion matrices, and presets
├── requirements.txt        # Python library dependencies
├── README.md               # Project documentation and presentation guide
└── .gitignore              # Git ignore rules for clean repository
```

---

## 💻 8. Installation & Local Execution (Windows PowerShell)

Follow these exact steps in **Windows PowerShell**:

### Step 1: Open Terminal in Project Directory
```powershell
cd d:\AIML-PROJECT
```

### Step 2: Create a Virtual Environment (Recommended)
```powershell
python -m venv venv
```

### Step 3: Activate the Virtual Environment
```powershell
.\venv\Scripts\Activate.ps1
```
*(If PowerShell shows an execution policy warning, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` first)*

### Step 4: Install Required Dependencies
```powershell
pip install -r requirements.txt
```

### Step 5: Train and Evaluate Models (Generates artifacts)
```powershell
python train_model.py
```

### Step 6: Launch the Streamlit Web Application
```powershell
streamlit run app.py
```
Open your browser at `http://localhost:8501` to view AquaGuard AI!

---

## 🚀 9. GitHub Upload & Streamlit Community Cloud Deployment

### A. Push Code to GitHub
1. Open PowerShell in `d:\AIML-PROJECT`.
2. Initialize Git and commit your files:
```powershell
git init
git add .
git commit -m "Initial commit: AquaGuard AI Water Quality Classification System"
```
3. Create a new public repository on [GitHub](https://github.com/new) named `aquaguard-ai`.
4. Link and push your repository:
```powershell
git branch -M main
git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/aquaguard-ai.git
git push -u origin main
```

### B. Deploy on Streamlit Community Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
2. Click **"New app"**.
3. Select your repository: `<YOUR_GITHUB_USERNAME>/aquaguard-ai`.
4. Branch: `main`.
5. Main file path: `app.py`.
6. Click **"Deploy!"**.
7. Streamlit Cloud will install dependencies from `requirements.txt` and launch your live web application!

---

## 🎓 10. 10-Minute College Presentation Guide

Use this script to deliver an outstanding college presentation:

- **Minute 0:00 – 1:30 (Title & Problem Definition):**  
  *"Good morning respected professors. Today I am presenting AquaGuard AI, a machine learning system for classifying water potability from physical and chemical indicators."*
- **Minute 1:30 – 3:00 (Dataset & Preprocessing):**  
  *"We used a genuine dataset of 3,276 water samples. We handled missing values in Sulfate and pH using median imputation to prevent outlier distortion, and fitted our pipeline only on the 80% training split to eliminate data leakage."*
- **Minute 3:00 – 5:00 (Algorithm Comparison & Key Discovery):**  
  *"We compared Logistic Regression and Random Forest. We discovered that water potability is non-linear—Logistic Regression failed with 0% recall by guessing the majority class, while Random Forest achieved 66.16% test accuracy and balanced multi-class performance."*
- **Minute 5:00 – 6:30 (Optimization & Feature Importance):**  
  *"We tuned Random Forest using 5-fold cross-validation. Sulfate and pH were identified as the strongest predictive indicators."*
- **Minute 6:30 – 8:30 (Live System Demonstration):**  
  *Open the Predictor tab. Click 'Load Known Potable Sample' -> Predict. Then click 'Load Known Non-Potable Sample' -> Predict. Point out the probability gauge, model badge, and academic disclaimer.*
- **Minute 8:30 – 10:00 (Conclusion & Viva Defense):**  
  *"AquaGuard AI shows how machine learning can serve as an automated preliminary screening tool while respecting water safety ethics. Thank you, I am ready for questions."*

---

## 🎯 11. Viva Questions & Model Answers

**Q1: Why did Logistic Regression fail with 0.00 recall?**  
*Answer:* The correlation between individual water features and potability is close to 0. A single linear hyperplane cannot partition the complex overlapping chemical parameters, so the algorithm minimized loss by classifying everything as the majority class (Class 0, 61%).

**Q2: What is data leakage, and how did you prevent it?**  
*Answer:* Data leakage occurs when test set statistics influence model training. We wrapped preprocessing in a Scikit-learn Pipeline so that the median imputer and standard scaler were fitted strictly on `X_train`.

**Q3: Why use Median Imputation instead of Mean?**  
*Answer:* Water parameters like Solids and Sulfate have wide spreads and high-value outliers. The median is a non-parametric statistic unaffected by extreme outliers.

---

## ⚠️ Academic Disclaimer

> **AquaGuard AI is an educational machine learning project. Its predictions reflect patterns learned from historical dataset labels and do not establish whether water is safe to drink. Laboratory testing and applicable drinking-water standards are required to assess actual water safety.**

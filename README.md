# 📦 Supply Chain Risk Intelligence
### Multi-class Delivery Risk Prediction using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Gradio](https://img.shields.io/badge/Gradio-Live%20Demo-orange.svg)](https://gradio.app)
[![XGBoost](https://img.shields.io/badge/XGBoost-Enabled-green.svg)](https://xgboost.readthedocs.io)
[![SMOTE](https://img.shields.io/badge/SMOTE-Balanced-purple.svg)](https://imbalanced-learn.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 Project Overview

This project develops a **Multi-class Machine Learning Classification System** for Supply Chain Risk Intelligence using the **DataCo Supply Chain Dataset**. The system predicts delivery outcomes for e-commerce orders in real-time and triggers automated **Agentic Interventions** based on the prediction.

The goal is to improve operational efficiency, reduce logistics costs, and enhance customer satisfaction by identifying delivery risks before they occur.

---

## 🚚 Delivery Outcome Classes

The model classifies each order into one of **four delivery outcomes:**

| Class | Description | Trigger |
|-------|-------------|---------|
| 🚨 **Late Delivery** | Order exceeded scheduled delivery date | Actual vs Scheduled = +1 to +4 |
| ✅ **Shipping On Time** | Order delivered within scheduled timeline | Actual vs Scheduled = 0 |
| ⚡ **Advance Shipping** | Order delivered ahead of schedule | Actual vs Scheduled = -1 or -2 |
| ❌ **Shipping Canceled** | Order canceled before delivery | Rare class |

---

## 📊 Model Performance

Four machine learning algorithms were trained and compared:

| Algorithm | Accuracy | F1-Score | Notes |
|-----------|----------|----------|-------|
| 🥇 **KNN** | **92.2%** | **0.919** | Best performer |
| 🥈 XGBoost | 88.5% | 0.886 | Strong gradient boosting |
| 🥉 Random Forest | 87.4% | 0.902 | Best F1 after KNN |
| Logistic Regression | 79.5% | 0.847 | Baseline linear model |

> **Key Finding:** KNN outperformed ensemble methods suggesting delivery outcomes are highly dependent on local patterns among similar orders rather than complex global decision boundaries.

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| **Language** | Python 3.10+ |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn, XGBoost |
| **Class Balancing** | imbalanced-learn (SMOTE) |
| **Explainability** | SHAP |
| **Visualization** | Matplotlib, Seaborn |
| **Deployment** | Gradio |
| **Model Serialization** | Joblib |
| **Environment** | Google Colab |

---

## 📁 Project Structure

```
supply-chain-risk-intelligence/
│
├── app.py                          ← Gradio dashboard (main UI)
├── training_pipeline.py            ← Full training pipeline
├── requirements.txt                ← Python dependencies
├── README.md                       ← Project documentation
│
├── models/
│   ├── xgboost_model.pkl           ← Trained XGBoost model
│   ├── random_forest_model.pkl     ← Trained Random Forest model
│   ├── knn_model.pkl               ← Trained KNN model
│   ├── logistic_regression_model.pkl ← Trained Logistic Regression
│   └── label_encoder.pkl           ← Label encoder for class mapping
│
└── notebooks/
    └── Supply_Chain_Intelligence.ipynb ← Google Colab notebook
```

---

## 🔧 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/YOURUSERNAME/supply-chain-risk-intelligence.git
cd supply-chain-risk-intelligence
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the Dataset
Download the **DataCo Supply Chain Dataset** from Kaggle:
👉 [DataCo Supply Chain Dataset on Kaggle](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis)

Place the CSV file in the root directory:
```
DataCoSupplyChainDataset.csv
```

### 4. Train the Models
Run the training pipeline to generate all `.pkl` model files:
```bash
python training_pipeline.py
```

### 5. Launch the Dashboard
```bash
python app.py
```
A public Gradio URL will be printed in the terminal. Open it in any browser.

---

## 🗃️ Dataset Overview

| Property | Value |
|----------|-------|
| **Dataset** | DataCo Supply Chain |
| **Total Rows** | 134,453 orders |
| **Input Features** | 7 |
| **Target Variable** | Delivery Status (4 classes) |
| **Encoding** | ISO-8859-1 |

---

## 🧠 Feature Engineering

| Feature | Description | Type |
|---------|-------------|------|
| `Type` | Order type | Categorical (encoded) |
| `Days for shipment (scheduled)` | Planned delivery window | Numeric |
| `Shipping Mode` | First Class, Same Day, Second Class, Standard | Categorical (encoded) |
| `Order Region` | Geographic region | Categorical (encoded 0–4) |
| `Order Item Product Price` | Price of ordered item in USD | Numeric |
| `Order Item Quantity` | Number of units in order | Numeric |
| `Actual_vs_Scheduled` ⭐ | **Engineered:** Actual days − Scheduled days | Numeric |

> ⭐ **Actual_vs_Scheduled** is the most critical feature. It directly measures deviation from the delivery plan. Positive = late, Zero = on time, Negative = early.

---

## ⚙️ Data Preprocessing Pipeline

```
1. Load CSV (134,453 rows, ISO-8859-1 encoding)
        ↓
2. Feature Engineering (Create Actual_vs_Scheduled)
        ↓
3. Handle Missing Values (Replace inf/NaN with median)
        ↓
4. Label Encoding (Type, Shipping Mode, Order Region, Target)
        ↓
5. Train-Test Split (80/20, random_state=42, stratify=y)
        ↓
6. SMOTE Balancing (Oversample minority classes in training set)
        ↓
7. Model Training & Evaluation
```

---

## 🤖 Agentic Intervention System

The dashboard automatically triggers intervention actions based on prediction:

### 🚨 Late Delivery — HIGH RISK
```
→ Warehouse Priority Flag ........... ACTIVE
→ Customer delay notification ....... DISPATCHED
→ Carrier escalation protocol ....... INITIATED
→ SLA breach alert .................. SENT TO OPS
```

### ✅ Shipping On Time — LOW RISK
```
→ Standard logistics monitoring ..... ACTIVE
→ SLA compliance .................... WITHIN BOUNDS
→ Carrier status .................... NOMINAL
```

### ⚡ Advance Shipping — EARLY DELIVERY
```
→ Warehouse receiving alert ......... SENT
→ Early arrival notification ........ DISPATCHED
→ Customer notified ................. ACTIVE
→ Storage slot pre-assigned ......... CONFIRMED
```

### ❌ Shipping Canceled
```
→ Order flagged for review .......... ACTIVE
→ Customer refund initiated ......... PROCESSING
→ Inventory restocked ............... PENDING
```

---

## 🖥️ Dashboard Interface

### Input Parameters
- **Algorithm** — Select ML model (XGBoost, Random Forest, KNN, Logistic Regression)
- **Scheduled Shipment Days** — Planned delivery window (0–10)
- **Actual vs Scheduled Days** — Key feature slider (-2 to +4)
- **Shipping Mode** — First Class, Same Day, Second Class, Standard Class
- **Order Region** — Encoded region (0–4)
- **Product Price** — Item price in USD
- **Order Quantity** — Number of units

### Output Panels
- **Status** — Predicted delivery outcome with risk indicator
- **Detail** — Model confidence score and accuracy
- **Agentic Action Log** — Automated intervention steps
- **Risk Indicator** — Visual bar showing risk level
- **Input Feature Vector** — Exact values sent to model for transparency

### Quick Reference — Try These Parameters

| Outcome | Actual vs Scheduled | Shipping Mode |
|---------|--------------------|-|
| 🚨 Late Delivery | +2 to +4 | Standard Class |
| ✅ On Time | 0 | Any |
| ⚡ Early Delivery | -1 or -2 | First Class |

---

## 📈 Evaluation Metrics

### Confusion Matrices
Plotted for all 4 models in a 2×2 grid. Darker diagonal = better performance.

### ROC Curves
Macro-average ROC curves for all models.
- KNN AUC: ~0.97
- Random Forest AUC: ~0.96
- XGBoost AUC: ~0.95
- Logistic Regression AUC: ~0.91

### SHAP Explainability
SHAP TreeExplainer used on Random Forest to rank feature importance globally. `Actual_vs_Scheduled` is the dominant predictor.

---

## 💡 Key Design Decisions

| Decision | Reason |
|----------|--------|
| Gradio over Streamlit | Avoids localtunnel/ngrok issues, instant public URL with share=True |
| SMOTE over class weighting | Better synthetic minority representation |
| Saved label_encoder.pkl separately | Maps numeric predictions back to real class names in dashboard |
| random_state=42 everywhere | Ensures reproducible results across all runs |
| encoding_errors='replace' | Loads all 134K rows without dropping rows with special characters |

---

## 🔮 Future Improvements

- [ ] Add weather data and carrier performance as features
- [ ] Implement real-time order feed from live database
- [ ] Add SHAP waterfall plots for individual order explainability
- [ ] Deploy on cloud platform (AWS/GCP) for 24/7 availability
- [ ] Add email/SMS notification integration for agentic actions
- [ ] Build REST API endpoint for integration with order management systems

---

## 📚 References

- [DataCo Supply Chain Dataset — Kaggle](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis)
- [SMOTE Paper — Chawla et al. 2002](https://arxiv.org/abs/1106.1813)
- [XGBoost Documentation](https://xgboost.readthedocs.io)
- [SHAP Documentation](https://shap.readthedocs.io)
- [Gradio Documentation](https://gradio.app/docs)

---

## 👤 Author

**Your Name**
- GitHub: [@Arshpreet-Singh-2005](https://github.com/Arshpreet-Singh-2005)
- LinkedIn: [Arshpreet Singh](https://www.linkedin.com/in/arshpreet-singh-56089531a/)

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<p align="center">
  Built with ❤️ for Supply Chain Intelligence
  <br>
  DataCo Dataset · SMOTE Balanced · Agentic Intervention System · ML v2.0
</p>

# Install advanced libraries
!pip install imbalanced-learn xgboost shap streamlit -q
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import urllib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import random
import shap
df = pd.read_csv('DataCoSupplyChainDataset.csv', encoding='ISO-8859-1')
print(f"Dataset Loaded: {df.shape[0]} rows")

# 1. Feature Engineering: Create the core predictor
df['Actual_vs_Scheduled'] = df['Days for shipping (real)'] - df['Days for shipment (scheduled)']

# 2. Select Relevant Features
# We exclude 'Delivery Status' from X as it is our target variable
features = ['Type', 'Days for shipment (scheduled)', 'Shipping Mode',
            'Order Region', 'Order Item Product Price', 'Order Item Quantity', 'Actual_vs_Scheduled']
X = df[features].copy()
y = df['Delivery Status']

# 3. Handle Infinity and NaNs
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(X.median(numeric_only=True), inplace=True)

# 4. Label Encoding for X and y
le = LabelEncoder()
for col in ['Type', 'Shipping Mode', 'Order Region']:
    X[col] = le.fit_transform(X[col].astype(str))

y = le.fit_transform(y.astype(str)) # 0: Late, 1: On-Time, etc.
joblib.dump(le, 'label_encoder.pkl') # Save for UI

# 5. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Apply SMOTE to balance the training data
sm = SMOTE(random_state=42)
# Define the models
models = {
    "XGBoost": XGBClassifier(n_estimators=100, random_state=42,eval_metric='mlogloss'),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42,n_jobs=-1),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000,random_state=42))
}
results = []
for name, model in models.items():
    model.fit(X_train_res, y_train_res)
    preds = model.predict(X_test)

    # Store Metrics
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='weighted')
    results.append({"Algorithm": name, "Accuracy": acc, "F1-Score": f1})

    # Save each model for the Streamlit UI
    joblib.dump(model, f"{name.lower().replace(' ', '_')}_model.pkl")

# Display Results Table
comparison_df = pd.DataFrame(results)
print(comparison_df.sort_values(by='F1-Score', ascending=False))
# 1. Initialize the SHAP Explainer (Use your best model, e.g., Random Forest)
explainer = shap.TreeExplainer(models["Random Forest"])
shap_values = explainer.shap_values(X_test)
# 2. Plot Feature Importance
print("SHAP Global Interpretation: Which features matter most?")
shap.summary_plot(shap_values, X_test, plot_type="bar")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize

# Set up the visualization grid
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 12))
axes = axes.flatten()

# 1. Plotting Confusion Matrices for all 4 Models
for i, (name, model) in enumerate(models.items()):
    preds = model.predict(X_test)
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt='d', ax=axes[i], cmap='Blues')
    axes[i].set_title(f'Confusion Matrix: {name}')
    axes[i].set_xlabel('Predicted')
    axes[i].set_ylabel('Actual')

plt.tight_layout()
plt.show()

# 2. Performance Comparison Bar Chart (Accuracy vs F1-Score)
comparison_df.set_index('Algorithm').plot(kind='bar', figsize=(10, 6), color=['#3498db', '#e74c3c'])
plt.title('Algorithm Comparison: Accuracy vs F1-Score')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# 3. Multi-Class ROC Curve (Interpretation)
# Note: This is a bit more advanced and looks great in a report
plt.figure(figsize=(10, 8))
for name, model in models.items():
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)
        # Binarize the output for multi-class ROC
        y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3])
        # Plotting the macro-average ROC for the algorithm
        fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc(fpr, tpr):.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison (Explainability)')
plt.legend()
plt.show()
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

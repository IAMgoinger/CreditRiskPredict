import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from data_EDA.processed import X_new, X_new_test, y_train, y_test
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, RocCurveDisplay
)
import warnings

warnings.filterwarnings("ignore")

# Инициализация моделей
log_reg = LogisticRegression(max_iter=1000, random_state=42)
tree = DecisionTreeClassifier(max_depth=5, min_samples_split=10, random_state=42)
rf = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42, n_jobs=-1)

models = {
    "Logistic Regression": log_reg,
    "Decision Tree": tree,
    "Random Forest": rf
}

# Обучение и оценка
for name, model in models.items():
    model.fit(X_new, y_train)

    y_pred = model.predict(X_new_test)
    y_proba = model.predict_proba(X_new_test)[:, 1]

    print(f"\n{name}")
    print("-" * 40)
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, y_proba):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

# ROC Curve
plt.figure(figsize=(8, 6))
for name, model in models.items():
    RocCurveDisplay.from_estimator(model, X_new_test, y_test, name=name, ax=plt.gca())
plt.title("ROC Curve Comparison")
plt.grid(True)
plt.tight_layout()
plt.show()

# Веса признаков для логистической регрессии
feature_names = X_new.columns
coef_df = pd.DataFrame({
    "feature": feature_names,
    "coefficient": log_reg.coef_[0]
})
coef_df = coef_df.sort_values(by="coefficient", key=abs, ascending=False)

print("\nLogistic Regression Coefficients by Absolute Value:")
print(coef_df.head(15))

# График важности признаков
coef_df.head(15).plot(kind='barh', x='feature', y='coefficient', legend=False, color='teal')
plt.title("Feature Importances (Logistic Regression)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
plt.close('all')
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def load_data(filepath):
    df = pd.read_csv(filepath)
    if "Id" in df.columns:
        df = df.drop(columns=["Id"])
    X = df.drop(columns=["Species"])
    y = df["Species"]
    return X, y


def main():
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "Iris.csv")
    X, y = load_data(data_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Regresión Logística": LogisticRegression(max_iter=200, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(kernel="rbf", probability=True, random_state=42),
    }

    print("=== Resultados de Clasificación (Dataset Iris) ===\n")
    for name, model in models.items():
        if name == "Random Forest":
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)
        else:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        roc_auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")

        print(f"Modelo: {name}")
        print(f"  - Accuracy : {acc:.4f}")
        print(f"  - F1-Score : {f1:.4f}")
        print(f"  - ROC-AUC  : {roc_auc:.4f}\n")


if __name__ == "__main__":
    main()

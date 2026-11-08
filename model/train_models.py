import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

def main():
    # 1. Load Data
    data_path = 'public_health_surveillance_dataset.csv'
    if not os.path.exists(data_path):
        data_path = '../public_health_surveillance_dataset.csv'
    
    df = pd.read_csv(data_path)
    
    # 2. Preprocess & Clean Missing Values
    for col in ['Medical_History', 'Reported_Symptoms', 'Diagnosis']:
        df[col] = df[col].fillna('None')
        
    date_cols = ['Date_of_Onset', 'Date_of_Data_Collection', 'Vaccination_Campaign_Dates']
    df_clean = df.drop(columns=date_cols)
    
    target = 'Vaccination_Hesitancy'
    y = (df_clean[target] == 'Yes').astype(int)
    X = df_clean.drop(columns=[target])
    
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'str']).columns.tolist()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features)
        ])
    
    # 3. Train Test Split (80% / 20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Save test dataset sample for Streamlit deployment
    test_df = X_test.copy()
    test_df['Vaccination_Hesitancy'] = np.where(y_test == 1, 'Yes', 'No')
    test_df.head(500).to_csv('test_data.csv', index=False)
    
    # Fit Preprocessor
    X_train_prep = preprocessor.fit_transform(X_train)
    X_test_prep = preprocessor.transform(X_test)
    
    os.makedirs('model', exist_ok=True)
    joblib.dump(preprocessor, 'model/scaler_encoder.pkl')
    
    # 4. Fast Model Definitions
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
        "kNN": KNeighborsClassifier(n_neighbors=15),
        "Naïve Bayes": GaussianNB(),
        # Fast Linear SVM with Probability Calibration
        "Support Vector Machine": CalibratedClassifierCV(LinearSVC(dual=False, max_iter=2000, random_state=42))
    }
    
    results = []
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_prep, y_train)
        
        y_pred = model.predict(X_test_prep)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test_prep)[:, 1]
        else:
            y_prob = model.decision_function(X_test_prep)
            
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)
        
        results.append({
            "ML Model Name": name,
            "Accuracy": f"{acc:.7f}",
            "AUC": f"{auc:.7f}",
            "Precision": f"{prec:.7f}",
            "Recall": f"{rec:.7f}",
            "F1": f"{f1:.7f}",
            "MCC": f"{mcc:.7f}"
        })
        
        filename_key = name.lower().replace(" ", "_").replace("ï", "i")
        joblib.dump(model, f'model/{filename_key}.pkl')
        
    results_df = pd.DataFrame(results)
    print("\n--- Model Evaluation Summary (7 Decimal Places) ---")
    print(results_df.to_string(index=False))

if __name__ == '__main__':
    main()
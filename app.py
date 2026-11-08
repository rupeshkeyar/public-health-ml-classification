import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef, 
    confusion_matrix, classification_report
)

st.set_page_config(
    page_title="Public Health ML Classifier App",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Public Health Surveillance: Classification Models Evaluator")
st.markdown("""
This interactive web application evaluates **5 Machine Learning Classification Models** on public health surveillance test data.
""")

st.sidebar.header("⚙️ Configuration Controls")

# File Upload Section
uploaded_file = st.sidebar.file_drop_target = st.sidebar.file_uploader("Upload Test Dataset (CSV)", type=["csv"])

# Model Selection Dropdown
model_choice = st.sidebar.selectbox(
    "Select ML Classification Model",
    ["Logistic Regression", "Decision Tree", "kNN", "Naïve Bayes", "Support Vector Machine"]
)

# Load pretrained preprocessor and selected model
@st.cache_resource
def load_artifacts(model_name):
    file_map = {
        "Logistic Regression": "logistic_regression.pkl",
        "Decision Tree": "decision_tree.pkl",
        "kNN": "knn.pkl",
        "Naïve Bayes": "naive_bayes.pkl",
        "Support Vector Machine": "support_vector_machine.pkl"
    }
    
    prep_path = os.path.join("model", "scaler_encoder.pkl")
    model_path = os.path.join("model", file_map[model_name])
    
    preprocessor = joblib.load(prep_path) if os.path.exists(prep_path) else None
    model = joblib.load(model_path) if os.path.exists(model_path) else None
    
    return preprocessor, model

if uploaded_file is not None:
    df_test = pd.read_csv(uploaded_file)
    st.success(f"Successfully loaded dataset with {df_test.shape[0]} rows and {df_test.shape[1]} columns.")
    
    with st.expander("🔍 Preview Input Test Dataset"):
        st.dataframe(df_test.head(10))
        
    if 'Vaccination_Hesitancy' in df_test.columns:
        y_true = (df_test['Vaccination_Hesitancy'] == 'Yes').astype(int)
        X_input = df_test.drop(columns=['Vaccination_Hesitancy'])
        
        # Clean missing values
        for col in ['Medical_History', 'Reported_Symptoms', 'Diagnosis']:
            if col in X_input.columns:
                X_input[col] = X_input[col].fillna('None')
                
        date_cols = ['Date_of_Onset', 'Date_of_Data_Collection', 'Vaccination_Campaign_Dates']
        X_input = X_input.drop(columns=[c for c in date_cols if c in X_input.columns])
        
        preprocessor, model = load_artifacts(model_choice)
        
        if preprocessor is not None and model is not None:
            try:
                X_prep = preprocessor.transform(X_input)
                y_pred = model.predict(X_prep)
                
                if hasattr(model, "predict_proba"):
                    y_prob = model.predict_proba(X_prep)[:, 1]
                else:
                    y_prob = model.decision_function(X_prep)
                    
                # Calculate metrics
                acc = accuracy_score(y_true, y_pred)
                auc = roc_auc_score(y_true, y_prob)
                prec = precision_score(y_true, y_pred, zero_division=0)
                rec = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                mcc = matthews_corrcoef(y_true, y_pred)
                
                st.subheader(f"📊 Evaluation Metrics: {model_choice}")
                m1, m2, m3, m4, m5, m6 = st.columns(6)
                m1.metric("Accuracy", f"{acc:.7f}")
                m2.metric("AUC Score", f"{auc:.7f}")
                m3.metric("Precision", f"{prec:.7f}")
                m4.metric("Recall", f"{rec:.7f}")
                m5.metric("F1 Score", f"{f1:.7f}")
                m6.metric("MCC Score", f"{mcc:.7f}")
                
                st.markdown("---")
                col_left, col_right = st.columns(2)
                
                with col_left:
                    st.subheader("🎯 Confusion Matrix")
                    cm = confusion_matrix(y_true, y_pred)
                    fig, ax = plt.subplots(figsize=(5, 4))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                                xticklabels=['No (0)', 'Yes (1)'], 
                                yticklabels=['No (0)', 'Yes (1)'], ax=ax)
                    plt.xlabel("Predicted Label")
                    plt.ylabel("True Label")
                    st.pyplot(fig)
                    
                with col_right:
                    st.subheader("📄 Classification Report")
                    report_dict = classification_report(y_true, y_pred, output_dict=True)
                    report_df = pd.DataFrame(report_dict).transpose()
                    st.dataframe(report_df.style.highlight_max(axis=0))
                    
            except Exception as e:
                st.error(f"Error during preprocessing or prediction: {str(e)}")
        else:
            st.warning("Model binaries not found in `model/` folder. Please execute `python model/train_models.py` first.")
    else:
        st.error("Uploaded CSV missing target column `Vaccination_Hesitancy`.")
else:
    st.info("👈 Please upload `test_data.csv` using the sidebar menu to view classification evaluations.")
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

# Page Configuration
st.set_page_config(
    page_title="Public Health Risk Intelligence Portal",
    page_icon="🩺",
    layout="wide"
)

# Custom Styling for Card Containers & Metrics
st.markdown("""
<style>
    .stApp {
        background-color: #F8FAFC;
    }
    .metric-container {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        text-align: center;
    }
    .metric-title {
        font-size: 0.85rem;
        color: #64748B;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .metric-val {
        font-size: 1.6rem;
        color: #0F172A;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# Application Banner Header
st.markdown("## MACHINE LEARNING (S1-26_DSEZG565)")
st.markdown("### ASSIGNMENT # 2")
st.markdown('<p style="font-size: 20px; font-weight: 600; color: BLUE; margin-bottom: 0px;">RUPESH K R | 2025NS04125</p>', unsafe_allow_html=True)
st.markdown('<hr>', unsafe_allow_html=True)
col_header, col_status = st.columns([4, 1])
with col_header:
    st.title("🩺 Public Health Risk Surveillance Portal")
    st.caption("Multi-Model Comparative Classification & Analytics Interface")

st.markdown("---")

# Sidebar Configuration
st.sidebar.markdown("### 📥 Data & Model Settings")
uploaded_file = st.sidebar.file_uploader("Upload Test Dataset (test_data.csv)", type=["csv"])

selected_model = st.sidebar.selectbox(
    "Select Classification Model:",
    ["Logistic Regression", "Decision Tree", "kNN", "Naïve Bayes", "Support Vector Machine"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Display Options")
show_raw_preds = st.sidebar.checkbox("Show Prediction Table", value=True)
show_prob_dist = st.sidebar.checkbox("Show Prediction Probabilities", value=True)

# Helper function to load model assets
@st.cache_resource
def get_artifacts(model_name):
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

# Main Workflow
if uploaded_file is not None:
    df_test = pd.read_csv(uploaded_file)
    st.success(f"Successfully loaded dataset with {df_test.shape[0]} rows and {df_test.shape[1]} columns.")
    
    if 'Vaccination_Hesitancy' in df_test.columns:
        y_true = (df_test['Vaccination_Hesitancy'] == 'Yes').astype(int)
        X_input = df_test.drop(columns=['Vaccination_Hesitancy'])
        
        # Clean missing values & dates
        for col in ['Medical_History', 'Reported_Symptoms', 'Diagnosis']:
            if col in X_input.columns:
                X_input[col] = X_input[col].fillna('None')
                
        date_cols = ['Date_of_Onset', 'Date_of_Data_Collection', 'Vaccination_Campaign_Dates']
        X_input = X_input.drop(columns=[c for c in date_cols if c in X_input.columns])
        
        preprocessor, model = get_artifacts(selected_model)
        
        if preprocessor and model:
            X_prep = preprocessor.transform(X_input)
            y_pred = model.predict(X_prep)
            
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_prep)[:, 1]
            else:
                y_prob = model.decision_function(X_prep)
                
            # Compute evaluation metrics (formatted to 7 decimal places)
            acc = accuracy_score(y_true, y_pred)
            auc = roc_auc_score(y_true, y_prob)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            mcc = matthews_corrcoef(y_true, y_pred)
            
            # --- SECTION 1: METRIC SCORECARDS ---
            st.markdown(f"### 🎯 Evaluation Metrics (`{selected_model}`)")
            
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            
            def render_card(col, title, val):
                col.markdown(f"""
                <div class="metric-container">
                    <div class="metric-title">{title}</div>
                    <div class="metric-val">{val:.7f}</div>
                </div>
                """, unsafe_allow_html=True)
                
            render_card(m1, "Accuracy", acc)
            render_card(m2, "AUC Score", auc)
            render_card(m3, "Precision", prec)
            render_card(m4, "Recall", rec)
            render_card(m5, "F1 Score", f1)
            render_card(m6, "MCC Score", mcc)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # --- SECTION 2: VISUAL ANALYTICS ---
            c_left, c_right = st.columns([1, 1])
            
            with c_left:
                st.markdown("#### 📌 Confusion Matrix")
                cm = confusion_matrix(y_true, y_pred)
                fig, ax = plt.subplots(figsize=(4.5, 3.5))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', #YlGnBu
                            xticklabels=['No (0)', 'Yes (1)',], 
                            yticklabels=['No (0)', 'Yes (1)'], ax=ax, 
                            annot_kws={"size": 9, "weight": "regular"})

                ax.set_xticklabels(['No (0)', 'Yes (1)'], fontsize=8)
                ax.set_yticklabels(['No (0)', 'Yes (1)'], fontsize=8)

                plt.ylabel("True Label", fontsize=9, fontweight="bold")
                plt.xlabel("Predicted Label", fontsize=9, fontweight="bold")
                ax.tick_params(labelsize=8)

                cbar = ax.collections[0].colorbar
                cbar.ax.tick_params(labelsize=8)
                st.pyplot(fig)
            with c_right:
                st.markdown("#### Classification Report")
                report_dict = classification_report(y_true, y_pred, output_dict=True)
                report_df = pd.DataFrame(report_dict).transpose()
                st.dataframe(report_df.style.format("{:.7f}"))  

            # --- SECTION 3: INTERACTIVE PREDICTION RESULTS ---
            c_bottom_left, c_bottom_right = st.columns([1, 1])  
            with c_bottom_left:
                if show_prob_dist:
                    st.markdown("#### 📈 Predicted Risk Score Distribution")
                    fig_dist, ax_dist = plt.subplots(figsize=(4.5, 3.5))
                    sns.histplot(y_prob, kde=True, color="#2563EB", bins=20, ax=ax_dist)
                    plt.xlabel("Predicted Probability of Vaccination Hesitancy", fontsize=9, fontweight="bold")
                    plt.ylabel("Patient Count", fontsize=9, fontweight="bold")
                    ax_dist.tick_params(labelsize=8)
                    st.pyplot(fig_dist)
            with c_bottom_right:
                if show_raw_preds:
                    st.markdown("---")
                    st.markdown("### 📋 Model Prediction Output Explorer")
                    res_df = df_test.copy()
                    res_df['Predicted_Hesitancy'] = np.where(y_pred == 1, 'Yes', 'No')
                    res_df['Risk_Probability'] = np.round(y_prob, 5)
                    
                    # Filter by prediction class
                    filter_choice = st.selectbox("Filter Predictions:", ["All Records", "Predicted Yes (Hesitant)", "Predicted No (Non-Hesitant)"])
                    if filter_choice == "Predicted Yes (Hesitant)":
                        res_df = res_df[res_df['Predicted_Hesitancy'] == 'Yes']
                    elif filter_choice == "Predicted No (Non-Hesitant)":
                        res_df = res_df[res_df['Predicted_Hesitancy'] == 'No']
                        
                    st.dataframe(res_df[['Age', 'Gender', 'Location', 'Vaccination_Hesitancy', 'Predicted_Hesitancy', 'Risk_Probability']].head(100), use_container_width=True)  


                
        else:
            st.error("Model or Preprocessor file not found in `model/` folder.")
    else:
        st.error("Uploaded dataset is missing the target column `Vaccination_Hesitancy`.")
else:
    st.info("👈 Upload your `test_data.csv` file from the sidebar to inspect model performance metrics and predictions.")

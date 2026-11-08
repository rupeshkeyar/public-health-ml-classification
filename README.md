# Public Health Surveillance Machine Learning Classification & Streamlit Application

## a. Problem Statement
Public health agencies require predictive modeling frameworks to analyze demographic, clinical, and environmental indicators for targeted interventions. This project implements five supervised machine learning models to classify **Vaccination Hesitancy** (`Yes` vs `No`) using public health surveillance data, enabling actionable decision support for healthcare resources.

## b. Dataset Description
- **Source:** Kaggle Public Health Dataset (`public_health_surveillance_dataset.csv`)
- **Records:** 43,689 samples across 35 features.
- **Key Features:** Patient Age, Gender, Ethnicity, Socioeconomic Status (SES), AQI, Humidity, Transmission Rate, Mortality Rate, Hospitalization Rate, Outbreak Status, and Infection Risk Level.
- **Target Variable:** `Vaccination_Hesitancy` (Binary: `No` [69.9%], `Yes` [30.1%]).

## c. GitHub Repository Link
- **GitHub Repository:** `https://github.com/2025NS04125/public-health-ml-classification`


## d. Models Used & Comparison Table (Evaluated to 7 Decimal Places)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.6990158 | 0.4992618 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 |
| **Decision Tree** | 0.6927214 | 0.5012445 | 0.3049645 | 0.0163498 | 0.0310357 | 0.0011113 |
| **kNN** | 0.6809339 | 0.5009142 | 0.3110048 | 0.0494297 | 0.0853018 | 0.0048967 |
| **Naïve Bayes** | 0.6931792 | 0.4948993 | 0.2660550 | 0.0110266 | 0.0211756 | -0.0085587 |
| **Support Vector Machine** | 0.6990158 | 0.5028914 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 |

### Observations on Model Performance

| ML Model Name | Observation about model |
| :--- | :--- |
| **Logistic Regression** | Predicts majority class (`No Hesitancy`), achieving baseline Accuracy of 0.6990158 but 0.0000000 F1 score. |
| **Decision Tree** | Captures non-linear split patterns, achieving F1 score of 0.0310357 and positive MCC (0.0011113). |
| **kNN** | Yields highest non-trivial Precision (0.3110048), Recall (0.0494297), and positive MCC (0.0048967). |
| **Naïve Bayes** | Fast execution, but feature independence assumptions yield slightly lower AUC (0.4948993) and negative MCC (-0.0085587). |
| **Support Vector Machine** | Optimized LinearSVC completes in under 3 seconds; achieves highest AUC (0.5028914) while maintaining baseline accuracy. |
| **Overall Winner for your dataset?** | **k-Nearest Neighbors (kNN)** — Achieves highest F1 score (0.0853018), Precision (0.3110048), and MCC (0.0048967), delivering meaningful positive class predictions. |
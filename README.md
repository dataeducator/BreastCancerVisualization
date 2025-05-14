# 📌 Breast Cancer Visualization: Predicting Breast Cancer with Machine Learning
<img width="953" alt="breast_cancer_prediction" src="https://github.com/user-attachments/assets/fe2e411e-d664-44f5-84fb-b36db7f19fc0" />

## 📌 Disclaimer:

This Jupyter notebook and its contents are intended solely for educational purposes. The analysis and results presented should not be interpreted as medical advice. This model has not been reviewed or endorsed by any professional medical organization.

The findings are for illustrative purposes only, and users should not rely on these predictions for clinical decision-making. Consult a licensed medical professional for diagnosis and treatment. The dataset used may not fully represent real-world clinical scenarios, and predictions should be interpreted with caution.

The author and contributors of this notebook disclaim any liability for the accuracy, completeness, or efficacy of the information provided.

## 📌 Overview:

The primary objective of this project is to develop a linear regression model capable of predicting tumor area (Area Mean) based on various diagnostic features.

Leveraging machine learning, Zenith Medical Analytics aims to explore how different cellular characteristics influence tumor size, which could contribute to better understanding of tumor growth patterns and potential risk factors.




## 📌 Business Understanding

**Problem Statement:**

Early and accurate detection of breast cancer is vital for improving patient survival rates. Leveraging machine learning, this project aims to build robust models to classify breast cancer as malignant or benign, focusing on maximizing predictive accuracy and interpretability for clinical use.

**Stakeholder:**

Healthcare providers, oncologists, and data scientists interested in diagnostic support tools.

**Business Case:**

As part of an initiative to enhance diagnostic capabilities, this project explores the use of machine learning algorithms to assist in the early detection of breast cancer. By providing interpretable predictions, the models can support clinicians in making informed decisions, potentially leading to better patient outcomes.

---

## 📌 Data Understanding

**Data Description:**

The dataset used is the classic Breast Cancer Wisconsin (Diagnostic) Dataset, which contains features computed from digitized images of fine needle aspirate (FNA) of breast masses.

### Features

| Feature                   | Description                                             |
|---------------------------|---------------------------------------------------------|
| `id`                      | Unique identifier                                       |
| `diagnosis`               | Target variable (M = malignant, B = benign)             |
| `radius_mean`             | Mean of distances from center to points on the perimeter|
| `texture_mean`            | Standard deviation of gray-scale values                 |
| `perimeter_mean`          | Mean size of the core tumor                             |
| `area_mean`               | Mean area of the tumor                                  |
| `smoothness_mean`         | Mean of local variation in radius lengths               |
| ...                       | ...                                                     |
| `fractal_dimension_worst` | "Worst" or largest value for fractal dimension          |

*Note: The dataset includes 30 real-valued features computed for each cell nucleus.*

### Data Exploration

The dataset is visualized and explored to identify patterns and relationships between features and the diagnosis outcome. For example:

- Malignant tumors tend to have larger mean radius, perimeter, and area compared to benign tumors.
- Certain features, such as `concavity_worst` and `compactness_mean`, are more pronounced in malignant cases.

---

## 📌 Data Preparation

**Data Cleaning and Preprocessing:**

- Removed unnecessary columns (e.g., `id`).
- Encoded target variable (`diagnosis`) as binary (1 = malignant, 0 = benign).
- Checked for and handled missing values.
- Scaled features for model compatibility.

**Visualization:**

- Used seaborn and matplotlib to plot feature distributions and correlations.
- Created pairplots and heatmaps to visualize relationships and feature importance.

---

## 📌 Modeling

This project implements several machine learning algorithms for binary classification:

- Logistic Regression
- K-Nearest Neighbors
- Support Vector Machine
- Decision Tree
- Random Forest
- XGBoost

**Workflow:**

1. Split data into training and test sets.
2. Train models and tune hyperparameters.
3. Evaluate models using metrics such as accuracy, recall, precision, and ROC-AUC.

**Sample Results:**

| Model                | Accuracy | Recall | Precision | ROC-AUC |
|----------------------|----------|--------|-----------|---------|
| Logistic Regression  | 0.97     | 0.96   | 0.97      | 0.99    |
| Random Forest        | 0.98     | 0.97   | 0.98      | 0.99    |
| XGBoost              | 0.98     | 0.97   | 0.98      | 0.99    |

*Note: Actual results may vary; refer to the notebook for detailed metrics.*

---

## 📌 Evaluation

Feature importance analysis highlights which features most influence the model's predictions. For example, `worst perimeter`, `mean concave points`, and `worst radius` are consistently among the top predictors for malignancy.

Visualizations such as SHAP plots and confusion matrices are provided to interpret model behavior and performance.

---

## Deployment

**Web Application (Work in Progress):**

The app can be accessed here [[https://the-mmc-mammo.streamlit.app/](https://the-mmc-mammo-insight.streamlit.app/)]
---

## Recommendations

- **Use Ensemble Models:** Random Forest and XGBoost consistently offer high performance and robustness.
- **Feature Selection:** Focus on the most influential features for streamlined and interpretable models.
- **Model Interpretability:** Utilize SHAP or LIME for explaining predictions to clinicians.
- **Continuous Improvement:** Regularly retrain models with new data to maintain accuracy.

---

## Future Work

- **Advanced Feature Engineering:** Explore dimensionality reduction (PCA), interaction terms, and synthetic data generation.
- **Model Deployment:** Finalize and deploy the web application for clinical or educational use.
- **User Feedback:** Gather feedback from users to improve the interface and model performance.
- **Integration with Electronic Health Records (EHR):** Explore integration for real-world clinical deployment.

---

## Contact

For questions or collaboration, please contact:

**Tenicka Norwood**  
tenicka.norwood@gmail.com

---

*For more details, please see the full analysis in the notebook and the streamlit application.*

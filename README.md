# Medical Tabular ML Pipeline

A research-oriented machine learning pipeline for binary medical classification using the Breast Cancer Wisconsin (Diagnostic) dataset.

The project implements a complete workflow from exploratory data analysis and preprocessing to model training, cross-validation, hyperparameter tuning, final evaluation, and error analysis.

---

## Project Overview

This project is designed as a practical foundation for medical machine learning research.

The main objectives are:

- Perform exploratory data analysis (EDA)
- Apply appropriate preprocessing techniques
- Train and compare multiple machine learning models
- Use stratified cross-validation
- Perform hyperparameter tuning
- Evaluate models using multiple medical classification metrics
- Analyze malignant-case errors
- Generate reproducible figures and result tables

---

## Dataset

### Breast Cancer Wisconsin (Diagnostic)

The dataset is loaded directly using `scikit-learn`.

- Samples: 569
- Features: 30
- Target classes: 2
- Malignant: `0`
- Benign: `1`

The dataset contains numerical features computed from digitized images of fine needle aspirate (FNA) samples of breast masses.

---

## Project Structure

```text
medical-tabular-ml-pipeline/
│
├── .venv/
│
├── .vscode/
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_model_training.ipynb
│
├── src/
│
├── results/
│   ├── figures/
│   │   ├── logistic_regression_confusion_matrix.png
│   │   ├── random_forest_confusion_matrix.png
│   │   ├── svm_confusion_matrix.png
│   │   ├── xgboost_confusion_matrix.png
│   │   ├── roc_curve_comparison.png
│   │   └── pr_curve_comparison.png
│   │
│   └── tables/
│       └── model_comparison.csv
│
├── README.md
└── .gitignore


Workflow

The project follows this pipeline:

Dataset
   ↓
EDA
   ↓
Train/Test Split
   ↓
Preprocessing
   ↓
Baseline Models
   ↓
Stratified 5-Fold Cross-Validation
   ↓
Hyperparameter Tuning
   ↓
Final Model
   ↓
Untouched Test Set
   ↓
Error Analysis
   ↓
Model Comparison




Exploratory Data Analysis

EDA was performed to understand:

Dataset shape
Feature types
Missing values
Duplicate records
Feature distributions
Target-class distribution
Feature correlations
Feature-target relationships

No missing values or duplicate rows were found in the dataset.





Preprocessing

The following preprocessing steps were applied:

Train/Test Split

The dataset was divided into:

80% training data
20% test data

Stratified splitting was used to preserve class proportions.




Feature Scaling

Standardization was applied to models that are sensitive to feature scale:

Logistic Regression
SVM

Tree-based models were trained without scaling:

Random Forest
XGBoost

The scaler was fitted only on the training data to avoid data leakage.





Machine Learning Models

Four classification models were evaluated:

1. Logistic Regression

Used as a linear baseline model.

2. Random Forest

An ensemble of decision trees using bootstrap sampling and random feature selection.

3. Support Vector Machine

An RBF-kernel SVM was used for nonlinear classification.

4. XGBoost

A gradient-boosted tree-based model was used as an additional nonlinear baseline.





Evaluation Metrics

The models were evaluated using:

Accuracy
Precision
Recall
Specificity
F1-Score
ROC-AUC
PR-AUC

Because this is a medical classification problem, malignant-case errors were also analyzed separately.




Cross-Validation

Stratified 5-Fold Cross-Validation was used on the training set.




The test set remained untouched during:

Cross-validation
Hyperparameter tuning
Model selection





This was done to reduce the risk of data leakage and obtain a more reliable estimate of generalization performance.






Hyperparameter Tuning

Grid Search with Stratified 5-Fold Cross-Validation was used.




Logistic Regression

Best C:

C = 0.1


Random Forest

Best parameters:

n_estimators = 100
max_depth = None
min_samples_split = 2
min_samples_leaf = 1



SVM

Best parameters:

C = 10
gamma = 0.01
kernel = RBF



XGBoost

Best parameters:

learning_rate = 0.05
max_depth = 3
n_estimators = 200
subsample = 1.0



Final Test Results

The final models were evaluated on the untouched test set.

Model	Test Accuracy	Precision	Recall	Malignant Sensitivity	Specificity	F1	ROC-AUC	PR-AUC

Logistic Regression	97.37%	97.26%	98.61%	95.24%	95.24%	97.93%	99.57%	99.74%

Random Forest	95.61%	95.89%	97.22%	92.86%	92.86%	96.55%	99.37%	99.62%

SVM	98.25%	98.61%	98.61%	97.62%	97.62%	98.61%	99.77%	99.86%

XGBoost	94.74%	94.59%	97.22%	90.48%	90.48%	95.89%	99.27%	99.58%

Note: Recall and Precision above use the dataset's class 1 (Benign) as the positive class. Malignant Sensitivity treats class 0 (Malignant) as the target class.


Malignant Error Analysis

There were 42 malignant cases in the test set.

Model	Malignant Correctly Detected	Malignant Missed
Logistic Regression	40	2
Random Forest	39	3
SVM	41	1
XGBoost	38	4

A malignant miss refers to an actual malignant sample being predicted as benign.

Results and Figures

Generated results are stored in:

results/tables/

and:

results/figures/



The project includes:

Model comparison table
Confusion matrices
ROC curve comparison
Precision-Recall curve comparison
Reproducibility

The experiments use fixed random seeds where applicable.

Example:

random_state=42

The final test set is kept separate from cross-validation and hyperparameter tuning.




How to Run
1. Clone the repository
git clone <YOUR_GITHUB_REPOSITORY_URL>
2. Create the virtual environment
uv venv --python 3.14 .venv
3. Activate the environment

Windows:

.venv\Scripts\activate
4. Install dependencies
uv pip install ipykernel pandas numpy matplotlib seaborn scikit-learn xgboost
5. Open the project in VS Code

Open the project folder and select the .venv Python kernel.

6. Run notebooks in order
01_eda.ipynb
      ↓
02_preprocessing.ipynb
      ↓
03_model_training.ipynb



Future Work

This tabular ML pipeline is intended as a foundation for a future multimodal healthcare AI research project.

Potential future directions include:

Clinical metadata-based prediction
Dermoscopic image classification
Image + clinical metadata multimodal fusion
Explainable AI using SHAP and Grad-CAM
Skin-tone subgroup evaluation
Fairness analysis
External validation
More diverse medical datasets




License

This project is intended for educational and research purposes.
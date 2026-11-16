# ============================================================
# MERCHANT CHURN PREDICTION
# Improved Machine Learning Workflow
# ============================================================

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

warnings.filterwarnings("ignore")


# ============================================================
# 1. PROJECT PATHS
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

DATA_PATH = os.path.join(
    PROJECT_DIR,
    "data",
    "processed",
    "Churn Model.csv"
)

RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
IMAGES_DIR = os.path.join(PROJECT_DIR, "images")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)


# ============================================================
# 2. CONFIGURATION
# ============================================================

TARGET = "churn_flag"
ID_COLUMN = "merchant_id"

TEST_SIZE = 0.30
RANDOM_STATE = 42

N_ESTIMATORS = 400

# These columns may directly reveal the churn label,
# depending on how churn_flag was generated.
#
# Excluding them is a conservative safeguard.
# It does NOT guarantee that all target leakage is removed.

LEAKAGE_PRONE_COLUMNS = [
    "days_since_last_transaction",
    "volume_last_30d",
    "txns_last_30d",
    "txns_prev_30d",
    "volume_prev_30d",
    "volume_change_pct_30d"
]

# Other columns that should not be predictive features.
EXCLUDED_COLUMNS = [
    ID_COLUMN,
    TARGET
]


# ============================================================
# 3. LOAD DATASET
# ============================================================

print("\n" + "=" * 65)
print("MERCHANT CHURN PREDICTION")
print("=" * 65)

print("\nLoading merchant churn dataset...")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

print(f"Dataset loaded: {DATA_PATH}")
print(f"Original dataset shape: {df.shape}")


# ============================================================
# 4. DATA VALIDATION
# ============================================================

if TARGET not in df.columns:
    raise ValueError(
        f"Target column '{TARGET}' is missing."
    )

if df[TARGET].isna().any():
    raise ValueError(
        "Target column contains missing values. "
        "Resolve them before training."
    )

if df[TARGET].nunique() != 2:
    raise ValueError(
        "This script requires a binary churn target."
    )

# Ensure target contains 0 and 1.
target_values = set(df[TARGET].unique())

if target_values != {0, 1}:
    raise ValueError(
        f"Expected target values 0 and 1, found: {target_values}"
    )

df[TARGET] = df[TARGET].astype(int)

# Remove duplicate merchant records.
if ID_COLUMN in df.columns:

    duplicate_count = df[ID_COLUMN].duplicated().sum()

    if duplicate_count > 0:

        print(
            f"\nDuplicate merchant records found: "
            f"{duplicate_count}"
        )

        df = df.drop_duplicates(
            subset=[ID_COLUMN],
            keep="first"
        )

        print("Duplicate merchant records removed.")


print(f"\nValidated dataset shape: {df.shape}")


# ============================================================
# 5. TARGET DISTRIBUTION
# ============================================================

print("\n" + "=" * 65)
print("CHURN TARGET DISTRIBUTION")
print("=" * 65)

target_counts = df[TARGET].value_counts().sort_index()

target_percentages = (
    df[TARGET]
    .value_counts(normalize=True)
    .sort_index() * 100
)

print("\nTarget counts:")
print(target_counts)

print("\nTarget percentages:")
print(target_percentages.round(2))

print(
    f"\nTotal merchants: {len(df):,}"
)

print(
    f"Churned merchants: "
    f"{(df[TARGET] == 1).sum():,}"
)

print(
    f"Non-churned merchants: "
    f"{(df[TARGET] == 0).sum():,}"
)


# ============================================================
# 6. FEATURE SELECTION
# ============================================================

print("\n" + "=" * 65)
print("FEATURE PREPARATION")
print("=" * 65)

# Exclude identifiers and target.
excluded = set(EXCLUDED_COLUMNS)

# Exclude potential target leakage features.
excluded.update(LEAKAGE_PRONE_COLUMNS)

available_excluded = [
    col for col in df.columns
    if col in excluded
]

X = df.drop(
    columns=available_excluded,
    errors="ignore"
)

y = df[TARGET].copy()

print("\nExcluded columns:")

for col in available_excluded:
    print(f"- {col}")

if X.empty:
    raise ValueError(
        "No predictive features remain after exclusions."
    )


# Remove columns that contain no useful information.
constant_columns = [
    col for col in X.columns
    if X[col].nunique(dropna=False) <= 1
]

if constant_columns:

    print("\nRemoving constant columns:")

    for col in constant_columns:
        print(f"- {col}")

    X = X.drop(columns=constant_columns)


# Drop columns that are entirely missing.
empty_columns = [
    col for col in X.columns
    if X[col].isna().all()
]

if empty_columns:

    print("\nRemoving completely empty columns:")

    for col in empty_columns:
        print(f"- {col}")

    X = X.drop(columns=empty_columns)


# Identify numeric and categorical columns.
numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

# Convert unsupported datetime columns to strings.
datetime_features = X.select_dtypes(
    include=["datetime", "datetimetz"]
).columns.tolist()

for col in datetime_features:

    X[col] = X[col].astype(str)

    categorical_features.append(col)


print("\nFeature preparation results:")

print(f"Numeric features: {len(numeric_features)}")
print(f"Categorical features: {len(categorical_features)}")

print(f"Total predictive features: {X.shape[1]}")

print("\nSelected features:")

for col in X.columns:
    print(f"- {col}")


if X.shape[1] == 0:
    raise ValueError(
        "No usable features available for model training."
    )


# ============================================================
# 7. TRAIN-TEST SPLIT
# ============================================================

print("\n" + "=" * 65)
print("TRAIN-TEST SPLIT")
print("=" * 65)

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=TEST_SIZE,

    random_state=RANDOM_STATE,

    stratify=y
)

print(f"\nTraining records: {len(X_train)}")
print(f"Testing records: {len(X_test)}")

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())


# ============================================================
# 8. PREPROCESSING PIPELINE
# ============================================================

numeric_transformer = Pipeline(

    steps=[

        (
            "imputer",
            SimpleImputer(strategy="median")
        ),

        (
            "scaler",
            StandardScaler()
        )

    ]

)


categorical_transformer = Pipeline(

    steps=[

        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),

        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )

    ]

)


transformers = []

if numeric_features:

    transformers.append(

        (
            "num",
            numeric_transformer,
            numeric_features
        )

    )


if categorical_features:

    transformers.append(

        (
            "cat",
            categorical_transformer,
            categorical_features
        )

    )


preprocessor = ColumnTransformer(

    transformers=transformers,

    remainder="drop"

)


# ============================================================
# 9. RANDOM FOREST MODEL
# ============================================================

print("\n" + "=" * 65)
print("TRAINING RANDOM FOREST MODEL")
print("=" * 65)

random_forest = RandomForestClassifier(

    n_estimators=N_ESTIMATORS,

    max_depth=8,

    min_samples_split=10,

    min_samples_leaf=5,

    max_features="sqrt",

    class_weight="balanced",

    random_state=RANDOM_STATE,

    n_jobs=-1

)


model = Pipeline(

    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "classifier",
            random_forest
        )

    ]

)


print("\nTraining model...")

model.fit(
    X_train,
    y_train
)

print("Model training completed successfully.")


# ============================================================
# 10. TEST SET PREDICTIONS
# ============================================================

print("\nGenerating test predictions...")

y_pred = model.predict(X_test)

y_probability = model.predict_proba(X_test)[:, 1]


# ============================================================
# 11. MODEL EVALUATION
# ============================================================

print("\n" + "=" * 65)
print("MODEL EVALUATION RESULTS")
print("=" * 65)

accuracy = accuracy_score(
    y_test,
    y_pred
)

balanced_accuracy = balanced_accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)

pr_auc = average_precision_score(
    y_test,
    y_probability
)


# Majority-class baseline.
majority_class = y_train.mode()[0]

baseline_predictions = np.full(
    len(y_test),
    majority_class
)

baseline_accuracy = accuracy_score(
    y_test,
    baseline_predictions
)

baseline_balanced_accuracy = balanced_accuracy_score(
    y_test,
    baseline_predictions
)


print(f"\nAccuracy: {accuracy:.4f}")

print(
    f"Balanced Accuracy: "
    f"{balanced_accuracy:.4f}"
)

print(f"Precision: {precision:.4f}")

print(f"Recall: {recall:.4f}")

print(f"F1 Score: {f1:.4f}")

print(f"ROC-AUC Score: {roc_auc:.4f}")

print(f"PR-AUC Score: {pr_auc:.4f}")


print("\nMAJORITY CLASS BASELINE")

print(
    f"Baseline Accuracy: "
    f"{baseline_accuracy:.4f}"
)

print(
    f"Baseline Balanced Accuracy: "
    f"{baseline_balanced_accuracy:.4f}"
)


# ============================================================
# 12. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 65)
print("CLASSIFICATION REPORT")
print("=" * 65)

report = classification_report(

    y_test,

    y_pred,

    labels=[0, 1],

    target_names=[
        "Not Churned",
        "Churned"
    ],

    zero_division=0
)

print(report)


# ============================================================
# 13. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(

    y_test,

    y_pred,

    labels=[0, 1]

)

print("\nCONFUSION MATRIX")
print(cm)


tn, fp, fn, tp = cm.ravel()

print("\nCONFUSION MATRIX INTERPRETATION")

print(f"True Negatives: {tn}")

print(f"False Positives: {fp}")

print(f"False Negatives: {fn}")

print(f"True Positives: {tp}")


# ============================================================
# 14. STRATIFIED CROSS-VALIDATION
# ============================================================

print("\n" + "=" * 65)
print("STRATIFIED CROSS-VALIDATION")
print("=" * 65)

minority_count = y.value_counts().min()

n_splits = min(5, int(minority_count))

if n_splits < 2:

    print(
        "Not enough minority-class records "
        "for cross-validation."
    )

else:

    cv = StratifiedKFold(

        n_splits=n_splits,

        shuffle=True,

        random_state=RANDOM_STATE

    )

    scoring = {

        "accuracy": "accuracy",

        "balanced_accuracy": "balanced_accuracy",

        "precision": "precision",

        "recall": "recall",

        "f1": "f1",

        "roc_auc": "roc_auc",

        "average_precision": "average_precision"

    }

    cv_results = cross_validate(

        model,

        X,

        y,

        cv=cv,

        scoring=scoring,

        n_jobs=-1,

        error_score="raise"

    )

    print(
        f"\nNumber of folds: {n_splits}"
    )

    for metric in scoring:

        values = cv_results[
            f"test_{metric}"
        ]

        print(

            f"{metric}: "
            f"{values.mean():.4f} "
            f"+/- {values.std():.4f}"

        )


# ============================================================
# 15. FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 65)
print("FEATURE IMPORTANCE")
print("=" * 65)

fitted_preprocessor = model.named_steps[
    "preprocessor"
]

fitted_classifier = model.named_steps[
    "classifier"
]

feature_names = fitted_preprocessor.get_feature_names_out()

importance_values = fitted_classifier.feature_importances_

feature_importance = pd.DataFrame({

    "feature": feature_names,

    "importance": importance_values

})

feature_importance = feature_importance.sort_values(

    by="importance",

    ascending=False

).reset_index(drop=True)


print("\nTOP 10 IMPORTANT FEATURES")

print(
    feature_importance.head(10).to_string(
        index=False
    )
)


feature_importance.to_csv(

    os.path.join(
        RESULTS_DIR,
        "churn_feature_importance.csv"
    ),

    index=False

)


# ============================================================
# 16. CONFUSION MATRIX VISUALIZATION
# ============================================================

plt.figure(figsize=(7, 6))

display = ConfusionMatrixDisplay(

    confusion_matrix=cm,

    display_labels=[
        "Not Churned",
        "Churned"
    ]

)

display.plot(
    cmap="Blues",
    values_format="d"
)

plt.title(
    "Merchant Churn Prediction - Confusion Matrix"
)

plt.tight_layout()

plt.savefig(

    os.path.join(
        IMAGES_DIR,
        "confusion_matrix_churn.png"
    ),

    dpi=300,

    bbox_inches="tight"

)

plt.close()


# ============================================================
# 17. FEATURE IMPORTANCE VISUALIZATION
# ============================================================

top_features = feature_importance.head(10).copy()

plt.figure(figsize=(10, 7))

plt.barh(

    top_features["feature"],

    top_features["importance"]

)

plt.gca().invert_yaxis()

plt.xlabel("Feature Importance")

plt.ylabel("Features")

plt.title(
    "Top 10 Merchant Churn Prediction Features"
)

plt.tight_layout()

plt.savefig(

    os.path.join(
        IMAGES_DIR,
        "churn_feature_importance.png"
    ),

    dpi=300,

    bbox_inches="tight"

)

plt.close()


# ============================================================
# 18. SAVE TEST SET PREDICTIONS
# ============================================================

test_predictions = pd.DataFrame({

    "actual_churn_flag": y_test.values,

    "predicted_churn_flag": y_pred,

    "churn_probability": y_probability

})

test_predictions.insert(

    0,

    "merchant_id",

    df.loc[
        X_test.index,
        ID_COLUMN
    ].values if ID_COLUMN in df.columns else X_test.index

)

test_predictions["prediction_correct"] = (

    test_predictions["actual_churn_flag"]

    ==

    test_predictions["predicted_churn_flag"]

)


test_predictions.to_csv(

    os.path.join(
        RESULTS_DIR,
        "churn_test_predictions.csv"
    ),

    index=False

)


# ============================================================
# 19. TRAIN FINAL MODEL ON ALL RECORDS
# ============================================================

print("\n" + "=" * 65)
print("TRAINING FINAL MODEL")
print("=" * 65)

final_model = model

final_model.fit(
    X,
    y
)

print(
    "Final model trained on the complete dataset."
)


# ============================================================
# 20. GENERATE RETROSPECTIVE MERCHANT RISK SCORES
# ============================================================

print("\nGenerating merchant churn risk scores...")

all_probabilities = final_model.predict_proba(X)[:, 1]

all_predictions = final_model.predict(X)


merchant_predictions = pd.DataFrame({

    "merchant_id": (
        df[ID_COLUMN].values
        if ID_COLUMN in df.columns
        else df.index
    ),

    "actual_churn_flag": y.values,

    "churn_probability": all_probabilities,

    "predicted_churn_flag": all_predictions

})


# Project-defined risk categories.
# These thresholds are not calibrated business probabilities.

merchant_predictions["Churn_Risk_Level"] = np.select(

    [

        merchant_predictions["churn_probability"] >= 0.66,

        merchant_predictions["churn_probability"] >= 0.33

    ],

    [

        "High Risk",

        "Medium Risk"

    ],

    default="Low Risk"

)


# These are retrospective scores because the final model
# was trained using all available records, including these rows.

merchant_predictions["score_type"] = (
    "Retrospective_in_sample"
)


# ============================================================
# 21. RISK DISTRIBUTION
# ============================================================

print("\n" + "=" * 65)
print("MERCHANT CHURN RISK DISTRIBUTION")
print("=" * 65)

risk_distribution = (

    merchant_predictions["Churn_Risk_Level"]

    .value_counts()

)

for risk_level in [

    "High Risk",

    "Medium Risk",

    "Low Risk"

]:

    count = risk_distribution.get(
        risk_level,
        0
    )

    percentage = (
        count / len(merchant_predictions)
    ) * 100

    print(

        f"{risk_level}: "
        f"{count} "
        f"({percentage:.2f}%)"

    )


# ============================================================
# 22. TOP 20 HIGH-RISK MERCHANTS
# ============================================================

print("\nTOP 20 HIGH-RISK MERCHANTS")

high_risk_merchants = (

    merchant_predictions[

        merchant_predictions["Churn_Risk_Level"]

        ==

        "High Risk"

    ]

    .sort_values(

        by="churn_probability",

        ascending=False

    )

    .head(20)

)


if high_risk_merchants.empty:

    print("No high-risk merchants identified.")

else:

    print(

        high_risk_merchants.to_string(

            index=False

        )

    )


# ============================================================
# 23. SAVE ALL MERCHANT PREDICTIONS
# ============================================================

merchant_predictions.to_csv(

    os.path.join(
        RESULTS_DIR,
        "churn_predictions_with_risk.csv"
    ),

    index=False

)


# ============================================================
# 24. SAVE FINAL MODEL
# ============================================================

MODEL_PATH = os.path.join(

    MODELS_DIR,

    "random_forest_churn_pipeline.pkl"

)

joblib.dump(

    final_model,

    MODEL_PATH

)

print("\nFinal model saved:")
print(MODEL_PATH)


# ============================================================
# 25. SAVE EVALUATION METRICS
# ============================================================

metrics = pd.DataFrame({

    "metric": [

        "Accuracy",

        "Balanced Accuracy",

        "Precision",

        "Recall",

        "F1 Score",

        "ROC-AUC",

        "PR-AUC",

        "Baseline Accuracy",

        "Baseline Balanced Accuracy"

    ],

    "value": [

        accuracy,

        balanced_accuracy,

        precision,

        recall,

        f1,

        roc_auc,

        pr_auc,

        baseline_accuracy,

        baseline_balanced_accuracy

    ]

})


metrics.to_csv(

    os.path.join(
        RESULTS_DIR,
        "churn_model_metrics.csv"
    ),

    index=False

)


# ============================================================
# 26. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("CHURN PREDICTION WORKFLOW COMPLETED")
print("=" * 65)

print("\nGENERATED FILES:")

generated_files = [

    "results/churn_predictions_with_risk.csv",

    "results/churn_test_predictions.csv",

    "results/churn_feature_importance.csv",

    "results/churn_model_metrics.csv",

    "models/random_forest_churn_pipeline.pkl",

    "images/confusion_matrix_churn.png",

    "images/churn_feature_importance.png"

]

for file in generated_files:

    print(f"- {os.path.join(PROJECT_DIR, file)}")


print("\nIMPORTANT NOTES:")

print(
    "1. Test metrics represent performance on the held-out test set."
)

print(
    "2. All-merchant risk scores are retrospective in-sample estimates."
)

print(
    "3. Risk categories use project-defined thresholds."
)

print(
    "4. Model performance does not establish real-world fraud "
    "or churn prediction accuracy."
)

print(
    "5. Confirm the churn label-generation logic before "
    "claiming that target leakage has been eliminated."
)

print("\nAll churn prediction operations completed successfully.")
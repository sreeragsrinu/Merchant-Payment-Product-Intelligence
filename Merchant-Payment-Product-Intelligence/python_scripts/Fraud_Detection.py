# coding: utf-8

"""
MERCHANT & PAYMENT PRODUCT INTELLIGENCE
FRAUD DETECTION AND TRANSACTION RISK SCORING

Input:
    data/processed/Fraud Model.csv

Outputs:
    results/fraud_alert.csv
    results/high_risk_transactions.csv
    images/fraud_patterns.png
    images/fraud_hours.png

Note:
    Isolation Forest identifies anomalous transactions.
    It does not confirm that a transaction is fraudulent.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "Fraud Model.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
IMAGES_DIR = PROJECT_ROOT / "images"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
CONTAMINATION_RATE = 0.02


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\nLoading fraud detection dataset...")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

required_columns = [
    "transaction_id",
    "merchant_id",
    "transaction_timestamp",
    "transaction_amount",
    "merchant_std_amount",
    "time_since_last_tx_sec",
    "merchant_age_days",
    "is_early_lifecycle_tx",
    "merchant_failure_rate_7d",
    "payment_method"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

initial_rows = len(df)

print(f"Rows loaded: {initial_rows:,}")


# ============================================================
# 3. DATA CLEANING
# ============================================================

df["transaction_timestamp"] = pd.to_datetime(
    df["transaction_timestamp"],
    errors="coerce"
)

df["transaction_amount"] = pd.to_numeric(
    df["transaction_amount"],
    errors="coerce"
)

df["merchant_id"] = (
    df["merchant_id"]
    .astype("string")
    .str.strip()
)

# Preserve original missing time-gap information.
first_purchase_mask = df["time_since_last_tx_sec"].isna()

df = df.dropna(
    subset=[
        "transaction_id",
        "merchant_id",
        "transaction_timestamp",
        "transaction_amount"
    ]
).copy()

if df.empty:
    raise ValueError(
        "No valid transaction records remain after cleaning."
    )

# Numeric conversion.
numeric_columns = [
    "merchant_std_amount",
    "time_since_last_tx_sec",
    "merchant_age_days",
    "is_early_lifecycle_tx",
    "merchant_failure_rate_7d"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

# Remove invalid infinite values.
df[numeric_columns] = df[numeric_columns].replace(
    [np.inf, -np.inf],
    np.nan
)

# First purchase indicator.
df["is_first_purchase"] = (
    first_purchase_mask
    .reindex(df.index)
    .fillna(False)
    .astype(int)
)

# Fill missing numeric values safely.
for col in numeric_columns:

    median_value = df[col].median()

    if pd.isna(median_value):
        median_value = 0

    df[col] = df[col].fillna(median_value)

# Prevent negative standard deviations and time gaps.
df["merchant_std_amount"] = (
    df["merchant_std_amount"].clip(lower=0)
)

df["time_since_last_tx_sec"] = (
    df["time_since_last_tx_sec"].clip(lower=0)
)

df["merchant_age_days"] = (
    df["merchant_age_days"].clip(lower=0)
)

df["is_early_lifecycle_tx"] = (
    df["is_early_lifecycle_tx"].clip(0, 1)
)

df["merchant_failure_rate_7d"] = (
    df["merchant_failure_rate_7d"].clip(lower=0)
)

print(f"Rows retained: {len(df):,}")
print(
    f"Rows removed: {initial_rows - len(df):,}"
)


# ============================================================
# 4. TRANSACTION TIME FEATURES
# ============================================================

df["hour_of_day"] = (
    df["transaction_timestamp"].dt.hour
)

df["day_of_week"] = (
    df["transaction_timestamp"].dt.dayofweek
)

df["is_weekend"] = (
    df["day_of_week"].isin([5, 6]).astype(int)
)

df["is_night"] = (
    df["hour_of_day"].between(0, 4).astype(int)
)


# ============================================================
# 5. MERCHANT BEHAVIOURAL FEATURES
# ============================================================

# Calculate merchant average transaction amount.
merchant_average = (
    df.groupby("merchant_id")["transaction_amount"]
    .mean()
)

# Mapping avoids the previous overlapping-column join error.
df["merchant_avg_amount"] = (
    df["merchant_id"].map(merchant_average)
)

overall_median = df["transaction_amount"].median()

if pd.isna(overall_median):
    overall_median = 0

df["merchant_avg_amount"] = (
    df["merchant_avg_amount"].fillna(overall_median)
)

# Avoid division by zero.
safe_average = (
    df["merchant_avg_amount"].replace(0, np.nan)
)

df["amount_to_avg_ratio"] = (
    df["transaction_amount"] / safe_average
).replace(
    [np.inf, -np.inf],
    np.nan
).fillna(0)

# Merchant-specific amount deviation.
safe_std = (
    df["merchant_std_amount"].replace(0, np.nan)
)

df["z_score_amount"] = (
    (
        df["transaction_amount"]
        - df["merchant_avg_amount"]
    ) / safe_std
).replace(
    [np.inf, -np.inf],
    np.nan
).fillna(0)

# Transaction frequency indicators.
df["rapid_repeat_tx"] = (
    df["time_since_last_tx_sec"] < 60
).astype(int)

df["burst_tx"] = (
    df["time_since_last_tx_sec"] < 300
).astype(int)

df["rapid_succession"] = df["burst_tx"]

# Compatibility feature.
df["amount_vs_avg"] = df["amount_to_avg_ratio"]


# ============================================================
# 6. PAYMENT METHOD ENCODING
# ============================================================

df["payment_method"] = (
    df["payment_method"]
    .astype("string")
    .fillna("unknown")
    .str.strip()
    .str.lower()
)

df = pd.get_dummies(
    df,
    columns=["payment_method"],
    prefix="method",
    dtype=int
)

expected_methods = [
    "method_card",
    "method_ussd",
    "method_bank"
]

for col in expected_methods:

    if col not in df.columns:
        df[col] = 0


# ============================================================
# 7. MODEL FEATURES
# ============================================================

fraud_features = [
    "transaction_amount",
    "merchant_avg_amount",
    "is_weekend",
    "merchant_std_amount",
    "time_since_last_tx_sec",
    "hour_of_day",
    "merchant_age_days",
    "is_early_lifecycle_tx",
    "merchant_failure_rate_7d",
    "amount_to_avg_ratio",
    "z_score_amount",
    "rapid_repeat_tx",
    "burst_tx",
    "is_first_purchase",
    "method_card",
    "method_ussd",
    "method_bank"
]

X = df[fraud_features].copy()

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

for col in X.columns:

    X[col] = pd.to_numeric(
        X[col],
        errors="coerce"
    )

    median_value = X[col].median()

    if pd.isna(median_value):
        median_value = 0

    X[col] = X[col].fillna(median_value)

X = X.fillna(0)

print("\nFraud detection dataset:", X.shape)


# ============================================================
# 8. FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# ============================================================
# 9. ISOLATION FOREST
# ============================================================

print("\nTraining Isolation Forest...")

model = IsolationForest(
    n_estimators=200,
    contamination=CONTAMINATION_RATE,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

model.fit(X_scaled)

# Isolation Forest:
# 1  = Normal
# -1 = Anomaly

predictions = model.predict(X_scaled)

df["anomaly_score"] = (
    model.decision_function(X_scaled)
)

df["is_fraud"] = (
    predictions == -1
).astype(int)

df["anomaly_label"] = np.where(
    df["is_fraud"] == 1,
    "Flagged Anomaly",
    "Normal"
)


# ============================================================
# 10. FRAUD DETECTION SUMMARY
# ============================================================

fraud_tx = df[
    df["is_fraud"] == 1
].copy()

normal_tx = df[
    df["is_fraud"] == 0
].copy()

total_transactions = len(df)
flagged_transactions = len(fraud_tx)

anomaly_rate = (
    flagged_transactions / total_transactions
    if total_transactions > 0
    else 0
)

print("\nFRAUD DETECTION SUMMARY")
print("-" * 45)

print(
    f"Total transactions: {total_transactions:,}"
)

print(
    f"Flagged as anomalies: {flagged_transactions:,} "
    f"({anomaly_rate:.2%})"
)

if not fraud_tx.empty:

    print(
        "Average flagged amount: "
        f"{fraud_tx['transaction_amount'].mean():,.2f}"
    )

else:

    print("Average flagged amount: N/A")

if not normal_tx.empty:

    print(
        "Average normal amount: "
        f"{normal_tx['transaction_amount'].mean():,.2f}"
    )

else:

    print("Average normal amount: N/A")

if not fraud_tx.empty:

    print(
        "Weekend share of flagged transactions: "
        f"{fraud_tx['is_weekend'].mean():.1%}"
    )

else:

    print("Weekend share of flagged transactions: N/A")

print(
    "Weekend share of all transactions: "
    f"{df['is_weekend'].mean():.1%}"
)


# ============================================================
# 11. FRAUD PATTERN VISUALIZATION
# ============================================================

plt.figure(figsize=(10, 6))

if not normal_tx.empty:

    plt.hist(
        normal_tx["transaction_amount"],
        bins=50,
        alpha=0.7,
        label="Normal"
    )

if not fraud_tx.empty:

    plt.hist(
        fraud_tx["transaction_amount"],
        bins=50,
        alpha=0.7,
        label="Flagged Anomaly"
    )

plt.xlabel("Transaction Amount")
plt.ylabel("Frequency")

plt.title(
    "Transaction Amount Distribution: "
    "Normal vs Flagged"
)

plt.legend()
plt.tight_layout()

plt.savefig(
    IMAGES_DIR / "fraud_patterns.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 12. TRANSACTION HOURLY ANALYSIS
# ============================================================

hours = range(24)

hour_normal = (
    normal_tx.groupby("hour_of_day")
    .size()
    .reindex(hours, fill_value=0)
)

hour_fraud = (
    fraud_tx.groupby("hour_of_day")
    .size()
    .reindex(hours, fill_value=0)
)

plt.figure(figsize=(10, 6))

plt.plot(
    hours,
    hour_normal.values,
    marker="o",
    label="Normal"
)

plt.plot(
    hours,
    hour_fraud.values,
    marker="o",
    label="Flagged Anomaly"
)

plt.xlabel("Hour of Day")
plt.ylabel("Number of Transactions")

plt.title(
    "Transactions by Hour: Normal vs Flagged"
)

plt.xticks(list(hours))
plt.legend()
plt.tight_layout()

plt.savefig(
    IMAGES_DIR / "fraud_hours.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 13. FRAUD RISK SCORING
# ============================================================

def calculate_risk_score(data):

    """
    Rule-based transaction risk score.

    Maximum score: 100

    This is a prioritisation score, not a
    statistically calibrated fraud probability.
    """

    score = np.zeros(len(data), dtype=float)

    # Rule 1: Unusually high amount.
    high_amount = (
        data["transaction_amount"]
        >
        (
            data["merchant_avg_amount"]
            + 3 * data["merchant_std_amount"]
        )
    )

    score += np.where(
        high_amount,
        40,
        0
    )

    # Rule 2: Transaction in configured near-limit range.
    near_limit = (
        data["transaction_amount"].between(
            75000,
            99000
        )
    )

    score += np.where(
        near_limit,
        25,
        0
    )

    # Rule 3: New merchant with high-value transaction.
    new_merchant = (
        (data["merchant_age_days"] < 30)
        &
        (data["transaction_amount"] > 50000)
    )

    score += np.where(
        new_merchant,
        15,
        0
    )

    # Rule 4: Rapid transaction succession.
    rapid_transaction = (
        data["time_since_last_tx_sec"] < 300
    )

    score += np.where(
        rapid_transaction,
        10,
        0
    )

    # Rule 5: Elevated merchant failure rate.
    high_failure_rate = (
        data["merchant_failure_rate_7d"] > 0.05
    )

    score += np.where(
        high_failure_rate,
        10,
        0
    )

    return np.clip(
        score,
        0,
        100
    )


df["fraud_risk_score"] = 0.0

flagged_mask = (
    df["is_fraud"] == 1
)

if flagged_mask.any():

    df.loc[
        flagged_mask,
        "fraud_risk_score"
    ] = calculate_risk_score(
        df.loc[flagged_mask]
    )


# ============================================================
# 14. FRAUD ACTION CLASSIFICATION
# ============================================================

def classify_risk(score):

    if score >= 60:
        return "BLOCK"

    elif score >= 40:
        return "REVIEW"

    else:
        return "APPROVE"


df["fraud_action"] = "APPROVE"

df.loc[
    flagged_mask,
    "fraud_action"
] = (
    df.loc[
        flagged_mask,
        "fraud_risk_score"
    ].apply(classify_risk)
)

df["fraud_flag"] = (
    df["is_fraud"].astype(int)
)

df["merchant_fraud_flag"] = (
    df.groupby("merchant_id")["is_fraud"]
    .transform("max")
    .astype(int)
)


# ============================================================
# 15. EXPORT FRAUD RESULTS
# ============================================================

fraud_output = (
    RESULTS_DIR / "fraud_alert.csv"
)

high_risk_output = (
    RESULTS_DIR / "high_risk_transactions.csv"
)

df.to_csv(
    fraud_output,
    index=False
)

high_risk_fraud = (
    df[df["is_fraud"] == 1]
    .sort_values(
        by=[
            "fraud_risk_score",
            "anomaly_score",
            "transaction_amount"
        ],
        ascending=[
            False,
            True,
            False
        ]
    )
)

high_risk_fraud.to_csv(
    high_risk_output,
    index=False
)


# ============================================================
# 16. TOP 10 FLAGGED TRANSACTIONS
# ============================================================

print("\nTOP 10 FLAGGED TRANSACTIONS")

display_columns = [
    "transaction_id",
    "merchant_id",
    "transaction_amount",
    "hour_of_day",
    "anomaly_score",
    "fraud_risk_score",
    "fraud_action"
]

if high_risk_fraud.empty:

    print("No transactions were flagged.")

else:

    print(
        high_risk_fraud[
            display_columns
        ].head(10).to_string(index=False)
    )


# ============================================================
# 17. RISK ACTION SUMMARY
# ============================================================

print("\nRISK ACTION SUMMARY")
print("-" * 45)

action_summary = (
    df.loc[flagged_mask, "fraud_action"]
    .value_counts()
    .reindex(
        ["BLOCK", "REVIEW", "APPROVE"],
        fill_value=0
    )
)

for action, count in action_summary.items():

    print(
        f"{action}: {count:,}"
    )


# ============================================================
# 18. FINAL OUTPUT
# ============================================================

print("\nFILES SAVED SUCCESSFULLY")

print(f"- {fraud_output}")
print(f"- {high_risk_output}")

print(
    f"- {IMAGES_DIR / 'fraud_patterns.png'}"
)

print(
    f"- {IMAGES_DIR / 'fraud_hours.png'}"
)

print("\nFraud detection workflow completed successfully.")

print(
    "\nIMPORTANT: Flagged anomalies are not confirmed "
    "fraudulent transactions. They require investigation "
    "or verification against labelled outcomes."
)
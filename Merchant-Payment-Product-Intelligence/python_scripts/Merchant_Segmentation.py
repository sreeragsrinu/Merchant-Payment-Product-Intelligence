"""
Merchant Segmentation and Churn Analysis
Project: Merchant & Payment Product Intelligence Platform

Expected project structure:
Merchant-Payment-Product-Intelligence/
├── data/
│   └── processed/
│       └── Merchant Segmentation.csv
├── results/
│   └── churn_predictions_with_risk.csv
├── images/
└── python_scripts/
    └── Merchant_Segmentation.py

Run from VS Code or terminal:
python python_scripts/Merchant_Segmentation.py
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------
# 1. PROJECT PATHS
# ---------------------------------------------------------

# This file is expected to be inside the "python_scripts" folder.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

SEGMENTATION_FILE = (
    PROJECT_ROOT / "data" / "processed" / "Merchant Segmentation.csv"
)
CHURN_FILE = PROJECT_ROOT / "results" / "churn_predictions_with_risk.csv"

RESULTS_DIR = PROJECT_ROOT / "results"
IMAGES_DIR = PROJECT_ROOT / "images"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# 2. LOAD AND VALIDATE DATA
# ---------------------------------------------------------

if not SEGMENTATION_FILE.exists():
    raise FileNotFoundError(
        f"Could not find merchant segmentation data:\n{SEGMENTATION_FILE}"
    )

df = pd.read_csv(SEGMENTATION_FILE)

required_columns = [
    "merchant_id",
    "total_transaction_volume",
    "avg_transaction_size",
    "transaction_frequency",
    "avg_monthly_transactions",
    "payment_method_diversity",
    "success_rate",
    "merchant_age_days",
]

missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

print("\n--- Dataset overview ---")
print(f"Number of merchant records: {len(df):,}")
print(f"Number of columns: {len(df.columns)}")
print(df.head())
print("\nDataset information:")
df.info()

# Check merchant ID quality.
if df["merchant_id"].isna().any():
    raise ValueError("merchant_id contains missing values.")

if df["merchant_id"].duplicated().any():
    raise ValueError("Duplicate merchant_id values found in segmentation data.")


# ---------------------------------------------------------
# 3. PREPARE FEATURES
# ---------------------------------------------------------

feature_columns = [
    "total_transaction_volume",
    "avg_transaction_size",
    "transaction_frequency",
    "avg_monthly_transactions",
    "payment_method_diversity",
    "success_rate",
    "merchant_age_days",
]

features = df[feature_columns].copy()

# Convert feature columns to numeric and replace missing values
# with each column's median.
for col in feature_columns:
    features[col] = pd.to_numeric(features[col], errors="coerce")
    features[col] = features[col].fillna(features[col].median())

if features.isna().any().any():
    raise ValueError("Missing values remain after median imputation.")

# Standardise features so variables with larger scales do not
# dominate the clustering model.
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)


# ---------------------------------------------------------
# 4. ELBOW METHOD
# ---------------------------------------------------------

inertia_values = []
k_values = range(2, min(10, len(df)))

for k in k_values:
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    model.fit(features_scaled)
    inertia_values.append(model.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(list(k_values), inertia_values, marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method – Merchant Segmentation")
plt.tight_layout()
plt.savefig(IMAGES_DIR / "merchant_elbow_method.png", dpi=300)
plt.close()


# ---------------------------------------------------------
# 5. K-MEANS CLUSTERING
# ---------------------------------------------------------

# Three clusters are used for the initial segmentation.
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(features_scaled)

# Give clusters interpretable labels based on their average
# transaction volume. This avoids assuming that KMeans cluster
# numbers always have the same meaning.
cluster_volume = df.groupby("Cluster")["total_transaction_volume"].mean()
ordered_clusters = cluster_volume.sort_values().index.tolist()

segment_names = {
    ordered_clusters[0]: "Low-Volume Merchants",
    ordered_clusters[1]: "Mid-Volume Merchants",
    ordered_clusters[2]: "High-Volume Merchants",
}

df["Segment_Name"] = df["Cluster"].map(segment_names)


# ---------------------------------------------------------
# 6. MERCHANT SEGMENT SUMMARY
# ---------------------------------------------------------

summary = (
    df.groupby(["Cluster", "Segment_Name"])
    .agg(
        merchants=("merchant_id", "nunique"),
        avg_transaction_volume=("total_transaction_volume", "mean"),
        avg_transaction_size=("avg_transaction_size", "mean"),
        avg_transaction_frequency=("transaction_frequency", "mean"),
        avg_monthly_transactions=("avg_monthly_transactions", "mean"),
        avg_payment_method_diversity=("payment_method_diversity", "mean"),
        avg_success_rate=("success_rate", "mean"),
        avg_merchant_age_days=("merchant_age_days", "mean"),
    )
    .round(2)
    .reset_index()
)

summary.to_csv(RESULTS_DIR / "merchant_segment_summary.csv", index=False)

print("\n--- Merchant segment summary ---")
print(summary.to_string(index=False))


# ---------------------------------------------------------
# 7. MERCHANT-LEVEL SEGMENTATION OUTPUT
# ---------------------------------------------------------

segmented_output = RESULTS_DIR / "merchants_with_segments.csv"
df.to_csv(segmented_output, index=False)

print(f"\nSaved merchant segmentation to: {segmented_output}")


# ---------------------------------------------------------
# 8. VISUALISE MERCHANT SEGMENTS
# ---------------------------------------------------------

plt.figure(figsize=(9, 6))
sns.scatterplot(
    data=df,
    x="merchant_age_days",
    y="total_transaction_volume",
    hue="Segment_Name",
    palette="viridis",
    alpha=0.75
)
plt.title("Merchant Segments: Transaction Volume vs Merchant Age")
plt.xlabel("Merchant Age (Days)")
plt.ylabel("Total Transaction Volume")
plt.tight_layout()
plt.savefig(IMAGES_DIR / "merchant_behavioral_segments.png", dpi=300)
plt.close()


# ---------------------------------------------------------
# 9. REVENUE / TRANSACTION VOLUME SHARE BY SEGMENT
# ---------------------------------------------------------

volume_by_segment = (
    df.groupby("Segment_Name")["total_transaction_volume"]
    .sum()
    .reset_index(name="total_transaction_volume")
)

total_volume = volume_by_segment["total_transaction_volume"].sum()

if total_volume > 0:
    volume_by_segment["volume_share_pct"] = (
        volume_by_segment["total_transaction_volume"] / total_volume * 100
    )
else:
    volume_by_segment["volume_share_pct"] = 0

volume_by_segment.to_csv(
    RESULTS_DIR / "transaction_volume_share_by_segment.csv",
    index=False
)

plt.figure(figsize=(8, 6))
plt.pie(
    volume_by_segment["total_transaction_volume"],
    labels=volume_by_segment["Segment_Name"],
    autopct="%1.1f%%",
    startangle=140
)
plt.title("Transaction Volume Share by Merchant Segment")
plt.tight_layout()
plt.savefig(IMAGES_DIR / "transaction_volume_share_by_segment.png", dpi=300)
plt.close()


# ---------------------------------------------------------
# 10. CHURN ANALYSIS BY MERCHANT SEGMENT
# ---------------------------------------------------------

if CHURN_FILE.exists():
    churn_df = pd.read_csv(CHURN_FILE)

    churn_required_columns = [
        "merchant_id",
        "churn_flag",
        "churn_probability",
    ]
    missing_churn_columns = [
        col for col in churn_required_columns if col not in churn_df.columns
    ]

    if missing_churn_columns:
        raise ValueError(
            f"Missing required columns in churn file: {missing_churn_columns}"
        )

    churn_df = churn_df[churn_required_columns].copy()

    if churn_df["merchant_id"].duplicated().any():
        raise ValueError("Duplicate merchant_id values found in churn data.")

    segmentation_with_churn = df.merge(
        churn_df,
        on="merchant_id",
        how="left",
        validate="one_to_one"
    )

    churn_summary = (
        segmentation_with_churn
        .groupby("Segment_Name")
        .agg(
            merchants=("merchant_id", "nunique"),
            merchants_with_churn_prediction=("churn_flag", "count"),
            predicted_churned_merchants=("churn_flag", "sum"),
            average_churn_probability=("churn_probability", "mean"),
        )
    )

    churn_summary["predicted_churn_rate_pct"] = (
        churn_summary["predicted_churned_merchants"]
        / churn_summary["merchants_with_churn_prediction"]
        * 100
    )

    churn_summary = churn_summary.round(2)
    churn_summary.to_csv(
        RESULTS_DIR / "churn_summary_by_merchant_segment.csv"
    )

    print("\n--- Churn summary by merchant segment ---")
    print(churn_summary)

    segmentation_with_churn.to_csv(
        RESULTS_DIR / "merchant_segments_with_churn.csv",
        index=False
    )

else:
    print(
        "\nChurn file not found. Skipping churn analysis. "
        f"Expected location: {CHURN_FILE}"
    )


# ---------------------------------------------------------
# 11. SEGMENT DISTRIBUTION
# ---------------------------------------------------------

segment_counts = (
    df["Segment_Name"]
    .value_counts()
    .rename_axis("Segment_Name")
    .reset_index(name="merchant_count")
)

segment_counts.to_csv(
    RESULTS_DIR / "merchant_segment_distribution.csv",
    index=False
)

print("\n--- Merchant count by segment ---")
print(segment_counts.to_string(index=False))

print("\nMerchant segmentation analysis completed successfully.")
print(f"Results folder: {RESULTS_DIR}")
print(f"Charts folder: {IMAGES_DIR}")

# coding: utf-8

import pandas as pd
import numpy as np
import uuid
import random
import argparse
from pathlib import Path

# ============================================================
# 1. CONFIGURATION
# ============================================================

RANDOM_SEED = 42

NUM_MERCHANTS = 800

P_API_KEY = 0.85
P_TEST_TXN = 0.75
P_LIVE_TXN = 0.55
P_REGULAR_USAGE = 0.35

START_DATE = pd.Timestamp("2024-01-01")
SIMULATION_DAYS = 240

FAILURE_RATE = 0.08
DAILY_CHURN_PROBABILITY = 0.002

SQL_DRIVER = "ODBC Driver 17 for SQL Server"
SQL_SERVER = r"THOTH\SQLEXPRESS"
SQL_DATABASE = "PaystackFintechDB"

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


# ============================================================
# 2. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


def generate_id(prefix, length=10):
    return f"{prefix}_{uuid.uuid4().hex[:length]}"


# ============================================================
# 3. MERCHANT GENERATION
# ============================================================

def generate_merchants():

    merchants = []

    for _ in range(NUM_MERCHANTS):

        signup_ts = START_DATE + pd.Timedelta(
            days=np.random.randint(0, 60)
        )

        merchants.append({

            "merchant_id": generate_id("m"),

            "signup_timestamp": signup_ts,

            "country": "NG",

            "industry": random.choice([
                "Ecommerce",
                "SaaS",
                "Education",
                "Logistics",
                "Fintech"
            ]),

            "business_type": random.choice([
                "SME",
                "Enterprise"
            ]),

            "signup_channel": random.choice([
                "Web",
                "Referral",
                "Sales"
            ])

        })

    return pd.DataFrame(merchants)


# ============================================================
# 4. API KEY GENERATION
# ============================================================

def generate_api_keys(merchants_df):

    api_keys = []

    for _, merchant in merchants_df.iterrows():

        if np.random.rand() < P_API_KEY:

            # TEST API KEY
            api_keys.append({

                "api_key_id": generate_id("k"),

                "merchant_id": merchant["merchant_id"],

                "created_timestamp":
                    merchant["signup_timestamp"]
                    + pd.Timedelta(days=1),

                "environment": "TEST",

                "key_type": "secret"

            })

            # LIVE API KEY
            if np.random.rand() < 0.65:

                api_keys.append({

                    "api_key_id": generate_id("k"),

                    "merchant_id": merchant["merchant_id"],

                    "created_timestamp":
                        merchant["signup_timestamp"]
                        + pd.Timedelta(days=5),

                    "environment": "LIVE",

                    "key_type": "secret"

                })

    return pd.DataFrame(api_keys, columns=[
        "api_key_id",
        "merchant_id",
        "created_timestamp",
        "environment",
        "key_type"
    ])


# ============================================================
# 5. LIVE TRANSACTION HELPER
# ============================================================

def create_live_transaction(merchant_id, transaction_day):

    # Apply failure probability to every LIVE transaction

    status = (
        "FAILED"
        if np.random.rand() < FAILURE_RATE
        else "SUCCESS"
    )

    failure_reason = (
        random.choice([
            "Insufficient Funds",
            "Bank Timeout",
            "Network Error",
            "Invalid Card"
        ])
        if status == "FAILED"
        else None
    )

    return {

        "transaction_id": generate_id("t", 12),

        "merchant_id": merchant_id,

        "transaction_timestamp": transaction_day,

        "environment": "LIVE",

        "status": status,

        "amount": np.random.randint(2000, 80000),

        "currency": "NGN",

        "payment_method": random.choice([
            "card",
            "bank",
            "ussd"
        ]),

        "failure_reason": failure_reason

    }


# ============================================================
# 6. TRANSACTION GENERATION
# ============================================================

def generate_transactions(merchants_df):

    transactions = []

    for _, merchant in merchants_df.iterrows():

        merchant_id = merchant["merchant_id"]

        signup_date = merchant["signup_timestamp"]

        # ----------------------------------------------------
        # TEST TRANSACTIONS
        # ----------------------------------------------------

        if np.random.rand() < P_TEST_TXN:

            num_test = np.random.randint(5, 15)

            for i in range(num_test):

                transactions.append({

                    "transaction_id": generate_id("t", 12),

                    "merchant_id": merchant_id,

                    "transaction_timestamp":
                        signup_date + pd.Timedelta(days=2 + i),

                    "environment": "TEST",

                    "status": "SUCCESS",

                    "amount": 0.00,

                    "currency": "NGN",

                    "payment_method": "card",

                    "failure_reason": None

                })

        # ----------------------------------------------------
        # LIVE ACTIVATION
        # ----------------------------------------------------

        if np.random.rand() >= P_LIVE_TXN:
            continue

        activation_day = signup_date + pd.Timedelta(
            days=np.random.randint(7, 20)
        )

        # Initial LIVE transaction
        activation_transaction = create_live_transaction(
            merchant_id,
            activation_day
        )

        transactions.append(activation_transaction)

        # Continue usage only after successful activation
        if activation_transaction["status"] == "FAILED":
            continue

        # ----------------------------------------------------
        # MERCHANT USAGE PATTERN
        # ----------------------------------------------------

        is_power_user = (
            np.random.rand() < P_REGULAR_USAGE
        )

        current_day = activation_day + pd.Timedelta(days=1)

        last_active_day = (
            activation_day
            + pd.Timedelta(days=SIMULATION_DAYS)
        )

        while current_day < last_active_day:

            # Simulated merchant activity stop
            if np.random.rand() < DAILY_CHURN_PROBABILITY:
                break

            if is_power_user:

                daily_txns = np.random.randint(1, 4)

            else:

                daily_txns = np.random.randint(0, 2)

            for _ in range(daily_txns):

                live_transaction = create_live_transaction(
                    merchant_id,
                    current_day
                )

                transactions.append(live_transaction)

            current_day += pd.Timedelta(days=1)

    return pd.DataFrame(transactions, columns=[
        "transaction_id",
        "merchant_id",
        "transaction_timestamp",
        "environment",
        "status",
        "amount",
        "currency",
        "payment_method",
        "failure_reason"
    ])


# ============================================================
# 7. DATA VALIDATION
# ============================================================

def validate_data(
    merchants_df,
    api_keys_df,
    transactions_df
):

    datasets = {

        "Merchants": (merchants_df, "merchant_id"),

        "API Keys": (api_keys_df, "api_key_id"),

        "Transactions": (transactions_df, "transaction_id")

    }

    for name, (df, id_column) in datasets.items():

        print(f"\n{name} dataset")

        print("Shape:", df.shape)

        print("\nMissing values:")

        print(df.isnull().sum())

        if df[id_column].duplicated().any():

            raise ValueError(
                f"Duplicate IDs found in {name}"
            )

    # Foreign key validation

    merchant_ids = set(merchants_df["merchant_id"])

    if not set(api_keys_df["merchant_id"]).issubset(merchant_ids):

        raise ValueError(
            "API keys contain unknown merchant IDs"
        )

    if not set(transactions_df["merchant_id"]).issubset(merchant_ids):

        raise ValueError(
            "Transactions contain unknown merchant IDs"
        )

    # Transaction status validation

    if not transactions_df["status"].isin(
        ["SUCCESS", "FAILED"]
    ).all():

        raise ValueError(
            "Invalid transaction status found"
        )

    # Failure reason validation

    failed = transactions_df["status"] == "FAILED"

    successful = transactions_df["status"] == "SUCCESS"

    if transactions_df.loc[
        failed, "failure_reason"
    ].isnull().any():

        raise ValueError(
            "Failed transactions must have failure reasons"
        )

    if transactions_df.loc[
        successful, "failure_reason"
    ].notnull().any():

        raise ValueError(
            "Successful transactions should not have failure reasons"
        )

    # TEST transaction validation

    test_transactions = transactions_df[
        transactions_df["environment"] == "TEST"
    ]

    if not (test_transactions["amount"] == 0).all():

        raise ValueError(
            "TEST transactions must have zero amount"
        )

    print("\nData validation completed successfully.")


# ============================================================
# 8. ACTIVATION ANALYSIS
# ============================================================

def calculate_activation_rate(
    merchants_df,
    transactions_df
):

    successful_live = transactions_df[
        (transactions_df["environment"] == "LIVE")
        & (transactions_df["status"] == "SUCCESS")
    ]

    activated = successful_live["merchant_id"].nunique()

    activation_rate = (
        activated / len(merchants_df)
        if len(merchants_df) > 0
        else 0
    )

    print("\nACTIVATION ANALYSIS")

    print("Total merchants:", len(merchants_df))

    print("Activated merchants:", activated)

    print(f"Activation rate: {activation_rate:.2%}")


# ============================================================
# 9. TRANSACTION PERFORMANCE ANALYSIS
# ============================================================

def transaction_analysis(transactions_df):

    live_transactions = transactions_df[
        transactions_df["environment"] == "LIVE"
    ]

    total_live = len(live_transactions)

    successful = (
        live_transactions["status"] == "SUCCESS"
    ).sum()

    failed = (
        live_transactions["status"] == "FAILED"
    ).sum()

    success_rate = (
        successful / total_live
        if total_live > 0
        else 0
    )

    failure_rate = (
        failed / total_live
        if total_live > 0
        else 0
    )

    print("\nLIVE TRANSACTION PERFORMANCE")

    print("Total LIVE transactions:", total_live)

    print("Successful transactions:", successful)

    print("Failed transactions:", failed)

    print(f"Success rate: {success_rate:.2%}")

    print(f"Failure rate: {failure_rate:.2%}")

    print("\nFailure reason breakdown:")

    print(
        live_transactions[
            live_transactions["status"] == "FAILED"
        ]["failure_reason"].value_counts()
    )


# ============================================================
# 10. SAVE DATASETS
# ============================================================

def save_datasets(
    merchants_df,
    api_keys_df,
    transactions_df
):

    merchants_df.to_csv(
        RAW_DATA_DIR / "merchants.csv",
        index=False
    )

    api_keys_df.to_csv(
        RAW_DATA_DIR / "api keys.csv",
        index=False
    )

    transactions_df.to_csv(
        RAW_DATA_DIR / "transactions.csv",
        index=False
    )

    print("\nCSV files saved successfully.")

    print("Location:", RAW_DATA_DIR)


# ============================================================
# 11. SQL SERVER INSERTION
# ============================================================

def insert_into_sql_server(
    merchants_df,
    api_keys_df,
    transactions_df
):

    import pyodbc

    connection_string = (

        f"DRIVER={{{SQL_DRIVER}}};"

        f"SERVER={SQL_SERVER};"

        f"DATABASE={SQL_DATABASE};"

        "Trusted_Connection=yes;"

    )

    conn = pyodbc.connect(connection_string)

    try:

        cursor = conn.cursor()

        cursor.fast_executemany = True

        # ----------------------------------------------------
        # MERCHANT INSERTION
        # ----------------------------------------------------

        insert_merchants = """

        INSERT INTO merchants (

            merchant_id,
            signup_timestamp,
            country,
            industry,
            business_type,
            signup_channel

        )

        VALUES (?, ?, ?, ?, ?, ?)

        """

        merchant_rows = list(
            merchants_df.itertuples(index=False, name=None)
        )

        if merchant_rows:

            cursor.executemany(
                insert_merchants,
                merchant_rows
            )

        print("Merchants inserted:", len(merchant_rows))

        # ----------------------------------------------------
        # API KEY INSERTION
        # ----------------------------------------------------

        insert_api_keys = """

        INSERT INTO api_keys (

            api_key_id,
            merchant_id,
            created_timestamp,
            environment,
            key_type

        )

        VALUES (?, ?, ?, ?, ?)

        """

        api_rows = list(
            api_keys_df.itertuples(index=False, name=None)
        )

        if api_rows:

            cursor.executemany(
                insert_api_keys,
                api_rows
            )

        print("API keys inserted:", len(api_rows))

        # ----------------------------------------------------
        # TRANSACTION INSERTION
        # ----------------------------------------------------

        insert_transactions = """

        INSERT INTO transactions (

            transaction_id,
            merchant_id,
            transaction_timestamp,
            environment,
            status,
            amount,
            currency,
            payment_method,
            failure_reason

        )

        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)

        """

        transaction_rows = list(
            transactions_df.itertuples(index=False, name=None)
        )

        batch_size = 1000

        for start in range(
            0,
            len(transaction_rows),
            batch_size
        ):

            batch = transaction_rows[
                start:start + batch_size
            ]

            cursor.executemany(
                insert_transactions,
                batch
            )

            print(
                f"Inserted {min(start + batch_size, len(transaction_rows)):,}"
                f" / {len(transaction_rows):,} transactions"
            )

        conn.commit()

        print("\nSQL Server insertion completed successfully.")

    except Exception:

        conn.rollback()

        raise

    finally:

        conn.close()


# ============================================================
# 12. MAIN EXECUTION
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--load-sql",
        action="store_true",
        help="Insert generated data into SQL Server"
    )

    args = parser.parse_args()

    print("\nGenerating merchant dataset...")

    merchants_df = generate_merchants()

    print("\nGenerating API keys...")

    api_keys_df = generate_api_keys(merchants_df)

    print("\nGenerating transactions...")

    transactions_df = generate_transactions(merchants_df)

    # Data validation

    validate_data(
        merchants_df,
        api_keys_df,
        transactions_df
    )

    # Activation analysis

    calculate_activation_rate(
        merchants_df,
        transactions_df
    )

    # Transaction performance

    transaction_analysis(transactions_df)

    # Dataset summary

    print("\nFINAL DATASET SUMMARY")

    print("Merchants:", len(merchants_df))

    print("API Keys:", len(api_keys_df))

    print("Transactions:", len(transactions_df))

    # Save CSV files

    save_datasets(
        merchants_df,
        api_keys_df,
        transactions_df
    )

    # Optional SQL insertion

    if args.load_sql:

        print("\nConnecting to SQL Server...")

        insert_into_sql_server(
            merchants_df,
            api_keys_df,
            transactions_df
        )


if __name__ == "__main__":

    main()
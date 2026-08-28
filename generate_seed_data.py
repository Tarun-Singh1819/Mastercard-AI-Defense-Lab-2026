import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import uuid
import random

fake = Faker('en_IN')

# Reproducibility seeds
np.random.seed(42)
random.seed(42)

# =====================================================================
# CONSTANTS
# =====================================================================
STRUCTURING_THRESHOLD = 10000  # INR threshold
STRUCTURING_LOW = STRUCTURING_THRESHOLD * 0.85  # ₹8,500
STRUCTURING_HIGH = STRUCTURING_THRESHOLD         # ₹10,000

# Rail-to-account-type mapping (ensures realistic rail usage)
RAIL_ACCOUNT_TYPES = {
    "BANK": ["SAVINGS", "CURRENT"],
    "UPI": ["SAVINGS", "CURRENT", "WALLET"],
    "WALLET": ["WALLET"],
    "CRYPTO": ["CRYPTO"]
}


# =====================================================================
# 1. NODE GENERATION (Stateful Accounts, Shared Devices, & Subnets)
# =====================================================================
def generate_account_nodes(n_normal=200, n_mules=50):
    """
    Generates two types of account nodes:
    
    Legitimate accounts:
    - 1:1 device mapping (each person has their own phone)
    - Old accounts (1-8 years)
    - Full KYC
    
    Mule accounts:
    - Clustered device/IP sharing (5 mules per cluster share 2 devices)
    - Young accounts (15-90 days)
    - Minimal/No KYC
    - Account types include CRYPTO for cross-rail exit
    """
    accounts = []

    # --- Legitimate Accounts ---
    for _ in range(n_normal):
        accounts.append({
            'account_id': str(uuid.uuid4())[:8],
            'account_type': random.choice(['SAVINGS', 'CURRENT', 'WALLET']),
            'account_age_days': random.randint(365, 3000),
            'kyc_status': 'FULL',
            'device_hash': f"DEV_LEGIT_{str(uuid.uuid4())[:8]}",
            'ip_subnet': f"192.168.{random.randint(1, 255)}.0/24",
            'is_mule': 0
        })

    # --- Mule Accounts (Clustered) ---
    n_clusters = max(1, n_mules // 5)
    shared_devices = [f"DEV_MULE_{str(uuid.uuid4())[:6]}" for _ in range(n_clusters * 2)]
    shared_ips = [f"10.0.{random.randint(1, 10)}.0/24" for _ in range(n_clusters)]

    # Distribute account types so each cluster has at least one CRYPTO account
    # for cross-rail exit capability
    mule_type_pool = ['SAVINGS', 'WALLET', 'CRYPTO', 'SAVINGS', 'WALLET']

    for i in range(n_mules):
        cluster_idx = i % n_clusters
        position_in_cluster = i // n_clusters

        # First mule in each cluster gets CRYPTO (guaranteed exit rail)
        if position_in_cluster == 0:
            acc_type = 'CRYPTO'
        else:
            acc_type = random.choice(mule_type_pool)

        accounts.append({
            'account_id': str(uuid.uuid4())[:8],
            'account_type': acc_type,
            'account_age_days': random.randint(15, 90),
            'kyc_status': random.choice(['MIN', 'NONE']),
            'device_hash': random.choice(shared_devices[cluster_idx * 2:(cluster_idx + 1) * 2]),
            'ip_subnet': shared_ips[cluster_idx],
            'is_mule': 1
        })

    return pd.DataFrame(accounts)


def get_account_lookup(accounts_df):
    """Fast O(1) metadata lookup table to enforce stateful signals."""
    return accounts_df.set_index('account_id').to_dict(orient='index')


def get_mules_by_type(accounts_df):
    """Returns a dict: account_type -> list of mule account_ids."""
    mule_df = accounts_df[accounts_df['is_mule'] == 1]
    result = {}
    for acc_type in mule_df['account_type'].unique():
        result[acc_type] = mule_df[mule_df['account_type'] == acc_type]['account_id'].tolist()
    return result


def is_structured(amount):
    """
    Structuring = amount intentionally kept in [threshold*0.85, threshold] range.
    
    Per solution_design.md (Section 4.2):
    is_structured = 1 if amount ∈ [₹8,500 , ₹10,000]
    
    NOT just 'amount >= 8000' — that would flag legitimate ₹12K transfers.
    """
    return 1 if STRUCTURING_LOW <= amount <= STRUCTURING_HIGH else 0


# =====================================================================
# 2. EDGE GENERATION (Stateful Transactions Engine)
# =====================================================================

def generate_legitimate_transactions(accounts_df, n=5000):
    """
    Generates normal transaction baseline.
    
    Key properties:
    - Amount follows log-normal distribution (₹100 - ₹50K, skewed left)
    - Rail weights: UPI 50%, BANK 30%, WALLET 20% (India-realistic)
    - Dwell time: 1 hour to 5 days (normal users don't rush)
    - Device/IP pulled from account's ACTUAL metadata (stateful)
    - No shared_hash (legitimate txns don't correlate cross-rail)
    """
    account_lookup = get_account_lookup(accounts_df)
    normal_accs = accounts_df[accounts_df['is_mule'] == 0]['account_id'].tolist()
    records = []

    for _ in range(n):
        sender_id = random.choice(normal_accs)
        receiver_id = random.choice(normal_accs)
        while receiver_id == sender_id:
            receiver_id = random.choice(normal_accs)

        sender_meta = account_lookup[sender_id]
        amount = float(np.random.lognormal(mean=7.5, sigma=1.2))

        records.append({
            'txn_id': str(uuid.uuid4()),
            'timestamp': fake.date_time_between(start_date='-30d', end_date='now'),
            'sender': sender_id,
            'receiver': receiver_id,
            'amount': round(amount, 2),
            'rail': random.choices(['BANK', 'UPI', 'WALLET'], weights=[0.3, 0.5, 0.2])[0],
            'device_hash': sender_meta['device_hash'],
            'ip_subnet': sender_meta['ip_subnet'],
            'dwell_time_seconds': random.randint(3600, 86400 * 5),
            'is_structured': is_structured(amount),
            'shared_hash': None,
            'is_fraud': 0
        })
    return pd.DataFrame(records)


def generate_vector_5_1_mule_networks(accounts_df, n_networks=10):
    """
    Vector 5.1: AI-Orchestrated Mule Network Simulation
    
    Attack flow:
    1. Fan-Out: Source (Victim) → Distributor Mules (amounts structured < ₹10K)
    2. Fan-In:  Distributor Mules → Aggregator Mule (fee deducted per hop)
    3. Noise:   Behavioral camouflage txns (Swiggy/Amazon) injected per mule
    
    Key signals embedded:
    - Shared device_hash across mules in same cluster
    - Shared ip_subnet across mules in same cluster
    - Low dwell_time (60-600s) on fan-in hops
    - Amounts structured in [₹8,500 - ₹9,800] range
    - Fee deduction per hop (₹100-₹300)
    """
    account_lookup = get_account_lookup(accounts_df)
    mule_accs = accounts_df[accounts_df['is_mule'] == 1]['account_id'].tolist()
    normal_accs = accounts_df[accounts_df['is_mule'] == 0]['account_id'].tolist()
    records = []

    for _ in range(n_networks):
        source_acc = random.choice(normal_accs)       # Victim source
        aggregator_acc = random.choice(mule_accs)     # Aggregator mule

        # Select 3-5 distributor mules per network (excluding aggregator)
        available_mules = [m for m in mule_accs if m != aggregator_acc]
        distributor_mules = random.sample(available_mules, k=min(random.randint(3, 5), len(available_mules)))

        base_time = fake.date_time_between(start_date='-15d', end_date='-1d')
        stolen_total = random.uniform(30000, 50000)
        split_amount = stolen_total / len(distributor_mules)

        # --- PHASE 1: Fan-Out (Victim → Distributor Mules) ---
        for mule in distributor_mules:
            mule_meta = account_lookup[mule]
            source_meta = account_lookup[source_acc]

            # Structure amount: keep below ₹10K, with variance
            amount = round(min(split_amount + random.uniform(-500, 500), 9800), 2)
            fanout_time = base_time + timedelta(minutes=random.randint(10, 120))

            records.append({
                'txn_id': str(uuid.uuid4()),
                'timestamp': fanout_time,
                'sender': source_acc,
                'receiver': mule,
                'amount': amount,
                'rail': 'BANK',
                'device_hash': source_meta['device_hash'],
                'ip_subnet': source_meta['ip_subnet'],
                'dwell_time_seconds': random.randint(300, 1800),
                'is_structured': is_structured(amount),
                'shared_hash': None,
                'is_fraud': 1
            })

            # --- PHASE 2: Fan-In (Distributor Mule → Aggregator) ---
            layer_dwell = random.randint(60, 600)
            fanin_time = fanout_time + timedelta(seconds=layer_dwell)
            pass_through_amount = round(amount - random.uniform(100, 300), 2)

            records.append({
                'txn_id': str(uuid.uuid4()),
                'timestamp': fanin_time,
                'sender': mule,
                'receiver': aggregator_acc,
                'amount': pass_through_amount,
                'rail': random.choice(['BANK', 'UPI']),
                'device_hash': mule_meta['device_hash'],
                'ip_subnet': mule_meta['ip_subnet'],
                'dwell_time_seconds': layer_dwell,
                'is_structured': is_structured(pass_through_amount),
                'shared_hash': None,
                'is_fraud': 1
            })

            # --- PHASE 3: Behavioral Noise (Camouflage Txns) ---
            for _ in range(random.randint(1, 3)):
                noise_time = fanin_time + timedelta(hours=random.randint(2, 24))
                noise_amount = round(random.uniform(50, 800), 2)
                records.append({
                    'txn_id': str(uuid.uuid4()),
                    'timestamp': noise_time,
                    'sender': mule,
                    'receiver': random.choice(normal_accs),
                    'amount': noise_amount,
                    'rail': random.choice(['UPI', 'WALLET']),
                    'device_hash': mule_meta['device_hash'],
                    'ip_subnet': mule_meta['ip_subnet'],
                    'dwell_time_seconds': random.randint(3600, 43200),
                    'is_structured': 0,
                    'shared_hash': None,
                    'is_fraud': 0  # Legitimate noise — not fraud
                })

    return pd.DataFrame(records)


def generate_vector_5_2_cross_rail(accounts_df, n_chains=15):
    """
    Vector 5.2: Cross-Rail Arbitrage (BANK → UPI → WALLET → CRYPTO)
    
    Attack flow:
    - Money enters via BANK, hops across UPI → WALLET → CRYPTO
    - Each hop uses an account whose type MATCHES the rail
    - All hops share a correlation hash (shared_hash) for cross-rail linking
    - Low dwell time (25-90s) — money moves FAST
    - Small fee deducted per hop
    
    Key signals embedded:
    - shared_hash connects all hops in one chain
    - Hyper-velocity dwell times (<90s)
    - Amount decreasing across hops (fee deduction)
    - 4 different rails in one chain
    """
    account_lookup = get_account_lookup(accounts_df)
    mule_accs = accounts_df[accounts_df['is_mule'] == 1]['account_id'].tolist()
    mules_by_type = get_mules_by_type(accounts_df)
    records = []
    rails = ["BANK", "UPI", "WALLET", "CRYPTO"]

    for _ in range(n_chains):
        stolen_amount = random.uniform(40000, 90000)
        chain_shared_hash = f"CORR_{str(uuid.uuid4())[:8]}"
        base_time = fake.date_time_between(start_date='-10d', end_date='now')

        # Pick initial sender (any mule)
        current_sender = random.choice(mule_accs)

        for rail in rails:
            # Pick receiver whose account_type is COMPATIBLE with this rail
            compatible_types = RAIL_ACCOUNT_TYPES[rail]
            compatible_mules = []
            for acc_type in compatible_types:
                compatible_mules.extend(mules_by_type.get(acc_type, []))

            # Remove current sender from candidates
            compatible_mules = [m for m in compatible_mules if m != current_sender]

            if not compatible_mules:
                # Fallback: use any mule if no compatible one found
                compatible_mules = [m for m in mule_accs if m != current_sender]

            current_receiver = random.choice(compatible_mules)
            sender_meta = account_lookup[current_sender]

            fee = random.uniform(50, 150)
            stolen_amount = max(100.0, stolen_amount - fee)

            dwell = random.randint(25, 90)
            base_time += timedelta(seconds=dwell)

            records.append({
                'txn_id': str(uuid.uuid4()),
                'timestamp': base_time,
                'sender': current_sender,
                'receiver': current_receiver,
                'amount': round(stolen_amount, 2),
                'rail': rail,
                'device_hash': sender_meta['device_hash'],
                'ip_subnet': sender_meta['ip_subnet'],
                'dwell_time_seconds': dwell,
                'is_structured': is_structured(stolen_amount),
                'shared_hash': chain_shared_hash,
                'is_fraud': 1
            })

            # Next hop: receiver becomes sender
            current_sender = current_receiver

    return pd.DataFrame(records)


# =====================================================================
# 3. DATASET GENERATION PIPELINE
# =====================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  MASTERCARD AI DEFENSE LAB — SEED DATA GENERATOR")
    print("  Vectors: 5.1 (Mule Networks) + 5.2 (Cross-Rail)")
    print("=" * 60)

    # ─── Step 1: Generate Account Nodes ───
    accounts_df = generate_account_nodes(n_normal=200, n_mules=50)
    print(f"\n📊 Accounts Generated:")
    print(f"   Total:      {len(accounts_df)}")
    print(f"   Legitimate: {(accounts_df['is_mule'] == 0).sum()}")
    print(f"   Mule:       {(accounts_df['is_mule'] == 1).sum()}")
    print(f"   By Type:    {accounts_df['account_type'].value_counts().to_dict()}")

    # ─── Step 2: Generate Transactions (Edges) ───
    legit_df = generate_legitimate_transactions(accounts_df, n=5000)
    v51_df = generate_vector_5_1_mule_networks(accounts_df, n_networks=15)
    v52_df = generate_vector_5_2_cross_rail(accounts_df, n_chains=20)

    print(f"\n📊 Transactions Generated:")
    print(f"   Legitimate:       {len(legit_df)} txns")
    print(f"   V5.1 Mule:        {len(v51_df)} txns (fraud: {v51_df['is_fraud'].sum()})")
    print(f"   V5.2 Cross-Rail:  {len(v52_df)} txns (fraud: {v52_df['is_fraud'].sum()})")

    # ─── Step 3: Combine & Shuffle ───
    transactions_df = pd.concat([legit_df, v51_df, v52_df], ignore_index=True)
    transactions_df = transactions_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # ─── Step 4: Summary Stats ───
    total_fraud = transactions_df['is_fraud'].sum()
    fraud_ratio = transactions_df['is_fraud'].mean() * 100
    structured_count = transactions_df['is_structured'].sum()
    cross_rail_txns = transactions_df[transactions_df['shared_hash'].notna()]

    print(f"\n{'=' * 60}")
    print(f"  FINAL SEED DATASET SUMMARY")
    print(f"{'=' * 60}")
    print(f"   Total Transactions:     {len(transactions_df)}")
    print(f"   Fraud Transactions:     {total_fraud} ({fraud_ratio:.2f}%)")
    print(f"   Structured Transactions:{structured_count}")
    print(f"   Cross-Rail Chains:      {cross_rail_txns['shared_hash'].nunique()} chains, {len(cross_rail_txns)} txns")
    print(f"   Rail Distribution:      {transactions_df['rail'].value_counts().to_dict()}")
    print(f"   Amount Stats:")
    print(f"     Mean:   ₹{transactions_df['amount'].mean():,.2f}")
    print(f"     Median: ₹{transactions_df['amount'].median():,.2f}")
    print(f"     Std:    ₹{transactions_df['amount'].std():,.2f}")

    # ─── Step 5: Save ───
    accounts_df.to_csv("seed_accounts_nodes.csv", index=False)
    transactions_df.to_csv("seed_transactions_edges.csv", index=False)
    print(f"\n✅ Saved: seed_accounts_nodes.csv ({len(accounts_df)} rows)")
    print(f"✅ Saved: seed_transactions_edges.csv ({len(transactions_df)} rows)")

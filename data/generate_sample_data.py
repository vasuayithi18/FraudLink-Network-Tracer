"""
generate_sample_data.py
-----------------------
Generates a realistic sample financial transaction dataset conforming to the
standard PaySim synthetic financial dataset schema.

This allows the 'FraudLink + Network Tracer' application to run out-of-the-box
with zero configuration, while demonstrating realistic transaction graphs,
mule rings, balance-draining patterns, and normal commerce.
"""

import os
import random
import numpy as np
import pandas as pd

def generate_paysim_sample(
    num_transactions: int = 5000,
    output_path: str = "data/paysim_sample.csv",
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate a synthetic financial dataset matching PaySim schema.
    
    Schema:
    - step: Simulation hour (1 to 744)
    - type: CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER
    - amount: Transaction amount
    - nameOrig: Originator account ID (C...)
    - oldbalanceOrg: Originator balance before transaction
    - newbalanceOrig: Originator balance after transaction
    - nameDest: Destination account ID (C... or M...)
    - oldbalanceDest: Destination balance before transaction
    - newbalanceDest: Destination balance after transaction
    - isFraud: Ground-truth anomaly indicator (1 or 0)
    - isFlaggedFraud: Rule-based indicator (transfer > 200,000)
    """
    np.random.seed(seed)
    random.seed(seed)
    
    # Pools of normal accounts and merchants
    num_normal_users = int(num_transactions * 0.45)
    num_merchants = int(num_transactions * 0.15)
    
    normal_users = [f"C{1000000 + i}" for i in range(num_normal_users)]
    merchants = [f"M{5000000 + i}" for i in range(num_merchants)]
    
    # Anomaly / Mule rings (for graph tracing & unsupervised anomaly detection)
    mule_master_1 = "C9990001"
    mule_master_2 = "C9990002"
    mules_group_1 = [f"C88800{i:02d}" for i in range(1, 12)]
    mules_group_2 = [f"C77700{i:02d}" for i in range(1, 10)]
    
    # Store account balances in state dict to maintain balance consistency
    balances = {}
    def get_balance(acc: str, default_min=5000, default_max=150000):
        if acc not in balances:
            balances[acc] = round(float(np.random.uniform(default_min, default_max)), 2)
        return balances[acc]
    
    records = []
    
    # 1. Generate normal transactions (approx 95%)
    normal_count = int(num_transactions * 0.94)
    txn_types = ["PAYMENT", "CASH_OUT", "CASH_IN", "TRANSFER", "DEBIT"]
    type_weights = [0.38, 0.32, 0.18, 0.08, 0.04]
    
    for i in range(normal_count):
        step = random.randint(1, 720) # 30 days
        ttype = np.random.choice(txn_types, p=type_weights)
        sender = random.choice(normal_users)
        
        if ttype == "PAYMENT":
            receiver = random.choice(merchants)
            amount = round(float(np.random.exponential(scale=2500) + 50), 2)
        elif ttype == "CASH_IN":
            # CASH_IN increases sender balance
            receiver = sender
            sender = random.choice(normal_users)
            amount = round(float(np.random.uniform(1000, 75000)), 2)
        elif ttype == "TRANSFER":
            receiver = random.choice(normal_users)
            while receiver == sender:
                receiver = random.choice(normal_users)
            amount = round(float(np.random.uniform(2000, 80000)), 2)
        elif ttype == "CASH_OUT":
            receiver = random.choice(merchants)
            amount = round(float(np.random.uniform(500, 45000)), 2)
        else: # DEBIT
            receiver = random.choice(merchants)
            amount = round(float(np.random.uniform(100, 10000)), 2)
            
        old_orig = get_balance(sender, 10000, 200000)
        
        if ttype == "CASH_IN":
            new_orig = round(old_orig + amount, 2)
        else:
            # Check sufficient balance
            if old_orig < amount:
                amount = round(old_orig * 0.8, 2)
            new_orig = max(0.0, round(old_orig - amount, 2))
        balances[sender] = new_orig
        
        # Receiver balances
        old_dest = get_balance(receiver, 0, 100000)
        if str(receiver).startswith("M"):
            # Merchants in PaySim don't track balance (0.0)
            old_dest = 0.0
            new_dest = 0.0
        else:
            new_dest = round(old_dest + amount, 2)
            balances[receiver] = new_dest
            
        records.append({
            "step": step,
            "type": ttype,
            "amount": amount,
            "nameOrig": sender,
            "oldbalanceOrg": old_orig,
            "newbalanceOrig": new_orig,
            "nameDest": receiver,
            "oldbalanceDest": old_dest,
            "newbalanceDest": new_dest,
            "isFraud": 0,
            "isFlaggedFraud": 0
        })

    # 2. Inject Anomaly Patterns (approx 6% of transactions)
    # Pattern A: Mule Ring / Structuring (Fan-in then Rapid Cash Out)
    # Multiple accounts send money to mule master, which dumps it all out
    for m in mules_group_1:
        step = random.randint(100, 105)
        amount = round(float(np.random.uniform(45000, 95000)), 2)
        old_m = get_balance(m, 100000, 150000)
        new_m = round(old_m - amount, 2)
        balances[m] = new_m
        
        old_dest = get_balance(mule_master_1, 5000, 10000)
        new_dest = round(old_dest + amount, 2)
        balances[mule_master_1] = new_dest
        
        records.append({
            "step": step,
            "type": "TRANSFER",
            "amount": amount,
            "nameOrig": m,
            "oldbalanceOrg": old_m,
            "newbalanceOrig": new_m,
            "nameDest": mule_master_1,
            "oldbalanceDest": old_dest,
            "newbalanceDest": new_dest,
            "isFraud": 1,
            "isFlaggedFraud": 0
        })
        
    # Mule Master 1 empties all accumulated funds
    step = 106
    master_bal = balances.get(mule_master_1, 500000.0)
    drain_amount = round(master_bal, 2)
    records.append({
        "step": step,
        "type": "CASH_OUT",
        "amount": drain_amount,
        "nameOrig": mule_master_1,
        "oldbalanceOrg": master_bal,
        "newbalanceOrig": 0.0,
        "nameDest": "C9990099",
        "oldbalanceDest": 0.0,
        "newbalanceDest": drain_amount,
        "isFraud": 1,
        "isFlaggedFraud": 1 if drain_amount > 200000 else 0
    })
    balances[mule_master_1] = 0.0

    # Pattern B: Rapid Account Balance Draining (Classic PaySim Fraud)
    # A stolen account transfers entire balance and then cashes out immediately
    for _ in range(35):
        step = random.randint(150, 600)
        stolen_victim = f"C{random.randint(6000000, 6009999)}"
        mule_acc = f"C{random.randint(7000000, 7009999)}"
        stolen_bal = round(float(np.random.uniform(210000, 980000)), 2)
        
        # Transfer out entire balance
        records.append({
            "step": step,
            "type": "TRANSFER",
            "amount": stolen_bal,
            "nameOrig": stolen_victim,
            "oldbalanceOrg": stolen_bal,
            "newbalanceOrig": 0.0,
            "nameDest": mule_acc,
            "oldbalanceDest": 0.0,
            "newbalanceDest": stolen_bal,
            "isFraud": 1,
            "isFlaggedFraud": 1 if stolen_bal > 200000 else 0
        })
        
        # Cash out immediately
        records.append({
            "step": step + 1,
            "type": "CASH_OUT",
            "amount": stolen_bal,
            "nameOrig": mule_acc,
            "oldbalanceOrg": stolen_bal,
            "newbalanceOrig": 0.0,
            "nameDest": f"C{random.randint(8000000, 8009999)}",
            "oldbalanceDest": 0.0,
            "newbalanceDest": stolen_bal,
            "isFraud": 1,
            "isFlaggedFraud": 0
        })

    # Pattern C: Fan-out Laundering Ring (Mule Master 2 splits big amount to many mules)
    master_2_initial = 850000.0
    balances[mule_master_2] = master_2_initial
    for m in mules_group_2:
        step = random.randint(220, 225)
        amount = round(master_2_initial / len(mules_group_2), 2)
        old_src = balances[mule_master_2]
        new_src = max(0.0, round(old_src - amount, 2))
        balances[mule_master_2] = new_src
        
        old_m = get_balance(m, 500, 2000)
        new_m = round(old_m + amount, 2)
        balances[m] = new_m
        
        records.append({
            "step": step,
            "type": "TRANSFER",
            "amount": amount,
            "nameOrig": mule_master_2,
            "oldbalanceOrg": old_src,
            "newbalanceOrig": new_src,
            "nameDest": m,
            "oldbalanceDest": old_m,
            "newbalanceDest": new_m,
            "isFraud": 1,
            "isFlaggedFraud": 0
        })

    df = pd.DataFrame(records)
    # Shuffle slightly while keeping chronological sense
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df = df.sort_values(by=["step"]).reset_index(drop=True)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} transactions in '{output_path}'.")
    print(f"Unique Senders (nameOrig): {df['nameOrig'].nunique()}")
    print(f"Unique Receivers (nameDest): {df['nameDest'].nunique()}")
    print(f"Ground-truth Fraud rows: {df['isFraud'].sum()} ({df['isFraud'].mean()*100:.2f}%)")
    
    return df

if __name__ == "__main__":
    generate_paysim_sample()

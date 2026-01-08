# score_predictions.py
import pandas as pd
import json
import os
from sklearn.metrics import accuracy_score, classification_report

# --- CONFIG ---
TRAIN_CSV = "./data/train.csv"  # <--- Ground Truth Source
PREDICTIONS_FILE = "data/final_predictions_vector.jsonl"  # <--- Your Pipeline Results


def main():
    print("--- Scoring Pipeline Results ---")

    # 1. Load Ground Truth from CSV
    if not os.path.exists(TRAIN_CSV):
        print(f"❌ Error: {TRAIN_CSV} not found.")
        return

    print(f"Loading ground truth from {TRAIN_CSV}...")
    df_truth = pd.read_csv(TRAIN_CSV)

    # Ensure ID is string for matching
    df_truth['id'] = df_truth['id'].astype(str)

    # Create Map: { "1": "consistent", "2": "contradict" }
    truth_map = dict(zip(df_truth['id'], df_truth['label']))

    # 2. Load Predictions
    if not os.path.exists(PREDICTIONS_FILE):
        print(f"❌ Error: {PREDICTIONS_FILE} not found.")
        return

    print(f"Loading predictions from {PREDICTIONS_FILE}...")
    preds_map = {}

    # Load JSON list from file
    with open(PREDICTIONS_FILE, 'r') as f:
        data = json.load(f)

    for item in data:
        # Normalize ID to string
        pid = str(item.get('id'))
        # Normalize verdict to lowercase (consistent/contradict)
        verdict = item.get('verdict', 'consistent').lower().strip()
        preds_map[pid] = verdict

    # 3. Align and Score
    y_true = []
    y_pred = []

    match_count = 0
    missing_count = 0

    for uid, true_label in truth_map.items():
        if uid in preds_map:
            # Normalize strings
            t = str(true_label).strip().lower()
            p = preds_map[uid]

            y_true.append(t)
            y_pred.append(p)
            match_count += 1
        else:
            missing_count += 1

    print(f"Matched {match_count} samples. Missing {missing_count}.")

    if not y_true:
        print("No matching data to score.")
        return

    # 4. Report
    acc = accuracy_score(y_true, y_pred)
    print(f"\n🏆 Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))


if __name__ == "__main__":
    main()
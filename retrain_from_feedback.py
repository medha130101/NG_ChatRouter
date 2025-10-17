# retrain_from_feedback.py
import os
import json
import pandas as pd
from model_utils import train

ORIG_CSV = "example_data.csv"
FEEDBACK_DIR = "feedback"
MERGED_CSV = "merged_for_retrain.csv"

def merge_feedback():
    df = pd.read_csv(ORIG_CSV)
    feedback_path = os.path.join(FEEDBACK_DIR, "feedback_log.jsonl")
    if not os.path.exists(feedback_path):
        print("No feedback found. Using original dataset.")
        return ORIG_CSV
    rows = []
    with open(feedback_path, "r") as f:
        for line in f:
            j = json.loads(line.strip())
            rows.append(j)
    print("Found feedback entries (please enrich feedback with original text for retraining):")
    for r in rows:
        print(r)
    return ORIG_CSV

if __name__ == "__main__":
    merged = merge_feedback()
    train(merged, output_dir=os.getenv("MODEL_DIR", "saved_model"))

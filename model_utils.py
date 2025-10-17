# Import Libraries
import os
import torch
import pandas as pd 
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from datasets import Dataset
import evaluate

from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

# Environment Variables and Constants
MODEL_NAME = os.getenv("MODEL_NAME", "distilbert-base-uncased")
EPOCHS = int(os.getenv("EPOCHS", 3))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 16))
MAX_LENGTH = int(os.getenv("MAX_LENGTH", 128))
MODEL_DIR = os.getenv("MODEL_DIR", "saved_model")
metric = evaluate.load("f1")

# Load Data Function
def load_data(csv_path):
    df = df = pd.read_csv(csv_path, quotechar='"', escapechar='\\')
    assert "text" in df.columns and "intent" in df.columns and "department" in df.columns
    return df

# Prepare Datasets Function
def prepare_datasets(df):
    intents = sorted(df['intent'].unique().tolist())
    intent2id = {lab:i for i,lab in enumerate(intents)}
    id2intent = {v:k for k,v in intent2id.items()}
    print(intents)
    print(intent2id)
    print(id2intent)
    print(df.head())
    df["label"] = df["intent"].map(intent2id)
    # Note - Stratified split may fail if some classes have very few samples.
    # train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
    try:
        train_df, test_df = train_test_split(
            df, test_size=0.2, random_state=42, stratify=df["label"]
        )
    except ValueError as e:
        print("Stratified split failed:", e)
    print("Falling back to random split (no stratification).")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_ds = Dataset.from_pandas(train_df[["text", "label"]])
    test_ds = Dataset.from_pandas(test_df[["text","label"]])
    return train_ds, test_ds, intent2id, id2intent
    
    
# Tokenization Function
def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH)

# Metrics Computation Function
def compute_metrics(eval_pred):
    logits,labels = eval_pred
    preds = logits.argmax(-1)
    f1 = metric.compute(predictions=preds, references=labels, average="weighted")["f1"]
    acc = (preds == labels).mean()
    return {"accuracy": acc, "f1": f1}


# Training Function


def train(csv_path, output_dir=MODEL_DIR):
    df = load_data(csv_path)
    train_ds, test_ds, intent2id, id2intent = prepare_datasets(df)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = train_ds.map(lambda examples: tokenize_function(examples, tokenizer), batched=True)
    test_ds = test_ds.map(lambda examples: tokenize_function(examples, tokenizer), batched=True)
    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(intent2id))

    training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir=f"{output_dir}/logs",
)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )

    trainer.train()
    os.makedirs(output_dir, exist_ok=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    import json
    with open(os.path.join(output_dir, "intent2id.json"), "w") as f:
        json.dump(intent2id, f)
    with open(os.path.join(output_dir, "id2intent.json"), "w") as f:
        json.dump(id2intent, f)
    print("Training complete. Model saved to", output_dir)
    
if __name__ == "__main__":
    import argparse
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="Data/example_data.csv")
    parser.add_argument("--output_dir", type=str, default="saved_model")

    # ignore unrecognized args (like --f)
    args, unknown = parser.parse_known_args()
    train(args.csv, args.output_dir)
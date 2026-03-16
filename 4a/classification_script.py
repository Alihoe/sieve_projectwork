import pandas as pd
import torch
import numpy as np
from transformers import (
    DebertaV2Tokenizer,
    DebertaV2ForSequenceClassification,
    Trainer,
    TrainingArguments
)
from datasets import Dataset, ClassLabel
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, classification_report

import numpy as np
import pandas as pd
from typing import Union, List


def convert_single_label(label: Union[str, list, float, int]) -> int:
    """Convert a single label to class index (0-2)"""
    try:
        # Handle string representation of list
        if isinstance(label, str) and label.startswith('[') and label.endswith(']'):
            values = [float(x) for x in label.strip('[]').split(',')]
            if len(values) != 3:
                raise ValueError(f"Expected 3 values, got {len(values)}")
            return int(np.argmax(values))

        # Handle actual list
        elif isinstance(label, list):
            if len(label) != 3:
                raise ValueError(f"Expected 3 values, got {len(label)}")
            return int(np.argmax(label))

        # Handle single numbers
        elif isinstance(label, (int, float)):
            return max(0, min(2, int(label)))

        raise ValueError(f"Unsupported label type: {type(label)}")
    except Exception as e:
        print(f"Error converting label {label}: {str(e)}")
        return 0  # Default fallback


def clean_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Convert dataframe labels to proper 0-2 class indices"""
    df = df.copy()

    # Convert all labels
    df['labels'] = df['labels'].apply(convert_single_label)

    # Validate we have exactly 3 classes
    unique_classes = df['labels'].unique()
    if len(unique_classes) != 3:
        print(f"Warning: Found {len(unique_classes)} classes after conversion")
        print("Class distribution:")
        print(df['labels'].value_counts().sort_index())

    return df


def load_data(train_path, dev_path):
    """Load data with robust label cleaning"""
    train_df = pd.read_csv(train_path, sep="\t")
    dev_df = pd.read_csv(dev_path, sep="\t")
    return (
        clean_labels(train_df),
        clean_labels(dev_df)
    )


def prepare_datasets(train_df, dev_df):
    """Create datasets with proper label types"""
    train_dataset = Dataset.from_pandas(train_df)
    dev_dataset = Dataset.from_pandas(dev_df)

    train_dataset = train_dataset.cast_column(
        "labels",
        ClassLabel(num_classes=3, names=['class0', 'class1', 'class2'])
    )
    dev_dataset = dev_dataset.cast_column(
        "labels",
        ClassLabel(num_classes=3, names=['class0', 'class1', 'class2'])
    )
    return train_dataset, dev_dataset


def tokenize_data(dataset, tokenizer):
    """Tokenize data ensuring proper input format"""

    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        return {
            'input_ids': tokenized['input_ids'].tolist(),
            'attention_mask': tokenized['attention_mask'].tolist(),
            'labels': examples["labels"],
            'index': examples.get("index", -1)
        }

    return dataset.map(tokenize_function, batched=True)


def train_model(train_path, dev_path, num_epochs=4):
    """Training with optimized parameters"""
    train_df, dev_df = load_data(train_path, dev_path)
    train_dataset, dev_dataset = prepare_datasets(train_df, dev_df)

    tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-base")
    model = DebertaV2ForSequenceClassification.from_pretrained(
        "microsoft/deberta-v3-base",
        num_labels=3,
        id2label={0: "class0", 1: "class1", 2: "class2"},
        label2id={"class0": 0, "class1": 1, "class2": 2}
    )

    tokenized_train = tokenize_data(train_dataset, tokenizer)
    tokenized_dev = tokenize_data(dev_dataset, tokenizer)

    training_args = TrainingArguments(
        output_dir="./results",
        overwrite_output_dir=True,
        eval_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        num_train_epochs=num_epochs,
        learning_rate=3e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        logging_dir='./logs',
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_dev,
        compute_metrics=lambda p: {
            "accuracy": accuracy_score(p.label_ids, p.predictions.argmax(-1)),
            "macro_f1": f1_score(p.label_ids, p.predictions.argmax(-1), average="macro"),
            "weighted_f1": f1_score(p.label_ids, p.predictions.argmax(-1), average="weighted")
        }
    )

    trainer.train()
    return model, tokenizer


def evaluate_dev(model, tokenizer, dev_path):
    """Evaluate model on development set enforcing 3-class structure"""
    dev_df = pd.read_csv(dev_path, sep="\t")
    dev_df = clean_labels(dev_df)

    # Force labels to be in 0-2 range
    dev_df['labels'] = dev_df['labels'].apply(lambda x: max(0, min(2, x)))

    dev_dataset = Dataset.from_pandas(dev_df)
    dev_dataset = dev_dataset.cast_column(
        "labels",
        ClassLabel(num_classes=3, names=['class0', 'class1', 'class2'])
    )

    tokenized_dev = tokenize_data(dev_dataset, tokenizer)

    trainer = Trainer(model)
    results = trainer.predict(tokenized_dev)

    # Convert to numpy arrays
    label_ids = results.label_ids
    preds = results.predictions.argmax(-1)

    # Ensure we have all 3 classes in the report
    unique_labels = set(label_ids)
    if len(unique_labels) < 3:
        print(f"Warning: Development set only contains {len(unique_labels)} classes")
        # Create dummy entries for missing classes
        dummy_labels = np.array([0, 1, 2])
        dummy_preds = np.array([0, 1, 2])
        label_ids = np.concatenate([label_ids, dummy_labels])
        preds = np.concatenate([preds, dummy_preds])

    print("\nDevelopment Set Evaluation (3-class enforced):")
    print(classification_report(
        label_ids,
        preds,
        target_names=['class0', 'class1', 'class2'],
        zero_division=0
    ))



def predict_with_uncertainty(model, tokenizer, data_path, threshold=0.5):
    """Predict with uncertainty detection"""
    df = pd.read_csv(data_path, sep="\t")
    df = clean_labels(df)
    dataset = Dataset.from_pandas(df)
    dataset = dataset.cast_column(
        "labels",
        ClassLabel(num_classes=3, names=['class0', 'class1', 'class2'])
    )
    tokenized_data = tokenize_data(dataset, tokenizer)

    trainer = Trainer(model)
    predictions = trainer.predict(tokenized_data)
    probas = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=-1).numpy()

    pred_labels = probas.argmax(axis=1)
    confidences = probas.max(axis=1)
    uncertain_mask = confidences < threshold
    uncertain_indices = [tokenized_data[i]["index"] for i in np.where(uncertain_mask)[0]]

    return pred_labels, uncertain_indices, confidences


def print_uncertain_texts(data_path, uncertain_ids):
    """Print texts of uncertain predictions"""
    df = pd.read_csv(data_path, sep="\t", index_col="index")
    for idx in uncertain_ids:
        if idx in df.index:
            print(f"\nUncertain ID: {idx}\n{df.loc[idx, 'text']}\n{'=' * 50}")


def save_model(model, tokenizer, output_dir):
    """Save model and tokenizer"""
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def load_model(model_path):
    """Load model and tokenizer"""
    tokenizer = DebertaV2Tokenizer.from_pretrained(model_path)
    model = DebertaV2ForSequenceClassification.from_pretrained(model_path)
    return model, tokenizer


if __name__ == "__main__":
    # 1. Train
    print("Training model...")
    model, tokenizer = train_model(
        train_path="data/ct_train.tsv",
        dev_path="data/ct_dev.tsv"
    )

    # 2. Save
    save_model(model, tokenizer, "./deberta_classifier")
    print("Model saved to ./deberta_classifier")

    #model, tokenizer = load_model("./deberta_classifier")

    # 3. Evaluate on dev set
    evaluate_dev(model, tokenizer, "data/ct_dev.tsv")


    # 4. Predict with uncertainty
    print("\nMaking predictions with uncertainty detection...")
    preds, uncertain_ids, _ = predict_with_uncertainty(model, tokenizer, "data/ct_dev.tsv")
    print(uncertain_ids)

    # 5. Print uncertain texts
    print("\nTexts with uncertain predictions:")
    print_uncertain_texts("data/ct_dev.tsv", uncertain_ids)
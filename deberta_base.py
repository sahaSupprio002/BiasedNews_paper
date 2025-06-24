# 📦 Imports
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    DebertaForSequenceClassification,
    DebertaTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

# 📁 Upload your dataset
from google.colab import files
uploaded = files.upload()

# 📄 Load dataset
df = pd.read_csv("all_balance_data 3.csv")
df['label'] = LabelEncoder().fit_transform(df['category'])  # 'political' = 1, 'non-political' = 0

# 🔀 Stratified train/validation split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['title'].tolist(),
    df['label'].tolist(),
    test_size=0.2,
    stratify=df['label'],
    random_state=42
)

# 🔤 Tokenization
tokenizer = DebertaTokenizer.from_pretrained("microsoft/deberta-base")
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

# 📚 Dataset class
class HeadlineDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.encodings['input_ids'][idx]),
            'attention_mask': torch.tensor(self.encodings['attention_mask'][idx]),
            'labels': torch.tensor(self.labels[idx])
        }

train_dataset = HeadlineDataset(train_encodings, train_labels)
val_dataset = HeadlineDataset(val_encodings, val_labels)

# 🧠 Load model
model = DebertaForSequenceClassification.from_pretrained(
    "microsoft/deberta-base",
    num_labels=2,
    hidden_dropout_prob=0.2,          # Key change
    attention_probs_dropout_prob=0.2  # Key change
)

# 🧮 Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'precision': precision_score(labels, preds),
        'recall': recall_score(labels, preds),
        'f1': f1_score(labels, preds)
    }

# ⚙️ Training arguments
training_args = TrainingArguments(
    output_dir="./deberta_model_0.2drop",
    num_train_epochs=50,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=2e-6,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=2,
    logging_dir="./logs",
    report_to="none"
)

# 🏋️ Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
)

# 🚀 Train
train_result = trainer.train()

# 📊 Evaluate
metrics = trainer.evaluate()
print("Evaluation metrics:", metrics)

# 💾 Save the model
trainer.save_model("./deberta_model_0.2drop")
tokenizer.save_pretrained("./deberta_model_0.2drop")

# 📈 Plot learning curves
training_logs = trainer.state.log_history
train_loss = [log["loss"] for log in training_logs if "loss" in log]
eval_loss = [log["eval_loss"] for log in training_logs if "eval_loss" in log]

plt.plot(train_loss, label="Train Loss")
plt.plot(eval_loss, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.show()

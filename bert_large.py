import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    DataCollatorWithPadding, get_scheduler
)
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler
import random
import os

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load Dataset
csv_path = "/content/new_marged_news_balance.csv"
df = pd.read_csv(csv_path)
df.dropna(subset=["title", "category"], inplace=True)
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["category"])

# Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")

# Augmentation: Random masking
def random_mask(tokens, prob=0.1):
    return [token if random.random() > prob else tokenizer.mask_token for token in tokens]

# Dataset Class
class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=48):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, idx):
        text = self.texts[idx]
        if random.random() < 0.5:
            tokens = tokenizer.tokenize(text)
            tokens = random_mask(tokens, prob=0.1)
            text = tokenizer.convert_tokens_to_string(tokens)
        encodings = self.tokenizer(text, truncation=True, padding="max_length",
                                   max_length=self.max_length, return_tensors="pt")
        return {
            "input_ids": encodings["input_ids"].squeeze(),
            "attention_mask": encodings["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx]),
        }

    def __len__(self):
        return len(self.labels)

# Split data
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["title"].tolist(), df["label"].tolist(),
    test_size=0.2, random_state=42, stratify=df["label"]
)

# Create datasets and loaders
train_dataset = NewsDataset(train_texts, train_labels, tokenizer, max_length=48)
val_dataset = NewsDataset(val_texts, val_labels, tokenizer, max_length=48)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                          collate_fn=data_collator, pin_memory=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=32, collate_fn=data_collator,
                        pin_memory=True, num_workers=2)

# Model setup
num_labels = len(label_encoder.classes_)
model = BertForSequenceClassification.from_pretrained(
    "bert-large-uncased", num_labels=num_labels,
    hidden_dropout_prob=0.4, attention_probs_dropout_prob=0.4
)

# Freeze embedding layer
for param in model.bert.embeddings.parameters():
    param.requires_grad = False

# Freeze all layers except last 5 (layers 19–23)
for name, param in model.bert.named_parameters():
    if "encoder.layer." in name:
        layer_num = int(name.split("encoder.layer.")[1].split(".")[0])
        if layer_num < 19:
            param.requires_grad = False

model.to(device)

# Optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
num_epochs = 70
num_training_steps = num_epochs * len(train_loader)
lr_scheduler = get_scheduler(
    "linear", optimizer=optimizer,
    num_warmup_steps=int(0.1 * num_training_steps),
    num_training_steps=num_training_steps
)

# AMP and early stopping setup
scaler = GradScaler()
patience = 10
best_val_loss = float("inf")
patience_counter = 0
train_losses, val_losses = [], []

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} Training"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        with autocast():
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    print(f"Epoch {epoch + 1} Train Loss: {avg_train_loss:.4f}")

    # Validation
    model.eval()
    val_loss = 0
    preds, true_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with autocast():
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

            val_loss += loss.item()
            logits = outputs.logits
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    # Metrics
    accuracy = accuracy_score(true_labels, preds)
    precision = precision_score(true_labels, preds, average="weighted", zero_division=0)
    recall = recall_score(true_labels, preds, average="weighted", zero_division=0)
    f1 = f1_score(true_labels, preds, average="weighted", zero_division=0)

    print(f"Epoch {epoch + 1} Val Loss: {avg_val_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1 Score: {f1:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "/content/best_model.pt")
        print("Saved best model")
    else:
        patience_counter += 1
        print(f"Patience counter: {patience_counter}")
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

# Plotting
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.show()

# Load best model
model.load_state_dict(torch.load("/content/best_model.pt"))
model.eval()

# 1. Upload CSV in Google Colab
from google.colab import files
uploaded = files.upload()

# 2. Imports
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# For AMP (Automatic Mixed Precision)
from torch.cuda.amp import autocast, GradScaler

# 3. Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 4. Load and preprocess data
filename = list(uploaded.keys())[0]
df = pd.read_csv(filename).sample(frac=1.0, random_state=42).reset_index(drop=True)
df['label'] = LabelEncoder().fit_transform(df['category'])

# 5. Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 6. Enhanced Dataset Class
class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=120):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def clean_text(self, text):
        text = str(text)
        # Add any text cleaning steps here if needed
        return text.strip()

    def __getitem__(self, idx):
        text = self.clean_text(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# 7. Prepare data loaders
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['title'], df['label'],
    test_size=0.15,
    stratify=df['label'],
    random_state=42
)

train_dataset = NewsDataset(train_texts.tolist(), train_labels.tolist(), tokenizer)
val_dataset = NewsDataset(val_texts.tolist(), val_labels.tolist(), tokenizer)

batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# 8. Enhanced Model initialization
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=len(df['label'].unique()),
    hidden_dropout_prob=0.4,
    attention_probs_dropout_prob=0.4
)
model.to(device)

# 9. Optimizer with weight decay
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {
        'params': [p for n, p in model.named_parameters()
                  if not any(nd in n for nd in no_decay)],
        'weight_decay': 0.01
    },
    {
        'params': [p for n, p in model.named_parameters()
                  if any(nd in n for nd in no_decay)],
        'weight_decay': 0.0
    }
]

optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5, eps=1e-8)

# 10. Learning rate scheduler
epochs = 20
total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)

# 11. Loss function with label smoothing
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# 12. AMP Scaler
scaler = GradScaler()

# 13. Gradient clipping
max_grad_norm = 1.0

# 14. Evaluation function
def evaluate(model, data_loader):
    model.eval()
    total_loss = 0
    preds, true_labels = [], []

    with torch.no_grad():
        for batch in data_loader:
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'labels': batch['label'].to(device)
            }

            with autocast():
                outputs = model(**inputs)
                loss = outputs.loss

            total_loss += loss.item()
            preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
            true_labels.extend(inputs['labels'].cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(true_labels, preds)
    precision = precision_score(true_labels, preds, average='weighted')
    recall = recall_score(true_labels, preds, average='weighted')
    f1 = f1_score(true_labels, preds, average='weighted')

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'preds': preds,
        'true_labels': true_labels
    }

# 15. Training loop
def train_eval(model, train_loader, val_loader, epochs=20, patience=6):
    best_val_loss = float('inf')
    best_f1 = 0
    train_losses, val_losses = [], []
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        model.train()
        total_train_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()

            with autocast():
                outputs = model(
                    input_ids=batch['input_ids'].to(device),
                    attention_mask=batch['attention_mask'].to(device),
                    labels=batch['label'].to(device)
                )
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        val_metrics = evaluate(model, val_loader)
        val_losses.append(val_metrics['loss'])

        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_metrics['loss']:.4f}")
        print(f"Accuracy: {val_metrics['accuracy']:.4f} | F1: {val_metrics['f1']:.4f}")
        print(f"Precision: {val_metrics['precision']:.4f} | Recall: {val_metrics['recall']:.4f}")

        # Save best model based on F1 score (unchanged)
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            torch.save(model.state_dict(), 'best_model.pt')
            print("Model improved (F1) - saved!")

        # Early stopping based on validation loss (new)
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping after {patience} epochs without validation loss improvement")
                break

    # Load best model
    model.load_state_dict(torch.load('best_model.pt'))
    final_metrics = evaluate(model, val_loader)

    # Plotting
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Confusion Matrix
    plt.subplot(1, 2, 2)
    cm = confusion_matrix(val_metrics['true_labels'], val_metrics['preds'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.show()

    # Error analysis
    error_indices = [i for i, (pred, true) in enumerate(zip(val_metrics['preds'], val_metrics['true_labels']))
                    if pred != true]
    error_samples = val_texts.iloc[error_indices].tolist()
    pd.DataFrame({'text': error_samples}).to_csv('error_analysis.csv', index=False)
    print("\nError analysis saved to 'error_analysis.csv'")

# 16. Train the model
train_eval(model, train_loader, val_loader, epochs=20, patience=6)

# 17. Save the final model
torch.save(model.state_dict(), 'final_model.pt')
print("\nFinal model saved to 'final_model.pt'")
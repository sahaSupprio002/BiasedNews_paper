!pip install torch transformers pandas matplotlib scikit-learn tqdm


import os
import torch
import pandas as pd
import re
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import RobertaTokenizer, RobertaForSequenceClassification, get_scheduler
from google.colab import drive

# ========================
# MOUNT GOOGLE DRIVE
# ========================
drive.mount('/content/drive')

# ========================
# CONFIGURATION
# ========================
DATA_PATH = "/content/drive/MyDrive/Balanced_news_dataset.csv"
MODEL_DIR = "/content/drive/MyDrive/news_classifier_model_Final"
os.makedirs(MODEL_DIR, exist_ok=True)

# ========================
# Preprocessing
# ========================
label_mapping = {'non-political': 0, 'political': 1}
df = pd.read_csv(DATA_PATH, encoding='ISO-8859-1')
df = df[df['category'].isin(label_mapping.keys())].copy()
df['label'] = df['category'].map(label_mapping)
df = df.dropna(subset=['title'])
df = df[df['title'].str.strip().astype(bool)].copy()

def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s.,!?]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

df['title'] = df['title'].apply(clean_text)

# ========================
# Train-validation split
# ========================
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['title'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42, stratify=df['label'].tolist()
)

# ========================
# Tokenizer & Model
# ========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = RobertaTokenizer.from_pretrained('roberta-large')

model = RobertaForSequenceClassification.from_pretrained(
    'roberta-large',
    num_labels=2,
    hidden_dropout_prob=0.3,
    attention_probs_dropout_prob=0.3
).to(device)

# Unfreeze only top 5 layers
for name, param in model.roberta.named_parameters():
    if not any(layer in name for layer in ["layer.19", "layer.20", "layer.21", "layer.22", "layer.23"]):
        param.requires_grad = False

# ========================
# Dataset & DataLoader
# ========================
class HeadlineDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

train_dataset = HeadlineDataset(train_texts, train_labels, tokenizer)
val_dataset = HeadlineDataset(val_texts, val_labels, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# ========================
# Optimizer & Scheduler
# ========================
optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.05)
epochs = 15
patience = 4
train_losses, val_losses, f1_scores = [], [], []
best_val_loss = float('inf')
counter = 0

num_training_steps = epochs * len(train_loader)
num_warmup_steps = int(0.1 * num_training_steps)

lr_scheduler = get_scheduler(
    "linear", optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
)

# ========================
# Training Loop
# ========================
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        lr_scheduler.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation
    model.eval()
    val_loss = 0
    all_preds, all_labels_eval = [], []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            val_loss += outputs.loss.item()
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels_eval.extend(labels.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    epoch_f1 = f1_score(all_labels_eval, all_preds, average='weighted')
    f1_scores.append(epoch_f1)

    print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | F1 Score: {epoch_f1:.4f}")
    print("Classification Report:")
    print(classification_report(all_labels_eval, all_preds, target_names=label_mapping.keys(), digits=4))

    # Save best model based on val_loss
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        counter = 0
        best_model_path = os.path.join(MODEL_DIR, "best_model.pt")
        torch.save(model.state_dict(), best_model_path)
        print(f"✅ Best model saved to {best_model_path}")

        checkpoint_path = os.path.join(MODEL_DIR, "checkpoint.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict(),
        }, checkpoint_path)
        print(f"💾 Full checkpoint saved to {checkpoint_path}")
    else:
        counter += 1
        print(f"⚠️ Patience: {counter}/{patience}")
        if counter >= patience:
            print("🛑 Early stopping.")
            break

# ========================
# Final Evaluation
# ========================
print("Loading best model...")
model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "best_model.pt")))
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("Final Classification Report:")
print(classification_report(all_labels, all_preds, target_names=label_mapping.keys(), digits=4))

# ========================
# Plot Loss
# ========================
epochs_range = range(1, len(train_losses)+1)
plt.figure(figsize=(12, 6))
plt.plot(epochs_range, train_losses, label='Train Loss')
plt.plot(epochs_range, val_losses, label='Validation Loss')
plt.title("Training Metrics")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.tight_layout()
plot_path = os.path.join(MODEL_DIR, "loss_plot.png")
plt.savefig(plot_path)
plt.show()
print(f"📈 Loss plot saved to {plot_path}")

# ========================
# Save Metrics CSV
# ========================
metrics_df = pd.DataFrame({
    'epoch': list(epochs_range),
    'train_loss': train_losses,
    'val_loss': val_losses,
    'f1_score': f1_scores
})
metrics_path = os.path.join(MODEL_DIR, "training_metrics.csv")
metrics_df.to_csv(metrics_path, index=False)
print(f"📊 Training metrics saved to {metrics_path}")

print("🎉 Training completed successfully!")

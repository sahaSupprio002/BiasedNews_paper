!pip install -q transformers datasets evaluate scikit-learn matplotlib


# Import and seed setting remain unchanged
from google.colab import files
import pandas as pd
import io
import random
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import RobertaTokenizer, RobertaForSequenceClassification, TrainingArguments, Trainer
from transformers import EarlyStoppingCallback, get_linear_schedule_with_warmup
from torch.optim import AdamW
from datasets import Dataset
import evaluate
import matplotlib.pyplot as plt
from google.colab import drive

SEED = 42
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed(SEED)

# Load dataset
uploaded = files.upload()
filename = list(uploaded.keys())[0]
df = pd.read_csv(io.BytesIO(uploaded[filename]))
df = df[['title', 'category']].dropna()
le = LabelEncoder()
df['label'] = le.fit_transform(df['category'])

# Augmentation
def augment_text(text):
    if random.random() < 0.3:
        words = text.split()
        if len(words) > 3:
            del words[random.randint(0, len(words)-1)]
            return ' '.join(words)
    return text

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['title'].tolist(),
    df['label'].tolist(),
    test_size=0.2,
    stratify=df['label'],
    random_state=SEED
)
train_texts = [augment_text(text) for text in train_texts]

# Tokenize
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
def tokenize_batch(texts):
    return tokenizer(texts, padding='max_length', truncation=True, max_length=64)
train_encodings = tokenize_batch(train_texts)
val_encodings = tokenize_batch(val_texts)
train_dataset = Dataset.from_dict({
    'input_ids': train_encodings['input_ids'],
    'attention_mask': train_encodings['attention_mask'],
    'labels': train_labels
})
val_dataset = Dataset.from_dict({
    'input_ids': val_encodings['input_ids'],
    'attention_mask': val_encodings['attention_mask'],
    'labels': val_labels
})

# Metrics
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return {
        'accuracy': accuracy.compute(predictions=preds, references=labels)['accuracy'],
        'f1': f1.compute(predictions=preds, references=labels, average='weighted')['f1']
    }

# Model with added dropout
model = RobertaForSequenceClassification.from_pretrained(
    'roberta-base',
    num_labels=2,
    hidden_dropout_prob=0.3,
    attention_probs_dropout_prob=0.3
)

# Optimizer with stronger L2
optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.05)

# Training arguments
training_args = TrainingArguments(
    output_dir="./roberta-news-model-final",
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    num_train_epochs=10,
    learning_rate=1e-5,
    weight_decay=0.05,
    warmup_ratio=0.1,
    seed=SEED,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    save_total_limit=1,
    logging_dir='./logs',
    logging_strategy="epoch",
    report_to="none"
)

total_steps = len(train_dataset) * training_args.num_train_epochs // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(total_steps * 0.1),
    num_training_steps=total_steps
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    optimizers=(optimizer, scheduler),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# Train
trainer.train()

# Plot losses
logs = trainer.state.log_history
train_loss = [log['loss'] for log in logs if 'loss' in log]
eval_loss = [log['eval_loss'] for log in logs if 'eval_loss' in log]
epochs = list(range(1, len(eval_loss) + 1))
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss[:len(epochs)], label='Train Loss')
plt.plot(epochs, eval_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train vs Validation Loss')
plt.legend()
plt.grid(True)
plt.show()

# Save to Drive
drive.mount('/content/drive')
!cp -r ./roberta-news-model /content/drive/MyDrive/roberta-news-model-final

# Print final metrics
final_eval = trainer.evaluate()
print("\nFinal Evaluation:")
for key, value in final_eval.items():
    print(f"{key}: {value}")

print("\nHyperparameters:")
print(f"Learning Rate: {training_args.learning_rate}")
print(f"Weight Decay: {training_args.weight_decay}")
print(f"Dropout: 0.3")
print(f"EarlyStopping Patience: 2")

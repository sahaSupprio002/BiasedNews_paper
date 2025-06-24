# 2. Imports
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    DebertaTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    AutoConfig,
)
from transformers.models.deberta.modeling_deberta import DebertaModel
from torch import nn
import evaluate
import matplotlib.pyplot as plt

# 3. Load CSV and Prepare Dataset
df = pd.read_csv("Balanced_news_dataset.csv")  # Replace with your actual CSV
df = df[["title", "category"]].dropna()
df['label'] = df['category'].map({"non-political": 0, "political": 1})

dataset = Dataset.from_pandas(df)
dataset = dataset.train_test_split(test_size=0.2)
train_dataset = dataset['train']
eval_dataset = dataset['test']

# 4. Tokenization
tokenizer = DebertaTokenizer.from_pretrained("microsoft/deberta-large")

def tokenize_function(example):
    return tokenizer(example["title"], padding="max_length", truncation=True)

train_dataset = train_dataset.map(tokenize_function, batched=True)
eval_dataset = eval_dataset.map(tokenize_function, batched=True)

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# 5. Custom Model with Dropout 0.2
from transformers import PreTrainedModel

class CustomDebertaClassifier(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.deberta = DebertaModel(config)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[0][:, 0, :]  # CLS token
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}

# Load model with custom dropout
config = AutoConfig.from_pretrained("microsoft/deberta-large", num_labels=2)
model = CustomDebertaClassifier.from_pretrained("microsoft/deberta-large", config=config)

# 6. Evaluation Metrics
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = torch.argmax(torch.tensor(logits), dim=-1)

    acc = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]
    try:
        f1 = f1_metric.compute(predictions=preds, references=labels, average='weighted')["f1"]
    except:
        f1 = 0.0
    try:
        precision = precision_metric.compute(predictions=preds, references=labels, average='weighted')["precision"]
    except:
        precision = 0.0
    try:
        recall = recall_metric.compute(predictions=preds, references=labels, average='weighted')["recall"]
    except:
        recall = 0.0

    return {
        "accuracy": acc,
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
    }

# 7. Training Arguments
training_args = TrainingArguments(
    output_dir="./deberta-model-v2",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=50,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    logging_dir="./logs-v2",
    logging_strategy="epoch"
)

# 8. Trainer with Early Stopping
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
)

# 9. Train the Model
train_result = trainer.train()

# 10. Save Best Model
trainer.save_model("./best-deberta-model-v2")

# 11. Plot Learning Curves
logs = trainer.state.log_history
train_loss = [log['loss'] for log in logs if 'loss' in log and 'eval_loss' not in log]
eval_loss = [log['eval_loss'] for log in logs if 'eval_loss' in log]
epochs = list(range(1, len(eval_loss) + 1))

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_loss) + 1), train_loss, label="Train Loss")
plt.plot(epochs, eval_loss, label="Eval Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.show()

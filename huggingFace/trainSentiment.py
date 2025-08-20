from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import evaluate

# 1. Carregar dataset IMDB
dataset = load_dataset("imdb")

# 2. Tokenizer
model_checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

dataset = dataset.map(tokenize, batched=True)
dataset = dataset.rename_column("label", "labels")
dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# 3. Modelo pré-treinado
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)

# 4. Métrica
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return accuracy.compute(predictions=preds, references=labels)

# 5. Configuração de treino
training_args = TrainingArguments(
    output_dir="./results",
    do_eval=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    push_to_hub=True,
    hub_model_id="RaissaPaula/sentiment-model",
)

# 6. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"].shuffle(seed=42).select(range(10000)), 
    eval_dataset=dataset["test"].shuffle(seed=42).select(range(2000)),
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# 7. Treinar e publicar
trainer.train()
trainer.push_to_hub()


# Hugging Face Transformers: Full Tutorial

## 1. Install Required Libraries
!pip install transformers datasets torch

## 2. Pipeline Example (Sentiment Analysis)
from transformers import pipeline
classifier = pipeline("sentiment-analysis")
print(classifier("I love learning Transformers!"))

## 3. Load Pretrained Model and Tokenizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")

inputs = tokenizer("Transformers are amazing!", return_tensors="pt")
outputs = model(**inputs)
print(outputs.logits)

## 4. Load Dataset and Tokenize
from datasets import load_dataset
dataset = load_dataset("imdb")
print(dataset["train"][0])

tokenized = dataset.map(lambda x: tokenizer(x["text"], truncation=True, padding="max_length"), batched=True)

## 5. Fine-tune Model
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    num_train_epochs=1,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"].shuffle(seed=42).select(range(1000)),
    eval_dataset=tokenized["test"].select(range(500)),
)

trainer.train()

## 6. Save and Load Model
model.save_pretrained("my_model")
tokenizer.save_pretrained("my_model")

model = AutoModelForSequenceClassification.from_pretrained("my_model")
tokenizer = AutoTokenizer.from_pretrained("my_model")

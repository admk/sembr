import torch
import numpy as np
from transformers import (
    AutoModelForTokenClassification, TrainingArguments, Trainer)

from sembr.data.sembr2023 import SemBr2023
from sembr.data.process import DataCollatorForTokenClassificationWithTruncation


dataset = SemBr2023()
dataset.download_and_prepare()
train_dataset = dataset.as_dataset(split='train')
test_dataset = dataset.as_dataset(split='test')

train_dataset = train_dataset.remove_columns(['words', 'modes', 'indents'])
label_names = train_dataset.features['labels'].feature.names
id2label = {i: l for i, l in enumerate(label_names)}
label2id = {l: i for i, l in enumerate(label_names)}
num_classes = train_dataset.features['labels'].feature.num_classes

collator = DataCollatorForTokenClassificationWithTruncation(
    dataset.processor.tokenizer, padding='max_length', max_length=512)
batch = collator(train_dataset)

model_name = 'distilbert-base-uncased'
model = AutoModelForTokenClassification.from_pretrained(
    model_name, num_labels=num_classes, id2label=id2label, label2id=label2id)
model.resize_token_embeddings(len(dataset.processor.tokenizer))
model = model.to('mps')


def compute_metrics(result):
    logits, labels = result
    preds = np.argmax(logits, axis=2)
    idx = labels != -100
    logits, preds, labels = logits[idx], preds[idx], labels[idx]
    acc = (preds == labels).mean()
    loss = torch.nn.functional.cross_entropy(
        torch.Tensor(logits), torch.LongTensor(labels), reduction='mean')
    return {'accuracy': acc, 'loss': loss}


training_args = TrainingArguments(
    output_dir='runs',
    run_name='sembr2023-distilbert-base-uncased',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    evaluation_strategy="steps",
    eval_steps=1,
    save_strategy="steps",
    save_steps=200,
    load_best_model_at_end=True,
    logging_steps = 1,
    # use_mps_device=True,
    # push_to_hub=True,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=dataset.processor.tokenizer,
    data_collator=collator,
    compute_metrics=compute_metrics,
)
trainer.train()

import torch
import numpy as np

import datasets
from transformers import (
    AutoModelForTokenClassification,
    TrainingArguments, Trainer)

from sembr.process import (
    SemBrProcessor, DataCollatorForTokenClassificationWithTruncation)


def chunk_examples(examples):
    id_chunks, label_chunks = [], []
    max_length = 512
    overlap = int(max_length / 10)
    for ids, labels in zip(examples['input_ids'], examples['labels']):
        id_chunks += [
            ids[i:i + max_length] for i in range(0, len(ids), overlap)]
        label_chunks += [
            labels[i:i + max_length] for i in range(0, len(labels), overlap)]
    return {'input_ids': id_chunks, 'labels': label_chunks}


def compute_metrics(result):
    logits, labels = result
    preds = np.argmax(logits, axis=2)
    idx = labels != -100
    logits, preds, labels = logits[idx], preds[idx], labels[idx]
    acc = (preds == labels).mean()
    loss = torch.nn.functional.cross_entropy(
        torch.Tensor(logits), torch.LongTensor(labels), reduction='mean')
    preds_set = set(preds.nonzero()[0])
    labels_set = set(labels.nonzero()[0])
    nonzero_acc = len(preds_set & labels_set) / len(labels_set)
    nonzero_iou = len(preds_set & labels_set) / len(preds_set | labels_set)
    return {
        'accuracy': acc,
        'nonzero_acc': nonzero_acc,
        'nonzero_iou': nonzero_iou,
        'loss': loss,
        'num_preds': len(set(preds))
    }


dataset = datasets.load_dataset('./sembr/data/sembr2023.py', 'sembr2023')
train_dataset = dataset['train']
test_dataset = dataset['test']
train_dataset = train_dataset.map(
    chunk_examples, batched=True,
    remove_columns=['words', 'modes', 'indents', 'base_indent'])
test_dataset = test_dataset.map(
    chunk_examples, batched=True,
    remove_columns=['words', 'modes', 'indents', 'base_indent'])
print(f'{len(train_dataset)=}, {len(test_dataset)=}')
label_names = train_dataset.features['labels'].feature.names
id2label = {i: l for i, l in enumerate(label_names)}
label2id = {l: i for i, l in enumerate(label_names)}
num_classes = train_dataset.features['labels'].feature.num_classes

tokenizer = SemBrProcessor().tokenizer
collator = DataCollatorForTokenClassificationWithTruncation(
    tokenizer, padding='max_length', max_length=512)
batch = collator(train_dataset)

model_name = 'distilbert-base-uncased'
model = AutoModelForTokenClassification.from_pretrained(
    model_name, num_labels=num_classes,
    id2label=id2label, label2id=label2id)
model.resize_token_embeddings(len(tokenizer))
model = model.train().to('cuda')
# for n, p in model.named_parameters():
#     if 'classifier' not in n:
#         p.requires_grad = False

run_name = 'sembr2023-distilbert-base-uncased-full'
training_args = TrainingArguments(
    output_dir=f'runs/{run_name}',
    run_name=run_name,
    learning_rate=1e-5,
    lr_scheduler_type='cosine',
    per_device_train_batch_size=64,
    # per_device_eval_batch_size=256,
    weight_decay=1e-5,
    evaluation_strategy="steps",
    max_steps=5000,
    eval_steps=20,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=1,
    metric_for_best_model="nonzero_iou",
    load_best_model_at_end=True,
    logging_steps = 1,
    # push_to_hub=True,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=collator,
    compute_metrics=compute_metrics,
)
trainer.train()

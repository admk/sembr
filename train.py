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


def binary_metrics(fp, tp, fn, tn=None, prefix=None):
    def safe_div(a, b):
        return a / b if b > 0 else 0
    result = {
        'precision': safe_div(tp, tp + fp),
        'recall': safe_div(tp, tp + fn),
        'f1': safe_div(2 * tp, 2 * tp + fp + fn),
        'iou': safe_div(tp, tp + fp + fn),
    }
    if tn is not None:
        accuracies = {
            'accuracy': safe_div(tp + tn, tn + fn + tp + fp),
            'balanced_accuracy': (
                safe_div(tp, tp + fn) + safe_div(tn, tn + fp)) / 2,
        }
        result.update(accuracies)
    if prefix is not None:
        result = {f'{prefix}_{k}': v for k, v in result.items()}
    return result


def compute_metrics(result):
    logits, labels = result
    preds = np.argmax(logits, axis=2)
    idx = labels != -100
    logits, preds, labels = logits[idx], preds[idx], labels[idx]
    loss = torch.nn.functional.cross_entropy(
        torch.Tensor(logits), torch.LongTensor(labels), reduction='mean')
    preds_set = set(preds.nonzero()[0])
    labels_set = set(labels.nonzero()[0])
    fp = len(preds_set - labels_set)
    tp = len(labels_set & preds_set)
    fn = len(labels_set - preds_set)
    tn = len(set(range(len(preds))) - (labels_set | preds_set))
    metrics = {
        **binary_metrics(fp, tp, fn, tn),
        'overall_accuracy': (preds == labels).mean(),
        'loss': loss,
    }
    return metrics


dataset = datasets.load_dataset('./sembr/sembr2023.py', 'sembr2023')
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

run_name = 'sembr2023-distilbert-base-uncased'
training_args = TrainingArguments(
    output_dir=f'checkpoints/{run_name}',
    run_name=run_name,
    learning_rate=1e-5,
    lr_scheduler_type='cosine',
    per_device_train_batch_size=64,
    # per_device_eval_batch_size=256,
    weight_decay=1e-5,
    evaluation_strategy="steps",
    max_steps=3000,
    eval_steps=10,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=1,
    metric_for_best_model="f1",
    load_best_model_at_end=True,
    logging_steps=1,
    push_to_hub=True,
    hub_strategy='end',
    hub_model_id=f'admko/{run_name}',
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
trainer.push_to_hub()

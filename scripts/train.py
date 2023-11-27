import os
import sys
import functools

import torch
import numpy as np
import datasets
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    TrainingArguments, Trainer)

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from sembr.process import (
    SemBrProcessor, DataCollatorForTokenClassificationWithTruncation)


def _process_examples(examples, processor, tokenizer, mode_names):
    examples['modes'] = [
        [mode_names[i] for i in bm] for bm in examples['modes']]
    transposed = [dict(zip(examples, c)) for c in zip(*examples.values())]
    results = processor.tokenize_with_modes(tokenizer, transposed)
    for r in results:
        r['labels'] = labels = []
        indents = []
        for m, i in zip(r.pop('modes'), r.pop('indents')):
            i = min(i, MAX_INDENT)
            indents.append(i)
            if m == 'off':
                label = 'off'
            else:
                label = f'{m}-{i}'
            labels.append(LABEL2ID[label])
        r.pop('base_indent')
    keys = ['input_ids', 'labels']
    return {k: [d[k] for d in results] for k in keys}


def chunk_examples(examples):
    id_chunks, label_chunks = [], []
    max_length = 512
    overlap = int(max_length / 10)
    for ids, labels in zip(examples.pop('input_ids'), examples.pop('labels')):
        id_chunks += [
            ids[i:i + max_length] for i in range(0, len(ids), overlap)]
        label_chunks += [
            labels[i:i + max_length] for i in range(0, len(labels), overlap)]
    return {'input_ids': id_chunks, 'labels': label_chunks}


def process_dataset(dataset, processor, tokenizer):
    mode_names = dataset.features['modes'].feature.names
    removed_columns = [
        'flat_lines', 'modes', 'mode_offsets', 'indents', 'base_indent']
    process_examples = functools.partial(
        _process_examples,
        processor=processor, tokenizer=tokenizer, mode_names=mode_names)
    dataset = dataset.map(
        process_examples, batched=True, remove_columns=removed_columns)
    dataset = dataset.map(chunk_examples, batched=True)
    return dataset


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


def init_dataset(args):
    dataset = datasets.load_dataset('./sembr/sembr2023.py', 'sembr2023')
    processor = SemBrProcessor()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    processor.prepare_tokenizer(tokenizer)
    train_dataset = process_dataset(dataset['train'], processor, tokenizer)
    test_dataset = process_dataset(dataset['test'], processor, tokenizer)
    print(f'{len(train_dataset)=}, {len(test_dataset)=}')
    collator = DataCollatorForTokenClassificationWithTruncation(
        tokenizer, padding='max_length', max_length=512)
    return train_dataset, test_dataset, tokenizer, collator


def init_labels(max_indent):
    label_names = ['off'] + [
        f'{m}-{i}' for m in ['space', 'nospace']
        for i in range(max_indent + 1)]
    id2label = {i: l for i, l in enumerate(label_names)}
    label2id = {l: i for i, l in enumerate(label_names)}
    return id2label, label2id


def main(args):
    train_dataset, test_dataset, tokenizer, collator = init_dataset(args)
    id2label, label2id = init_labels(args.max_indent)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name, num_labels=len(id2label),
        id2label=id2label, label2id=label2id)
    model.config.__dict__['max_indent'] = args.max_indent
    model.resize_token_embeddings(len(tokenizer))
    if args.classifier_only:
        for n, p in model.named_parameters():
            if 'classifier' not in n:
                p.requires_grad = False
    run_name = f'sembr2023-{args.model_name}'
    training_args = TrainingArguments(
        output_dir=f'checkpoints/{run_name}',
        run_name=run_name,
        learning_rate=args.learning_rate,
        lr_scheduler_type='cosine',
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        weight_decay=1e-5,
        evaluation_strategy='steps',
        max_steps=args.max_steps,
        eval_steps=args.eval_steps,
        save_strategy='steps',
        save_steps=args.save_steps,
        save_total_limit=1,
        metric_for_best_model='f1',
        load_best_model_at_end=True,
        logging_steps=1,
        push_to_hub=args.hub_user is not None,
        hub_strategy='end',
        hub_model_id=f'{args.hub_user}/{run_name}',
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
    if args.hub_user is not None:
        trainer.push_to_hub()


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model-name', type=str)
    parser.add_argument('-co', '--classifier-only', action='store_true')
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-5)
    parser.add_argument('-tb', '--train-batch-size', type=int, default=64)
    parser.add_argument('-eb', '--eval-batch-size', type=int, default=128)
    parser.add_argument('-mi', '--max-indent', type=int, default=3)
    parser.add_argument('-hu', '--hub-user', type=str, default=None)
    parser.add_argument('-ms', '--max-steps', type=int, default=3000)
    parser.add_argument('-es', '--eval-steps', type=int, default=100)
    parser.add_argument('-ss', '--save-steps', type=int, default=100)
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())

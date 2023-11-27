import os
import sys
import argparse
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import datasets
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    TrainingArguments, Trainer)

from .process import (
    SemBrProcessor, DataCollatorForTokenClassificationWithTruncation)
from .dataset import process_dataset
from .utils import compute_metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-5)
    parser.add_argument('-tb', '--train-batch-size', type=int, default=64)
    parser.add_argument('-eb', '--eval-batch-size', type=int, default=128)
    parser.add_argument('-mi', '--max-indent', type=int, default=3)
    parser.add_argument('-hu', '--hub-user', type=str, default=None)
    parser.add_argument('-ms', '--max-steps', type=int, default=5000)
    parser.add_argument('-es', '--eval-steps', type=int, default=10)
    parser.add_argument('-ss', '--save-steps', type=int, default=100)
    parser.add_argument('-rt', '--report-to', type=str, default='all')
    parser.add_argument('-d', '--debug', action='store_true')
    args = parser.parse_args()
    if args.debug:
        import debugpy
        debugpy.listen(5678)
        print('Waiting for debugger...')
        debugpy.wait_for_client()
        args.report_to = 'none'
        args.hub_user = None
    return args


def init_dataset(args, label2id, max_length):
    dataset = datasets.load_dataset('./sembr/sembr2023.py', 'sembr2023')
    processor = SemBrProcessor()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    processor.prepare_tokenizer(tokenizer)
    train_dataset = process_dataset(
        dataset['train'], processor, tokenizer, args.max_indent, label2id)
    test_dataset = process_dataset(
        dataset['test'], processor, tokenizer, args.max_indent, label2id)
    print(f'{len(train_dataset)=}, {len(test_dataset)=}')
    collator = DataCollatorForTokenClassificationWithTruncation(
        tokenizer, padding='max_length', max_length=max_length)
    return train_dataset, test_dataset, tokenizer, collator


def init_model(model_name, max_indent):
    label_names = ['off'] + [
        f'{m}-{i}' for m in ['space', 'nospace']
        for i in range(max_indent + 1)]
    id2label = {i: l for i, l in enumerate(label_names)}
    label2id = {l: i for i, l in enumerate(label_names)}
    model = AutoModelForTokenClassification.from_pretrained(
        model_name, ignore_mismatched_sizes=True,
        num_labels=len(id2label), id2label=id2label, label2id=label2id)
    return model


def main(args):
    model = init_model(args.model, args.max_indent)
    max_length = model.config.max_position_embeddings
    train_dataset, test_dataset, tokenizer, collator = \
        init_dataset(args, model.config.label2id, max_length)
    model.config.__dict__['max_indent'] = args.max_indent
    model.resize_token_embeddings(len(tokenizer))
    model_name = args.model.split('/')[-1]
    run_name = f'sembr2023-{model_name}'
    training_args = TrainingArguments(
        output_dir=f'checkpoints/{run_name}',
        run_name=run_name,
        report_to=args.report_to,
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


if __name__ == '__main__':
    main(parse_args())

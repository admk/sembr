import os
import argparse
import functools

import datasets


def _process_examples(
    examples, processor, tokenizer, mode_names, max_indent, label2id
):
    examples['modes'] = [
        [mode_names[i] for i in bm] for bm in examples['modes']]
    transposed = [dict(zip(examples, c)) for c in zip(*examples.values())]
    results = processor.tokenize_with_modes(tokenizer, transposed)
    for r in results:
        r['labels'] = labels = []
        indents = []
        for m, i in zip(r.pop('modes'), r.pop('indents')):
            i = min(i, max_indent)
            indents.append(i)
            if m == 'off':
                label = 'off'
            else:
                label = f'{m}-{i}'
            labels.append(label2id[label])
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


def process_dataset(dataset, processor, tokenizer, max_indent, label2id):
    mode_names = dataset.features['modes'].feature.names
    removed_columns = [
        'flat_lines', 'modes', 'mode_offsets', 'indents', 'base_indent']
    process_examples = functools.partial(
        _process_examples,
        processor=processor, tokenizer=tokenizer, mode_names=mode_names,
        max_indent=max_indent, label2id=label2id)
    dataset = dataset.map(
        process_examples, batched=True, remove_columns=removed_columns)
    dataset = dataset.map(chunk_examples, batched=True)
    return dataset


def init_dataset():
    dataset_file = os.path.join(os.path.dirname(__file__), 'databuilder.py')
    return datasets.load_dataset(dataset_file)

def push_to_hub(dataset, hub_user, dataset_name, private):
    hub_path = f'{hub_user}/{dataset_name}'
    print(f'Pushing dataset to {hub_path}...')
    dataset.push_to_hub(hub_path, private=private)


def save_to_disk(dataset, save_dir, dataset_name):
    save_path = os.path.join(save_dir, dataset_name)
    print(f'Saving dataset to {save_path}...')
    dataset.save_to_disk(save_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--dataset-name', type=str, default='sembr')
    parser.add_argument('-s', '--save-dir', type=str, default='data')
    parser.add_argument('-u', '--hub-user', type=str, default=None)
    parser.add_argument('-p', '--private', action='store_true')
    return parser.parse_args()


def main(args=None):
    if args is None:
        args = parse_args()
    dataset = init_dataset()
    if args.save_dir is not None:
        save_to_disk(dataset, args.save_dir, args.dataset_name)
    if args.hub_user is not None:
        push_to_hub(dataset, args.hub_user, args.dataset_name, args.private)


if __name__ == '__main__':
    main()

import functools


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

import torch
from tqdm import trange

from transformers import DataCollatorForTokenClassification


def _tiled_inference(model, collator, results, batch_size, overlap_divisor):
    device = model.device
    max_length = model.config.max_position_embeddings
    overlap_length = int(max_length / overlap_divisor)
    input_ids = [{'input_ids': r['input_ids']} for r in results]
    num_paras = len(input_ids)
    lengths = [len(i['input_ids']) for i in input_ids]
    sorted_indices = sorted(
        range(num_paras), key=lambda i: lengths[i], reverse=True)
    logits = torch.zeros(
        (num_paras, max(lengths), model.config.num_labels), device=device)
    counts = torch.zeros(
        (num_paras, max(lengths)), dtype=torch.long, device=device)
    for b in trange(0, num_paras, batch_size):
        bslice = slice(b, min(num_paras, b + batch_size))
        bindices = sorted_indices[bslice]
        binids = [input_ids[i] for i in bindices]
        data = collator(binids, return_tensors='pt').to(device)
        num_tokens = data['input_ids'].shape[1]
        for i in range(0, num_tokens, max_length - overlap_length):
            islice = slice(i, min(num_tokens, i + max_length))
            inids = data['input_ids'][:, islice]
            attns = data['attention_mask'][:, islice]
            with torch.no_grad():
                outputs = model(
                    input_ids=inids, attention_mask=attns, return_dict=True)
            logits[bindices, islice] += outputs.logits
            counts[bindices, islice] += attns
    attns = counts > 0
    logits /= counts.unsqueeze(-1)
    logits[~attns] = 0
    return logits, attns


def _format_labels(id2label, preds, attns, results):
    modes, indents = [], []
    for i, (p, a) in enumerate(zip(preds, attns)):
        para_modes, para_indents = [], []
        for name in [id2label[int(t)] for t in p[a]]:
            if name == 'off':
                mode, indent = 'off', 0
            else:
                mode, indent = name.split('-')
            para_modes.append(mode)
            para_indents.append(int(indent))
        modes.append(para_modes)
        indents.append(para_indents)
    for r, m, i in zip(results, modes, indents):
        r['modes'] = m
        r['indents'] = i
    return results


def predict_argmax(logits, counts, tokens_per_line):
    return logits.argmax(dim=2)


def predict_logit_adjustment(logits, counts, tokens_per_line):
    delta = 1.0
    logits[:, :, 0] -= delta
    logits[:, :, 1:] += delta / logits.shape[2]
    return logits.argmax(dim=2)


def predict_greedy_linebreaks(logits, counts, tokens_per_line):
    if tokens_per_line is None:
        return logits.argmax(dim=2)
    has_long_lines = True
    while has_long_lines:
        has_long_lines = False
        for b in range(logits.shape[0]):
            modes, repeats = torch.unique_consecutive(
                logits[b].argmax(dim=1), return_counts=True)
            stops = torch.cumsum(repeats, dim=0)
            starts = torch.cat([
                torch.zeros(1, device=stops.device, dtype=stops.dtype),
                stops[:-1]])
            long_lines = (modes == 0) & (repeats > tokens_per_line)
            if long_lines.any():
                has_long_lines = True
            starts, stops = starts[long_lines], stops[long_lines]
            for s, e in zip(starts, stops):
                # find the token with the lowest "off" logit
                max_index = logits[b, s:e, 0].argmin(0)
                # force linebreak by reducing "off" by -1e6
                logits[b, s + max_index, 0] -= 1e6
    return logits.argmax(dim=2)


PREDICT_FUNC_MAP = {
    'argmax': predict_argmax,
    'logit_adjustment': predict_logit_adjustment,
    'greedy_linebreaks': predict_greedy_linebreaks,
}


def inference(
    text, tokenizer, model, processor,
    predict_func='argmax', tokens_per_line=10,
    batch_size=8, overlap_divisor=8,
):
    if text.strip() == '':
        return []
    collator = DataCollatorForTokenClassification(tokenizer, padding='longest')
    results = processor.parse_text(text, split=isinstance(text, str))
    results = processor.tokenize_with_modes(tokenizer, results)
    logits, counts = _tiled_inference(
        model, collator, results, batch_size, overlap_divisor)
    preds = PREDICT_FUNC_MAP[predict_func](logits, counts, tokens_per_line)
    return _format_labels(model.config.id2label, preds, counts, results)


def sembr(
    text, tokenizer, model, processor,
    predict_func='argmax', tokens_per_line=10,
    batch_size=8, overlap_divisor=8,
):
    results = inference(
        text, tokenizer, model, processor, predict_func, tokens_per_line,
        batch_size, overlap_divisor)
    return processor.generate(results, join=True)

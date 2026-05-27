from collections import deque

import torch
from tqdm import trange


DEFAULT_LENGTH_LOSS_WEIGHT = 0.05


def _parse_tokens_per_line(tokens_per_line):
    if tokens_per_line is None:
        return None
    length_loss_weight = DEFAULT_LENGTH_LOSS_WEIGHT
    if isinstance(tokens_per_line, str):
        tokens_per_line = tokens_per_line.strip()
        if '@' in tokens_per_line:
            tokens_per_line, length_loss_weight = tokens_per_line.split('@', 1)
            length_loss_weight = float(length_loss_weight)
        if ':' in tokens_per_line:
            lower, upper = tokens_per_line.split(':', 1)
            return int(lower), int(upper), length_loss_weight
        return int(tokens_per_line), int(tokens_per_line), length_loss_weight
    if isinstance(tokens_per_line, (tuple, list)):
        if len(tokens_per_line) not in (2, 3):
            raise ValueError(
                'tokens_per_line range must have two or three values.')
        if len(tokens_per_line) == 3:
            length_loss_weight = float(tokens_per_line[2])
        return int(tokens_per_line[0]), int(tokens_per_line[1]), length_loss_weight
    tokens_per_line = int(tokens_per_line)
    return tokens_per_line, tokens_per_line, length_loss_weight


def _line_length_loss(length, lower, upper, weight=1.0):
    if lower <= length <= upper:
        return 0.0
    if length < lower:
        return float(weight * (lower - length) ** 2)
    return float(weight * (length - upper) ** 2)


def _boundary_loss(start, off_logits, break_logits):
    if start == 0:
        return 0.0
    return float(off_logits[start].item() - break_logits[start].item())


class _LiChaoLine:
    def __init__(self, slope, intercept, index):
        self.slope = slope
        self.intercept = intercept
        self.index = index

    def value(self, x):
        return self.slope * x + self.intercept


class _LiChaoNode:
    def __init__(self):
        self.line = None
        self.left = None
        self.right = None


class _LiChaoMin:
    def __init__(self, x_left, x_right):
        self.x_left = x_left
        self.x_right = x_right
        self.root = _LiChaoNode()
        self.size = 0

    def add_line(self, line):
        self.size += 1
        self._add_line(self.root, self.x_left, self.x_right, line)

    def _add_line(self, node, left, right, line):
        if node.line is None:
            node.line = line
            return

        mid = (left + right) // 2
        current = node.line
        if line.value(mid) < current.value(mid):
            node.line, line = line, current
            current = node.line
        if left == right:
            return

        if line.value(left) < current.value(left):
            if node.left is None:
                node.left = _LiChaoNode()
            self._add_line(node.left, left, mid, line)
        elif line.value(right) < current.value(right):
            if node.right is None:
                node.right = _LiChaoNode()
            self._add_line(node.right, mid + 1, right, line)

    def query(self, x):
        if self.size == 0:
            return float('inf'), -1
        return self._query(self.root, self.x_left, self.x_right, x)

    def _query(self, node, left, right, x):
        if node is None:
            return float('inf'), -1

        best = (node.line.value(x), node.line.index)
        if left == right:
            return best

        mid = (left + right) // 2
        if x <= mid:
            child_best = self._query(node.left, left, mid, x)
        else:
            child_best = self._query(node.right, mid + 1, right, x)
        if child_best[0] < best[0]:
            return child_best
        return best


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
    line_range = _parse_tokens_per_line(tokens_per_line)
    if line_range is None:
        return logits.argmax(dim=2)
    _, tokens_per_line, _ = line_range
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


def predict_balanced_linebreaks(logits, counts, tokens_per_line):
    line_range = _parse_tokens_per_line(tokens_per_line)
    if line_range is None:
        return logits.argmax(dim=2)
    lower, upper, length_loss_weight = line_range
    if lower < 1 or upper < lower or length_loss_weight < 0:
        raise ValueError(
            'tokens_per_line must be positive, or a range like "8:12@0.05".')

    preds = torch.zeros(
        logits.shape[:2], dtype=torch.long, device=logits.device)
    for b in range(logits.shape[0]):
        num_tokens = int(counts[b].sum().item())
        if num_tokens == 0:
            continue

        row_logits = logits[b, :num_tokens]
        break_logits, break_labels = row_logits[:, 1:].max(dim=1)
        break_labels += 1
        off_logits = row_logits[:, 0]
        off_prefix = torch.cat([
            off_logits.new_zeros(1),
            torch.cumsum(off_logits, dim=0),
        ])

        dp = [float('inf')] * (num_tokens + 1)
        prev = [-1] * (num_tokens + 1)
        values = [float('inf')] * (num_tokens + 1)
        range_min = deque()
        long_hull = _LiChaoMin(0, max(0, num_tokens - upper))
        dp[0] = 0.0
        values[0] = 0.0
        for end in range(1, num_tokens + 1):
            range_start = end - lower
            if range_start >= 0:
                while (
                    range_min
                    and values[range_min[-1]] > values[range_start]
                ):
                    range_min.pop()
                range_min.append(range_start)

            long_start = end - upper - 1
            if long_start >= 0:
                line = _LiChaoLine(
                    -2.0 * length_loss_weight * long_start,
                    values[long_start] + length_loss_weight * long_start ** 2,
                    long_start,
                )
                long_hull.add_line(line)

            range_stop = end - upper
            while range_min and range_min[0] < range_stop:
                range_min.popleft()

            candidates = []
            for start in range(max(0, end - lower + 1), end):
                length = end - start
                loss = values[start] + _line_length_loss(
                    length, lower, upper, length_loss_weight)
                candidates.append((loss, start))

            if range_min:
                start = range_min[0]
                candidates.append((values[start], start))

            if long_hull.size:
                x = end - upper
                hull_value, start = long_hull.query(x)
                candidates.append((
                    hull_value + length_loss_weight * x ** 2,
                    start,
                ))

            best_value, best_start = min(candidates)
            dp[end] = best_value - float(off_prefix[end].item())
            prev[end] = best_start
            values[end] = dp[end] + float(off_prefix[end].item())
            if end < num_tokens:
                values[end] += _boundary_loss(end, off_logits, break_logits)

        start = prev[num_tokens]
        while start > 0:
            preds[b, start] = break_labels[start]
            start = prev[start]

    return preds


PREDICT_FUNC_MAP = {
    'argmax': predict_argmax,
    'logit_adjustment': predict_logit_adjustment,
    'greedy_linebreaks': predict_greedy_linebreaks,
    'balanced_linebreaks': predict_balanced_linebreaks,
}


def inference(
    text, tokenizer, model, processor,
    predict_func='argmax', tokens_per_line=10,
    batch_size=8, overlap_divisor=8,
):
    if text.strip() == '':
        return []
    from transformers import DataCollatorForTokenClassification

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

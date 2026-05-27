import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sembr.inference import (
    _line_length_loss,
    predict_balanced_linebreaks,
)


def test_balanced_linebreaks_uses_best_break_label():
    logits = torch.zeros((1, 8, 3))
    logits[:, :, 0] = 1.0
    logits[:, :, 1] = -1.0
    logits[:, :, 2] = -1.0
    logits[0, 4, 2] = 4.0
    counts = torch.ones((1, 8), dtype=torch.long)

    preds = predict_balanced_linebreaks(logits, counts, '4')

    assert preds.tolist() == [[0, 0, 0, 0, 2, 0, 0, 0]]


def test_balanced_linebreaks_accepts_line_length_range():
    logits = torch.zeros((1, 9, 2))
    logits[:, :, 0] = 1.0
    logits[:, :, 1] = -1.0
    counts = torch.ones((1, 9), dtype=torch.long)

    preds = predict_balanced_linebreaks(logits, counts, '4:5@1.0')

    assert preds.tolist() == [[0, 0, 0, 0, 1, 0, 0, 0, 0]]


def _predict_balanced_linebreaks_brute(logits, counts, tokens_per_line):
    lower, upper, weight = tokens_per_line
    preds = torch.zeros(logits.shape[:2], dtype=torch.long)
    for b in range(logits.shape[0]):
        num_tokens = int(counts[b].sum().item())
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
        dp[0] = 0.0
        for end in range(1, num_tokens + 1):
            for start in range(0, end):
                label_loss = -float((off_prefix[end] - off_prefix[start]).item())
                if start > 0:
                    label_loss += float(off_logits[start].item())
                    label_loss -= float(break_logits[start].item())
                loss = dp[start] + label_loss + _line_length_loss(
                    end - start, lower, upper, weight)
                if loss < dp[end]:
                    dp[end] = loss
                    prev[end] = start

        start = prev[num_tokens]
        while start > 0:
            preds[b, start] = break_labels[start]
            start = prev[start]
    return preds


def test_balanced_linebreaks_matches_brute_force_small_cases():
    generator = torch.Generator().manual_seed(1)
    logits = torch.randn((4, 12, 4), generator=generator)
    counts = torch.ones((4, 12), dtype=torch.long)

    preds = predict_balanced_linebreaks(logits, counts, (3, 5, 0.05))
    expected = _predict_balanced_linebreaks_brute(logits, counts, (3, 5, 0.05))

    assert preds.tolist() == expected.tolist()


def test_balanced_linebreaks_length_weight_can_be_tuned():
    logits = torch.zeros((1, 12, 2))
    logits[:, :, 0] = 2.0
    logits[:, :, 1] = -2.0
    logits[0, 3, 1] = 5.0
    counts = torch.ones((1, 12), dtype=torch.long)

    soft_preds = predict_balanced_linebreaks(logits, counts, '5:6@0.05')
    hard_preds = predict_balanced_linebreaks(logits, counts, '5:6@1.0')

    assert soft_preds[0, 3].item() == 1
    assert hard_preds[0, 3].item() == 0

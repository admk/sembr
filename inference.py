import sys

import torch
from tqdm import trange
from transformers import AutoModelForTokenClassification

from sembr.process import SemBrProcessor


try:
    file_name = sys.argv[1]
except IndexError:
    file_name = './data/test/mair.tex'

model_name = './runs/checkpoint-3560'
model = AutoModelForTokenClassification.from_pretrained(model_name)
max_length = model.config.max_position_embeddings
overlap_length = int(max_length / 8)

processor = SemBrProcessor()
with open(file_name, 'r', encoding='utf-8') as f:
    results = processor(f.read())
for result in results:
    # sliding window prediction for token length > 512
    num_tokens = len(result['input_ids'])
    modes, indents = [None] * num_tokens, [None] * num_tokens
    indent = 0
    for i in trange(0, num_tokens, max_length - overlap_length):
        input_ids = torch.LongTensor([result['input_ids'][i:i + max_length]])
        outputs = model(input_ids=input_ids, return_dict=True)
        preds = outputs.logits.argmax(dim=2)[0]
        for j, p in enumerate(preds):
            name = model.config.id2label[int(p)]
            if name == 'off':
                mode = 'off'
            else:
                mode, indent = name.split('-')
            cmode = modes[i + j]
            cindent = indents[i + j]
            if cmode is None or cmode == 'off':
                modes[i + j] = mode
            if cindent is None or cindent == 0:
                indents[i + j] = int(indent)
    if any(m is None for m in modes) or any(i is None for i in indents):
        raise ValueError('modes or indents contains Nones.')
    result['modes'] = modes
    result['indents'] = indents
print(processor.generate(results))

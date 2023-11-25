import sys

import torch
from transformers import AutoModelForTokenClassification

from sembr.data.process import SemBrProcessor


try:
    file_name = sys.argv[1]
except IndexError:
    file_name = './data/test/mair.tex'

model_name = './runs/checkpoint-2400'
model = AutoModelForTokenClassification.from_pretrained(model_name)

processor = SemBrProcessor()
with open(file_name, 'r', encoding='utf-8') as f:
    results = processor(f.read())
    for result in results:
        # sliding window prediction for token length > 512
        modes, indents = [], []
        indent = 0
        for i in range(0, len(result['input_ids']), 512):
            input_ids = torch.LongTensor([result['input_ids'][i:i + 512]])
            outputs = model(input_ids=input_ids, return_dict=True)
            preds = outputs.logits.argmax(dim=2)[0]
            for p in preds:
                name = model.config.id2label[int(p)]
                if name == 'off':
                    mode = 'off'
                else:
                    mode, indent = name.split('-')
                modes.append(mode)
                indents.append(int(indent))
        result['modes'] = modes
        result['indents'] = indents
    print(processor.generate(results))

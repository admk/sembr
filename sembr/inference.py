import torch
from tqdm import tqdm


def inference(text, tokenizer, model, processor):
    max_length = model.config.max_position_embeddings
    overlap_length = int(max_length / 8)
    results = processor(text)
    results = processor.tokenize_with_modes(tokenizer, results)
    for result in tqdm(results):
        # sliding window prediction for token length > 512
        num_tokens = len(result['input_ids'])
        modes, indents = [None] * num_tokens, [None] * num_tokens
        indent = 0
        for i in range(0, num_tokens, max_length - overlap_length):
            input_ids = [result['input_ids'][i:i + max_length]]
            input_ids = torch.tensor(
                input_ids, dtype=torch.long, device=model.device)
            with torch.no_grad():
                outputs = model(input_ids=input_ids, return_dict=True)
            preds = outputs.logits.argmax(dim=2)[0]
            for j, p in enumerate(preds):
                name = model.config.id2label[int(p)]
                if name == 'off':
                    mode = 'off'
                else:
                    mode, indent = name.split('-')
                cmode = modes[i + j]
                if cmode is None or cmode == 'off':
                    modes[i + j] = mode
                    indents[i + j] = int(indent)
        if any(m is None for m in modes) or any(i is None for i in indents):
            raise ValueError('modes or indents contains Nones.')
        result['modes'] = modes
        result['indents'] = indents
    return processor.generate(results)

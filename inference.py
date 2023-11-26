import argparse

import requests
from flask import Flask, request

import torch
from tqdm import tqdm
from transformers import AutoModelForTokenClassification

from sembr.process import SemBrProcessor


model_name = './checkpoints/sembr2023-distilbert-base-uncased-full/checkpoint-2800'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model-name', type=str, default=model_name)
    parser.add_argument('-i', '--input-file', type=str, default=None)
    parser.add_argument('-o', '--output-file', type=str, default=None)
    parser.add_argument('-w', '--words-per-line', type=int, default=10)
    parser.add_argument('-s', '--server', type=str, default='localhost:5000')
    parser.add_argument('-d', '--host-daemon', action='store_true')
    parser.add_argument('-p', '--port', type=int, default=5000)
    return parser.parse_args()


def init(model_name):
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    model.eval()
    if torch.cuda.is_available():
        model = model.to('cuda')
    elif torch.backends.mps.is_available():
        model = model.to('mps')
    processor = SemBrProcessor()
    return model, processor


def process(text, model, processor):
    max_length = model.config.max_position_embeddings
    overlap_length = int(max_length / 8)
    results = processor(text)
    for result in tqdm(results):
        # sliding window prediction for token length > 512
        num_tokens = len(result['input_ids'])
        modes, indents = [None] * num_tokens, [None] * num_tokens
        indent = 0
        for i in range(0, num_tokens, max_length - overlap_length):
            input_ids = [result['input_ids'][i:i + max_length]]
            input_ids = torch.tensor(
                input_ids, dtype=torch.long, device=model.device)
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


def rewrap(input_file, output_file, model, processor):
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    results = process(text, model, processor)
    if output_file is None:
        return results
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(results)


def start_server(port, model, processor):
    app = Flask(__name__)

    @app.route('/check')
    def check():
        return 'OK'

    @app.route('/rewrap', methods=['POST'])
    def rewrap():
        text = request.form['text']
        results = process(text, model, processor)
        return results

    app.run(port=port)


def check_server(server):
    if not server:
        return False
    try:
        status = requests.get(f'http://{server}/check', timeout=1)
    except requests.exceptions.ConnectionError:
        return False
    if status.status_code != 200 or status.text != 'OK':
        return False
    return True


def rewrap_on_server(input_file, output_file, server):
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    results = requests.post(
        f'http://{server}/rewrap', data={'text': text})
    if results.status_code != 200:
        raise ValueError(f'Error {results.status_code}: {results.text}')
    results = results.text
    if output_file is None:
        return results
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(results)


def main():
    args = parse_args()
    if check_server(args.server):
        return rewrap_on_server(args.input_file, args.output_file, args.server)
    model, processor = init(args.model_name)
    if args.host_daemon:
        return start_server(args.port, model, processor)
    return rewrap(args.input_file, args.output_file, model, processor)


if __name__ == '__main__':
    result = main()
    if result is not None:
        print(result)

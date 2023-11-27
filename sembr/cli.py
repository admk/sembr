import os
import sys
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    model_name = 'admko/sembr2023-bert-small'
    parser.add_argument('-m', '--model-name', type=str, default=model_name)
    parser.add_argument('-i', '--input-file', type=str, default=None)
    parser.add_argument('-o', '--output-file', type=str, default=None)
    parser.add_argument('-w', '--words-per-line', type=int, default=10)
    parser.add_argument('-s', '--server', type=str, default='127.0.0.1:5000')
    parser.add_argument('-l', '--listen', action='store_true')
    parser.add_argument('-p', '--port', type=int, default=5000)
    return parser.parse_args()


def init(model_name):
    import torch
    from transformers import (AutoTokenizer, AutoModelForTokenClassification)
    from .process import SemBrProcessor
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    model.eval()
    if torch.cuda.is_available():
        model = model.to('cuda')
    elif torch.backends.mps.is_available():
        model = model.to('mps')
    processor = SemBrProcessor()
    return tokenizer, model, processor



def start_server(port, tokenizer, model, processor):
    from flask import Flask, request

    app = Flask(__name__)

    @app.route('/check')
    def check():
        return 'OK'

    @app.route('/rewrap', methods=['POST'])
    def rewrap():
        from .inference import inference
        text = request.form['text']
        results = inference(text, tokenizer, model, processor)
        return results

    app.run(port=port)


def check_server(server):
    import requests
    if not server:
        return False
    try:
        status = requests.get(f'http://{server}/check', timeout=1)
    except requests.exceptions.ConnectionError:
        return False
    if status.status_code != 200 or status.text != 'OK':
        return False
    return True


def rewrap_on_server(text, server):
    import requests
    results = requests.post(
        f'http://{server}/rewrap', data={'text': text})
    if results.status_code != 200:
        raise ValueError(f'Error {results.status_code}: {results.text}')
    return results.text


def main(args):
    if args.listen:
        tokenizer, model, processor = init(args.model_name)
        return start_server(args.port, tokenizer, model, processor)
    if args.input_file is not None:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        text = sys.stdin.read()
    if check_server(args.server):
        result = rewrap_on_server(text, args.server)
    else:
        from .inference import inference
        tokenizer, model, processor = init(args.model_name)
        result = inference(text, tokenizer, model, processor)
    if args.output_file is None:
        print(result)
        return
    with open(args.output_file, 'w', encoding='utf-8') as f:
        f.write(result)


if __name__ == '__main__':
    main(parse_args())

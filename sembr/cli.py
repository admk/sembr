import os
import sys
import traceback


os.environ['TOKENIZERS_PARALLELISM'] = 'false'
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

STDIN_TTY = os.isatty(sys.stdin.fileno())
STDOUT_TTY = os.isatty(sys.stdout.fileno())
STDERR_TTY = os.isatty(sys.stderr.fileno())


def cli_parser():
    import argparse
    p = argparse.ArgumentParser(
        description='SemBr: Rewrap text with semantic breaks.')
    model_name = 'admko/sembr2023-bert-small'
    p.add_argument('-m', '--model-name', type=str, default=model_name)
    p.add_argument('-i', '--input-file', type=str, default=None)
    p.add_argument('-o', '--output-file', type=str, default=None)
    p.add_argument('-w', '--words-per-line', type=int, default=10)
    p.add_argument('-b', '--batch-size', type=int, default=8)
    p.add_argument('-d', '--overlap-divisor', type=int, default=8)
    p.add_argument(
        '-f', '--predict-func', type=str,
        choices=['argmax', 'breaks_first', 'logit_adjustment'],
        default='argmax')
    p.add_argument('-t', '--tokens-per-line', type=int, default=10)
    p.add_argument('-s', '--server', type=str, default='127.0.0.1')
    p.add_argument('-l', '--listen', action='store_true')
    p.add_argument('-p', '--port', type=int, default=8384)
    return p


def init(model_name):
    import torch
    from transformers import AutoTokenizer, AutoModelForTokenClassification
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
    base_rv = {
        'model': model.__class__.__name__,
        'tokenizer': tokenizer.__class__.__name__,
        'processor': processor.__class__.__name__,
    }

    @app.route('/check')
    def check():
        return {
            'status': 'success',
            **base_rv,
        }

    @app.route('/rewrap', methods=['POST'])
    def rewrap():
        from .inference import sembr
        form = request.form
        text = form['text']
        kwargs = {
            'batch_size': int(form.get('batch_size', 8)),
            'predict_func': form.get('predict_func', 'argmax'),
            'tokens_per_line': int(form.get('tokens_per_line', 10)),
            'overlap_divisor': int(form.get('overlap_divisor', 8)),
        }
        try:
            results = sembr(text, tokenizer, model, processor, **kwargs)
            return {
                'status': 'success',
                **base_rv,
                **kwargs,
                'text': results,
            }
        except Exception as e:
            return {
                'status': 'error',
                **base_rv,
                **kwargs,
                'error': str(e),
                'traceback': traceback.format_exc(),
            }

    app.run(port=port)


def _fetch(server, port, endpoint, method='get', data=None, timeout=None):
    import requests
    from requests.exceptions import ConnectionError, ReadTimeout
    try:
        results = getattr(requests, method.lower())(
            f'http://{server}:{port}/{endpoint}', data=data, timeout=timeout)
    except (ConnectionError, ReadTimeout) as e:
        raise RuntimeError(f'Connection Error: {e}')
    if results.status_code != 200:
        raise RuntimeError(
            f'Connection Error: {results.status_code}: {results.text}')
    data = results.json()
    if data['status'] != 'success':
        raise RuntimeError(
            f'Status: {data["status"]}\n'
            f'Exception: {data["error"]}\n'
            f'Traceback: {data.get("traceback")}')
    return data


def check_server(server, port):
    if not server:
        return False
    try:
        _fetch(server, port, 'check', timeout=0.3)
    except RuntimeError:
        return False
    return True


def rewrap_on_server(text, server, port, kwargs):
    data = {'text': text, **kwargs}
    response = _fetch(server, port, 'rewrap', 'post', data)
    return response['text']


def main():
    parser = cli_parser()
    args = parser.parse_args()
    if args.listen:
        tokenizer, model, processor = init(args.model_name)
        return start_server(args.port, tokenizer, model, processor)
    if args.input_file is not None:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            text = f.read()
    elif not STDIN_TTY:
        text = sys.stdin.read()
    else:
        parser.print_help()
        print('\nNo input file or stdin text provided.')
        return
    kwargs = {
        'batch_size': args.batch_size,
        'predict_func': args.predict_func,
        'tokens_per_line': args.tokens_per_line,
        'overlap_divisor': args.overlap_divisor,
    }
    if check_server(args.server, args.port):
        result = rewrap_on_server(text, args.server, args.port, kwargs)
    else:
        from .inference import sembr
        tokenizer, model, processor = init(args.model_name)
        result = sembr(text, tokenizer, model, processor, **kwargs)
    if args.output_file is None:
        print(result)
        return
    with open(args.output_file, 'w', encoding='utf-8') as f:
        f.write(result)


if __name__ == '__main__':
    main()

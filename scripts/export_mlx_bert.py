from pathlib import Path
import sys

from sembr.mlx_backend import (
    EXPORT_FORMAT,
    EXPORT_METADATA_FILE,
    EXPORT_VERSION,
    WEIGHTS_FILE,
    _dtype_from_name,
    _load_config,
    _load_safetensors_weights,
    MlxBertForTokenClassification,
)


def _save_pretrained_config_and_tokenizer(model_name, output_dir):
    from transformers import AutoConfig, AutoTokenizer

    try:
        config = AutoConfig.from_pretrained(model_name)
    except Exception:
        config = AutoConfig.from_pretrained(model_name, local_files_only=True)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, local_files_only=True)
    config.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def _flatten_exported_weights(weights, *, quantization, dtype_name):
    arrays = {}
    quantized_weights = {}
    for key, value in weights.items():
        if isinstance(value, tuple):
            qweight, scales, biases = value
            qweight_key = f'{key}.qweight'
            scales_key = f'{key}.scales'
            biases_key = f'{key}.biases'
            arrays[qweight_key] = qweight
            arrays[scales_key] = scales
            metadata = {'qweight': qweight_key, 'scales': scales_key}
            if biases is not None:
                arrays[biases_key] = biases
                metadata['biases'] = biases_key
            quantized_weights[key] = metadata
        else:
            arrays[key] = value
    metadata = {
        'format': EXPORT_FORMAT,
        'version': EXPORT_VERSION,
        'quantization': quantization,
        'dtype': dtype_name or 'float32',
        'quantized_weights': quantized_weights,
    }
    return arrays, metadata


def export_mlx_bert_token_classifier(
    model_name,
    output_dir,
    *,
    dtype=None,
    quantization='nvfp4',
    overwrite=False,
):
    import json
    import mlx.core as mx

    if quantization == 'none':
        raise ValueError('Export quantization must not be "none".')
    dtype_obj = _dtype_from_name(mx, dtype)
    config = _load_config(model_name)
    weights = _load_safetensors_weights(model_name, dtype_obj)
    model = MlxBertForTokenClassification(
        config, weights, quantization=quantization)

    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f'{output_dir} already exists and is not empty. '
            'Use --overwrite to replace exported files.')
    output_dir.mkdir(parents=True, exist_ok=True)

    arrays, metadata = _flatten_exported_weights(
        model.weights, quantization=quantization, dtype_name=dtype)
    mx.save_safetensors(str(output_dir / WEIGHTS_FILE), arrays)
    with (output_dir / EXPORT_METADATA_FILE).open('w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
        f.write('\n')
    _save_pretrained_config_and_tokenizer(model_name, output_dir)
    return output_dir


def cli_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description='Export a SemBr BERT model to pre-quantized MLX weights.')
    parser.add_argument(
        'output_dir',
        help='Directory to write the exported MLX model.')
    parser.add_argument(
        '-m', '--model-name',
        default='admko/sembr2023-bert-small',
        help='Hugging Face model name or local model directory.')
    parser.add_argument(
        '-q', '--quantization',
        default='nvfp4',
        choices=['affine', 'mxfp4', 'mxfp8', 'nvfp4'],
        help='MLX quantization mode to export.')
    parser.add_argument(
        '--dtype',
        default=None,
        help='Floating dtype for unquantized weights, e.g. float16.')
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Allow overwriting files in an existing output directory.')
    return parser


def main() -> int:
    parser = cli_parser()
    args = parser.parse_args()
    try:
        output_dir = export_mlx_bert_token_classifier(
            args.model_name,
            args.output_dir,
            dtype=args.dtype,
            quantization=args.quantization,
            overwrite=args.overwrite,
        )
    except Exception as e:
        print(f'Export error: {e}', file=sys.stderr)
        return 1
    print(output_dir)
    return 0


if __name__ == '__main__':
    sys.exit(main())

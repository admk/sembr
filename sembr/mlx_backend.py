from dataclasses import dataclass
import json
from pathlib import Path

from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError

EXPORT_FORMAT = 'sembr-mlx-bert-token-classifier'
EXPORT_VERSION = 1
EXPORT_METADATA_FILE = 'sembr_mlx.json'
WEIGHTS_FILE = 'model.safetensors'

_LINEAR_WEIGHT_SUFFIXES = (
    'attention.self.query.weight',
    'attention.self.key.weight',
    'attention.self.value.weight',
    'attention.output.dense.weight',
    'intermediate.dense.weight',
    'output.dense.weight',
    'classifier.weight',
)


def _dtype_from_name(mx, name):
    if name is None:
        return mx.float32
    try:
        return getattr(mx, name)
    except AttributeError:
        raise ValueError(f'Unsupported MLX dtype: {name!r}.') from None


def _resolve_model_file(model_name, filename, *, local_files_only=False):
    path = Path(model_name)
    if path.exists():
        candidate = path / filename if path.is_dir() else path
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f'Could not find {filename} in {path}.')

    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(
        model_name, filename, local_files_only=local_files_only))


def _load_config(model_name):
    from transformers import AutoConfig
    try:
        return AutoConfig.from_pretrained(model_name)
    except Exception:
        return AutoConfig.from_pretrained(model_name, local_files_only=True)


def _load_safetensors_weights(model_name, dtype):
    import mlx.core as mx

    try:
        model_path = _resolve_model_file(model_name, 'model.safetensors')
    except Exception:
        model_path = _resolve_model_file(
            model_name, 'pytorch_model.bin', local_files_only=True)
        return _load_torch_bin_weights(model_path, dtype)

    weights = mx.load(str(model_path))
    return {
        key: value.astype(dtype) if value.dtype in [mx.float32, mx.float16]
        else value
        for key, value in weights.items()
    }


def _load_torch_bin_weights(path, dtype):
    import mlx.core as mx
    import torch

    state = torch.load(path, map_location='cpu', weights_only=True)
    return {
        key: mx.array(value.numpy()).astype(dtype)
        for key, value in state.items()
    }


def export_metadata_path(model_name):
    paths = export_model_paths(model_name)
    if paths is None:
        return None
    metadata_path, _ = paths
    return metadata_path


def export_model_paths(model_name):
    path = Path(model_name)
    if path.is_dir():
        metadata_path = path / EXPORT_METADATA_FILE
        weights_path = path / WEIGHTS_FILE
        if metadata_path.exists() and weights_path.exists():
            return metadata_path, weights_path
        return None

    try:
        metadata_path = _resolve_model_file(model_name, EXPORT_METADATA_FILE)
    except (EntryNotFoundError, FileNotFoundError, LocalEntryNotFoundError):
        return None

    weights_path = metadata_path.with_name(WEIGHTS_FILE)
    if not weights_path.exists():
        weights_path = _resolve_model_file(model_name, WEIGHTS_FILE)
    return metadata_path, weights_path


def unflatten_exported_weights(arrays, metadata):
    if metadata.get('format') != EXPORT_FORMAT:
        raise ValueError(
            f'Unsupported MLX export format: {metadata.get("format")!r}.')
    if metadata.get('version') != EXPORT_VERSION:
        raise ValueError(
            f'Unsupported MLX export version: {metadata.get("version")!r}.')

    component_keys = set()
    for quantized in metadata.get('quantized_weights', {}).values():
        component_keys.update(quantized.values())

    weights = {
        key: value
        for key, value in arrays.items()
        if key not in component_keys
    }
    for key, quantized in metadata.get('quantized_weights', {}).items():
        weights[key] = (
            arrays[quantized['qweight']],
            arrays[quantized['scales']],
            arrays.get(quantized.get('biases')),
        )
    return weights


def load_exported_weights(model_name):
    paths = export_model_paths(model_name)
    if paths is None:
        return None
    metadata_path, weights_path = paths

    import mlx.core as mx

    with metadata_path.open('r', encoding='utf-8') as f:
        metadata = json.load(f)
    arrays = mx.load(str(weights_path))
    weights = unflatten_exported_weights(arrays, metadata)
    return weights, metadata


def _gelu(mx, x):
    return 0.5 * x * (1.0 + mx.erf(x / 1.4142135623730951))


def _linear(mx, x, weights, name, quantization):
    weight_key = f'{name}.weight'
    bias = weights.get(f'{name}.bias')
    weight = weights[weight_key]
    if quantization != 'none' and weight_key.endswith(_LINEAR_WEIGHT_SUFFIXES):
        weight, scales, biases = weight
        x = mx.quantized_matmul(
            x,
            weight,
            scales=scales,
            biases=biases,
            transpose=True,
            mode=quantization,
        )
    else:
        x = x @ weight.T
    if bias is not None:
        x = x + bias
    return x


def _layer_norm(mx, x, weights, name, eps):
    weight = weights[f'{name}.weight']
    bias = weights[f'{name}.bias']
    mean = mx.mean(x, axis=-1, keepdims=True)
    variance = mx.mean(mx.square(x - mean), axis=-1, keepdims=True)
    return ((x - mean) * mx.rsqrt(variance + eps)) * weight + bias


def _quantize_linear_weights(weights, mode):
    if mode == 'none':
        return weights
    if mode not in ['affine', 'mxfp4', 'mxfp8', 'nvfp4']:
        raise ValueError(f'Unsupported MLX quantization mode: {mode!r}.')

    import mlx.core as mx

    converted = dict(weights)
    for key, value in weights.items():
        if not key.endswith(_LINEAR_WEIGHT_SUFFIXES):
            continue
        quantized = mx.quantize(value, mode=mode)
        if len(quantized) == 2:
            qweight, scales = quantized
            biases = None
        else:
            qweight, scales, biases = quantized
        converted[key] = (qweight, scales, biases)
    return converted


@dataclass
class _MlxBertConfig:
    id2label: dict
    max_position_embeddings: int
    num_labels: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int
    layer_norm_eps: float
    hidden_act: str


class MlxBertForTokenClassification:
    def __init__(
        self, config, weights, *, quantization='none',
        weights_are_quantized=False,
    ):
        import mlx.core as mx

        if getattr(config, 'model_type', None) != 'bert':
            raise ValueError(
                'The MLX backend currently supports only BERT token '
                f'classification models, got {config.model_type!r}.')
        if config.hidden_act not in ['gelu']:
            raise ValueError(
                'The MLX backend currently supports only GELU BERT models, '
                f'got hidden_act={config.hidden_act!r}.')
        self.mx = mx
        self.config = _MlxBertConfig(
            id2label=config.id2label,
            max_position_embeddings=config.max_position_embeddings,
            num_labels=config.num_labels,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            layer_norm_eps=config.layer_norm_eps,
            hidden_act=config.hidden_act,
        )
        self.weights = (
            weights if weights_are_quantized
            else _quantize_linear_weights(weights, quantization)
        )
        self.quantization = quantization
        self.device = 'mlx'

    def eval(self):
        return self

    def __call__(self, *, input_ids, attention_mask, return_dict=True):
        logits = self.forward(input_ids, attention_mask)
        if return_dict:
            return type('MlxTokenClassifierOutput', (), {'logits': logits})()
        return (logits,)

    def forward(self, input_ids, attention_mask):
        mx = self.mx
        weights = self.weights
        positions = mx.arange(input_ids.shape[1])[None, :]
        token_types = mx.zeros_like(input_ids)
        x = (
            weights['bert.embeddings.word_embeddings.weight'][input_ids]
            + weights['bert.embeddings.position_embeddings.weight'][positions]
            + weights['bert.embeddings.token_type_embeddings.weight'][token_types]
        )
        x = _layer_norm(
            mx, x, weights, 'bert.embeddings.LayerNorm',
            self.config.layer_norm_eps)

        for layer_index in range(self.config.num_hidden_layers):
            x = self._encoder_layer(x, attention_mask, layer_index)
        return _linear(mx, x, weights, 'classifier', self.quantization)

    def _encoder_layer(self, x, attention_mask, layer_index):
        mx = self.mx
        weights = self.weights
        prefix = f'bert.encoder.layer.{layer_index}'
        attention_output = self._self_attention(x, attention_mask, prefix)
        attention_output = _linear(
            mx, attention_output, weights,
            f'{prefix}.attention.output.dense', self.quantization)
        x = _layer_norm(
            mx, attention_output + x, weights,
            f'{prefix}.attention.output.LayerNorm',
            self.config.layer_norm_eps)

        intermediate = _linear(
            mx, x, weights, f'{prefix}.intermediate.dense',
            self.quantization)
        intermediate = _gelu(mx, intermediate)
        output = _linear(
            mx, intermediate, weights, f'{prefix}.output.dense',
            self.quantization)
        return _layer_norm(
            mx, output + x, weights, f'{prefix}.output.LayerNorm',
            self.config.layer_norm_eps)

    def _self_attention(self, x, attention_mask, prefix):
        mx = self.mx
        weights = self.weights
        heads = self.config.num_attention_heads
        head_dim = self.config.hidden_size // heads

        query = _linear(
            mx, x, weights, f'{prefix}.attention.self.query',
            self.quantization)
        key = _linear(
            mx, x, weights, f'{prefix}.attention.self.key',
            self.quantization)
        value = _linear(
            mx, x, weights, f'{prefix}.attention.self.value',
            self.quantization)

        def split_heads(tensor):
            batch, seq_len, _ = tensor.shape
            tensor = tensor.reshape(batch, seq_len, heads, head_dim)
            return tensor.transpose(0, 2, 1, 3)

        query = split_heads(query)
        key = split_heads(key)
        value = split_heads(value)
        scores = (query @ key.transpose(0, 1, 3, 2)) / (head_dim ** 0.5)
        mask = attention_mask[:, None, None, :].astype(mx.bool_)
        scores = mx.where(mask, scores, mx.full(scores.shape, -10000.0))
        probs = mx.softmax(scores, axis=-1)
        context = probs @ value
        context = context.transpose(0, 2, 1, 3)
        return context.reshape(x.shape)


def load_mlx_bert_token_classifier(
    model_name, *, dtype=None, quantization='none'
):
    config = _load_config(model_name)
    exported = load_exported_weights(model_name)
    if exported is not None:
        weights, metadata = exported
        exported_quantization = metadata['quantization']
        if quantization != 'none' and quantization != exported_quantization:
            raise ValueError(
                'Requested MLX quantization '
                f'{quantization!r} does not match exported weights '
                f'{exported_quantization!r}.')
        return MlxBertForTokenClassification(
            config,
            weights,
            quantization=exported_quantization,
            weights_are_quantized=True)

    import mlx.core as mx

    dtype = _dtype_from_name(mx, dtype)
    weights = _load_safetensors_weights(model_name, dtype)
    return MlxBertForTokenClassification(
        config, weights, quantization=quantization)

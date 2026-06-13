_EXPORTS = {
    'DataCollatorForTokenClassificationWithTruncation': '.trainer',
    'SemBrDataBuilder': '.databuilder',
    'binary_metrics': '.utils',
    'chunk_examples': '.dataset',
    'compute_metrics': '.utils',
    'init_dataset': '.trainer',
    'init_model': '.trainer',
    'main': '.trainer',
    'parse_args': '.trainer',
    'process_dataset': '.dataset',
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    from importlib import import_module
    module = import_module(module_name, __name__)
    return getattr(module, name)

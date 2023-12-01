import os
import glob

import datasets

from .process import SemBrProcessor


logger = datasets.logging.get_logger(__name__)

MAX_INDENT = 10


class SemBr2023(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name='sembr2023',
            version=datasets.Version('1.0.0'),
            description='SemBr2023 dataset'),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.processor = SemBrProcessor()

    def _info(self):
        modes = ['off', 'space', 'nospace']
        indents = [str(i) for i in range(MAX_INDENT + 1)]
        return datasets.DatasetInfo(
            features=datasets.Features({
                'flat_lines': datasets.Value('string'),
                'modes': datasets.Sequence(
                    datasets.features.ClassLabel(names=modes)),
                'mode_offsets': datasets.Sequence(
                    datasets.Sequence(datasets.Value('int32'))),
                'indents': datasets.Sequence(
                    datasets.features.ClassLabel(names=indents)),
                'base_indent': datasets.Value('int32'),
            })
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={'root': './data/raw/train/'}),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={'root': './data/raw/test/'}),
        ]

    def _generate_examples(self, root):
        eid = 0
        for path in glob.glob(os.path.join(root, '*')):
            logger.info(f'Generating examples from {path!r}...')
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
            for p in self.processor(text):
                yield eid, p
                eid += 1

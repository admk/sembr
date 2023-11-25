import os
import glob

import datasets

from .process import SemBrProcessor


logger = datasets.logging.get_logger(__name__)

MAX_INDENT = 10


class SemBr2023(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="sembr2023",
            version=datasets.Version("1.0.0"),
            description="SemBr2023 dataset"),
    ]
    model_name = 'distilbert-base-uncased'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.processor = SemBrProcessor()

    def _info(self):
        modes = ['off', 'space', 'nospace']
        indents = [str(i) for i in range(MAX_INDENT)]
        labels = ['off'] + [f'{m}-{i}' for m in modes for i in indents]
        return datasets.DatasetInfo(
            features=datasets.Features({
                "input_ids": datasets.Sequence(datasets.Value("int32")),
                "words": datasets.Sequence(datasets.Value("string")),
                "modes": datasets.Sequence(
                    datasets.features.ClassLabel(names=modes)),
                "indents": datasets.Sequence(
                    datasets.features.ClassLabel(names=indents)),
                "labels": datasets.Sequence(
                    datasets.features.ClassLabel(names=labels)),
            })
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={'root': './data/train/'}),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={'root': './data/test/'}),
        ]

    def _generate_examples(self, root):
        eid = 0
        for path in glob.glob(os.path.join(root, "*.tex")):
            logger.info(f'Generating examples from {path!r}...')
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
            for p in self.processor(text):
                p['labels'] = labels = []
                for m, i in zip(p['modes'], p['indents']):
                    if m == 'off':
                        labels.append('off')
                    else:
                        labels.append(f'{m}-{i}')
                yield eid, p
                eid += 1

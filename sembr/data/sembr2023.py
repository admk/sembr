import os
import glob

import datasets

from .process import SemBrProcessor, SemBrTokenizer


logger = datasets.logging.get_logger(__name__)


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
        self.tokenizer = SemBrTokenizer().from_pretrained(self.model_name, use_fast=False)
        self.processor = SemBrProcessor(self.tokenizer)

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "sembr_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=['off', 'space', 'nospace', 'comment']
                        )
                    ),
                    "indent_levels": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=[str(i) for i in range(5)]
                        )
                    ),
                }
            ),
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
        for path in glob.glob(os.path.join(root, "*.tex")):
            logger.info(f'Generating examples from {path!r}')
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
            results = self.processor(text)
            for r in results:
                yield r

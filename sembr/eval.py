import glob

import datasets
import evaluate
from transformers import AutoModelForTokenClassification, AutoTokenizer

from .dataset import process_dataset
from .inference import inference
from .process import SemBrProcessor


def checkpoints():
    return glob.glob("checkpoints/*/checkpoint-*")


def eval_model(dataset, processor, checkpoint, metric):
    model = AutoModelForTokenClassification.from_pretrained(checkpoint)
    model = model.to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    paras = dataset['test']['flat_lines']
    results = inference(
        paras, tokenizer, model, processor, batch_size=8, overlap_divisor=8)
    generated = processor.generate(results, join=False)
    return metric.compute(predictions=generated, references=paras)


def main():
    dataset = datasets.load_from_disk('./data/sembr2023')
    processor = SemBrProcessor()
    wer = evaluate.load('wer')
    for checkpoint in checkpoints():
        metrics = eval_model(dataset, processor, checkpoint, wer)
        print(f'{checkpoint=}, {metrics=}')


if __name__ == '__main__':
    main()

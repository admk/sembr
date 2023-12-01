# Semantic Line Breaker (SemBr)

[![GitHub](https://img.shields.io/github/license/admk/sembr)](LICENSE)
[![python](https://img.shields.io/badge/Python-3.10-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![pytorch](https://img.shields.io/badge/PyTorch-2.1.0-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![PyPI](https://badge.fury.io/py/sembr.svg)](https://pypi.org/project/sembr)

```
> When writing text
> with a compatible markup language,
> add a line break
> after each substantial unit of thought.
```

## What is SemBr?

SemBr is a command-line tool
powered by [Transformer][transformers1] [models][transformers2]
that breaks lines in a text file at semantic boundaries.

### Installation

SemBr is available as a Python package on PyPI.
To install it,
simply run the following command in your terminal,
assuming that you have Python 3.10 or later installed:
```shell
pip install sembr
```

### Supported Platforms

SemBr is supported on Linux, Mac and Windows.
On machines with CUDA devices,
or on Apple Silicon Macs,
SemBr will use the GPU / Apple Neural Engine
to accelerate inference.

### Usage

To use SemBr,
run the following command in your terminal:
```shell
sembr -i <input_file> -o <output_file>
```
where `<input_file>` and `<output_file>`
are the paths to the input and output files respectively.

On the first run,
it will download the SemBr model
and cache it in `~/.cache/huggingface`.
Subsequent runs will check for updates
and use the cached model if it is up-to-date.

Alternatively,
you can pipe the input into `sembr`,
and the output can also be printed to the terminal:
```shell
cat <input_file> | sembr
```
This is especially useful if you want to use SemBr
with clipboard managers, for instance, on a Mac:
```shell
pbpaste | sembr | pbcopy
```
Or on Linux:
```shell
xclip -o | sembr | xclip -i
```

Additionally,
you can specify the following options
to customize the behavior of SemBr:

* `-m <model_name>`:
  The name of the Hugging Face model to use.
  - The default is
    [`admko/sembr2023-bert-small`][sembr-bert-small].
  - To use it offline,
    you can download the model from Hugging Face,
    and then specify the path to the model directory,
    or prepend `TRANSFORMERS_OFFLINE=1` to the command
    to use the cached model.
* `-l`:
  Serves the SemBr API on a local server.
  - Each instance of `sembr` run
    will detect if the API is accessible,
    and if not it will run the model on its own.
  - This option is useful
    to avoid the time taken to initialize the model
    by keeping it in memory in a separate process.
* `-p <port>`:
  The port to serve the SemBr API on.
  - The default is `8384`.
* `-s <ip>`:
  The IP address to serve the SemBr API on.
  - The default is `127.0.0.1`.

## What are Semantic Line Breaks?

[Semantic Line Breaks][sembr]
or [Semantic Linefeeds][semlf]
describe a set of conventions
for using insensitive vertical whitespace
to structure prose along semantic boundaries.

## Why use Semantic Line Breaks?

Semantic Line Breaks has the following advantages:

* Breaking lines by splitting clauses
  reflects the logical, grammatical and semantic structure
  of the text.

* It enhances the ease of editing and version control
  for a text file.
  Merge conflicts are less likely to occur
  when small changes are made,
  and the changes are easier to identify.

* Documents written with semantic line breaks
  are easier to navigate and edit
  with Vim and other text editors
  that use Vim keybindings.

* Semantic line breaks
  are invisible to readers.
  The final rendered output
  shows no changes to the source text.

## Why SemBr?

Converting existing text not written
with semantic line breaks
takes a long time to do it manually,
and it is surprisingly difficult
to do it automatically with rule-based methods.

### Challenges of rule-based methods

Rule-based heuristics do not work well
with the actual semantic structure of the text,
often leading to incorrect semantic boundaries.
Moreover,
these boundaries are hierarchical and nested,
and a rule-based approach
cannot capture this structure.
A semantic line break
may occur after a dependent clause,
but where to break clauses into lines
is challenging to determine
without syntactic and semantic reasoning capabilities.
For examples:

* A rule that breaks lines at punctuation marks
  will not work well with sentences
  that contain periods
  in abbreviations or mathematical expressions.

* Syntactic or semantic structures
  are not always easy to determine.
  "I like to eat apples and oranges
  because they are healthy."
  should be broken into lines as follows:
  ```
  > I like to eat apples and oranges
  > because they are healthy.
  ```
  rather than:
  ```
  > I like to eat apples
  > and oranges because they are healthy.
  ```

For this reason,
I have created SemBr,
which uses finetuned Transformer models
to predict line breaks at semantic boundaries.


## How does SemBr work?

SemBr uses a Transformer model
to predict line breaks at semantic boundaries.

A small dataset of text with semantic line breaks
was created from my existing LaTeX documents.
The dataset was split into training
(46,295 lines, 170,681 words and 1,492,952 characters)
and test
(2,187 lines, 7,564 words and 72,231 characters)
datasets.

The data was prepared
by extracting line breaks and indent levels
from the files,
and then converting the result
into strings of paragraphs with line breaks removed.
The data can then be tokenized using the tokenizer
and converted into a dataset with tokens,
where each token has a label
denoting if there is line break before it,
and the indent level of the token.

For LaTeX documents,
there are two types of line breaks:
one with a normal line break
that adds implicit spacing (e.g. `line a⏎line b`)
and one with no spacing (e.g. `line a%⏎line b`).
The data processor
also tries to preserve the LaTeX syntax of the text
by adding and removing comment symbols (`%`),
if necessary.

The pretrained masked language model
is then finetuned as a token classifier
on the training dataset
to predict the labels of the tokens.
We save the model with the best F1 score
on correctly predicting the existence of a line break
on the test set.
The finetuning logs for the following models
can be found on this [WandB][wandb] report:

* `distilbert-base-uncased`
  [[Pretrained]][distilbert-bu]
  [[Finetuned]][sembr-distilbert-bu]
* `distilbert-base-cased`
  [[Pretrained]][distilbert-bc]
  [[Finetuned]][sembr-distilbert-bc]
* `distilbert-base-uncased-finetuned-sst-2-english`
  [[Pretrained]][distilbert-bufs2e]
  [[Finetuned]][sembr-distilbert-bufs2e]
* `prajjwal1/bert-tiny`
  [[Pretrained]][bert-tiny]
  [[Finetuned]][sembr-bert-tiny]
* `prajjwal1/bert-mini`
  [[Pretrained]][bert-mini]
  [[Finetuned]][sembr-bert-mini]
* `prajjwal1/bert-small`
  [[Pretrained]][bert-small]
  [[Finetuned]][sembr-bert-small]


## Performance

Current inference speed on an M2 Macbook Pro
is about 850 words per second
on `bert-small` with the default options,
the memory usage is about 1.70 GB.

The link breaking accuracy is difficult to measure,
and the locations of line breaks
could also be subjective.
On the test set,
the per-token line break accuracy
of the models are >95%,
with ~80% F1 scores.
Because of the sparse nature of line breaks,
the accuracy is not a good metric
to measure the performance of the model,
and I used the F1 score instead
to save best models.

## Improvements and TODOs

* [ ] Support natural languages other than English.
* [ ] Support other markup languages such as Markdown.
* [ ] Some lines are too long without a line break.
      The inference algorithm can be improved
      to penalize long lines.
* [ ] Performance and accuracy benchmarking,
      and comparisons with related works.
* [ ] Improve inference speed.
* [ ] Reduce memory usage.
* [ ] Improve indent level prediction.
* [ ] Inference queue.

## Related Projects and References

Sentence splitting:
* https://code.google.com/archive/p/splitta/
* https://en.wikipedia.org/wiki/Sentence_boundary_disambiguation
* https://github.com/nipunsadvilkar/pySBD
* https://www.nltk.org/api/nltk.tokenize.sent_tokenize.html

Semantic line breaking:
* https://github.com/waldyrious/semantic-linebreaker
* https://github.com/bobheadxi/readable ([blog post](https://bobheadxi.dev/semantic-line-breaks/))
* https://github.com/chrisgrieser/obsidian-sembr


[transformers1]: https://huggingface.co/learn/nlp-course/chapter1/4
[transformers2]: https://lilianweng.github.io/posts/2020-04-07-the-transformer-family/

[sembr]: https://sembr.org
[semlf]: https://rhodesmill.org/brandon/2012/one-sentence-per-line

[wandb]: https://wandb.ai/admko/sembr2023

[distilbert-bu]: https://huggingface.co/distilbert-base-uncased
[distilbert-bc]: https://huggingface.co/distilbert-base-cased
[distilbert-bufs2e]: https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english
[bert-tiny]: https://huggingface.co/prajjwal1/bert-tiny
[bert-mini]: https://huggingface.co/prajjwal1/bert-mini
[bert-small]: https://huggingface.co/prajjwal1/bert-small
[sembr-distilbert-bu]: https://huggingface.co/admko/sembr2023-distilbert-base-uncased
[sembr-distilbert-bc]: https://huggingface.co/admko/sembr2023-distilbert-base-cased
[sembr-distilbert-bufs2e]: https://huggingface.co/admko/sembr2023-distilbert-base-uncased-finetuned-sst-2-english
[sembr-bert-tiny]: https://huggingface.co/admko/sembr2023-bert-tiny
[sembr-bert-mini]: https://huggingface.co/admko/sembr2023-bert-mini
[sembr-bert-small]: https://huggingface.co/admko/sembr2023-bert-small

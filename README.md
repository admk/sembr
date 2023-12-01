# Semantic Line Breaker (SemBr)

```
> When writing text
> with a compatible markup language,
> add a line break
> after each substantial unit of thought.
```

## What is SemBr?

SemBr is a tool
powered by [
    Transformer models
](https://huggingface.co/learn/nlp-course/chapter1/4)
that breaks lines in a text file
at semantic boundaries.

### Installation

SemBr is available as a Python package
on PyPI.
To install it,
simply run the following command
in your terminal,
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
run the following command
in your terminal:
```shell
sembr -i <input_file> -o <output_file>
```
where `<input_file>` is the path to the input file.

Alternatively,
you can pipe the input
into `sembr`,
and the output can also be printed
to the terminal:
```shell
cat <input_file> | sembr
```
This is especially useful
if you want to use SemBr
with clipboard managers,
for instance, on a Mac:
```shell
pbpaste | sembr | pbcopy
```

Additionally,
you can specify the following options
to customize the behavior of SemBr:
* `-m <model_name>`: The name of the Hugging Face model to use.
  The default is `admko/sembr2023-bert-small`.
* `-l`: Serves the SemBr API on a local server.
  Each instance of `sembr` run
  will detect if the API is accessible,
  and if not it will run the model on its own.
  This option is useful
  to avoid the time taken to initialize the model
  by keeping it in memory in a separate process.
* `-p <port>`: The port to serve the SemBr API on.
  The default is `8384`.
* `-s <ip>`: The IP address to serve the SemBr API on.
  The default is `127.0.0.1`.

## What are Semantic Line Breaks?

[Semantic Line Breaks](https://sembr.org)
or [
    Semantic Linefeeds
](https://rhodesmill.org/brandon/2012/one-sentence-per-line/)
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
to do it automatically
with rule-based methods.

### Challenges of rule-based methods

Rule-based heuristics do not work well
with the actual semantic structure of the text,
often leading to incorrect semantic boundaries.
Moreover,
semantic boundaries are hierarchical and nested,
and a rule-based approach
cannot capture this structure.
A semantic line break
may occur after a dependent clause,
but not all clauses should be broken into lines.
For examples:

* A rule that breaks lines at punctuation marks
  will not work well
  with sentences that contain
  periods in abbreviations or mathematical expressions.

* For example,
  "I like to eat apples and oranges
  because they are healthy."
  should be broken into lines as follows:
  ```
  I like to eat apples and oranges
  because they are healthy.
  ```
  rather than:
  ```
  I like to eat apples
  and oranges because they are healthy.
  ```

For this reason,
I have created SemBr,
which uses finetuned Transformer models
to predict line breaks at semantic boundaries.


## How does SemBr work?

SemBr uses a Transformer model
to predict line breaks
at semantic boundaries.
A small dataset of text
with semantic line breaks was created
from my existing LaTeX documents.
The dataset was split into training
(46,295 lines, 170,681 words and 1,492,952 characters)
and test
(2,187 lines, 7,564 words and 72,231 characters)
datasets.

The data was prepared
by extracting line breaks and indent levels
from the files,
and then converting the files
into strings of paragraphs
with line breaks removed.
The data can then be tokenized
using the tokenizer
and converted into a dataset
with tokens,
where each token has a label
denoting:
* no line break (label = 0), or
* a line break
  that adds a space in LaTeX documents
  at the token with an indent level
  (label in [0, 1, 2, ..., MAX_INDENT]), or
* a line break that adds no space
  (label in [MAX_INDENT + 1, MAX_INDENT + 2, ..., 2 * MAX_INDENT]).

The pretrained masked language model
is then finetuned as a token classifier
on the training dataset
to predict the labels
of the tokens.
We save the model
with the best F1 score
on correctly predicting line breaks of any kind
on the test set.
The finetuning logs
for all models including the following
can be found
on this [WandB](https://wandb.ai/admko/sembr2023) report:
* [`distilbert-base-uncased`](https://huggingface.co/distilbert-base-uncased).
* [`distilbert-base-cased`](https://huggingface.co/distilbert-base-cased).
* [`distilbert-base-uncased-finetuned-sst-2-english`](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english).
* [`prajjwal1/bert-tiny`](https://huggingface.co/prajjwal1/bert-tiny).
* [`prajjwal1/bert-small`](https://huggingface.co/prajjwal1/bert-small).


## Performance

Current inference speed
on an M2 Macbook Pro
is about 1,500 characters per second
on `bert-small`,
the memory usage is about 1.70 GB.


## Improvements and TODOs

* [ ] Support natural languages other than English.
* [ ] Support other markup languages
      such as Markdown.
* [ ] Some lines are too long
      without a line break.
      The inference algorithm
      can be improved to penalize long lines.
* [ ] Performance benchmarking.
* [ ] Improve inference speed.
* [ ] Reduce memory usage.
* [ ] Improve indent level prediction.
* [ ] Inference queue.

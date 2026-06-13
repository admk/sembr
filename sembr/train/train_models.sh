ts -G 1 \
    python -m sembr.train prajjwal1/bert-tiny \
    -hu=admko -lr=1e-4 -tb=64 -ms=1000
ts -G 1 \
    python -m sembr.train prajjwal1/bert-small \
    -hu=admko -lr=1e-4 -tb=64 -ms=1000
ts -G 1 \
    python -m sembr.train distilbert-base-uncased \
    -hu=admko -lr=1e-4 -tb=64 -ms=1000
ts -G 1 \
    python -m sembr.train distilbert-base-uncased-finetuned-sst-2-english \
    -hu=admko -lr=1e-4 -tb=64 -ms=1000
# ts -G 1 \
#     python -m sembr.train distilbert-base-cased \
#     -hu=admko -lr=1e-4 -tb=64 -ms=1000
# ts -G 1 \
#     python -m sembr.train distilbert-base-multilingual-cased \
#     -hu=admko -lr=1e-4 -tb=64 -ms=1000

import spacy

def perform_semantic_line_breaks(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    lines = []
    current_line = ""
    for token in doc:
        print(f"Token: {token.text}, POS: {token.pos_}, Metadata: {token._}")
    return lines

# Example usage
text = """
This is a sample text. It demonstrates how semantic line breaks
should be performed in accordance with the given specification.
A line break is needed after a sentence. A line break should also
occur after an independent clause, for example, separated by a comma.
Em dash --- is also a valid separator. A line break should also occur.
Mr. Smith went to the store. Dr. Johnson is a scientist.
This is an `inline markup`.
You should not break a hyphen-joined word.
This is a hyperlink: https://www.google.com.
Do not modify code blocks, such as this one:
```python
def foo():
    print("Hello! This is a code block.")
```
"""

result = perform_semantic_line_breaks(text)

for line in result:
    print(line)

from .processors.plaintext import PlainTextProcessor


class SemBrProcessor(PlainTextProcessor):
    def __call__(self, text):
        return self.parse_text(text)

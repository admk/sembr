import re
from dataclasses import dataclass

from transformers import AutoTokenizer, DataCollatorForTokenClassification


MAX_INDENT = 10
LABELS = ['off', 'space', 'nospace', 'comment']
id2label = {i: l for i, l in enumerate(LABELS)}
label2id = {l: i for i, l in enumerate(LABELS)}


@dataclass
class LabelAttr:
    mode: str = True
    indent: int = 0

    def __str__(self):
        return f'<{self.mode}-{self.indent}>'


class TokenFSM(object):
    states = ['space', 'nospace', 'normal', 'comment']

    def __init__(self):
        super().__init__()
        self.indent = 0
        self.state = 'space'

    @property
    def trail_space(self):
        return self.state == 'space'

    @property
    def in_comment(self):
        return self.state == 'comment'

    def transition(self, token):
        match (token, self.state):
            case ('[CLS]' | '[SEP]'), _:
                return False, None
            case ('[indent]', 'normal') | ('[indent]', 'comment'):
                return False, None
            case ('[indent]', 'space') | ('[indent]', 'nospace'):
                self.indent += 1
                return False, None
            case ('[newline]', 'space') | ('[newline]', 'nospace'):
                self.state = 'space'
                self.indent = 0
                return False, None
            case '[newline]', _:
                if self.state == 'normal':
                    self.state = 'space'
                elif self.state == 'comment':
                    self.state = 'nospace'
                self.indent = 0
                return False, None
            case '[par]', _:
                self.state = 'space'
                self.indent = 0
                return True, None
            case '%', _:
                attr = None
                if self.state in ['space', 'nospace']:
                    attr = LabelAttr('comment', self.indent)
                emit = self.state == 'comment'
                self.state = 'comment'
                return emit, attr
            case _:
                if self.state in ['space', 'nospace']:
                    if self.in_comment:
                        mode = 'comment'
                    else:
                        mode = self.state
                    attr = LabelAttr(mode, self.indent)
                    self.state = 'normal'
                    return True, attr
                return True, None


class SemBrTokenizer:
    tokenizer_name = 'distilbert-base-uncased'

    def __init__(self, spaces=4, **kwargs):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name, **kwargs)
        self.spaces = spaces
        self.replace_tokens = {
            # r'\n(?:\s*\n)+': '[par]',
            '\n': '[newline]',
            ' ' * self.spaces: '[indent]',
            # '\t': '[indent]',
        }
        self.reverse_replace_tokens = {
            v: k for k, v in self.replace_tokens.items()}
        self.tokenizer.add_tokens(list(self.replace_tokens.values()))

    def __call__(self, text, **kwargs):
        ftext = text
        for k, v in self.replace_tokens.items():
            if isinstance(v, str):
                ftext = re.sub(k, v, ftext)
        enc = self.tokenizer(ftext, return_offsets_mapping=True, **kwargs)
        enc['formatted_text'] = ftext
        return enc

    def convert_ids_to_tokens(self, ids, replace=True):
        tokens = self.tokenizer.convert_ids_to_tokens(ids)
        if not replace:
            return tokens
        return [self.reverse_replace_tokens.get(t, t) for t in tokens]

    def convert_tokens_to_ids(self, tokens):
        tokens = [self.replace_tokens.get(t, t) for t in tokens]
        return self.tokenizer.convert_tokens_to_ids(tokens)


class SemBrProcessor(object):
    def __init__(self, tokenizer, spaces=4, max_indent=MAX_INDENT):
        super().__init__()
        self.tokenizer = tokenizer
        self.spaces = spaces
        self.max_indent = max_indent

    def __call__(self, text):
        paras = re.split(r'\n(?:\s*\n)+', text)
        paragraphs = [
            self._process_paragraph(p.strip()) for p in paras if p.strip()]
        return paragraphs

    def _process_paragraph(self, text):
        enc = self.tokenizer(text)
        ids = enc.input_ids
        tokens = self.tokenizer.convert_ids_to_tokens(ids, replace=False)
        offsets = enc.offset_mapping
        fsm = TokenFSM()
        cursor = 0
        ftext = enc['formatted_text']
        mtext = []
        pids = []
        for pid, tok, (_, offset) in zip(ids, tokens, offsets):
            emit, attr = fsm.transition(tok)
            if attr is not None:
                attr.indent = min(attr.indent, self.max_indent)
                mtext.append(attr)
            if emit:
                if isinstance(emit, str):
                    pid = self.tokenizer.convert_tokens_to_ids(emit)
                mtext.append(ftext[cursor:offset])
                pids.append(pid)
            cursor = offset
        mtext.append(LabelAttr('off'))
        pattrs, ptext = {}, []
        # omit first break
        for t in mtext:
            if isinstance(t, str):
                ptext.append(t)
                pattrs[len(ptext) - 1] = LabelAttr('off')
            else:
                pattrs[len(ptext) - 1] = t
        pattrs = [pattrs.get(i, LabelAttr('off')) for i in range(len(ptext))]
        result = {
            'input_ids': pids,
            'words': ptext,
            'modes': [a.mode for a in pattrs],
            'indents': [a.indent for a in pattrs],
            # 'processed_text': ''.join(str(t) for t in mtext),
            # 'formatted_text': ftext,
        }
        return result

    def generate(self, words, modes, indents):
        text = []
        in_comment = False
        for w, m, i in zip(words, modes, indents):
            text.append(w)
            if m in ['space', 'nospace']:
                if not in_comment and m != 'space':
                    text.append('%')
                text.append('\n')
                in_comment = False
                if i:
                    text.append(' ' * (i * self.spaces))
            if m == 'comment':
                text.append('\n%')
                in_comment = True
        return ''.join(text)


class DataCollatorForTokenClassificationWithTruncation(
    DataCollatorForTokenClassification
):
    def __init__(self, tokenizer, max_length=512, **kwargs):
        super().__init__(tokenizer, **kwargs)
        self.max_length = max_length

    def __call__(self, features, return_tensors=None):
        truncated_features = []
        for f in features:
            truncated_features.append(
                {k: v[:self.max_length] for k, v in f.items()})
        return super().__call__(truncated_features, return_tensors)


if __name__ == '__main__':
    from icecream import ic
    test = open('./data/test/mair.tex', 'r').read()
    tokenizer = SemBrTokenizer()
    processor = SemBrProcessor(tokenizer)
    preped = processor(test)[2]
    tokens = tokenizer.convert_ids_to_tokens(preped['input_ids'])
    print(processor.generate(
        preped['words'], preped['modes'], preped['indents']))

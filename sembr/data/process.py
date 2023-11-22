import re
from dataclasses import dataclass

from transformers import PreTrainedTokenizerFast


LABELS = [
    'off',
    'space0', 'space1', 'space2', 'space3', 'space4',
    'nospace0', 'nospace1', 'nospace2', 'nospace3', 'nospace4',
    'comment0', 'comment1', 'comment2', 'comment3', 'comment4',
]
LABELS = [f'<{l}>' for l in LABELS]
id2label = {i: l for i, l in enumerate(LABELS)}
label2id = {l: i for i, l in enumerate(LABELS)}


@dataclass
class LabelAttr:
    br: bool
    indent: int = 0
    trail_space: bool = True
    comment: bool = False

    def __str__(self):
        return id2label[attr2id(self)]


def attr2id(label: LabelAttr):
    if not label.br:
        name = 'off'
    elif label.comment:
        name = f'comment{label.indent}'
    elif label.trail_space:
        name = f'space{label.indent}'
    else:
        name = f'nospace{label.indent}'
    return label2id[f'<{name}>']


def id2attr(label_id: int):
    name = id2label[label_id]
    if name == 'off':
        return LabelAttr(br=False)
    elif name.startswith('comment'):
        return LabelAttr(br=True, indent=int(name[-1]), comment=True)
    elif name.startswith('space'):
        return LabelAttr(br=True, indent=int(name[-1]), trail_space=True)
    elif name.startswith('nospace'):
        return LabelAttr(br=True, indent=int(name[-1]), trail_space=False)
    raise ValueError(f'Invalid label id: {label_id}.')


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
                    attr = LabelAttr(
                        br=True,
                        indent=self.indent,
                        trail_space=self.trail_space,
                        comment=True)
                emit = self.state == 'comment'
                self.state = 'comment'
                return emit, attr
            case _:
                if self.state in ['space', 'nospace']:
                    attr = LabelAttr(
                        br=True, indent=self.indent,
                        trail_space=self.trail_space, comment=self.in_comment)
                    self.state = 'normal'
                    return True, attr
                return True, None


class SemBrTokenizer(PreTrainedTokenizerFast):
    def __init__(self, *args, spaces=4, **kwargs):
        super().__init__(*args, **kwargs)
        self.spaces = spaces
        self.replace_tokens = {
            r'\n(?:\s*\n)+': '[par]',
            '\n': '[newline]',
            ' ' * self.spaces: '[indent]',
            # '\t': '[indent]',
        }
        self.add_tokens(list(self.replace_tokens.values()))

    def __call__(self, text, **kwargs):
        ftext = text
        for k, v in self.replace_tokens.items():
            if isinstance(v, str):
                ftext = re.sub(k, v, ftext)
        enc = super().__call__(ftext, return_offsets_mapping=True, **kwargs)
        enc['formatted_text'] = ftext
        return enc


class SemBrProcessor(object):
    def __init__(self, tokenizer, spaces=4):
        super().__init__()
        self.tokenizer = tokenizer
        self.spaces = spaces

    def __call__(self, text):
        paras = re.split(r'\n(?:\s*\n)+', text)
        paragraphs = [
            self._process_paragraph(p.strip()) for p in paras if p.strip()]
        return paragraphs

    def _process_paragraph(self, text):
        enc = tokenizer(text)
        ids = enc.input_ids
        tokens = tokenizer.convert_ids_to_tokens(ids)
        offsets = enc.offset_mapping
        fsm = TokenFSM()
        cursor = 0
        ftext = enc['formatted_text']
        mtext = []
        pids = []
        for pid, tok, (_, offset) in zip(ids, tokens, offsets):
            emit, attr = fsm.transition(tok)
            if attr is not None:
                mtext.append(attr)
            if emit:
                if isinstance(emit, str):
                    pid = tokenizer.convert_tokens_to_ids(emit)
                mtext.append(ftext[cursor:offset])
                pids.append(pid)
            cursor = offset
        mtext.append(LabelAttr(br=True))
        pattrs, ptext = {}, []
        # omit first break
        for t in mtext:
            if isinstance(t, str):
                ptext.append(t)
                pattrs[len(ptext) - 1] = LabelAttr(br=False)
            else:
                pattrs[len(ptext) - 1] = t
        pattrs = [pattrs.get(i, LabelAttr(br=False)) for i in range(len(ptext))]
        result = {
            'input_ids': pids,
            'words': ptext,
            'attrs': pattrs,
            'attr_ids': [attr2id(a) for a in pattrs],
            'processed_text': ''.join(str(t) for t in mtext),
            'formatted_text': ftext,
        }
        return result

    def generate(self, words, attrs):
        text = []
        in_comment = False
        for w, a in zip(words, attrs):
            text.append(w)
            if a.br:
                if not in_comment and not a.trail_space:
                    text.append('%')
                text.append('\n')
                in_comment = False
                if a.indent:
                    text.append(' ' * (a.indent * self.spaces))
            if a.comment:
                text.append('%')
                in_comment = True
        return ''.join(text)


if __name__ == '__main__':
    test = open('./data/test/mair.tex', 'r').read()
    model_name = 'distilbert-base-uncased'
    tokenizer = SemBrTokenizer.from_pretrained(model_name)
    processor = SemBrProcessor(tokenizer)
    preped = processor(test)[2]
    ptext = preped['words']
    pattrs = preped['attrs']
    pids = preped['input_ids']
    tokens = tokenizer.convert_ids_to_tokens(pids)
    print(processor.generate(ptext, pattrs))

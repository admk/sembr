import re

from transformers import AutoTokenizer, DataCollatorForTokenClassification


class SemBrProcessor(object):
    tokenizer_name = 'distilbert-base-uncased'

    def __init__(self, spaces=4):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.spaces = spaces
        self.replace_tokens = {
            # r'\n(?:\s*\n)+': '[par]',
            # '\t': '[indent]',
            # ' ' * self.spaces: '[indent]',
            '\\%': '[percent]',
            '\n': '[newline]',
        }
        self.reverse_replace_tokens = {
            v: k for k, v in self.replace_tokens.items()}
        self.tokenizer.add_tokens(list(self.replace_tokens.values()))

    def _process_specials(self, lines):
        for k, v in self.replace_tokens.items():
            # lines = [re.sub(k, v, l) for l in lines]
            lines = [l.replace(k, v) for l in lines]
        return lines

    def _process_indents(self, lines):
        nlines = []
        indents = []
        # get indent levels
        for line in lines:
            indent = 0
            for c in line:
                if c == ' ':
                    indent += 1
                elif c == '\t':
                    indent += self.spaces
                else:
                    break
            indent_level = int(indent / self.spaces)
            nlines.append(line[indent_level * self.spaces:].rstrip())
            indents.append(indent_level)
        return nlines, indents

    def _process_comments(self, lines, indents):
        # normalize comments, ["xxx % comment"] -> ["xxx", "% comment"]
        nclines = []
        ncindents = []
        for line, indent in zip(lines, indents):
            if '%' in line:
                normal, *comment = line.split('%')
                comment = '%'.join(comment).strip()
                if normal.strip():
                    if comment:
                        nclines += [normal, f'%{comment}']
                        ncindents += [indent, indent]
                        continue
                    line = f'{normal}%'
            nclines.append(line)
            ncindents.append(indent)
        return nclines, ncindents

    def _process_modes(self, lines):
        new_lines = []
        modes = []
        prev_status = 'start'
        for line in lines:
            if line.startswith('%'):
                status = 'comment'
            elif line.endswith('%'):
                status = 'percent'
                line = line.rstrip('%')
            else:
                status = 'normal'
            match (prev_status, status):
                case ('start', _):
                    pass
                case ('normal', _):
                    modes.append('space')
                case ('percent', _):
                    modes.append('nospace')
                case ('comment', 'normal'):
                    modes.append('break')
                case ('comment', 'percent'):
                    modes.append('break')
                case ('comment', 'comment'):
                    modes.append('comment')
                case (_, 'comment'):
                    modes.append('comment')
                case _:
                    raise ValueError(
                        'Unknown status transition: '
                        f'{prev_status} -> {status}.')
            new_lines.append(line)
            prev_status = status
        # last transition always force a break
        modes.append('break')
        return new_lines, modes

    def _flatten_with_modes(self, lines, modes):
        in_comment = 0
        flat_lines, flat_modes, offsets = [], [], []
        prev_len = flat_len = 0
        for line, mode in zip(lines, modes):
            if in_comment >= 1:
                line = re.sub(r'^\s*%', '', line)
            if mode == 'break':
                in_comment = 0
                line = f'{line}[newline]'
                mode = 'off'
            elif mode == 'comment':
                in_comment += 1
                mode = 'space'
            elif mode == 'space':
                line = f'{line} '
            elif mode == 'nospace':
                pass
            else:
                raise ValueError(f'Unknown mode: {mode}.')
            flat_lines.append(line)
            flat_modes.append(mode)
            prev_len = flat_len
            flat_len += len(line)
            offsets.append((prev_len, flat_len))
        return ''.join(flat_lines), flat_modes, offsets

    def _tokenize_with_modes(
        self, text, line_modes, line_modes_offsets, line_indents
    ):
        enc = self.tokenizer(text, return_offsets_mapping=True)
        words, modes, indents = [], [], []
        pos = mode_idx = 0
        # fill empty words in offset mapping
        offset_mapping = []
        for start, end in enc.offset_mapping:
            offset_mapping.append((min(start, pos), end))
            pos = end
        pos = 0
        input_ids = []
        for tid, (start, end) in zip(enc.input_ids, offset_mapping):
            if pos >= len(text):
                break
            mode_offset = line_modes_offsets[mode_idx][1]
            word = text[pos:end]
            input_ids.append(tid)
            words.append(word)
            indents.append(line_indents[mode_idx])
            pos = max(pos, end)
            if mode_offset >= end:
                modes.append('off')
                continue
            mode = line_modes[mode_idx]
            modes.append(mode)
            mode_idx += 1
            # current word is on a new line
            indents[-1] = line_indents[mode_idx]
        return input_ids, words, modes, indents

    def _process_paragraph(self, text):
        lines = text.split('\n')
        lines = self._process_specials(lines)
        lines, indents = self._process_indents(lines)
        base_indent = min(indents)
        indents = [i - base_indent for i in indents]
        lines, indents = self._process_comments(lines, indents)
        lines, modes = self._process_modes(lines)
        flat_lines, modes, modes_offsets = self._flatten_with_modes(lines, modes)
        input_ids, words, modes, indents = self._tokenize_with_modes(
            flat_lines, modes, modes_offsets, indents)
        result = {
            'input_ids': input_ids,
            'words': words,
            'modes': modes,
            'indents': indents,
            'base_indent': base_indent,
        }
        keys = ['input_ids', 'words', 'modes', 'indents']
        if len(set(len(result[k]) for k in keys)) != 1:
            len_dict = {k: len(result[k]) for k in keys}
            raise ValueError(
                f'Lengths do not match. Found: {len_dict}.')
        return result

    def __call__(self, text):
        paras = []
        for p in re.split(r'\n(?:\s*\n)+', text):
            if not p.strip():
                continue
            paras.append(self._process_paragraph(p))
        return paras

    def _replace_newlines(self, words, modes, indents):
        new_words, new_modes, new_indents = [], [], []
        next_mode = None
        for word, mode, indent in zip(words, modes, indents):
            if word == '[newline]':
                next_mode = 'break'
                continue
            if next_mode:
                # if mode != 'off':
                #     raise ValueError(
                #         f'Cannot set mode {next_mode} '
                #         f'when mode is {mode}.')
                mode = next_mode
                next_mode = None
            new_words.append(word)
            new_modes.append(mode)
            new_indents.append(indent)
        return new_words, new_modes, new_indents

    def _generate_lines(self, words, modes, indents):
        lbs = [
            (o, m) for o, m in enumerate(modes)
            if m in ('space', 'nospace', 'break')]
        if not lbs or lbs[-1][0] < len(words):
            lbs.append((len(words), 'space'))
        lines, line_indents  = [], []
        pos = in_comment = 0
        for o, m in lbs:
            line = ''.join(words[pos:o]).strip()
            if line.startswith('%'):
                in_comment = 1
            if m == 'nospace':
                line = f'{line}%'
            if m in ('space', 'break'):
                if in_comment > 1:
                    line = f'% {line}'
                if in_comment:
                    in_comment += 1
                if m == 'break':
                    in_comment = 0
            lines.append(line)
            line_indents.append(indents[pos:o])
            pos = o
        # line_indents = [Counter(l).most_common(1)[0][0] for l in line_indents]
        line_indents = [l[0] for l in line_indents]
        return lines, line_indents

    def _indent_lines(self, lines, indents, base_indent):
        spaces = ' ' * self.spaces
        return [
            f'{spaces * (i + base_indent)}{l}'
            for i, l in zip(indents, lines)]

    def _generate_paragraph(self, processed):
        words = processed['words']
        modes = processed['modes']
        indents = processed['indents']
        base_indent = processed['base_indent']
        words, modes, indents = self._replace_newlines(words, modes, indents)
        lines, indents = self._generate_lines(words, modes, indents)
        lines = self._indent_lines(lines, indents, base_indent)
        text = '\n'.join(lines)
        for k, v in self.reverse_replace_tokens.items():
            text = text.replace(k, v)
        return text

    def generate(self, paragraphs):
        return '\n\n'.join(self._generate_paragraph(p) for p in paragraphs)


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
    # test = open('./data/test/mair.tex', 'r').read()
    test = open('./data/example.tex', 'r').read()
    processor = SemBrProcessor()
    results = processor(test)
    print('--- Processed ---')
    print(processor.generate(results))
    for r in results:
        r['modes'] = ['off' if m != 'break' else m for m in r['modes']]
    print('--- Flattened ---')
    print(processor.generate(results))

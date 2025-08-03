"""
LaTeX processor refactored to use the new base processor architecture.
"""

import re
from typing import List, Dict, Any

from .base import BaseProcessor


class LaTeXProcessor(BaseProcessor):
    """
    LaTeX processor using the original SemBr logic.
    Refactored to inherit from BaseProcessor.
    """

    def _get_replace_tokens(self) -> Dict[str, str]:
        """
        Get LaTeX-specific token replacements.

        Returns:
            Dictionary mapping original tokens to replacement tokens
        """
        return {
            "\t": " " * self.spaces,
            "\\%": "[percent]",
            "\n": "[newline]",
        }

    def _identify_content_regions(self, text: str) -> List[Dict[str, Any]]:
        """
        For LaTeX, we use paragraph-based splitting rather than grammar parsing.

        Args:
            text: Input LaTeX text

        Returns:
            List of content regions (paragraphs)
        """
        # Split on double newlines to get paragraphs
        paragraphs = re.split(r"\n(?:\s*\n)+", text)
        regions = []
        for i, para in enumerate(paragraphs):
            if para.strip():
                regions.append(
                    {
                        "type": "content",
                        "content": para,
                        "index": i,
                    }
                )
        return regions

    def _process_comments(
        self, lines: List[str], indents: List[int]
    ) -> tuple[List[str], List[int]]:
        """
        Normalize LaTeX comments.

        Args:
            lines: List of text lines
            indents: List of indent levels

        Returns:
            Tuple of (processed_lines, processed_indents)
        """
        nclines = []
        ncindents = []
        for line, indent in zip(lines, indents):
            if "%" in line:
                normal, *comment = line.split("%")
                comment = "%".join(comment).strip()
                if normal.strip():
                    if comment:
                        nclines += [normal, f"%{comment}"]
                        ncindents += [indent, indent]
                        continue
                    line = f"{normal}%"
            nclines.append(line)
            ncindents.append(indent)
        return nclines, ncindents

    def _process_modes(self, lines: List[str]) -> tuple[List[str], List[str]]:
        """
        Process LaTeX mode transitions.

        Args:
            lines: List of text lines

        Returns:
            Tuple of (processed_lines, modes)
        """
        new_lines = []
        modes = []
        prev_status = "start"
        for line in lines:
            if line.startswith("%"):
                status = "comment"
            elif line.endswith("%"):
                status = "percent"
                line = line.rstrip("%")
            else:
                status = "normal"
            match (prev_status, status):
                case ("start", _):
                    pass
                case ("normal", _):
                    modes.append("space")
                case ("percent", _):
                    modes.append("nospace")
                case ("comment", "normal"):
                    modes.append("break")
                case ("comment", "percent"):
                    modes.append("break")
                case ("comment", "comment"):
                    modes.append("comment")
                case (_, "comment"):
                    modes.append("comment")
                case _:
                    raise ValueError(
                        f"Unknown status transition: {prev_status} -> {status}."
                    )
            new_lines.append(line)
            prev_status = status
        # Last transition always forces a break
        modes.append("break")
        return new_lines, modes

    def _flatten_with_modes(
        self, lines: List[str], modes: List[str]
    ) -> tuple[str, List[str], List[tuple[int, int]]]:
        """
        Flatten lines with LaTeX mode handling.

        Args:
            lines: List of text lines
            modes: List of modes

        Returns:
            Tuple of (flattened_text, flattened_modes, offsets)
        """
        in_comment = 0
        flat_lines, flat_modes, offsets = [], [], []
        prev_len = flat_len = 0
        for line, mode in zip(lines, modes):
            if in_comment >= 1:
                line = re.sub(r"^\s*%", "", line)
            if mode == "break":
                in_comment = 0
                line = f"{line}[newline]"
                mode = "off"
            elif mode == "comment":
                in_comment += 1
                mode = "space"
            elif mode == "space":
                line = f"{line} "
            elif mode == "nospace":
                pass
            else:
                raise ValueError(f"Unknown mode: {mode}.")
            flat_lines.append(line)
            flat_modes.append(mode)
            prev_len = flat_len
            flat_len += len(line)
            offsets.append((prev_len, flat_len))
        return "".join(flat_lines), flat_modes, offsets

    def _process_paragraph(self, text: str) -> Dict[str, Any]:
        """
        Process a single LaTeX paragraph.

        Args:
            text: Paragraph text

        Returns:
            Processed paragraph data
        """
        lines = text.split("\n")
        lines = self._process_specials(lines)
        lines, indents = self._process_indents(lines)
        base_indent = min(indents) if indents else 0
        indents = [i - base_indent for i in indents]
        lines, indents = self._process_comments(lines, indents)
        lines, modes = self._process_modes(lines)
        flat_lines, modes, mode_offsets = self._flatten_with_modes(lines, modes)
        return {
            "flat_lines": flat_lines,
            "modes": modes,
            "mode_offsets": mode_offsets,
            "indents": indents,
            "base_indent": base_indent,
        }

    def parse_text(self, text: str, split: bool = True) -> List[Dict[str, Any]]:
        """
        Parse LaTeX text into processable paragraphs.

        Args:
            text: Input LaTeX text
            split: Whether to split into paragraphs

        Returns:
            List of parsed paragraphs
        """
        text = text.replace("\t", " " * self.spaces)
        if split:
            text = re.split(r"\n(?:\s*\n)+", text)
        elif isinstance(text, str):
            raise ValueError("Text must be a list of strings if split=False.")
        paras = []
        for p in text:
            if not p.strip():
                continue
            paras.append(self._process_paragraph(p))
        return paras

    def _tokenize_with_modes(
        self,
        tokenizer,
        text: str,
        line_modes: List[str],
        line_mode_offsets: List[tuple[int, int]],
        line_indents: List[int],
    ) -> tuple[List[int], List[str], List[str], List[int]]:
        """
        Tokenize text with LaTeX mode information.

        Args:
            tokenizer: HuggingFace tokenizer
            text: Text to tokenize
            line_modes: List of modes
            line_mode_offsets: List of offset tuples
            line_indents: List of indent levels

        Returns:
            Tuple of (input_ids, words, modes, indents)
        """
        enc = tokenizer(text, return_offsets_mapping=True)
        words, modes, indents = [], [], []
        pos = mode_idx = 0
        # Fill empty words in offset mapping
        offset_mapping = []
        for start, end in enc.offset_mapping:
            offset_mapping.append((min(start, pos), end))
            pos = end
        pos = 0
        input_ids = []
        for tid, (start, end) in zip(enc.input_ids, offset_mapping):
            if pos >= len(text):
                break
            mode_offset = line_mode_offsets[mode_idx][1]
            word = text[pos:end]
            input_ids.append(tid)
            words.append(word)
            indents.append(line_indents[mode_idx])
            pos = max(pos, end)
            if mode_offset >= end:
                modes.append("off")
                continue
            mode = line_modes[mode_idx]
            modes.append(mode)
            mode_idx += 1
            # Current word is on a new line
            indents[-1] = line_indents[mode_idx]
        return input_ids, words, modes, indents

    def tokenize_with_modes(
        self, tokenizer, results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Tokenize parsed results with LaTeX mode information.

        Args:
            tokenizer: HuggingFace tokenizer
            results: Parsed paragraphs from parse_text

        Returns:
            Tokenized results with modes and indents
        """
        self.prepare_tokenizer(tokenizer)
        new_results = []
        for r in results:
            flat_lines = r["flat_lines"]
            modes = r["modes"]
            mode_offsets = r["mode_offsets"]
            indents = r["indents"]
            base_indent = r["base_indent"]
            input_ids, words, modes, indents = self._tokenize_with_modes(
                tokenizer, flat_lines, modes, mode_offsets, indents
            )
            tokenized = {
                "input_ids": input_ids,
                "words": words,
                "modes": modes,
                "indents": indents,
                "base_indent": base_indent,
            }
            # Validate lengths
            keys = ["input_ids", "words", "modes", "indents"]
            if len(set(len(tokenized[k]) for k in keys)) != 1:
                len_dict = {k: len(tokenized[k]) for k in keys}
                raise ValueError(f"Lengths do not match. Found: {len_dict}.")
            new_results.append(tokenized)
        return new_results

    def _generate_lines(
        self, words: List[str], modes: List[str], indents: List[int]
    ) -> tuple[List[str], List[int]]:
        """
        Generate lines from words and modes with LaTeX comment handling.

        Args:
            words: List of word tokens
            modes: List of modes
            indents: List of indent levels

        Returns:
            Tuple of (lines, line_indents)
        """
        lbs = [
            (o, m) for o, m in enumerate(modes)
            if m in ("space", "nospace", "break")
        ]
        if not lbs or lbs[-1][0] < len(words):
            lbs.append((len(words), "space"))
        lines, line_indents = [], []
        pos = in_comment = 0
        for o, m in lbs:
            line = "".join(words[pos:o]).strip()
            if line.startswith("%"):
                in_comment = 1
            if m == "nospace":
                line = f"{line}%"
            if m in ("space", "break"):
                if in_comment > 1:
                    line = f"% {line}"
                if in_comment:
                    in_comment += 1
                if m == "break":
                    in_comment = 0
            lines.append(line)
            line_indents.append(indents[pos] if pos < len(indents) else 0)
            pos = o
        return lines, line_indents

    def _generate_paragraph(self, processed: Dict[str, Any]) -> str:
        """
        Generate LaTeX text from processed paragraph.

        Args:
            processed: Processed paragraph data

        Returns:
            Generated LaTeX text
        """
        words = processed["words"]
        modes = processed["modes"]
        indents = processed["indents"]
        base_indent = processed["base_indent"]
        words, modes, indents = self._replace_newlines(words, modes, indents)
        lines, indents = self._generate_lines(words, modes, indents)
        lines = self._indent_lines(lines, indents, base_indent)
        text = "\n".join(lines)
        # Apply reverse token replacements
        for k, v in self.reverse_replace_tokens.items():
            text = text.replace(k, v)
        return text.rstrip()

    def generate(
        self, paragraphs: List[Dict[str, Any]], join: bool = True
    ) -> str:
        """
        Generate formatted LaTeX text from processed paragraphs.

        Args:
            paragraphs: Processed paragraph data
            join: Whether to join paragraphs with double newlines

        Returns:
            Generated LaTeX text
        """
        paragraphs = [self._generate_paragraph(p) for p in paragraphs]
        if join:
            return "\n\n".join(paragraphs)
        return paragraphs

from sembr.processors.plaintext import PlainTextProcessor


def test_processor_emits_configured_space_width():
    processor = PlainTextProcessor(spaces=2)

    assert processor._indent_lines(['wrapped'], [2], 1) == ['      wrapped']


def test_processor_emits_tabs_when_configured():
    processor = PlainTextProcessor(spaces=4, indent_type='tab')

    assert processor._indent_lines(['wrapped'], [2], 1) == ['\t\t\twrapped']


def test_processor_reads_tab_indentation_when_configured():
    processor = PlainTextProcessor(spaces=4, indent_type='tab')

    lines, indents = processor._process_indents(['\twrapped'])

    assert lines == ['wrapped']
    assert indents == [1]


def test_processor_auto_detects_two_space_indents():
    processor = PlainTextProcessor(spaces='auto')

    paragraphs = processor.parse_text('  one\n    two')

    assert processor._active_spaces == 2
    assert paragraphs[0]['indents'] == [0, 1]


def test_processor_auto_detects_four_space_indents():
    processor = PlainTextProcessor(spaces='auto')

    paragraphs = processor.parse_text('    one\n        two')

    assert processor._active_spaces == 4
    assert paragraphs[0]['indents'] == [0, 1]


def test_processor_auto_detects_eight_space_indents():
    processor = PlainTextProcessor(spaces='auto')

    paragraphs = processor.parse_text('        one\n                two')

    assert processor._active_spaces == 8
    assert paragraphs[0]['indents'] == [0, 1]

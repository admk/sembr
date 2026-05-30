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

from sembr.processors.markdown import MarkdownProcessor


def test_regions_after_non_ascii_text_are_located_by_character():
    text = 'Ünïcödé café.\n\n- Second — item\n  continued.\n\nThird.\n'
    processor = MarkdownProcessor(spaces=2)

    regions = processor.parse_text(text)

    assert [r['original_text'] for r in regions] == [
        'Ünïcödé café.', 'Second — item\n  continued.', 'Third.']
    assert [text[r['start_byte']:r['end_byte']] for r in regions] == [
        r['original_text'] for r in regions]
    assert regions[1]['continuation_indent'] == '  '

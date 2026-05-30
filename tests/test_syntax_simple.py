#!/usr/bin/env python3
"""
Simple test runner for markdown syntax that focuses on tree structure preservation.
This version doesn't require the full SemBr model and can run quickly.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pathlib import Path

try:
    import tree_sitter_markdown as tsmarkdown
    from tree_sitter import Language, Parser
except ImportError as e:
    raise ImportError(
        "tree-sitter and tree-sitter-markdown are required. "
        "Install with: pip install tree-sitter tree-sitter-markdown"
    ) from e

from sembr.processors.markdown import MarkdownProcessor


def get_tree_structure(parser, text: str) -> dict:
    """Extract tree structure for comparison."""
    tree = parser.parse(bytes(text, 'utf8'))
    
    def extract_node(node):
        return {
            'type': node.type,
            'children': [extract_node(child) for child in node.children],
        }
    
    return extract_node(tree.root_node)


def compare_trees(orig, proc, path="root"):
    """Compare two tree structures."""
    if orig['type'] != proc['type']:
        return False, f"Node type mismatch at {path}: {orig['type']} != {proc['type']}"
    
    if len(orig['children']) != len(proc['children']):
        return False, f"Child count mismatch at {path}: {len(orig['children'])} != {len(proc['children'])}"
    
    for i, (orig_child, proc_child) in enumerate(zip(orig['children'], proc['children'])):
        success, error = compare_trees(orig_child, proc_child, f"{path}.children[{i}]")
        if not success:
            return False, error
    
    return True, "OK"


def mock_process_minimal(processor, text: str) -> str:
    """Process text through SemBr pipeline without line breaks for tree structure testing."""
    # Just pass through the processor's generate method without actual processing
    # This tests if the basic parse->generate cycle preserves tree structure
    parsed = processor.parse_text(text)
    
    # Create minimal mock tokenized results (no line breaks)
    mock_tokenized = []
    for region in parsed:
        # Just use the original text as a single "word"
        original_text = region['original_text']
        
        mock_tokenized.append({
            'words': [original_text],
            'modes': ['off'],
            'indents': [0],
            'base_indent': region['base_indent'],
            'start_byte': region['start_byte'],
            'end_byte': region['end_byte'],
            'node_type': region['node_type'],
            'original_text': original_text,
            'continuation_prefix': region['continuation_prefix'],
            'continuation_indent': region['continuation_indent'],
        })
    
    return processor.generate(mock_tokenized)


def _check_markdown_file(processor, parser, filepath: Path) -> tuple[bool, str]:
    """Test a single markdown file."""
    with open(filepath, 'r') as f:
        original = f.read()
    
    try:
        # Process with minimal changes (no line breaks)
        processed = mock_process_minimal(processor, original)
        
        # Compare tree structures
        original_tree = get_tree_structure(parser, original)
        processed_tree = get_tree_structure(parser, processed)
        
        success, error = compare_trees(original_tree, processed_tree)
        
        if success:
            return True, f"✓ {filepath.name}: Tree structure preserved"
        else:
            return False, f"✗ {filepath.name}: {error}"
            
    except Exception as e:
        return False, f"✗ {filepath.name}: Exception - {str(e)}"


def test_markdown_fixtures_preserve_tree_structure():
    """Run tree-structure preservation checks on all markdown fixtures."""
    fixtures_dir = Path(__file__).parent / 'fixtures'
    processor = MarkdownProcessor()
    language = Language(tsmarkdown.language())
    parser = Parser(language)

    failures = []
    for test_file in sorted(fixtures_dir.glob('*.md')):
        success, message = _check_markdown_file(processor, parser, test_file)
        if not success:
            failures.append(message)

    assert not failures, '\n'.join(failures)


def main():
    """Run all markdown syntax tests."""
    # Set up
    fixtures_dir = Path(__file__).parent / 'fixtures'
    processor = MarkdownProcessor()
    
    language = Language(tsmarkdown.language())
    parser = Parser(language)
    
    print("Running Markdown Syntax Tests")
    print("=" * 40)
    
    # Get all test files
    test_files = list(fixtures_dir.glob('*.md'))
    
    if not test_files:
        print(f"No test files found in {fixtures_dir}")
        return 1
    
    # Run tests
    passed = 0
    failed = 0
    
    for test_file in sorted(test_files):
        success, message = _check_markdown_file(processor, parser, test_file)
        print(message)
        
        if success:
            passed += 1
        else:
            failed += 1
    
    print("\n" + "=" * 40)
    print(f"Results: {passed} passed, {failed} failed")
    
    if failed > 0:
        print("\nRunning detailed analysis for failures...")
        # Run actual SemBr processing for failed cases to see output
        for test_file in sorted(test_files):
            success, message = _check_markdown_file(processor, parser, test_file)
            if not success:
                print(f"\nAnalyzing {test_file.name}:")
                with open(test_file, 'r') as f:
                    original = f.read()
                processed = mock_process_minimal(processor, original)
                
                print("ORIGINAL:")
                print(original[:200] + "..." if len(original) > 200 else original)
                print("\nPROCESSED:")
                print(processed[:200] + "..." if len(processed) > 200 else processed)
                print("-" * 40)
    
    return 1 if failed > 0 else 0


if __name__ == '__main__':
    sys.exit(main())

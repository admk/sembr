#!/usr/bin/env python3
"""
Unit tests for markdown processor with advanced syntax.

Tests that process markdown through SemBr and verify the tree structure
remains consistent (only newlines should be inserted).
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest
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
from sembr.cli import init


class TestMarkdownSyntax(unittest.TestCase):
    """Test various markdown syntax elements through SemBr processing."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        cls.fixtures_dir = Path(__file__).parent / 'fixtures'
        cls.processor = MarkdownProcessor()
        
        # Initialize tree-sitter parser for verification
        cls.language = Language(tsmarkdown.language())
        cls.parser = Parser(cls.language)
        
        # Initialize SemBr model (this will load the default model)
        try:
            cls.model, cls.tokenizer = init()
        except Exception as e:
            # If model loading fails, we'll mock it for tree structure tests
            print(f"Warning: Could not load SemBr model: {e}")
            cls.model = None
            cls.tokenizer = None

    def _get_tree_structure(self, text: str) -> dict:
        """Extract tree structure for comparison."""
        tree = self.parser.parse(bytes(text, 'utf8'))
        
        def extract_node(node):
            return {
                'type': node.type,
                'children': [extract_node(child) for child in node.children],
                'start_byte': node.start_byte,
                'end_byte': node.end_byte,
            }
        
        return extract_node(tree.root_node)
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison (collapse multiple newlines)."""
        # Replace multiple consecutive newlines with double newlines
        import re
        normalized = re.sub(r'\n{3,}', '\n\n', text)
        return normalized.strip()
    
    def _process_with_sembr(self, text: str) -> str:
        """Process text through SemBr pipeline."""
        if self.model is None or self.tokenizer is None:
            # Mock processing: just apply basic wrapping for tree structure tests
            return self._mock_process(text)
        
        # Parse text
        parsed = self.processor.parse_text(text)
        
        # Tokenize
        tokenized = self.processor.tokenize_with_modes(self.tokenizer, parsed)
        
        # Apply model predictions (mock some line breaks for testing)
        for region in tokenized:
            predictions = [0] * len(region['words'])
            # Add line breaks every 6-8 words to simulate model predictions
            for i in range(6, len(predictions), 8):
                if i < len(predictions):
                    predictions[i] = 1
            
            # Update modes based on predictions
            for i, pred in enumerate(predictions):
                if pred == 1 and i < len(region['modes']):
                    region['modes'][i] = 'break'
        
        # Generate output
        return self.processor.generate(tokenized)
    
    def _mock_process(self, text: str) -> str:
        """Mock processing for when model is not available."""
        # Just parse through the processor to test tree structure preservation
        parsed = self.processor.parse_text(text)
        
        # Create mock tokenized results
        mock_tokenized = []
        for region in parsed:
            words = region['flat_lines'].split()
            mock_tokenized.append({
                'words': words,
                'modes': ['off'] * len(words),
                'indents': [0] * len(words),
                'base_indent': region['base_indent'],
                'start_byte': region['start_byte'],
                'end_byte': region['end_byte'],
                'node_type': region['node_type'],
                'original_text': region['original_text'],
                'continuation_prefix': region['continuation_prefix'],
                'continuation_indent': region['continuation_indent'],
            })
        
        return self.processor.generate(mock_tokenized)

    def _assert_tree_structure_preserved(self, original: str, processed: str, test_name: str):
        """Assert that tree structure is preserved between original and processed text."""
        # Get tree structures
        original_tree = self._get_tree_structure(original)
        processed_tree = self._get_tree_structure(processed)
        
        # Compare tree structures (ignoring byte positions)
        def compare_trees(orig, proc, path="root"):
            self.assertEqual(orig['type'], proc['type'], 
                           f"{test_name}: Node type mismatch at {path}")
            self.assertEqual(len(orig['children']), len(proc['children']),
                           f"{test_name}: Child count mismatch at {path}")
            
            for i, (orig_child, proc_child) in enumerate(zip(orig['children'], proc['children'])):
                compare_trees(orig_child, proc_child, f"{path}.children[{i}]")
        
        compare_trees(original_tree, processed_tree)
        print(f"✓ {test_name}: Tree structure preserved")

    def test_nested_quotes(self):
        """Test nested block quotes processing."""
        fixture_path = self.fixtures_dir / 'nested_quotes.md'
        with open(fixture_path, 'r') as f:
            original = f.read()
        
        processed = self._process_with_sembr(original)
        self._assert_tree_structure_preserved(original, processed, "Nested Quotes")
        
        # Additional checks for block quotes
        self.assertIn('>', processed)
        self.assertIn('> >', processed)
        print(f"Nested quotes test passed")

    def test_mixed_lists(self):
        """Test mixed list types processing."""
        fixture_path = self.fixtures_dir / 'mixed_lists.md'
        with open(fixture_path, 'r') as f:
            original = f.read()
        
        processed = self._process_with_sembr(original)
        self._assert_tree_structure_preserved(original, processed, "Mixed Lists")
        
        # Additional checks for lists
        self.assertIn('1.', processed)
        self.assertIn('-', processed)
        print(f"Mixed lists test passed")

    def test_task_lists(self):
        """Test task lists processing."""
        fixture_path = self.fixtures_dir / 'task_lists.md'
        with open(fixture_path, 'r') as f:
            original = f.read()
        
        processed = self._process_with_sembr(original)
        self._assert_tree_structure_preserved(original, processed, "Task Lists")
        
        # Additional checks for task lists
        self.assertIn('- [x]', processed)
        self.assertIn('- [ ]', processed)
        print(f"Task lists test passed")

    def test_tables(self):
        """Test table processing."""
        fixture_path = self.fixtures_dir / 'tables.md'
        with open(fixture_path, 'r') as f:
            original = f.read()
        
        processed = self._process_with_sembr(original)
        self._assert_tree_structure_preserved(original, processed, "Tables")
        
        # Additional checks for tables
        self.assertIn('|', processed)
        self.assertIn('---', processed)
        print(f"Tables test passed")

    def test_indented_code(self):
        """Test indented code blocks processing."""
        fixture_path = self.fixtures_dir / 'indented_code.md'
        with open(fixture_path, 'r') as f:
            original = f.read()
        
        processed = self._process_with_sembr(original)
        self._assert_tree_structure_preserved(original, processed, "Indented Code")
        
        # Additional checks for indented code
        lines = processed.split('\n')
        code_lines = [line for line in lines if line.startswith('    ') and line.strip()]
        self.assertTrue(len(code_lines) > 0, "Should preserve indented code blocks")
        print(f"Indented code test passed")

    def test_links(self):
        """Test links and references processing."""
        fixture_path = self.fixtures_dir / 'links.md'
        with open(fixture_path, 'r') as f:
            original = f.read()
        
        processed = self._process_with_sembr(original)
        self._assert_tree_structure_preserved(original, processed, "Links")
        
        # Additional checks for links
        self.assertIn('[', processed)
        self.assertIn('](', processed)
        self.assertIn('[1]:', processed)
        print(f"Links test passed")

    def test_inline_formatting(self):
        """Test complex inline formatting processing."""
        fixture_path = self.fixtures_dir / 'inline_formatting.md'
        with open(fixture_path, 'r') as f:
            original = f.read()
        
        processed = self._process_with_sembr(original)
        self._assert_tree_structure_preserved(original, processed, "Inline Formatting")
        
        # Additional checks for inline formatting
        self.assertIn('**', processed)
        self.assertIn('*', processed)
        self.assertIn('`', processed)
        print(f"Inline formatting test passed")

    def test_horizontal_rules(self):
        """Test horizontal rules processing."""
        fixture_path = self.fixtures_dir / 'horizontal_rules.md'
        with open(fixture_path, 'r') as f:
            original = f.read()
        
        processed = self._process_with_sembr(original)
        self._assert_tree_structure_preserved(original, processed, "Horizontal Rules")
        
        # Additional checks for horizontal rules
        self.assertIn('---', processed)
        self.assertIn('***', processed)
        self.assertIn('___', processed)
        print(f"Horizontal rules test passed")

    def test_headers(self):
        """Test headers and sections processing."""
        fixture_path = self.fixtures_dir / 'headers.md'  
        with open(fixture_path, 'r') as f:
            original = f.read()
        
        processed = self._process_with_sembr(original)
        self._assert_tree_structure_preserved(original, processed, "Headers")
        
        # Additional checks for headers
        self.assertIn('#', processed)
        self.assertIn('##', processed)
        self.assertIn('###', processed)
        print(f"Headers test passed")

    def test_comprehensive(self):
        """Test a comprehensive markdown document."""
        # Create a comprehensive test combining multiple elements
        comprehensive_md = """# Comprehensive Test

This is a paragraph with **bold** and *italic* text.

## Lists and Quotes

1. First item with [link](https://example.com)
   - Nested bullet with `code`
   - Another nested item
2. Second item

> Block quote with long text that should be wrapped properly
> > Nested quote

## Code and Tables

```python
def hello():
    print("world")
```

| Col1 | Col2 |
|------|------|
| A    | B    |

---

Final paragraph.
"""
        
        processed = self._process_with_sembr(comprehensive_md)
        self._assert_tree_structure_preserved(comprehensive_md, processed, "Comprehensive")
        print(f"Comprehensive test passed")


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)

#!/usr/bin/env python3
"""
Test runner that uses actual SemBr processing to verify line breaking behavior.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pathlib import Path
import subprocess


def test_with_actual_sembr(test_file: Path) -> tuple[bool, str]:
    """Test with actual SemBr CLI."""
    output_file = test_file.with_suffix('.output.md')

    try:
        # Run SemBr on the file
        result = subprocess.run([
            'uv', 'run', 'sembr',
            '--file-type', 'markdown',
            '-i', str(test_file),
            '-o', str(output_file)
        ], capture_output=True, text=True, cwd=test_file.parent.parent)

        if result.returncode != 0:
            return False, f"SemBr failed: {result.stderr}"

        # Read results
        with open(test_file, 'r') as f:
            original = f.read()
        with open(output_file, 'r') as f:
            processed = f.read()

        # Basic checks
        checks = []

        # Check that structure markers are preserved
        if '# ' in original and '# ' not in processed:
            checks.append("Headers missing")
        if '- ' in original and '- ' not in processed:
            checks.append("List markers missing")
        if '> ' in original and '> ' not in processed:
            checks.append("Block quote markers missing")
        if '```' in original and '```' not in processed:
            checks.append("Code block markers missing")
        if '|' in original and '|' not in processed:
            checks.append("Table markers missing")
        if '[' in original and ']' in original and ('[' not in processed or ']' not in processed):
            checks.append("Link markers missing")

        # Check that content exists
        if len(processed.strip()) < len(original.strip()) * 0.8:
            checks.append("Significant content loss")

        # Clean up
        if output_file.exists():
            output_file.unlink()

        if checks:
            return False, f"Issues: {', '.join(checks)}"
        else:
            return True, "All structural elements preserved"

    except Exception as e:
        return False, f"Exception: {str(e)}"


def main():
    """Run actual SemBr tests on all fixtures."""
    fixtures_dir = Path(__file__).parent / 'fixtures'
    test_files = list(fixtures_dir.glob('*.md'))

    print("Running Actual SemBr Processing Tests")
    print("=" * 45)

    passed = 0
    failed = 0

    for test_file in sorted(test_files):
        success, message = test_with_actual_sembr(test_file)
        status = "✓" if success else "✗"
        print(f"{status} {test_file.name}: {message}")

        if success:
            passed += 1
        else:
            failed += 1

    print("\n" + "=" * 45)
    print(f"Results: {passed} passed, {failed} failed")

    return 1 if failed > 0 else 0


if __name__ == '__main__':
    sys.exit(main())

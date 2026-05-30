#!/usr/bin/env python3
"""
Test runner that uses actual SemBr processing to verify line breaking behavior.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pathlib import Path
import subprocess
import socket
import tempfile
import time
import urllib.error
import urllib.request

import pytest


FIXTURES_DIR = Path(__file__).parent / 'fixtures'
FIXTURE_FILES = sorted(FIXTURES_DIR.glob('*.md'))
REPO_ROOT = Path(__file__).parent.parent


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def _wait_for_server(port: int, timeout: float = 60.0) -> None:
    deadline = time.time() + timeout
    url = f'http://127.0.0.1:{port}/check'
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0):
                return
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            pass
        time.sleep(0.5)
    raise RuntimeError(f'SemBr listen server did not become ready on port {port}.')


def _start_actual_sembr_listener():
    port = _find_free_port()
    env = os.environ.copy()
    env['XDG_CONFIG_HOME'] = str(REPO_ROOT / '.pytest-xdg')
    log = tempfile.NamedTemporaryFile(
        mode='w+', prefix='sembr-listen-', suffix='.log', delete=False)
    proc = subprocess.Popen(
        [
            'uv', 'run', 'sembr', '--listen',
            '-c', f'server.port={port}',
        ],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=log,
    )
    _wait_for_server(port)
    log.close()
    return proc, port, env, Path(log.name)


def _stop_actual_sembr_listener(proc):
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


@pytest.fixture(scope='session')
def actual_sembr_runtime():
    proc, port, env, log_path = _start_actual_sembr_listener()
    try:
        yield port, env, log_path
    finally:
        _stop_actual_sembr_listener(proc)


def _check_with_actual_sembr(test_file: Path, port: int, env: dict) -> tuple[bool, str]:
    """Test with actual SemBr CLI."""
    output_file = test_file.with_suffix('.output.md')

    try:
        # Run SemBr on the file
        result = subprocess.run([
            'uv', 'run', 'sembr',
            '--file-type', 'markdown',
            '-c', f'server.port={port}',
            '-i', str(test_file),
            '-o', str(output_file)
        ], capture_output=True, text=True, cwd=test_file.parent.parent, env=env)

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


@pytest.mark.parametrize(
    'test_file',
    FIXTURE_FILES,
    ids=[path.name for path in FIXTURE_FILES],
)
def test_actual_sembr_fixture(test_file, actual_sembr_runtime):
    """Run actual SemBr CLI processing on a markdown fixture."""
    port, env, log_path = actual_sembr_runtime
    success, message = _check_with_actual_sembr(test_file, port, env)
    if not success and log_path.exists():
        message = f'{message}\n\nServer log:\n{log_path.read_text()}'

    assert success, message


def main():
    """Run actual SemBr tests on all fixtures."""
    print("Running Actual SemBr Processing Tests")
    print("=" * 45)

    proc, port, env, _ = _start_actual_sembr_listener()
    passed = 0
    failed = 0

    try:
        for test_file in FIXTURE_FILES:
            success, message = _check_with_actual_sembr(test_file, port, env)
            status = "✓" if success else "✗"
            print(f"{status} {test_file.name}: {message}")

            if success:
                passed += 1
            else:
                failed += 1
    finally:
        _stop_actual_sembr_listener(proc)

    print("\n" + "=" * 45)
    print(f"Results: {passed} passed, {failed} failed")

    return 1 if failed > 0 else 0


if __name__ == '__main__':
    sys.exit(main())

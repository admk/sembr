import platform
import shutil
import sys


def normalized_os():
    if sys.platform == 'darwin':
        return 'macos'
    if sys.platform.startswith('linux'):
        return 'linux'
    if sys.platform.startswith(('win32', 'cygwin', 'msys')):
        return 'windows'
    return sys.platform.lower()


def normalized_machine():
    machine = platform.machine().lower()
    aliases = {
        'aarch64': 'arm64',
        'amd64': 'x86_64',
    }
    return aliases.get(machine, machine)


def platform_override_keys():
    raw_os = sys.platform.lower()
    os_name = normalized_os()
    machine = normalized_machine()
    os_keys = list(dict.fromkeys([raw_os, os_name]))
    keys = list(os_keys)
    if machine:
        keys.extend(f'{os_key}-{machine}' for os_key in os_keys)
    return tuple(keys)


def has_cuda_device_hint():
    return shutil.which('nvidia-smi') is not None


def recommended_backend_extra():
    os_name = normalized_os()
    machine = normalized_machine()
    if os_name == 'macos' and machine == 'arm64':
        return 'mlx'
    if os_name == 'linux' and has_cuda_device_hint():
        return 'cuda'
    return 'torch'


def install_command_for_extra(extra=None):
    extra = recommended_backend_extra() if extra is None else extra
    return f'uv tool install "sembr[{extra}]"'

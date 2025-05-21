from packaging import version


def min_torch_version(min_version: str) -> bool:
    """
    Returns True when torch's  version is at least `min_version`.
    """
    try:
        import torch
    except ImportError:
        return False

    installed_version = version.parse(torch.__version__.split("+")[0])
    min_version = version.parse(min_version)
    return installed_version >= min_version

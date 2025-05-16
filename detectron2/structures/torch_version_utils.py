import torch

import re


def is_fbcode():
    return not hasattr(torch.version, "git_version")

def parse_version(version_string):
    # Extract just the X.Y.Z part from the version string
    match = re.match(r"(\d+\.\d+\.\d+)", version_string)
    if match:
        version = match.group(1)
        return [int(x) for x in version.split(".")]
    else:
        raise ValueError(f"Invalid version string format: {version_string}")

def compare_versions(v1, v2):
    v1_parts = parse_version(v1)
    v2_parts = parse_version(v2)
    return (v1_parts > v2_parts) - (v1_parts < v2_parts)

def torch_version_at_least(min_version):
    return is_fbcode() or compare_versions(torch.__version__, min_version) >= 0

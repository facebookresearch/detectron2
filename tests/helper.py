# noqa D100
import unittest


def skipIfUnsupportedMinOpsetVersion(min_opset_version):
    """Skips tests for all versions below min_opset_version.

    if exporting the op is only supported after a specific version,
    add this wrapper to prevent running the test for opset_versions
    smaller than the currently tested opset_version
    """
    def skip_dec(func):
        def wrapper(self):
            if self.opset_version < min_opset_version:
                raise unittest.SkipTest(
                    f"Unsupported opset_version: {self.opset_version} < {min_opset_version}"
                )
            return func(self)

        return wrapper

    return skip_dec

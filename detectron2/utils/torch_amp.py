try:
    from torch.amp import autocast as _autocast

    def autocast(dtype=None):
        return _autocast("cuda", dtype=dtype)

except ImportError:
    from torch.cuda.amp import autocast  # torch < 2.0
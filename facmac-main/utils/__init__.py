import os, warnings
warnings.warn("Top-level package 'utils' is a compatibility shim; prefer importing 'src.utils' instead.", DeprecationWarning)
_this_dir = os.path.dirname(__file__)
_repo_root = os.path.dirname(_this_dir)
_real = os.path.join(_repo_root, 'src', 'utils')
if os.path.isdir(_real):
    __path__[:] = [_real]
else:
    pass

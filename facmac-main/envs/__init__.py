import os, warnings
# This shim is deprecated. Prefer importing the package via the project package
# name (the `src` package) when running as a module (e.g. `python -m src.main`).
warnings.warn("Top-level package 'envs' is a compatibility shim; prefer importing 'src.envs' instead.", DeprecationWarning)
_this_dir = os.path.dirname(__file__)
_repo_root = os.path.dirname(_this_dir)
_real = os.path.join(_repo_root, 'src', 'envs')
if os.path.isdir(_real):
    # Keep backward compatible by adjusting __path__, but warn so callers migrate.
    __path__[:] = [_real]
else:
    # fallback: keep default behaviour
    pass

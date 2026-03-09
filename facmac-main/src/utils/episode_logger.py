import json
import os
import time
from typing import Optional, Dict, Any, List

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # tensorboard may not be installed yet
    SummaryWriter = None  # type: ignore


def _json_safe(obj):
    """Recursively convert objects that json.dumps can't handle (e.g. numpy types)
    into plain Python builtins (int, float, str, list, dict, None).
    - numpy integers/floats -> int/float
    - numpy arrays -> lists
    - lists/tuples/dicts -> recursively converted
    - other unknown objects -> str(obj)
    """
    try:
        import numpy as _np
    except Exception:
        _np = None

    # None
    if obj is None:
        return None

    # numpy scalar -> python scalar
    if _np is not None:
        if isinstance(obj, (_np.integer,)):
            return int(obj)
        if isinstance(obj, (_np.floating,)):
            v = float(obj)
            # handle NaN/Inf gracefully
            if _np.isnan(v) or _np.isinf(v):
                return None
            return v
        if isinstance(obj, (_np.bool_,)):
            return bool(obj)
        if isinstance(obj, _np.ndarray):
            try:
                return _json_safe(obj.tolist())
            except Exception:
                # fallback to list conversion elementwise
                return [_json_safe(x) for x in obj]

    # basic python types
    if isinstance(obj, (bool, int, float, str)):
        return obj

    # lists / tuples / sets
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(x) for x in obj]

    # dict
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            try:
                key = str(k)
            except Exception:
                key = repr(k)
            out[key] = _json_safe(v)
        return out

    # try numeric conversion
    try:
        return float(obj)
    except Exception:
        pass

    # fallback to string
    try:
        return str(obj)
    except Exception:
        return None


class EpisodeDualLogger:
    """
    Write per-episode logs to both TensorBoard (scalars only) and JSONL (full details).

    - TensorBoard: only key scalar metrics, one point per episode via writer.add_scalar(tag, value, episode)
    - JSONL: a full JSON object per episode (one line per episode)
    """

    def __init__(
        self,
        base_results_dir: str,
        run_token: str,
        enable_tb: bool = True,
        enable_jsonl: bool = True,
        tb_subdir: str = "tensorboard",
        jsonl_subdir: str = "jsonl",
        jsonl_filename: str = "episodes.jsonl",
    ) -> None:
        self.enable_tb = bool(enable_tb and SummaryWriter is not None)
        self.enable_jsonl = bool(enable_jsonl)

        # Prepare directories
        self.tb_dir = os.path.join(base_results_dir, tb_subdir, run_token)
        self.jsonl_dir = os.path.join(base_results_dir, jsonl_subdir, run_token)
        if self.enable_tb:
            os.makedirs(self.tb_dir, exist_ok=True)
        if self.enable_jsonl:
            os.makedirs(self.jsonl_dir, exist_ok=True)

        # TensorBoard writer
        self.tb_writer: Optional[Any] = None
        if self.enable_tb and SummaryWriter is not None:
            self.tb_writer = SummaryWriter(log_dir=self.tb_dir)  # type: ignore

        # JSONL file handle
        self.jsonl_path = os.path.join(self.jsonl_dir, jsonl_filename)
        self._fh = None
        if self.enable_jsonl:
            # line-buffered text file; newline='\n' for consistent JSONL formatting
            self._fh = open(self.jsonl_path, mode="a", encoding="utf-8", buffering=1, newline="\n")

    def log_episode(self, episode: int, scalars: Dict[str, Any], details: Dict[str, Any], images: Dict[str, Any] = None) -> None:
        """
        - episode: episode index (int)
        - scalars: dict of scalar metrics to log to TensorBoard
        - details: dict with any JSON-serializable objects to store in JSONL
        - images: optional dict of images to write to TensorBoard. Values can be:
            - matplotlib.figure.Figure
            - numpy.ndarray (H,W) or (H,W,3) or (C,H,W) etc.
        """
        ts = int(time.time())

        # TB: write selected scalars
        if self.tb_writer is not None:
            for k, v in list(scalars.items()):
                try:
                    # accept bool/ints/floats; cast others when possible
                    if isinstance(v, (bool, int, float)):
                        self.tb_writer.add_scalar(k, float(v), global_step=episode)
                    else:
                        self.tb_writer.add_scalar(k, float(v), global_step=episode)
                except Exception:
                    # skip non-numeric
                    continue
            # also expose a wallclock timestamp for reference
            self.tb_writer.add_scalar("meta/timestamp", ts, global_step=episode)
            # Ensure the data is flushed to disk so TensorBoard can read it promptly
            try:
                self.tb_writer.flush()
            except Exception:
                try:
                    if hasattr(self.tb_writer, 'file_writer') and hasattr(self.tb_writer.file_writer, 'flush'):
                        self.tb_writer.file_writer.flush()
                except Exception:
                    pass

            # NEW: write images if present
            if images is not None:
                try:
                    # import here to avoid hard dependency unless used
                    import numpy as _np
                    import matplotlib
                    from matplotlib.figure import Figure
                except Exception:
                    _np = None
                    Figure = None

                for img_tag, img_val in list(images.items()):
                    try:
                        if Figure is not None and isinstance(img_val, Figure):
                            # write matplotlib figure directly
                            try:
                                self.tb_writer.add_figure(f"images/{img_tag}", img_val, global_step=episode)
                            except Exception:
                                # some SummaryWriter versions require figure canvas draw
                                pass
                        elif _np is not None and isinstance(img_val, _np.ndarray):
                            arr = img_val
                            # Normalize/reshape heuristics
                            try:
                                if arr.ndim == 3 and arr.shape[2] in (3, 4):
                                    # H,W,C -> use dataformats='HWC'
                                    self.tb_writer.add_image(f"images/{img_tag}", arr, global_step=episode, dataformats='HWC')
                                elif arr.ndim == 2:
                                    # H,W -> expand to HWC
                                    arr2 = arr[:, :, None]
                                    self.tb_writer.add_image(f"images/{img_tag}", arr2, global_step=episode, dataformats='HWC')
                                elif arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
                                    # C,H,W -> dataformats='CHW'
                                    self.tb_writer.add_image(f"images/{img_tag}", arr, global_step=episode, dataformats='CHW')
                                else:
                                    # fallback: try to convert to float image
                                    self.tb_writer.add_image(f"images/{img_tag}", arr, global_step=episode)
                            except Exception:
                                # last resort: try converting to uint8
                                try:
                                    arr_u8 = (arr * 255).astype('uint8') if arr.dtype != 'uint8' else arr
                                    self.tb_writer.add_image(f"images/{img_tag}", arr_u8, global_step=episode)
                                except Exception:
                                    continue
                        else:
                            # unsupported type -> try string representation as scalar
                            try:
                                self.tb_writer.add_text(f"images/{img_tag}", str(img_val), global_step=episode)
                            except Exception:
                                pass
                    except Exception:
                        # avoid crashing logging for any image error
                        continue

            # flush after writing images as well
            try:
                self.tb_writer.flush()
            except Exception:
                pass

        # JSONL: write full details per episode
        if self._fh is not None:
            record = {
                "episode": int(episode),
                "timestamp": ts,
            }
            if scalars:
                record.update({str(k): v for k, v in scalars.items()})
            if details:
                record.update({str(k): v for k, v in details.items()})
            try:
                # ensure nested numpy types and arrays are converted to Python builtins
                line = json.dumps(_json_safe(record), ensure_ascii=False)
            except TypeError:
                # fallback: attempt to coerce non-serializable values
                def _coerce(obj):
                    try:
                        json.dumps(obj)
                        return obj
                    except Exception:
                        try:
                            return float(obj)
                        except Exception:
                            try:
                                return str(obj)
                            except Exception:
                                return None
                record = {k: _coerce(v) for k, v in record.items()}
                line = json.dumps(record, ensure_ascii=False)
            self._fh.write(line + "\n")
            # flush to make it available for tail -f / readers
            self._fh.flush()

        # 只在每100轮保存一次json
        if self.enable_jsonl and episode > 0 and episode % 100 == 0:
            record = {
                "episode": int(episode),
                "timestamp": ts,
            }
            if scalars:
                record.update({str(k): v for k, v in scalars.items()})
            if details:
                record.update({str(k): v for k, v in details.items()})
            try:
                record = _json_safe(record)
            except Exception:
                pass
            fname = os.path.join(self.jsonl_dir, f"episodes_{episode}.json")
            try:
                with open(fname, "w", encoding="utf-8") as f:
                    json.dump(record, f, ensure_ascii=False, indent=2)
            except Exception:
                pass

    def close(self) -> None:
        if self.tb_writer is not None:
            try:
                self.tb_writer.flush()
            except Exception:
                pass
            try:
                self.tb_writer.close()
            except Exception:
                pass
            self.tb_writer = None
        if self._fh is not None:
            try:
                self._fh.flush()
            except Exception:
                pass
            try:
                self._fh.close()
            except Exception:
                pass
            self._fh = None

    # Context manager support (optional)
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def generate_loss_plot(self, output_filename: str = "loss_curve.png", keys: Optional[List[str]] = None) -> Optional[str]:
        """
        Read the JSONL file and generate a PNG loss curve plot for selected keys.
        - output_filename: name of PNG to write under the same jsonl_dir
        - keys: list of metric names to plot (defaults to ['critic_loss','pg_loss'])
        Returns the path to the written PNG, or None if failed.
        """
        try:
            import numpy as _np
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except Exception:
            return None

        if keys is None:
            keys = ["critic_loss", "pg_loss"]

        # Read JSONL
        data: Dict[str, list] = {k: [] for k in keys}
        episodes: List[int] = []
        try:
            with open(self.jsonl_path, mode="r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    ep = rec.get("episode")
                    if ep is None:
                        continue
                    episodes.append(int(ep))
                    for k in keys:
                        v = rec.get(k)
                        try:
                            data[k].append(float(v) if v is not None else _np.nan)
                        except Exception:
                            data[k].append(_np.nan)
        except Exception:
            return None

        if len(episodes) == 0:
            return None

        # Plot
        fig, ax = plt.subplots(figsize=(8, 5))
        for k in keys:
            if len(data.get(k, [])) == len(episodes):
                ax.plot(episodes, data[k], label=k)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss Curves")
        ax.legend()
        ax.grid(True, alpha=0.3)
        out_path = os.path.join(self.jsonl_dir, output_filename)
        try:
            fig.savefig(out_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            return out_path
        except Exception:
            try:
                plt.close(fig)
            except Exception:
                pass
            return None

"""
Plot UAV experiment summary metrics from results JSON files.

Usage (from repo root):
    python plot_rewards.py --dir results --out results/plots

If --dir is omitted it defaults to 'results'. The script will:
 - search for files matching episode_*.json under the directory
 - load per-episode summary metrics (success_rate, collision_rate, avg_latency_rate)
 - produce line plots (PNG) and an aggregate CSV
"""
import os
import glob
import json
import argparse
import csv
import math
import shutil

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except Exception as e:
    print('matplotlib not available:', e)
    plt = None


def discover_files(results_dir):
    pattern1 = os.path.join(results_dir, '**', 'episode_*.json')
    pattern2 = os.path.join(results_dir, '**', 'episodes_*.json')
    files = []
    files.extend(glob.glob(pattern1, recursive=True))
    files.extend(glob.glob(pattern2, recursive=True))
    files = sorted(set(files))
    return files


def _find_nearest_config_json(start_path, stop_at):
    """Search upward from start_path for a config.json file until stop_at directory."""
    cur = os.path.abspath(start_path)
    stop_at = os.path.abspath(stop_at)
    while True:
        cfg_path = os.path.join(cur, 'config.json')
        if os.path.isfile(cfg_path):
            return cfg_path
        if cur == stop_at or os.path.dirname(cur) == cur:
            break
        cur = os.path.dirname(cur)
    return None


def _build_run_metadata_index(results_root):
    """Scan results_root for run/config JSON files and return a list of tuples (dir, json_content, json_str).

    This helps mapping an episode's master_seed (from filename) to a run's metadata.
    """
    candidates = []
    # common sacred layout: results/sacred/<run_id>/config.json or run.json
    patterns = [os.path.join(results_root, '**', 'config.json'),
                os.path.join(results_root, '**', 'run.json'),
                os.path.join(results_root, '**', 'info.json')]
    seen = set()
    for pat in patterns:
        for f in glob.glob(pat, recursive=True):
            d = os.path.dirname(f)
            if f in seen:
                continue
            seen.add(f)
            try:
                with open(f, 'r', encoding='utf-8') as fh:
                    j = json.load(fh)
            except Exception:
                j = None
            s = json.dumps(j) if j is not None else ''
            candidates.append((d, j, s))
    return candidates


def filter_by_algo(filepaths, algo, results_root):
    """Return subset of filepaths whose content or nearby config indicates the given algo string.

    Heuristics:
    - If the episode JSON contains any string value that includes algo -> match.
    - Else look for nearest config.json upward from the episode file's directory up to results_root and inspect it.
    - Else check if the algorithm name appears in the file path.
    """
    if not algo:
        return filepaths
    kept = []
    algo_lower = str(algo).lower()
    # build run metadata index once
    run_meta = _build_run_metadata_index(results_root)

    for p in filepaths:
        matched = False
        # 1) check episode JSON content
        try:
            with open(p, 'r', encoding='utf-8') as fh:
                j = json.load(fh)
        except Exception:
            j = None
        if isinstance(j, dict):
            # search for algo substring in any stringified value
            try:
                for v in j.values():
                    if isinstance(v, str) and algo_lower in v.lower():
                        matched = True
                        break
                    # nested dict or list -> stringify check
                    if not matched and algo_lower in str(v).lower():
                        matched = True
                        break
            except Exception:
                matched = False
        if matched:
            kept.append(p)
            continue

        # 2) Try mapping by master_seed extracted from filename to run metadata
        import re
        m = re.match(r'.*episode_\d+_(?P<master>\d+)_(?P<sub>\d+)\.json$', os.path.basename(p))
        if m:
            master_seed = m.group('master')
            subseed = m.group('sub')
            # search run_meta for this seed (as number or string) and check content
            for (d, j, s) in run_meta:
                try:
                    if j is None:
                        # fallback to string search
                        if master_seed in s:
                            if algo_lower in s.lower():
                                matched = True
                                break
                        continue
                    # search values recursively in j
                    js = json.dumps(j).lower()
                    if master_seed.lower() in js:
                        if algo_lower in js:
                            matched = True
                            break
                except Exception:
                    continue
        if matched:
            kept.append(p)
            continue

        # 3) check path
        if algo_lower in p.lower():
            kept.append(p)
            continue

    return kept


def load_summaries(files):
    rows = []
    import re
    for f in files:
        try:
            with open(f, 'r', encoding='utf-8') as fh:
                j = json.load(fh)
        except Exception:
            continue
        if not isinstance(j, dict):
            continue

        # episode_id: prefer explicit key, fall back to 'episode', else parse from filename
        episode_id = j.get('episode_id')
        if episode_id is None:
            episode_id = j.get('episode')
        if episode_id is None:
            m = re.search(r'episodes?_(\d+)\.json$', os.path.basename(f))
            if m:
                try:
                    episode_id = int(m.group(1))
                except Exception:
                    episode_id = None

        # total_reward: prefer explicit key, else compute from rewards list
        total_reward = j.get('total_reward')
        if total_reward is None:
            # compatible with older keys
            er = j.get('episode_rewards') or j.get('step_rewards')
            if isinstance(er, list) and len(er) > 0:
                try:
                    import numpy as _np
                    arr = _np.array(er, dtype=float)
                    total_reward = float(_np.nansum(arr))
                except Exception:
                    ssum = 0.0
                    ok = False
                    for v in er:
                        try:
                            if v is None:
                                continue
                            fv = float(v)
                            if math.isnan(fv):
                                continue
                            ssum += fv
                            ok = True
                        except Exception:
                            continue
                    total_reward = float(ssum) if ok else None

        row = {
            'file': f,
            'episode_id': episode_id,
            'success_rate': j.get('success_rate'),
            'collision_rate': j.get('collision_rate'),
            'critic_loss': j.get('critic_loss'),
            'actor_loss': j.get('actor_loss'),
            'total_reward': total_reward,
            # extras retained for plotting/backward-compat (not written to CSV)
            'timestamp': j.get('timestamp') or 0,
            'episode_total_reward': total_reward,
        }
        rows.append(row)
    return rows


def write_csv(rows, out_csv):
    if not rows:
        return
    os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
    # Fixed column order per user request
    fieldnames = ['file', 'episode_id', 'success_rate', 'collision_rate', 'critic_loss', 'actor_loss', 'total_reward']
    with open(out_csv, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for r in rows:
            # Ensure only requested keys are written and in order
            writer.writerow({k: r.get(k) for k in fieldnames})


def plot_lines(rows, out_dir, algo=None):
    if plt is None:
        print('matplotlib is not available, skipping plotting')
        return
    os.makedirs(out_dir, exist_ok=True)

    x = list(range(len(rows)))
    def series(key):
        s = []
        for r in rows:
            v = r.get(key)
            if v is None:
                s.append(float('nan'))
            else:
                try:
                    s.append(float(v))
                except Exception:
                    s.append(float('nan'))
        return s

    # 只保留四类曲线
    episode_reward = series('episode_total_reward')
    critic_loss = series('critic_loss')
    success = series('success_rate')
    collision = series('collision_rate')

    # Helper to save a single plot
    def save_plot(y, ylabel, fname):
        plt.figure(figsize=(8,4))
        # ensure numeric arrays and draw lines connecting points
        try:
            import numpy as _np
            x_arr = _np.array(x, dtype=float)
            y_arr = _np.array(y, dtype=float)
        except Exception:
            x_arr = x
            y_arr = y
        plt.plot(x_arr, y_arr, marker='o', linestyle='-')
        plt.xlabel('episode_index')
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        # include algo prefix in filename if provided
        if algo:
            base, ext = os.path.splitext(fname)
            fname = f"{base}_{algo}{ext}"
        out = os.path.join(out_dir, fname)
        plt.savefig(out)
        plt.close()
        print('Saved', out)

    save_plot(episode_reward, 'episode_total_reward', 'episode_total_reward.png')
    save_plot(critic_loss, 'critic_loss', 'critic_loss.png')
    save_plot(success, 'success_rate', 'success_rate.png')
    save_plot(collision, 'collision_rate', 'collision_rate.png')
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='results', help='results directory to scan')
    parser.add_argument('--out', type=str, default=None, help='output directory for plots (default: <dir>/plots)')
    parser.add_argument('--algo', type=str, default=None, help='filter by algorithm name (substring match)')
    parser.add_argument('--algos', type=str, default=None, help='filter by multiple algorithm names (comma-separated)')
    args = parser.parse_args()

    results_dir = args.dir
    out_dir = args.out or os.path.join(results_dir, 'plots')

    # Support multiple algorithms via --algos (comma-separated) or single --algo
    algos = []
    if args.algos:
        algos = [a.strip() for a in args.algos.split(',') if a.strip()]
    elif args.algo:
        algos = [args.algo]

    if not algos:
        # default: no filtering, process all files into out_dir
        files = discover_files(results_dir)
        print('Found', len(files), 'episode files under', results_dir)
        if not files:
            print('No episode files found - make sure your results dir and filename pattern match (episode_*.json)')
            return
        rows = load_summaries(files)
        csv_path = os.path.join(out_dir, 'aggregate.csv')
        write_csv(rows, csv_path)
        print('Wrote CSV to', csv_path)
        plot_lines(rows, out_dir)
        return

    # For each requested algorithm, filter files and generate plots under a subdirectory
    for algo in algos:
        files = discover_files(results_dir)
        files = filter_by_algo(files, algo, results_dir)
        print(f"Algorithm '{algo}': found {len(files)} episode files")
        if not files:
            print(f"No files found for algorithm '{algo}', skipping.")
            continue

        # Organize per-algo directories under the results root so JSON and plots are grouped
        # Base algo directory (under results_dir) e.g., results/<algo>
        algo_base = os.path.join(results_dir, algo)
        json_dir = os.path.join(algo_base, 'json')
        plots_dir = os.path.join(algo_base, 'plots')
        os.makedirs(json_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)

        # Copy matching episode JSONs into the per-algo json folder (skip if already exists)
        copied_files = []
        for src in files:
            try:
                bn = os.path.basename(src)
                dst = os.path.join(json_dir, bn)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
                copied_files.append(dst)
            except Exception as e:
                print(f"Failed to copy {src} to {json_dir}: {e}")

        # Load summaries from the copied files to ensure plotting uses only these JSONs
        rows = load_summaries(copied_files)

        # Write CSV and plots into the per-algo plots folder
        csv_path = os.path.join(plots_dir, f'aggregate_{algo}.csv')
        write_csv(rows, csv_path)
        print('Wrote', csv_path)
        plot_lines(rows, plots_dir, algo=algo)

if __name__ == '__main__':
    main()

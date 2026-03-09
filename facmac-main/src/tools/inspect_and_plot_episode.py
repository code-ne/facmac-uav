import os
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Project root is two levels up from this script (src/tools)
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'results'))
PLOTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'plots'))
os.makedirs(PLOTS_DIR, exist_ok=True)

def find_first_episode_file():
    files = [f for f in os.listdir(RESULTS_DIR) if f.startswith('episode_') and f.endswith('.json')]
    if not files:
        return None
    files.sort()
    return os.path.join(RESULTS_DIR, files[0])


def looks_like_xyz_list(v):
    # detect lists of numeric triplets
    try:
        arr = np.array(v)
        if arr.ndim == 2 and arr.shape[1] == 3 and np.issubdtype(arr.dtype, np.number):
            return True
    except Exception:
        pass
    return False


def find_trajectory_candidates(obj, path=''):
    candidates = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            candidates += find_trajectory_candidates(v, path + '/' + k)
    elif isinstance(obj, list):
        # Check if it's T x N x 3 or N x T x 3 or T x 3 (single agent) or N x 3 (one time step)
        try:
            a = np.array(obj)
            if a.ndim == 3 and a.shape[2] == 3:
                candidates.append((path, 'T,N,3', a.shape))
            elif a.ndim == 2 and a.shape[1] == 3:
                candidates.append((path, 'T,3', a.shape))
            elif a.ndim == 2 and a.shape[0] == 3:
                candidates.append((path, '3,T', a.shape))
        except Exception:
            pass
        # recurse first-level to find nested lists
        for i, item in enumerate(obj[:10]):
            candidates += find_trajectory_candidates(item, path + f'[{i}]')
    return candidates


def load_and_inspect(path):
    print('Loading', path)
    with open(path, 'r', encoding='utf-8') as f:
        j = json.load(f)
    print('Top-level keys:', list(j.keys()))
    cands = find_trajectory_candidates(j)
    print('Found trajectory-like candidates:')
    for p, kind, shape in cands:
        print('  ', p, kind, shape)
    return j, cands


def plot_candidate(j, candidate, out_name):
    path, kind, shape = candidate
    # resolve path (simple parser)
    def resolve(obj, path):
        if path == '':
            return obj
        parts = path.strip('/').split('/')
        cur = obj
        for p in parts:
            if p == '':
                continue
            if '[' in p:
                name, rest = p.split('[', 1)
                idx = int(rest.split(']')[0])
                cur = cur.get(name) if isinstance(cur, dict) else cur
                cur = cur[idx]
            else:
                if isinstance(cur, dict):
                    cur = cur.get(p)
                else:
                    return None
        return cur

    data = resolve(j, path)
    if data is None:
        print('Failed to resolve path', path)
        return False
    arr = np.array(data)
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection='3d')
    if arr.ndim == 3 and arr.shape[2] == 3:
        # assume shape T x N x 3 or N x T x 3; pick orientation with larger T as time
        T, N, _ = arr.shape
        if N > T:
            # maybe N x T x 3 -> transpose
            arr = arr.transpose(1,0,2)
            T, N, _ = arr.shape
        for i in range(N):
            traj = arr[:, i, :]
            ax.plot(traj[:,0], traj[:,1], traj[:,2], label=f'UAV {i}')
            ax.scatter(traj[0,0], traj[0,1], traj[0,2], marker='o', s=20, color='k')
            ax.scatter(traj[-1,0], traj[-1,1], traj[-1,2], marker='*', s=50)
    elif arr.ndim == 2 and arr.shape[1] == 3:
        ax.plot(arr[:,0], arr[:,1], arr[:,2], label='traj')
        ax.scatter(arr[0,0], arr[0,1], arr[0,2], marker='o', s=20, color='k')
        ax.scatter(arr[-1,0], arr[-1,1], arr[-1,2], marker='*', s=50)
    else:
        print('Unhandled array shape for plotting:', arr.shape)
        return False
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.legend()
    plt.title('Episode trajectories')
    outp = os.path.join(PLOTS_DIR, out_name)
    plt.savefig(outp)
    plt.close()
    print('Saved plot to', outp)
    return True


if __name__ == '__main__':
    path = find_first_episode_file()
    if path is None:
        print('No episode files found in', RESULTS_DIR)
        raise SystemExit(1)
    j, cands = load_and_inspect(path)
    if not cands:
        print('No trajectory-like arrays found in the episode JSON.')
        raise SystemExit(2)
    # choose the largest candidate (by product of dims)
    best = max(cands, key=lambda c: (c[2][0] if len(c[2])>0 else 0) * (c[2][1] if len(c[2])>1 else 1))
    print('Plotting best candidate:', best)
    ok = plot_candidate(j, best, 'episode_first_traj.png')
    if not ok:
        print('Plotting failed for candidate')
        raise SystemExit(3)
    print('Done')

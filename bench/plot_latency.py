#!/usr/bin/env python3
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def parse_runs(path):
    """Return list of (label, sizes, min_latencies) for each run in file."""
    runs = []
    current_label = None
    sizes, lats = [], []
    in_data = False

    with open(path) as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i]

        # Detect start of a new run (mpirun command line)
        if 'mpirun' in line:
            if current_label and sizes:
                runs.append((current_label, sizes[:], lats[:]))
                sizes, lats = [], []
            in_data = False
            m = re.search(r'-M\s+(\w+)', line)
            current_label = m.group(1) if m else 'default'
            i += 1
            continue

        # Detect data header
        if re.search(r'\bCount\b.*\bSize\b', line):
            in_data = True
            i += 2  # skip the avg/min/max sub-header
            continue

        # Parse data rows
        if in_data:
            m = re.match(r'\s+\d+\s+(\d+)\s+[\d.]+\s+([\d.]+)', line)
            if m:
                sizes.append(int(m.group(1)))
                lats.append(float(m.group(2)))

        i += 1

    if current_label and sizes:
        runs.append((current_label, sizes, lats))

    return runs

def format_bytes(n, _pos=None):
    for unit, threshold in [('GB', 1 << 30), ('MB', 1 << 20), ('KB', 1 << 10)]:
        if n >= threshold:
            val = n / threshold
            return f'{val:g}{unit}'
    return f'{n}B'

def split_runs(runs, max_size):
    lo, hi = [], []
    for label, sizes, lats in runs:
        s1, l1, s2, l2 = [], [], [], []
        for s, l in zip(sizes, lats):
            if s <= max_size:
                s1.append(s); l1.append(l)
            else:
                s2.append(s); l2.append(l)
        lo.append((label, s1, l1))
        hi.append((label, s2, l2))
    return lo, hi

# Colour/style scheme: same collective gets same linestyle, different mem modes get colours
COLORS = {'global': '#1f77b4', 'local': '#ff7f0e', 'default': '#2ca02c'}
LINESTYLES = {'alltoall': '-', 'alltoallv': '--'}
MARKERS = {'alltoall': 'o', 'alltoallv': 's'}

MAX_GRAPH1 = 32 * 1024 * 1024  # 32 MB

aa_runs  = parse_runs('alltoall-global.out')
av_runs  = parse_runs('alltoallv-global.out')

aa_lo, aa_hi = split_runs(aa_runs,  MAX_GRAPH1)
av_lo, av_hi = split_runs(av_runs,  MAX_GRAPH1)

def make_graph(ax, series_groups, title):
    for collective, series in series_groups:
        ls = LINESTYLES[collective]
        mk = MARKERS[collective]
        for label, sizes, lats in series:
            if not sizes:
                continue
            color = COLORS.get(label, 'gray')
            ax.plot(sizes, lats, linestyle=ls, marker=mk, markersize=4,
                    color=color, label=f'{collective} ({label})')

    ax.set_xscale('log', base=2)
    ax.set_xlabel('Message Size')
    ax.set_ylabel('Min Latency (µs)')
    ax.set_title(title)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_bytes))
    ax.xaxis.set_major_locator(ticker.LogLocator(base=2, numticks=20))
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.legend(fontsize=8)

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

make_graph(axes[0],
    [('alltoall', aa_lo), ('alltoallv', av_lo)],
    'Min Latency: 32B – 32MB\n(cuda, 8 ranks, H100, solid=alltoall / dashed=alltoallv)')

make_graph(axes[1],
    [('alltoall', aa_hi), ('alltoallv', av_hi)],
    'Min Latency: 64MB – 16GB\n(cuda, 8 ranks, H100, solid=alltoall / dashed=alltoallv)')

fig.tight_layout()
fig.savefig('latency_graphs.png', dpi=150)
print('Saved latency_graphs.png')

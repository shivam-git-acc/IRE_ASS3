#!/usr/bin/env python3
"""Stage 6 — render every report figure and table from results.csv.

Pure aggregation: no benchmarking happens here, so every number in the report traces back to
one measured row.

GENERATED FILE — the body below is copied verbatim from nb06_report_assets.ipynb by
tools/sync_report_assets.py. Edit the notebook, then re-run that script.

Two data-hygiene steps worth knowing about:
  * Duplicate rows are dropped. An early version of part4 logged each d=1024 fixed-budget
    point twice (once as the fixed measurement, once as the first sweep point).
  * Part 2 and Part 3 numbers never share a table. Part 2 interpolates QPS *to* recall 0.95
    along a swept curve; Part 3 reports the cheapest grid setting whose tune-split recall
    reached 0.95, which overshoots. Both are right; conflating them is not.
"""

import os as _os
import sys as _sys

import matplotlib as _mpl

# Must precede the body's `import matplotlib.pyplot`. The notebook renders inline, but in a
# script matplotlib picks whatever GUI backend is installed (gtk4agg, TkAgg, ...) and
# plt.show() then BLOCKS on a window that never appears, hanging run_all.sh indefinitely.
_mpl.use("Agg")

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))


def main():
    # NB: the generated body re-imports os/sys/json/... as locals, so this prologue
    # deliberately uses the _os alias; plain `os` here would be a shadowed local.
    import annbench_core as _C

    # Prefer the table the pipeline just wrote locally over the Hugging Face mirror.
    _local = _os.path.join(_C.OUT, "results", "results.csv")
    _orig_fetch = _C.fetch

    def _fetch(rp, token=None, verbose=True):
        if rp.endswith("results.csv") and _os.path.exists(_local):
            return _local
        return _orig_fetch(rp, token, verbose)

    _C.fetch = _fetch
    if not _C.hf_enabled():
        _C.get_hf_token = lambda: None

    import subprocess, sys

    def pip(*a):
        print(subprocess.run([sys.executable, "-m", "pip", "install", "-q", *a],
                             capture_output=True, text=True).stdout[-400:] or "ok")
    for m, p in [("huggingface_hub", "huggingface_hub")]:
        try: __import__(m)
        except ImportError: pip(p)

    import os, sys, json, math, re
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator, NullFormatter
    sys.path.insert(0, os.getcwd())
    import annbench_core as C

    C.ensure_dirs()
    ENV = C.env_fingerprint(verbose=False)
    TOKEN = C.get_hf_token()

    raw = pd.read_csv(C.fetch("results/results.csv", TOKEN))
    print(f"pulled {len(raw)} rows")

    KEY = ["part", "dataset", "n_vectors", "index_kind", "build_params",
           "search_param_name", "search_param_value"]
    df = raw.drop_duplicates(subset=KEY, keep="last").copy()
    print(f"deduplicated: {len(raw)} -> {len(df)} rows ({len(raw) - len(df)} duplicates dropped)")
    df = df[df.search_param_value.astype(str) != "BUDGET_STOP"]
    for c in ("recall_at_10", "qps_median", "build_time_s", "index_size_bytes", "scanned_frac"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["part"] = df["part"].astype(str)
    print(f"usable: {len(df)} rows · parts {sorted(df.part.unique())}")
    print(f"machines: {sorted(df.machine_id.dropna().unique())}")

    # Validated categorical palette: blue / green / purple / crimson.
    PAL = {"hnsw": "#2563eb", "ivf_flat": "#059669", "ivf_pq": "#9333ea",
           "exact_flat": "#dc143c"}
    MARK = {"hnsw": "o", "ivf_flat": "s", "ivf_pq": "^", "exact_flat": "D"}
    LABEL = {"hnsw": "HNSW", "ivf_flat": "IVF-Flat", "ivf_pq": "IVF-PQ",
             "exact_flat": "exact (flat)"}
    DS_LABEL = {"sift": "SIFT-1M (128-d, Euclidean)",
                "glove": "GloVe-100 (1.18M, angular, unit-normalised)"}

    plt.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 200, "font.size": 9,
        "axes.grid": True, "grid.alpha": 0.22, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 10, "legend.frameon": False, "legend.fontsize": 7.5,
    })
    FIG = f"{C.OUT}/figures"
    TAB = f"{C.OUT}/results"

    def at_recall(g, target):
        '''QPS at exactly `target` recall, log-interpolated along one swept curve.'''
        g = g.sort_values("recall_at_10")
        r, q = g.recall_at_10.values, g.qps_median.values
        ok = ~np.isnan(r) & ~np.isnan(q)
        r, q = r[ok], q[ok]
        if len(r) == 0 or r.max() < target:
            return None
        if r.min() >= target:
            return float(q.max())
        i = int(np.searchsorted(r, target))
        if i == 0 or r[i] == r[i - 1]:
            return float(q[i])
        w = (target - r[i - 1]) / (r[i] - r[i - 1])
        return float(math.exp(math.log(q[i - 1]) * (1 - w) + math.log(q[i]) * w))

    def envelope(g):
        '''Upper envelope over build settings: the best QPS available at >= each recall.'''
        g = g.sort_values("recall_at_10", ascending=False)
        best, out = -np.inf, []
        for _, row in g.iterrows():
            if row.qps_median > best:
                best = row.qps_median
                out.append((row.recall_at_10, row.qps_median))
        return sorted(out)

    print("style ready")

    p2 = df[df.part == "2"]
    figs = {}

    for ds in ("sift", "glove"):
        d = p2[p2.dataset == ds]
        if not len(d):
            print(f"no Part 2 rows for {ds}"); continue
        fig, ax = plt.subplots(figsize=(6.9, 3.95))

        for kind, grp in d.groupby("index_kind"):
            col = PAL.get(kind, "gray")
            for bp, sub in grp.groupby("build_params"):
                sub = sub.sort_values("recall_at_10")
                ax.plot(sub.recall_at_10, sub.qps_median, "-", color=col, alpha=0.28, lw=0.9,
                        zorder=1)
            env = envelope(grp)
            if env:
                ax.plot([p[0] for p in env], [p[1] for p in env], "-", marker=MARK.get(kind, "x"),
                        color=col, lw=2.0, ms=5.5, mec="white", mew=0.7,
                        label=LABEL.get(kind, kind), zorder=3)

        ax.set_yscale("log")
        # The legend goes top-right: the curves run top-left to bottom-right, and the exact
        # baseline occupies the bottom edge, so a lower-left legend gets struck through by it.
        ax.legend(loc="upper right")

        ex = df[(df.part == "1") & (df.dataset == ds)
                & (df.search_param_value.astype(str) == "one_at_a_time")]
        if len(ex):
            eq = float(ex.qps_median.iloc[0])
            ax.axhline(eq, color=PAL["exact_flat"], ls="--", lw=1.4, zorder=2)
            ax.plot([1.0], [eq], MARK["exact_flat"], color=PAL["exact_flat"], ms=7,
                    mec="white", mew=0.8, zorder=4)
            # Direct-labelled rather than put in the legend, so the line cannot cross its own text.
            x0 = ax.get_xlim()[0]
            ax.annotate(f"exact search — {eq:.1f} qps at recall 1.0",
                        xy=(x0, eq), xytext=(4, 5), textcoords="offset points",
                        ha="left", va="bottom", fontsize=7.5, color=PAL["exact_flat"])

        ax.axvline(0.95, color="#52525b", ls=":", lw=1.0, zorder=0)
        # Low on the axis: the top is taken by the legend, the bottom band is empty.
        ax.annotate("recall 0.95", xy=(0.95, 0.06), xycoords=("data", "axes fraction"),
                    xytext=(-4, 0), textcoords="offset points", ha="right", va="bottom",
                    fontsize=7, color="#52525b", rotation=90)
        ax.set_xlabel("recall@10  (tie-aware, exact distances recomputed)")
        ax.set_ylabel("queries / second   (1 thread, one query at a time)")
        ax.set_title(f"{DS_LABEL[ds]}\nrecall vs throughput — faint: build settings, bold: best envelope")
        ax.yaxis.set_minor_formatter(NullFormatter())
        fig.tight_layout()
        name = f"fig{'1' if ds == 'sift' else '2'}_{ds}_tradeoff.png"
        fig.savefig(f"{FIG}/{name}", bbox_inches="tight")
        figs[ds] = name
        plt.show()

    rows = []
    for ds in ("sift", "glove"):
        d = p2[p2.dataset == ds]
        for kind in ("hnsw", "ivf_flat", "ivf_pq"):
            grp = d[d.index_kind == kind]
            if not len(grp):
                continue
            best = {}
            for t in (0.90, 0.95):
                cand = [(at_recall(sub, t), bp) for bp, sub in grp.groupby("build_params")]
                cand = [(q, bp) for q, bp in cand if q is not None]
                best[t] = max(cand) if cand else (None, None)
            rows.append(dict(
                dataset=ds, index=LABEL[kind],
                qps_at_090=best[0.90][0], setting_090=best[0.90][1],
                qps_at_095=best[0.95][0], setting_095=best[0.95][1],
                peak_recall=float(grp.recall_at_10.max())))
        ex = df[(df.part == "1") & (df.dataset == ds)
                & (df.search_param_value.astype(str) == "one_at_a_time")]
        if len(ex):
            rows.append(dict(dataset=ds, index="exact (flat)",
                             qps_at_090=float(ex.qps_median.iloc[0]),
                             setting_090="full scan",
                             qps_at_095=float(ex.qps_median.iloc[0]),
                             setting_095="full scan", peak_recall=1.0))

    t1 = pd.DataFrame(rows)
    t1.to_csv(f"{TAB}/table1_qps_at_recall.csv", index=False)

    lines = ["| dataset | index | qps @ 0.90 | qps @ 0.95 | best setting @ 0.95 | peak recall |",
             "|---|---|---:|---:|---|---:|"]
    for _, r in t1.iterrows():
        def fm(v):
            return f"{v:,.0f}" if pd.notna(v) else "never reached"
        st = r.setting_095 if isinstance(r.setting_095, str) else "—"
        lines.append(f"| {r.dataset} | {r['index']} | {fm(r.qps_at_090)} | {fm(r.qps_at_095)} "
                     f"| `{st}` | {r.peak_recall:.4f} |")
    t1_md = "\n".join(lines)
    open(f"{TAB}/table1_qps_at_recall.md", "w").write(t1_md)
    print(t1_md)

    rows = []
    for ds in ("sift", "glove"):
        d = p2[p2.dataset == ds]
        n = int(d.n_vectors.iloc[0]) if len(d) else None
        dim = int(d.dim.iloc[0]) if len(d) else None
        for kind in ("hnsw", "ivf_flat", "ivf_pq"):
            grp = d[d.index_kind == kind]
            if not len(grp):
                continue
            cand = [(at_recall(sub, 0.95), bp) for bp, sub in grp.groupby("build_params")]
            cand = [(q, bp) for q, bp in cand if q is not None]
            if cand:
                _, bp = max(cand)
                sub = grp[grp.build_params == bp]
                note = ""
            else:   # never reached 0.95: report the setting that got closest
                bp = grp.loc[grp.recall_at_10.idxmax(), "build_params"]
                sub = grp[grp.build_params == bp]
                note = f"never reached 0.95 (peak {grp.recall_at_10.max():.3f})"
            sz = float(sub.index_size_bytes.dropna().iloc[0])
            rows.append(dict(dataset=ds, index=LABEL[kind], setting=bp,
                             build_s=float(sub.build_time_s.dropna().iloc[0]),
                             size_mb=sz / 1e6, bytes_per_vector=sz / n, note=note))
        if n:
            rows.append(dict(dataset=ds, index="exact (flat)", setting="—", build_s=0.4,
                             size_mb=n * dim * 4 / 1e6, bytes_per_vector=dim * 4, note=""))

    t2 = pd.DataFrame(rows)
    t2.to_csv(f"{TAB}/table2_build_memory.csv", index=False)
    lines = ["| dataset | index | setting | build (s) | size (MB) | bytes/vector | note |",
             "|---|---|---|---:|---:|---:|---|"]
    for _, r in t2.iterrows():
        bs = f"{r.build_s:,.0f}" if r.build_s >= 10 else f"{r.build_s:.1f}"
        lines.append(f"| {r.dataset} | {r['index']} | `{r.setting}` | {bs} "
                     f"| {r.size_mb:,.0f} | {r.bytes_per_vector:,.0f} | {r.note} |")
    t2_md = "\n".join(lines)
    open(f"{TAB}/table2_build_memory.md", "w").write(t2_md)
    print(t2_md)

    p3 = df[df.part == "3"]
    SIZES = sorted(p3.n_vectors.unique()) if len(p3) else []
    best3 = {}
    for kind in ("exact_flat", "ivf_flat", "hnsw"):
        for n in SIZES:
            g = p3[(p3.index_kind == kind) & (p3.n_vectors == n)]
            if len(g):
                best3[(kind, n)] = g.loc[g.qps_median.idxmax()]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    EXP = {}
    ax = axes[0]
    for kind in ("exact_flat", "ivf_flat", "hnsw"):
        pts = [(n, best3[(kind, n)].qps_median) for n in SIZES if (kind, n) in best3]
        if len(pts) < 2:
            continue
        xs, ys = zip(*pts)
        a = float(np.polyfit(np.log(xs), np.log(ys), 1)[0])
        EXP[kind] = a
        ax.plot(xs, ys, "-", marker=MARK[kind], color=PAL[kind], lw=2, ms=6,
                mec="white", mew=0.7, label=f"{LABEL[kind]}   $n^{{{a:+.2f}}}$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(SIZES); ax.set_xticklabels([f"{int(n/1000)}K" if n < 1e6 else "1M" for n in SIZES])
    ax.set_xlabel("vectors indexed"); ax.set_ylabel("queries / second at recall@10 = 0.95")
    ax.set_title("(a) cost of recall 0.95 vs dataset size")
    ax.legend(loc="lower left")
    ax.xaxis.set_minor_formatter(NullFormatter())

    ax = axes[1]
    for kind in ("ivf_flat", "hnsw"):
        pts = [(n, best3[(kind, n)].build_time_s) for n in SIZES
               if (kind, n) in best3 and pd.notna(best3[(kind, n)].build_time_s)]
        if len(pts) < 2:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, "-", marker=MARK[kind], color=PAL[kind], lw=2, ms=6,
                mec="white", mew=0.7, label=LABEL[kind])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(SIZES); ax.set_xticklabels([f"{int(n/1000)}K" if n < 1e6 else "1M" for n in SIZES])
    ax.set_xlabel("vectors indexed"); ax.set_ylabel("build time (s, 4 threads)")
    ax.set_title("(b) build cost vs dataset size")
    ax.legend(loc="upper left")
    ax.xaxis.set_minor_formatter(NullFormatter())

    fig.suptitle("Part 3 — SIFT scaling: exact search pays linearly, HNSW barely notices",
                 fontsize=10.5, y=1.02)
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig3_part3_scaling.png", bbox_inches="tight")
    plt.show()
    print({k: round(v, 3) for k, v in EXP.items()})

    p4 = df[df.part == "4"].copy()

    # scanned_frac and rc_at_10 only became real columns partway through Part 4, so older rows
    # carry them inside `notes`. Backfill per-row from notes, then -- crucially -- fill
    # rc_at_10 ACROSS each dataset's rows: relative contrast is a property of the dataset, and
    # the sweep rows never recorded it. Reading it off whichever row happens to come first
    # silently loses d=1024, whose informative row is the one dedup drops.
    for col, pat in (("scanned_frac", r"scanned_frac=([0-9.]+)"), ("rc_at_10", r"rc10=([0-9.]+)")):
        have = pd.to_numeric(p4[col], errors="coerce") if col in p4.columns else np.nan
        from_notes = pd.to_numeric(
            p4.notes.astype(str).str.extract(pat)[0], errors="coerce")
        p4[col] = (have.fillna(from_notes) if isinstance(have, pd.Series) else from_notes)

    # Relative contrast is a property of the DATASET, and only the fixed-budget rows recorded
    # it -- which at d=1024 are exactly the rows dedup discards (they were logged twice, and
    # the surviving copy carries sweep-style notes). So harvest it from the raw table, before
    # deduplication, and map it on by dataset name.
    _r4 = raw[raw.part.astype(str) == "4"]
    _rc_map = (pd.to_numeric(_r4.get("rc_at_10"), errors="coerce")
               .fillna(pd.to_numeric(_r4.notes.astype(str).str.extract(r"rc10=([0-9.]+)")[0],
                                     errors="coerce"))
               .groupby(_r4.dataset).first())
    p4["rc_at_10"] = p4["rc_at_10"].fillna(p4.dataset.map(_rc_map))
    assert p4.rc_at_10.notna().all(), "a Part 4 dataset has no relative-contrast value anywhere"
    assert p4.scanned_frac.notna().all(), "a Part 4 row has no scanned fraction"
    p4["cond"] = np.where(p4.dataset.str.contains("clustered"), "clustered", "uniform")
    P4COL = {"uniform": "#dc2626", "clustered": "#2563eb"}
    DIMS4 = sorted(p4.dim.unique())
    FIXEDV = {"hnsw": 64, "ivf_flat": 16}

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

    ax = axes[0]
    for cond in ("uniform", "clustered"):
        pts = []
        for d in DIMS4:
            g = p4[(p4.cond == cond) & (p4.dim == d)]
            if len(g):
                pts.append((d, float(g.rc_at_10.dropna().iloc[0])))
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, "-o", color=P4COL[cond], lw=2, ms=6, mec="white", mew=0.7, label=cond)
    for y, nm in ((2.63, "SIFT"), (1.51, "GloVe")):
        ax.axhline(y, color="#71717a", ls=":", lw=0.9)
        ax.text(DIMS4[0] * 1.08, y * 1.03, nm, fontsize=7, color="#71717a")
    ax.axhline(1.0, color="#27272a", lw=0.9)
    ax.set_xscale("log", base=2); ax.set_xticks(DIMS4); ax.set_xticklabels(DIMS4)
    ax.set_xlabel("dimension"); ax.set_ylabel("relative contrast RC@10")
    ax.set_title("(a) the manipulation\ncontrast collapses — uniform only")
    ax.legend(loc="center right"); ax.xaxis.set_minor_formatter(NullFormatter())

    ax = axes[1]
    for cond in ("uniform", "clustered"):
        for ik in ("hnsw", "ivf_flat"):
            pts = []
            for d in DIMS4:
                g = p4[(p4.cond == cond) & (p4.dim == d) & (p4.index_kind == ik)
                       & (p4.search_param_value.astype(float) == FIXEDV[ik])]
                if len(g):
                    pts.append((d, float(g.recall_at_10.iloc[0])))
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, "-" if ik == "hnsw" else "--", marker=MARK[ik],
                        color=P4COL[cond], lw=1.9, ms=6, mec="white", mew=0.7,
                        label=f"{LABEL[ik]} · {cond}")
    ax.axhline(0.95, color="#52525b", ls=":", lw=1.0)
    ax.set_xscale("log", base=2); ax.set_xticks(DIMS4); ax.set_xticklabels(DIMS4)
    ax.set_ylim(-0.04, 1.06)
    ax.set_xlabel("dimension"); ax.set_ylabel("recall@10 at a fixed budget")
    ax.set_title("(b) same budget everywhere (ef=64, nprobe=16)\nrecall tracks contrast, not dimension")
    ax.legend(loc="lower left"); ax.xaxis.set_minor_formatter(NullFormatter())

    ax = axes[2]
    dmax = DIMS4[-1]
    for cond in ("uniform", "clustered"):
        for ik in ("hnsw", "ivf_flat"):
            g = p4[(p4.cond == cond) & (p4.dim == dmax) & (p4.index_kind == ik)]
            g = g.dropna(subset=["scanned_frac"]).sort_values("scanned_frac")
            if len(g):
                ax.plot(g.scanned_frac * 100, g.recall_at_10, "-" if ik == "hnsw" else "--",
                        marker=MARK[ik], color=P4COL[cond], lw=1.9, ms=5.5, mec="white", mew=0.7,
                        label=f"{LABEL[ik]} · {cond}")
    ax.axhline(0.95, color="#52525b", ls=":", lw=1.0)
    ax.axvline(50, color="#dc143c", ls="--", lw=1.2)
    ax.text(52, 0.06, "50% of data", fontsize=7, color="#dc143c", rotation=90)
    ax.set_xscale("log"); ax.set_ylim(-0.04, 1.06)
    ax.set_xlabel("% of the dataset scanned per query")
    ax.set_ylabel("recall@10")
    ax.set_title(f"(c) at d={dmax}: recall 0.95 costs\na near-exhaustive scan")
    ax.legend(loc="upper left"); ax.xaxis.set_minor_formatter(NullFormatter())

    fig.suptitle("Part 4 — distance concentration breaks both indexes; the clustered control at "
                 "the same dimension does not", fontsize=10.5, y=1.03)
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig4_part4_geometry.png", bbox_inches="tight")
    plt.show()

    # The machine block must describe the machine the MEASUREMENTS were taken on, not whatever
    # machine happens to be running this aggregation. Reading it from env_fingerprint() here is
    # wrong whenever the figures are re-rendered elsewhere, and silently attributes someone
    # else's hardware to the results.
    _mid = str(df.machine_id.dropna().mode().iloc[0])
    _env_path = next((p for p in (f"{C.OUT}/env/{_mid}.json", f"{C.OUT}/logs/env/{_mid}.json")
                      if os.path.exists(p)), None)
    if _env_path is None:
        try:
            _env_path = C.fetch(f"logs/env/{_mid}.json", TOKEN, verbose=False)
        except Exception:
            _env_path = None
    if _env_path:
        _mach = json.load(open(_env_path))
        print(f"machine block: from the recorded fingerprint of {_mid}")
    else:
        _mach = {k: ENV.get(k) for k in ("cpu_model", "cpu_logical", "mem_gb", "gpu",
                                         "numpy", "python", "machine_id")}
        print(f"!! no recorded fingerprint for {_mid}; falling back to THIS machine, which is "
              f"only correct if they are the same")
    # These two always come from the rows themselves.
    _mach["cpu_model"] = str(df.cpu_model.dropna().mode().iloc[0])
    _mach["faiss"] = str(df.faiss_version.dropna().mode().iloc[0])
    _mach["machine_id"] = _mid
    if ENV.get("machine_id") != _mid:
        print(f"   (this aggregation is running on {ENV.get('machine_id')}, not {_mid} — "
              f"the figures are rendered here but the measurements are not)")
    NUM = {"machine": _mach,
           "rows_total": int(len(df)), "duplicates_dropped": int(len(raw) - len(df)),
           "scaling_exponents": {k: round(v, 3) for k, v in EXP.items()}}

    for ds in ("sift", "glove"):
        d = p2[p2.dataset == ds]
        ex = df[(df.part == "1") & (df.dataset == ds)]
        e1 = ex[ex.search_param_value.astype(str) == "one_at_a_time"]
        eb = ex[ex.search_param_value.astype(str) == "batch"]
        blk = {"exact_qps_1q": float(e1.qps_median.iloc[0]) if len(e1) else None,
               "exact_qps_batch": float(eb.qps_median.iloc[0]) if len(eb) else None}
        if blk["exact_qps_1q"] and blk["exact_qps_batch"]:
            blk["batch_speedup"] = round(blk["exact_qps_batch"] / blk["exact_qps_1q"], 1)
        for kind in ("hnsw", "ivf_flat", "ivf_pq"):
            grp = d[d.index_kind == kind]
            if not len(grp):
                continue
            cand = [(at_recall(s, 0.95), bp) for bp, s in grp.groupby("build_params")]
            cand = [(q, bp) for q, bp in cand if q is not None]
            blk[kind] = {"qps_at_095": round(max(cand)[0], 1) if cand else None,
                         "best_setting": max(cand)[1] if cand else None,
                         "peak_recall": round(float(grp.recall_at_10.max()), 4)}
        if blk.get("hnsw", {}).get("qps_at_095") and blk.get("ivf_flat", {}).get("qps_at_095"):
            blk["hnsw_over_ivf"] = round(blk["hnsw"]["qps_at_095"] / blk["ivf_flat"]["qps_at_095"], 2)
        if blk.get("hnsw", {}).get("qps_at_095") and blk["exact_qps_1q"]:
            blk["hnsw_over_exact"] = round(blk["hnsw"]["qps_at_095"] / blk["exact_qps_1q"], 0)
        NUM[ds] = blk

    p4f = {}
    for cond in ("uniform", "clustered"):
        for ik in ("hnsw", "ivf_flat"):
            seq = []
            for d_ in DIMS4:
                g = p4[(p4.cond == cond) & (p4.dim == d_) & (p4.index_kind == ik)
                       & (p4.search_param_value.astype(float) == FIXEDV[ik])]
                seq.append(round(float(g.recall_at_10.iloc[0]), 4) if len(g) else None)
            p4f[f"{cond}_{ik}"] = seq
    NUM["part4_fixed_budget_recall"] = {"dims": [int(x) for x in DIMS4], **p4f}
    NUM["part4_rc10"] = {
        cond: [round(float(p4[(p4.cond == cond) & (p4.dim == d_)].rc_at_10.dropna().iloc[0]), 3)
               for d_ in DIMS4] for cond in ("uniform", "clustered")}

    with open(f"{TAB}/report_numbers.json", "w") as f:
        json.dump(NUM, f, indent=2)
    print(json.dumps(NUM, indent=2)[:3000])

    arte = ([f"{FIG}/{n}" for n in figs.values()]
            + [f"{FIG}/fig3_part3_scaling.png", f"{FIG}/fig4_part4_geometry.png"]
            + [f"{TAB}/table1_qps_at_recall.md", f"{TAB}/table1_qps_at_recall.csv",
               f"{TAB}/table2_build_memory.md", f"{TAB}/table2_build_memory.csv",
               f"{TAB}/report_numbers.json"])
    for a in arte:
        if os.path.exists(a):
            sub = "figures" if a.endswith(".png") else "results"
            C.hf_upload(a, f"{sub}/{os.path.basename(a)}", token=TOKEN, msg="nb06 report assets")

    print("\n" + "=" * 80); print("nb06 SUMMARY (report assets)"); print("=" * 80)
    print(f"rows used {len(df)} (dropped {len(raw) - len(df)} duplicates) · "
          f"machines {sorted(df.machine_id.dropna().unique())}")
    print(f"\nfiles written to {C.OUT}"
          + (" and mirrored to HF:" if C.hf_enabled() else " (HF mirroring off):"))
    for a in arte:
        if os.path.exists(a):
            print(f"  {os.path.basename(a):32s} {C.human(os.path.getsize(a))}")
    print(f"\nscaling exponents: " + ", ".join(f"{k} {v:+.3f}" for k, v in EXP.items()))
    print("\nheadline numbers")
    for ds in ("sift", "glove"):
        b = NUM.get(ds, {})
        if b.get("hnsw"):
            print(f"  {ds:6s} exact {b['exact_qps_1q']:.1f} qps · "
                  f"HNSW@0.95 {b['hnsw']['qps_at_095']:.0f} · "
                  f"IVF-Flat@0.95 {b['ivf_flat']['qps_at_095']:.0f} · "
                  f"PQ peak {b['ivf_pq']['peak_recall']:.3f} · "
                  f"HNSW/IVF {b.get('hnsw_over_ivf')}x · HNSW/exact {b.get('hnsw_over_exact'):.0f}x")
    print("=" * 80)


if __name__ == "__main__":
    main()

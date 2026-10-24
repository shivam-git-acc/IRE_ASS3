"""
annbench_core.py -- shared core for CS4.406 Assignment 3 (ANN benchmarking).

Every notebook writes an identical copy of this file, so each notebook runs standalone.

Conventions used everywhere in this project:
  * All distances are SQUARED L2, computed on float32 vectors. GloVe is unit-normalised
    first, so squared-L2 order == cosine order. We never rely on the scale of the
    distances stored in the ann-benchmarks HDF5 files (euclidean vs cosine units);
    those files are used for their neighbour IDS only, as an independent cross-check.
  * k = 10 everywhere.
  * Queries 0..4999  = TUNE split (parameter picking).
    Queries 5000..9999 = EVAL split (every number that appears in the report).
  * Timing is single-thread, one query at a time. See bench_single_query.
"""

import json
import os
import platform
import re
import subprocess
import sys
import time
import hashlib

import numpy as np

K = 10
TIE_EPS = 1e-6
TUNE = slice(0, 5000)
EVAL = slice(5000, 10000)


def set_split(n_queries):
    """Re-point the tune/eval halves. The full datasets ship 10,000 queries, but quick mode
    uses fewer, and the halves must move with them so no reported number is ever measured
    on a query that was used for tuning."""
    global TUNE, EVAL
    half = n_queries // 2
    TUNE, EVAL = slice(0, half), slice(half, n_queries)
    return TUNE, EVAL

HF_REPO = "donbosoc/ire-a3-ann"
HF_REPO_TYPE = "dataset"

def _pick_scratch():
    """Scratch for big files: /kaggle/temp does not count against the 20 GB output quota,
    but it is not guaranteed to exist or be writable, so fall back rather than crash.
    A3_SCRATCH overrides it entirely, which is how run_all.sh keeps everything in-tree."""
    for cand in ([os.environ["A3_SCRATCH"]] if os.environ.get("A3_SCRATCH") else []) + [
            "/kaggle/temp/a3", "/kaggle/working/tmp/a3", "/tmp/a3"]:
        try:
            os.makedirs(cand, exist_ok=True)
            probe = os.path.join(cand, ".w")
            with open(probe, "w") as f:
                f.write("x")
            os.remove(probe)
            return cand
        except Exception:
            continue
    raise RuntimeError("no writable scratch directory found")


TMP = _pick_scratch()
OUT = os.environ.get("A3_OUT") or ("/kaggle/working" if os.path.isdir("/kaggle/working")
                                   else os.path.abspath("./out"))


def ensure_dirs():
    for p in (TMP, f"{TMP}/data", f"{TMP}/gt", f"{TMP}/idx",
              f"{OUT}/results", f"{OUT}/figures", f"{OUT}/logs", f"{OUT}/env"):
        os.makedirs(p, exist_ok=True)


# --------------------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------------------

def env_fingerprint(verbose=True):
    """Capture the machine + library versions behind every measurement.

    Kaggle VMs are shared and the CPU model can differ between sessions, so this goes
    into every results row and the report states it rather than claiming a quiet machine.
    """
    info = {}
    try:
        cpuinfo = open("/proc/cpuinfo").read()
        models = re.findall(r"^model name\s*:\s*(.+)$", cpuinfo, re.M)
        info["cpu_model"] = models[0].strip() if models else "unknown"
        info["cpu_logical"] = len(models)
    except Exception:
        info["cpu_model"], info["cpu_logical"] = platform.processor() or "unknown", os.cpu_count()
    try:
        mem_kb = int(re.search(r"MemTotal:\s*(\d+)", open("/proc/meminfo").read()).group(1))
        info["mem_gb"] = round(mem_kb / 1048576, 1)
    except Exception:
        info["mem_gb"] = None

    info["python"] = sys.version.split()[0]
    info["numpy"] = np.__version__
    for mod in ("faiss", "torch", "h5py"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            info[mod] = None
    try:
        import torch
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        info["gpu"] = None
    try:
        info["nvidia_smi"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        info["nvidia_smi"] = None

    # Stable id for "which machine produced this row".
    key = f"{info['cpu_model']}|{info['cpu_logical']}|{info['mem_gb']}"
    info["machine_id"] = hashlib.sha1(key.encode()).hexdigest()[:8]

    if verbose:
        print(f"machine_id : {info['machine_id']}")
        print(f"cpu        : {info['cpu_model']}  ({info['cpu_logical']} logical)")
        print(f"memory     : {info['mem_gb']} GB")
        print(f"gpu        : {info['gpu']}")
        print(f"versions   : python {info['python']}, numpy {info['numpy']}, "
              f"faiss {info['faiss']}, torch {info['torch']}, h5py {info['h5py']}")
    return info


def set_single_thread():
    """Pin FAISS to one thread. Every benchmark number in this project is single-thread."""
    import faiss
    faiss.omp_set_num_threads(1)
    n = faiss.omp_get_max_threads()
    assert n == 1, f"expected 1 FAISS thread, got {n}"
    return n


# --------------------------------------------------------------------------------------
# Hugging Face
# --------------------------------------------------------------------------------------

_TOKEN_NAMES = ["HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGINGFACEHUB_API_TOKEN",
                "HF", "hf_token", "huggingface"]


def hf_enabled():
    """HF mirroring is a convenience for running across Kaggle sessions, never a
    requirement. Everything this project writes also lands on local disk, so with no token
    (the normal case for someone reproducing the results) the mirror is simply skipped."""
    if os.environ.get("A3_NO_HF"):
        return False
    try:
        get_hf_token()
        return True
    except Exception:
        return False


def get_hf_token():
    """Find the HF token in Kaggle secrets, then the environment. Raises if absent."""
    tried = []
    try:
        from kaggle_secrets import UserSecretsClient
        client = UserSecretsClient()
        for name in _TOKEN_NAMES:
            try:
                tok = client.get_secret(name)
                if tok:
                    print(f"HF token: Kaggle secret '{name}'")
                    return tok.strip()
            except Exception:
                tried.append(f"secret:{name}")
    except Exception:
        tried.append("kaggle_secrets unavailable")
    for name in _TOKEN_NAMES:
        tok = os.environ.get(name)
        if tok:
            print(f"HF token: environment variable '{name}'")
            return tok.strip()
        tried.append(f"env:{name}")
    raise RuntimeError(
        "No Hugging Face token found. Add it under Notebook > Add-ons > Secrets with "
        f"label 'HF_TOKEN' and attach it to this notebook. Tried: {tried}")


def hf_api(token=None):
    from huggingface_hub import HfApi
    return HfApi(token=token or get_hf_token())


def hf_ensure_repo(token=None):
    api = hf_api(token)
    api.create_repo(HF_REPO, repo_type=HF_REPO_TYPE, exist_ok=True, private=False)
    return api


def hf_upload(local_path, path_in_repo, token=None, msg=None, quiet=False, timeout=420):
    """Upload one file, but never let a network stall hang the session.

    huggingface_hub has no hard wall-clock cap on an upload, and a stalled connection will
    block forever. A 12-hour Kaggle session was lost to exactly that. The work runs on a
    daemon thread with a join timeout: if it overruns we warn and carry on, because the
    local copy of everything we upload is always intact on disk.
    """
    import threading

    if not hf_enabled():
        return False
    size = os.path.getsize(local_path) / 1e6
    if not quiet:
        print(f"  -> hf://{HF_REPO}/{path_in_repo}  ({size:.1f} MB)", flush=True)

    box = {}

    def work():
        try:
            hf_api(token).upload_file(
                path_or_fileobj=local_path, path_in_repo=path_in_repo,
                repo_id=HF_REPO, repo_type=HF_REPO_TYPE,
                commit_message=msg or f"add {path_in_repo}")
            box["ok"] = True
        except Exception as exc:
            box["err"] = exc

    th = threading.Thread(target=work, daemon=True)
    th.start()
    # Big files legitimately take longer, so scale the allowance with size.
    th.join(timeout + size * 2)
    if th.is_alive():
        print(f"  ! upload of {path_in_repo} still running after "
              f"{timeout + size * 2:.0f}s -- continuing; the local copy is intact",
              flush=True)
        return False
    if "err" in box:
        print(f"  ! upload of {path_in_repo} failed: {box['err']}", flush=True)
        return False
    return True


def hf_download(path_in_repo, token=None, local_dir=None):
    """Return a local path for a repo file, or None if it is not in the repo."""
    from huggingface_hub import hf_hub_download
    if not hf_enabled():
        return None
    try:
        return hf_hub_download(repo_id=HF_REPO, repo_type=HF_REPO_TYPE,
                               filename=path_in_repo, token=token or get_hf_token(),
                               local_dir=local_dir or f"{TMP}/hf")
    except Exception as e:
        if "404" in str(e) or "EntryNotFound" in type(e).__name__:
            return None
        raise


def fetch(repo_path, token=None, verbose=True):
    """Download one prepared artifact from the HF mirror, erroring clearly if nb00 never ran."""
    p = hf_download(repo_path, token=token, local_dir=f"{TMP}/hf")
    if p is None:
        raise FileNotFoundError(
            f"hf://{HF_REPO}/{repo_path} not found -- run nb00_setup_data first")
    if verbose:
        print(f"  {repo_path:44s} {human(os.path.getsize(p))}")
    return p


def load_local(name, with_base=True, with_stored=False):
    """Load what prepare_data.py wrote to local disk (the path run_all.sh uses)."""
    d = {}
    if with_base:
        d["base"] = np.load(f"{TMP}/data/{name}_base.npy")
    d["query"] = np.load(f"{TMP}/data/{name}_query.npy")
    d["gt_ids"] = np.load(f"{TMP}/gt/{name}_top100_ids.npy")
    d["gt_dists"] = np.load(f"{TMP}/gt/{name}_top100_dists.npy")
    if with_stored:
        d["stored"] = np.load(f"{TMP}/data/{name}_stored_neighbors.npy")
    return d


def load_prepared(name, token=None, with_base=True, with_stored=False, verbose=True):
    """Load a dataset prepared by nb00: base, all 10k queries, and our exact top-100 GT.

    Use d["query"][TUNE] for parameter picking and d["query"][EVAL] for reported numbers.
    """
    d = {}
    if with_base:
        d["base"] = np.load(fetch(f"data/{name}_base.npy", token, verbose))
    d["query"] = np.load(fetch(f"data/{name}_query.npy", token, verbose))
    d["gt_ids"] = np.load(fetch(f"gt/{name}_top100_ids.npy", token, verbose))
    d["gt_dists"] = np.load(fetch(f"gt/{name}_top100_dists.npy", token, verbose))
    if with_stored:
        d["stored"] = np.load(fetch(f"data/{name}_stored_neighbors.npy", token, verbose))
    return d


# --------------------------------------------------------------------------------------
# Exact ground truth on the GPU
# --------------------------------------------------------------------------------------

def exact_gt_gpu(base, queries, k=100, chunk=None, mem_budget_gb=None,
                 refine=True, verbose=True):
    """Brute-force top-k by squared L2, selected on the GPU, then refined exactly.

    The GPU pass uses the (|q|^2 + |b|^2 - 2 q.b) expansion in float32, which is fast but
    loses a few ULPs. Selection errors from that can only happen between points that are
    equal to within those ULPs -- i.e. ties, which our recall metric forgives anyway. The
    returned DISTANCES are then recomputed by direct subtraction in float64, so the
    threshold that recall@10 depends on is exact.

    The distance block is (nq, chunk) float32, which at 10k queries is 400 MB per 10k
    base vectors, so a fixed chunk would OOM a 16 GB T4 outright. `chunk` is derived from
    the free VRAM and the query count unless given explicitly, and the block is written
    into ONE preallocated buffer with in-place ops -- writing the expression out as
    `q_sq + b_sq - 2*q@b.T` would allocate three such blocks per iteration instead.

    Returns (ids int32 (nq,k), dists float64 (nq,k)) sorted ascending.
    """
    import torch
    assert base.dtype == np.float32 and queries.dtype == np.float32
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    n, d = base.shape
    nq = queries.shape[0]
    k = min(k, n)

    # torch.topk allocates a workspace about the size of its input, so the real peak is
    # ~2x the distance block. Budget a fifth of free VRAM to leave headroom for that.
    if mem_budget_gb is None:
        if dev == "cuda":
            torch.cuda.empty_cache()
            mem_budget_gb = max(0.25, 0.20 * torch.cuda.mem_get_info()[0] / (1 << 30))
        else:
            mem_budget_gb = 2.0
    auto = max(1024, int(mem_budget_gb * (1 << 30) / (nq * 4)))
    chunk = min(chunk, auto) if chunk else auto
    chunk = min(chunk, n)

    q = torch.from_numpy(np.ascontiguousarray(queries).copy()).to(dev)
    q_sq = (q * q).sum(1, keepdim=True)
    buf = torch.empty((nq, chunk), device=dev, dtype=torch.float32)

    best_d = best_i = None
    t0 = time.perf_counter()
    if verbose:
        print(f"    gt: {n:,} x {nq:,} in blocks of {chunk:,} "
              f"(~{nq * chunk * 4 / 1e9:.2f} GB/block) on {dev}")
    for s in range(0, n, chunk):
        blk = torch.from_numpy(np.ascontiguousarray(base[s:s + chunk]).copy()).to(dev)
        m = blk.shape[0]
        dist = buf[:, :m]
        torch.mm(q, blk.T, out=dist)
        dist.mul_(-2.0).add_(q_sq).add_((blk * blk).sum(1)[None, :]).clamp_(min=0)
        kk = min(k, m)
        dv, di = torch.topk(dist, kk, dim=1, largest=False)
        di = di + s
        if best_d is None:
            best_d, best_i = dv.clone(), di
        else:
            cd, ci = torch.cat([best_d, dv], 1), torch.cat([best_i, di], 1)
            best_d, pos = torch.topk(cd, k, dim=1, largest=False)
            best_i = torch.gather(ci, 1, pos)
            del cd, ci, pos
        del blk, dv, di
        if verbose:
            print(f"\r    gt: {min(s + chunk, n):,}/{n:,} vectors", end="", flush=True)
    if verbose:
        print(f"\r    gt: {n:,}/{n:,} vectors in {time.perf_counter() - t0:.1f}s on {dev}")

    ids = best_i.cpu().numpy().astype(np.int32)
    approx = best_d.cpu().numpy().astype(np.float64)
    del q, q_sq, best_i, best_d, buf
    if dev == "cuda":
        torch.cuda.empty_cache()
    if not refine:
        return ids, approx

    dists = exact_dists_for_ids(base, queries, ids)
    order = np.argsort(dists, axis=1, kind="stable")
    ids = np.take_along_axis(ids, order, axis=1)
    dists = np.take_along_axis(dists, order, axis=1)
    return ids, dists


def exact_gt(base, queries, k=100, verbose=True):
    """Exact top-k, on the GPU when there is one and through FAISS when there is not.

    The GPU path is far faster for the big ground-truth jobs, but torch is not a hard
    dependency: FAISS's own flat index gives the same ids, and in both cases the returned
    distances are recomputed exactly in float64 so the recall threshold is not approximate.
    """
    try:
        import torch
        if torch.cuda.is_available():
            return exact_gt_gpu(base, queries, k=k, verbose=verbose)
    except ImportError:
        pass

    import faiss
    if verbose:
        print(f"    gt: {len(base):,} x {len(queries):,} via faiss (no GPU)", flush=True)
    ix = faiss.IndexFlatL2(base.shape[1])
    ix.add(base)
    _, ids = ix.search(np.ascontiguousarray(queries), min(k, len(base)))
    ids = ids.astype(np.int32)
    dists = exact_dists_for_ids(base, queries, ids)
    order = np.argsort(dists, axis=1, kind="stable")
    return np.take_along_axis(ids, order, axis=1), np.take_along_axis(dists, order, axis=1)


def exact_dists_for_ids(base, queries, ids, block=512):
    """Exact squared-L2 distance from each query to each of its listed ids, in float64.

    Direct subtraction (no q^2+b^2-2qb expansion), so there is no cancellation error.
    ids == -1 (index returned fewer than k) yields +inf. Chunked over queries to bound
    the (block, k, d) temporary.
    """
    nq, kk = ids.shape
    out = np.empty((nq, kk), dtype=np.float64)
    for s in range(0, nq, block):
        e = min(s + block, nq)
        idx = ids[s:e]
        valid = idx >= 0
        vecs = base[np.where(valid, idx, 0)].astype(np.float64)   # (b, k, d)
        diff = vecs - queries[s:e].astype(np.float64)[:, None, :]
        out[s:e] = np.where(valid, np.einsum("bkd,bkd->bk", diff, diff), np.inf)
    return out


# --------------------------------------------------------------------------------------
# Recall@10, tie-aware  (Part 1 asks for this specifically)
# --------------------------------------------------------------------------------------

def recall_at_k(retrieved_ids, retrieved_true_d, gt_dists, k=K, eps=TIE_EPS):
    """Per-query recall@k that does not punish ties at the k-th neighbour.

    A returned id counts as correct when its EXACT distance is <= the exact distance of
    the true k-th neighbour (plus a tiny slack). If the true 10th and 11th neighbours sit
    at the same distance, returning either one is right, so membership of the stored
    top-10 id list is the wrong test -- distance is the right one.

    Duplicate ids are credited once, so an index that returns the same vector twice
    cannot inflate its score.

    retrieved_ids    : (nq, k') ids from the index; -1 means "fewer than k' results".
    retrieved_true_d : (nq, k') EXACT squared-L2 distances for those ids (inf for -1).
    gt_dists         : (nq, >=k) exact squared-L2 distances of the true neighbours, ascending.
    returns          : (nq,) float64 in [0, 1].
    """
    retrieved_ids = np.asarray(retrieved_ids)
    retrieved_true_d = np.asarray(retrieved_true_d, dtype=np.float64)
    gt_dists = np.asarray(gt_dists, dtype=np.float64)
    if retrieved_ids.ndim == 1:
        retrieved_ids = retrieved_ids[None, :]
        retrieved_true_d = retrieved_true_d[None, :]
        gt_dists = gt_dists[None, :]
    assert retrieved_ids.shape == retrieved_true_d.shape
    assert gt_dists.shape[1] >= k, f"need >= {k} ground-truth distances"

    dk = gt_dists[:, k - 1][:, None]
    thresh = dk * (1.0 + eps) + eps           # relative slack, plus absolute for dk == 0
    hit = (retrieved_ids >= 0) & (retrieved_true_d <= thresh)

    # Count DISTINCT hit ids per row.
    masked = np.where(hit, retrieved_ids, -1)
    masked = np.sort(masked, axis=1)
    dup = np.zeros(masked.shape, dtype=bool)
    dup[:, 1:] = masked[:, 1:] == masked[:, :-1]
    n_hit = ((masked >= 0) & ~dup).sum(1)
    return np.minimum(n_hit, k) / float(k)


def recall_from_ids(retrieved_ids, base, queries, gt_dists, k=K):
    """recall_at_k convenience wrapper that recomputes the exact distances itself.

    Needed for IVF-PQ, whose returned distances are approximations of the real ones and
    must never be fed to the recall metric.
    """
    d = exact_dists_for_ids(base, queries, np.asarray(retrieved_ids))
    return recall_at_k(retrieved_ids, d, gt_dists, k=k)


# --------------------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------------------

def bench_single_query(search_fn, queries, repeats=3, warmup=50):
    """Time `search_fn` one query at a time, single thread; report the median run.

    Per-query timers cost ~50-100 ns each against queries of 10^2-10^4 us, i.e. under
    0.1%, so QPS is taken as n / sum(per-query latencies) from the same pass that
    produces the latency percentiles.
    """
    nq = len(queries)
    for i in range(min(warmup, nq)):
        search_fn(queries[i:i + 1])

    runs = []
    for _ in range(repeats):
        lat = np.empty(nq, dtype=np.float64)
        for i in range(nq):
            t0 = time.perf_counter_ns()
            search_fn(queries[i:i + 1])
            lat[i] = time.perf_counter_ns() - t0
        runs.append(lat)

    qps = np.array([nq / (r.sum() / 1e9) for r in runs])
    med = runs[int(np.argsort(qps)[len(qps) // 2])]       # latencies of the median run
    return dict(
        qps_median=float(np.median(qps)),
        qps_runs=[round(float(x), 2) for x in qps],
        qps_spread_pct=round(float(100 * (qps.max() - qps.min()) / np.median(qps)), 2),
        lat_p50_ms=float(np.percentile(med, 50) / 1e6),
        lat_p95_ms=float(np.percentile(med, 95) / 1e6),
        n_queries=int(nq), repeats=int(repeats),
    )


def bench_batch(search_fn, queries, repeats=3, warmup=1):
    """Time one call that passes every query at once (Part 1's batched comparison)."""
    nq = len(queries)
    for _ in range(warmup):
        search_fn(queries)
    secs = []
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        search_fn(queries)
        secs.append((time.perf_counter_ns() - t0) / 1e9)
    secs = np.array(secs)
    return dict(qps_median=float(nq / np.median(secs)),
                qps_runs=[round(float(nq / s), 2) for s in secs],
                wall_s_median=float(np.median(secs)),
                n_queries=int(nq), repeats=int(repeats))


# --------------------------------------------------------------------------------------
# Results table
# --------------------------------------------------------------------------------------

COLUMNS = [
    "run_id", "timestamp", "part", "dataset", "n_vectors", "dim", "metric",
    "index_kind", "build_params", "search_param_name", "search_param_value",
    "recall_at_10", "recall_std", "qps_median", "qps_runs", "qps_spread_pct",
    "lat_p50_ms", "lat_p95_ms", "build_time_s", "train_time_s", "add_time_s",
    "index_size_bytes", "bytes_per_vector", "n_queries", "repeats", "threads",
    "query_split", "machine_id", "cpu_model", "faiss_version", "scanned_frac",
    "rc_at_10", "notes",
]


class ResultsLog:
    """Append-only results table, mirrored to HF after every few rows.

    One row per benchmark run, as the deliverable requires. Resumability comes from this
    file: `done()` lets a restarted notebook skip measurements already in the table,
    which is what makes a killed Kaggle session cheap.
    """

    KEY = ("part", "dataset", "n_vectors", "index_kind", "build_params",
           "search_param_name", "search_param_value")

    def __init__(self, env, token=None, path=f"{OUT}/results/results.csv",
                 repo_path="results/results.csv", flush_every=5, pull=True):
        import pandas as pd
        self.pd, self.env, self.token = pd, env, token
        self.path, self.repo_path, self.flush_every = path, repo_path, flush_every
        self._since_flush = 0
        os.makedirs(os.path.dirname(path), exist_ok=True)

        df = None
        if pull and hf_enabled():
            try:
                remote = hf_download(repo_path, token=token)
                if remote:
                    df = pd.read_csv(remote)
                    print(f"results.csv: pulled {len(df)} existing rows from HF")
            except Exception as e:
                print(f"results.csv: could not pull from HF ({e}); starting fresh")
        if df is None and os.path.exists(path):
            df = pd.read_csv(path)
            print(f"results.csv: {len(df)} existing rows found locally")
        if df is None:
            df = pd.DataFrame(columns=COLUMNS)
            print("results.csv: starting a new table")
        for c in COLUMNS:
            if c not in df.columns:
                df[c] = None
        self.df = df[COLUMNS]
        self.df.to_csv(self.path, index=False)

    @staticmethod
    def _sv(x):
        """Normalise a search-param value for comparison: pandas may read an all-numeric
        column back as float, turning 40 into '40.0' and breaking resume detection."""
        t = str(x)
        if t.endswith(".0") and t[:-2].lstrip("-").isdigit():
            t = t[:-2]
        return t

    @staticmethod
    def _norm(params):
        return json.dumps(params, sort_keys=True, separators=(",", ":")) \
            if isinstance(params, dict) else str(params)

    def done(self, part, dataset, n_vectors, index_kind, build_params,
             search_param_name, search_param_value):
        """True when this exact measurement is already in the table."""
        if len(self.df) == 0:
            return False
        want = (str(part), str(dataset), self._sv(n_vectors), str(index_kind),
                self._norm(build_params), str(search_param_name),
                self._sv(search_param_value))
        have = self.df[list(self.KEY)].astype(str).apply(
            lambda col: col.map(self._sv) if col.name in ("n_vectors", "search_param_value")
            else col)
        return bool((have == self.pd.Series(want, index=list(self.KEY))).all(axis=1).any())

    def add(self, *, part, dataset, n_vectors, dim, metric, index_kind, build_params,
            search_param_name, search_param_value, recall, timing, build_time_s=None,
            train_time_s=None, add_time_s=None, index_size_bytes=None,
            query_split="eval", threads=1, scanned_frac=None, rc_at_10=None,
            notes="", flush=False):
        recall = np.asarray(recall, dtype=np.float64)
        timing = timing or {}
        row = {
            "run_id": f"{int(time.time() * 1000)}-{np.random.randint(1e6):06d}",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "part": part, "dataset": dataset, "n_vectors": int(n_vectors), "dim": int(dim),
            "metric": metric, "index_kind": index_kind,
            "build_params": self._norm(build_params),
            "search_param_name": search_param_name, "search_param_value": search_param_value,
            "recall_at_10": round(float(recall.mean()), 6),
            "recall_std": round(float(recall.std()), 6),
            "qps_median": (round(float(timing["qps_median"]), 3)
                           if timing.get("qps_median") is not None else None),
            "qps_runs": json.dumps(timing.get("qps_runs")),
            "qps_spread_pct": timing.get("qps_spread_pct"),
            "lat_p50_ms": timing.get("lat_p50_ms"),
            "lat_p95_ms": timing.get("lat_p95_ms"),
            "build_time_s": build_time_s, "train_time_s": train_time_s,
            "add_time_s": add_time_s, "index_size_bytes": index_size_bytes,
            "bytes_per_vector": (round(index_size_bytes / n_vectors, 2)
                                 if index_size_bytes else None),
            "n_queries": timing.get("n_queries"), "repeats": timing.get("repeats"),
            "threads": int(threads), "query_split": query_split,
            "machine_id": self.env.get("machine_id"), "cpu_model": self.env.get("cpu_model"),
            "faiss_version": self.env.get("faiss"), "scanned_frac": scanned_frac,
            "rc_at_10": rc_at_10, "notes": notes,
        }
        # .loc[] append rather than concat: concat against a frame holding all-NA columns
        # raises a FutureWarning on every single row, which floods the notebook log.
        self.df.loc[len(self.df)] = self.pd.Series(row)
        self.df.to_csv(self.path, index=False)
        self._since_flush += 1
        if flush or self._since_flush >= self.flush_every:
            self.flush()
        return row

    def flush(self):
        self._since_flush = 0
        if not hf_enabled():
            return
        t0 = time.time()
        try:
            hf_upload(self.path, self.repo_path, token=self.token,
                      msg=f"results.csv: {len(self.df)} rows", quiet=True)
        except Exception as e:
            print(f"  ! results.csv upload failed ({e}); the local copy is intact")
        dt = time.time() - t0
        if dt > 30:
            print(f"  (results.csv sync took {dt:.0f}s)", flush=True)


# --------------------------------------------------------------------------------------
# Misc
# --------------------------------------------------------------------------------------

def index_size_bytes(index, path=None):
    """Serialise a FAISS index to disk and measure it. Memory footprint ~= file size for
    these index types, and it is the one number we can take reproducibly."""
    import faiss
    path = path or f"{TMP}/idx/_size_probe.faiss"
    faiss.write_index(index, path)
    n = os.path.getsize(path)
    os.remove(path)
    return n


def normalise(x):
    """Unit-normalise rows in place-ish. Zero rows are left as zeros."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    nrm = np.linalg.norm(x, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    return (x / nrm).astype(np.float32)


def relative_contrast(gt_dists, mean_dist_sq, k=1):
    """RC = (mean distance to a random point) / (distance to the k-th neighbour).

    Reported on true (non-squared) distances, the usual convention. Values near 1 mean
    the nearest neighbour is barely nearer than anything else, which is exactly when
    graph and cell indexes stop working.
    """
    dk = np.sqrt(np.asarray(gt_dists, dtype=np.float64)[:, k - 1])
    dk = np.where(dk == 0, np.finfo(np.float64).tiny, dk)
    return np.sqrt(np.asarray(mean_dist_sq, dtype=np.float64)) / dk


def mean_sq_dist_to_random(base, queries, n_sample=100_000, seed=0):
    """Mean squared distance from each query to a random sample of the base."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(base), size=min(n_sample, len(base)), replace=False)
    sample = base[np.sort(idx)].astype(np.float32)
    s_sq = (sample * sample).sum(1)
    out = np.empty(len(queries), dtype=np.float64)
    for s in range(0, len(queries), 512):
        e = min(s + 512, len(queries))
        q = queries[s:e].astype(np.float32)
        d = (q * q).sum(1)[:, None] + s_sq[None, :] - 2.0 * (q @ sample.T)
        out[s:e] = np.maximum(d, 0).mean(1)
    return out


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024

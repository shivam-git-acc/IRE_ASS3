#!/usr/bin/env python3
"""Regenerate make_report_assets.py from nb06_report_assets.ipynb.

The notebooks are how these experiments were actually run on Kaggle; the scripts are what
run_all.sh drives. For the aggregation stage the two share a body verbatim, generated here, so
a change to the notebook cannot silently leave the pipeline behind.

    python tools/sync_report_assets.py          # rewrite make_report_assets.py
    python tools/sync_report_assets.py --check  # fail if it is out of date (for CI)
"""

import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
CODE = HERE.parent
NB = CODE / "notebooks" / "nb06_report_assets.ipynb"
TARGET = CODE / "make_report_assets.py"

HEADER = '''#!/usr/bin/env python3
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

'''

FOOTER = '\n\nif __name__ == "__main__":\n    main()\n'


def render():
    nb = json.loads(NB.read_text())
    cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    cells = [c for c in cells
             if not c.startswith("%%writefile") and "pip install" not in c[:400]]
    body = []
    for cell in cells:
        body += [("    " + line).rstrip() for line in cell.rstrip().split("\n")] + [""]
    return HEADER + "\n".join(body) + FOOTER


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero if the generated file is stale")
    a = ap.parse_args()
    want = render()
    if a.check:
        have = TARGET.read_text() if TARGET.exists() else ""
        if have != want:
            print(f"{TARGET.name} is out of date; run: python tools/sync_report_assets.py")
            return 1
        print(f"{TARGET.name} is up to date with {NB.name}")
        return 0
    TARGET.write_text(want)
    print(f"regenerated {TARGET} from {NB.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Build report.pdf from the measured results.

    python build_report.py --roll 2025202030 --name "Shivam Patel"

Every quoted number is read from results/report_numbers.json and results/results.csv rather
than typed in, so the report cannot drift from the data. Figures are embedded as base64 and
the mechanism diagrams are inline SVG, so report.html is self-contained. Rendering uses
headless Chrome.
"""

import argparse
import base64
import json
import pathlib
import re
import shutil
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
RES = HERE.parent / "results"
FIGS = RES / "figures"
OUTDIR = HERE.parent / "report"
OUTDIR.mkdir(exist_ok=True)
sys.path.insert(0, str(HERE))


def img(name):
    return "data:image/png;base64," + base64.b64encode((FIGS / name).read_bytes()).decode()


def cell(c):
    """Tidy one markdown cell for print. 'never reached' is an em dash in a throughput column,
    but in the note column it carries the peak recall and must keep it."""
    c = c.replace("`", "").strip()
    c = re.sub(r"never reached 0\.95 \(peak ([0-9.]+)\)", r"ceiling \1", c)
    if c == "never reached":
        return "&mdash;"
    # Build settings arrive as the JSON blob results.csv stores; render them as parameters.
    if c.startswith("{") and c.endswith("}"):
        parts = re.findall(r'"([A-Za-z_]+)"\s*:\s*([0-9]+)', c)
        pretty = {"M": "M", "efConstruction": "efC", "nlist": "nlist", "m": "m",
                  "nbits": "nbits"}
        keep = [(pretty.get(k, k), v) for k, v in parts if k != "nbits"]
        return ", ".join(f"{k}={int(v):,}" for k, v in keep)
    return c


def md_table(md, cls=""):
    rows = md.strip().split("\n")
    head = [h.strip() for h in rows[0].strip("|").split("|")]
    body = [[c for c in r.strip("|").split("|")] for r in rows[2:]]
    th = "".join(f"<th>{h}</th>" for h in head)
    tb = "".join("<tr>" + "".join(f"<td>{cell(c)}</td>" for c in r) + "</tr>" for r in body)
    return f'<table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{tb}</tbody></table>'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--roll", default="2025202030")
    ap.add_argument("--name", default="Shivam Patel")
    a = ap.parse_args()

    import diagrams
    N = json.loads((RES / "report_numbers.json").read_text())
    m, s, g = N["machine"], N["sift"], N["glove"]
    e = N["scaling_exponents"]
    p4r, p4c = N["part4_fixed_budget_recall"], N["part4_rc10"]

    # Only the body goes through .format(); the stylesheet is left alone, since CSS rules
    # are full of braces that str.format would try to interpret as fields.
    body = (HERE / "report_body.html").read_text().format(
        roll=a.roll, name=a.name,
        cpu=m["cpu_model"], cores=m["cpu_logical"], mem=m["mem_gb"], gpu=m["gpu"],
        faiss=m["faiss"], numpy=m["numpy"], py=m["python"], mid=m["machine_id"],
        rows=N["rows_total"],
        s_exact=s["exact_qps_1q"], s_batch=s["exact_qps_batch"], s_sp=s["batch_speedup"],
        g_exact=g["exact_qps_1q"], g_batch=g["exact_qps_batch"], g_sp=g["batch_speedup"],
        s_hnsw=s["hnsw"]["qps_at_095"], s_ivf=s["ivf_flat"]["qps_at_095"],
        s_hi=s["hnsw_over_ivf"], s_he=int(s["hnsw_over_exact"]), s_pq=s["ivf_pq"]["peak_recall"],
        g_hnsw=g["hnsw"]["qps_at_095"], g_ivf=g["ivf_flat"]["qps_at_095"],
        g_hi=g["hnsw_over_ivf"], g_he=int(g["hnsw_over_exact"]), g_pq=g["ivf_pq"]["peak_recall"],
        e_exact=e["exact_flat"], e_ivf=e["ivf_flat"], e_hnsw=e["hnsw"],
        rc_u=" → ".join(f"{x:.2f}" for x in p4c["uniform"]),
        rc_c=" → ".join(f"{x:.2f}" for x in p4c["clustered"]),
        u_h=" → ".join(f"{x:.3f}" for x in p4r["uniform_hnsw"]),
        u_i=" → ".join(f"{x:.3f}" for x in p4r["uniform_ivf_flat"]),
        c_h=" → ".join(f"{x:.3f}" for x in p4r["clustered_hnsw"]),
        c_i=" → ".join(f"{x:.3f}" for x in p4r["clustered_ivf_flat"]),
        table1=md_table((RES / "table1_qps_at_recall.md").read_text()),
        table2=md_table((RES / "table2_build_memory.md").read_text()),
        fig1=img("fig1_sift_tradeoff.png"), fig2=img("fig2_glove_tradeoff.png"),
        fig3=img("fig3_part3_scaling.png"), fig4=img("fig4_part4_geometry.png"),
        d_proto=diagrams.protocol_diagram(),
    )
    html = ('<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
            '<title>Benchmarking ANN Search</title>\n<style>\n'
            + (HERE / "report_style.css").read_text()
            + '\n</style></head><body>\n' + body + '\n</body></html>\n')
    (OUTDIR / "report.html").write_text(html)
    print(f"wrote {OUTDIR / 'report.html'}")

    chrome = next((c for c in ("google-chrome", "chromium", "chromium-browser")
                   if shutil.which(c)), None)
    if not chrome:
        print("no Chrome/Chromium found; open report.html and print to PDF manually")
        return 1
    out = OUTDIR / "report.pdf"
    r = subprocess.run([chrome, "--headless", "--disable-gpu", "--no-sandbox",
                        "--no-pdf-header-footer", f"--print-to-pdf={out}",
                        str(OUTDIR / "report.html")],
                       capture_output=True, text=True, timeout=180)
    if not out.exists():
        print(r.stderr[-2000:])
        return 1
    stamp_page_numbers(out, f"{a.name} ({a.roll})")
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB)")
    return verify(html, out)


def stamp_page_numbers(pdf, left_text):
    """Draw "name (roll)" and "N / M" along the foot of every page.

    Chrome cannot do this itself: its print path ignores CSS @page margin boxes, and its own
    --print-to-pdf footer is not configurable from the command line. So the numbers are
    stamped onto the finished PDF instead. If pypdf is not installed the report is still
    perfectly valid, just unnumbered, so this never fails the build.
    """
    try:
        from pypdf import PdfReader, PdfWriter
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import A4
    except ImportError:
        print("  (pypdf/reportlab not installed — page numbers skipped)")
        return

    import io
    reader = PdfReader(str(pdf))
    n = len(reader.pages)
    overlay = io.BytesIO()
    cv = canvas.Canvas(overlay, pagesize=A4)
    w, h = A4
    for i in range(1, n + 1):
        cv.setFont("Helvetica", 7.4)
        cv.setFillColorRGB(0.45, 0.47, 0.52)
        cv.drawString(48, 24, left_text)
        cv.setFillColorRGB(0.08, 0.24, 0.47)          # the document's navy
        cv.setFont("Helvetica-Bold", 7.6)
        cv.drawRightString(w - 48, 24, f"{i} / {n}")
        cv.setStrokeColorRGB(0.80, 0.82, 0.85)
        cv.setLineWidth(0.4)
        cv.line(48, 32, w - 48, 32)
        cv.showPage()
    cv.save()
    overlay.seek(0)

    writer = PdfWriter()
    marks = PdfReader(overlay)
    for page, mark in zip(reader.pages, marks.pages):
        page.merge_page(mark)
        writer.add_page(page)
    with open(pdf, "wb") as f:
        writer.write(f)
    print(f"  stamped page numbers on {n} pages")


def verify(html, pdf):
    """Check that nothing in the HTML was dropped on its way into the PDF.

    An earlier layout gave each page a fixed height, and Chrome silently clipped whatever
    overran it — the Part 4 control callout, the single most important result in that
    section, vanished from the PDF while remaining in the HTML. Nothing in the build failed.
    So every heading, figure caption and callout is now checked for presence in the rendered
    text, and the build reports a failure if any of them is missing.
    """
    if not shutil.which("pdftotext"):
        print("  (pdftotext not installed — skipping the clipping check)")
        return 0
    txt = subprocess.run(["pdftotext", str(pdf), "-"], capture_output=True, text=True).stdout
    flat = re.sub(r"\s+", " ", txt)

    def norm(x):
        x = re.sub(r"<[^>]+>", " ", x)   # tags become spaces: <span>4</span>Part -> "4 Part"
        x = (x.replace("&amp;", "&").replace("&mdash;", "—").replace("&nbsp;", " ")
             .replace("&#8594;", "→"))
        return re.sub(r"\s+", " ", x).strip()

    sentinels = []
    # Headings carry a <span class="num"> that pdftotext may place on its own line, so the
    # sentinel is the heading text WITHOUT its number.
    for pat in (
            # headings, minus their <span class="num"> which pdftotext may split onto its
            # own line
            r'<h2>(?:<span class="num">[^<]*</span>)?(.*?)</h2>',
            r'<h3>(?:<span class="num">[^<]*</span>)?(.*?)</h3>',
            # figure and table captions, minus the "Figure N:" label
            r'<figcaption>(?:<span class="lbl">[^<]*</span>)?(.*?)</figcaption>',
            r'<div class="tcap">(?:<span class="lbl">[^<]*</span>)?(.*?)</div>',
            # callout titles and their bodies -- this is the content that went missing
            r'<div class="box"><span class="bt">(.*?)</span>',
            r'<div class="box"><span class="bt">[^<]*</span>\s*<p>(.{0,160})',
            # the five report questions
            r'<span class="q">(.*?)</span>'):
        sentinels += [norm(m) for m in re.findall(pat, html, re.S)]

    missing = []
    for sraw in sentinels:
        probe = " ".join(sraw.split()[:6])
        if len(probe) < 8:
            continue
        if probe not in flat:
            missing.append(sraw[:70])
    pages = flat.count("\f") + 1 if "\f" in txt else txt.count("\f") + 1
    if missing:
        print(f"  !! {len(missing)} block(s) present in the HTML but MISSING from the PDF:")
        for mm in missing:
            print(f"     - {mm}")
        print("     (content was clipped — fix the layout before submitting)")
        return 1
    print(f"  checked {len(sentinels)} headings, captions and callouts — all present in the PDF")
    return 0



if __name__ == "__main__":
    sys.exit(main())

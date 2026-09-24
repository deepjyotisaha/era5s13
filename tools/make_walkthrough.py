"""Build the Session 13 report page from README.md, the EXECUTED notebook and summary.json.

Same series as the Session 11 and 12 walkthroughs. The page opens with the headline numbers,
then the README's own write-up (the answer, the findings, how reversibility works, the setup,
the limits) with the figures and two drawn diagrams, then every cell with the transcript it
printed on Colab. Nothing is typed here by hand: the prose comes from README.md (whose numbers
come from summary.json via tools/make_readme.py), the diagrams' numbers from summary.json, the
transcripts from the executed .ipynb, the figures from outputs/.

Usage:
    python tools/make_walkthrough.py                       # newest executed notebook
    python tools/make_walkthrough.py path/to/executed.ipynb
"""
import base64
import glob
import html
import io
import json
import os
import re
import sys
import tokenize
import zipfile
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loop_section import loop_section  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
OUT = os.path.join(ROOT, "session13_walkthrough.html")
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
LIST = re.compile(r"^\s*(?:[-*]|\d+\.)\s+")
IMG = re.compile(r"^!\[([^\]]*)\]\(([^)]+)\)\s*$")

# README chapters that belong to the repo, not the report: the cells section covers them.
SKIP_CHAPTERS = ("Files", "How to run", "Acceptance checks")
KICKER = {"The answer in one table": "the answer", "What the brief asked, answered": "the brief",
          "Findings": "findings", "How the reversible stack saves memory": "how it works",
          "The setup": "setup", "Limitations": "limits"}


def newest_executed():
    c = sorted(glob.glob(os.path.join(ROOT, "*_v*_executed*.ipynb")))
    return c[-1] if c else sorted(glob.glob(os.path.join(ROOT, "*.ipynb")))[-1]


def highlight(src):
    """Python syntax highlighting via the stdlib tokenizer (strings with '#' stay strings)."""
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return html.escape(src)
    KW = {"False", "None", "True", "and", "as", "assert", "async", "await", "break", "class",
          "continue", "def", "del", "elif", "else", "except", "finally", "for", "from",
          "global", "if", "import", "in", "is", "lambda", "nonlocal", "not", "or", "pass",
          "raise", "return", "try", "while", "with", "yield"}
    lines = src.split("\n")
    spans = []
    for t in toks:
        if t.type == tokenize.COMMENT:
            spans.append((t.start, t.end, "c"))
        elif t.type == tokenize.STRING:
            spans.append((t.start, t.end, "s"))
        elif t.type == tokenize.NUMBER:
            spans.append((t.start, t.end, "n"))
        elif t.type == tokenize.NAME and t.string in KW:
            spans.append((t.start, t.end, "k"))
    for (sr, sc), (er, ec), cls in sorted(spans, reverse=True):
        if sr != er:
            lines[sr - 1] = lines[sr - 1][:sc] + f"\x00{cls}\x01" + lines[sr - 1][sc:]
            lines[er - 1] = lines[er - 1][:ec] + "\x02" + lines[er - 1][ec:]
        else:
            L = lines[sr - 1]
            lines[sr - 1] = L[:sc] + f"\x00{cls}\x01" + L[sc:ec] + "\x02" + L[ec:]
    out = html.escape("\n".join(lines))
    out = re.sub(r"\x00(\w)\x01", lambda m: f'<span class="{m.group(1)}">', out)
    return out.replace("\x02", "</span>")


def inline(s):
    s = html.escape(s, quote=False)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    # External links stay links; repo-relative ones would 404 on a published page, so they keep
    # their text only.
    s = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r'<a href="\2">\1</a>', s)
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<![*\w])\*([^*\s][^*]*)\*(?![*\w])", r"<em>\1</em>", s)
    return s


def _starts_block(ln):
    return ln.startswith(("```", "|", "#", ">", "![")) or bool(LIST.match(ln))


def data_uri(path):
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def md_to_html(md, hooks=None):
    """The Markdown subset the notebook and README use: headings, tables, fences, lists,
    blockquotes, images, paragraphs, inline code/bold/em/links. `hooks` maps a substring of a
    paragraph or an image path to extra HTML placed right after that block."""
    hooks = hooks or {}
    lines, out, i = md.split("\n"), [], 0

    def after(text):
        for k, v in hooks.items():
            if k in text:
                out.append(v)

    while i < len(lines):
        ln = lines[i]
        if not ln.strip() or re.match(r"^-{3,}\s*$", ln):
            i += 1
            continue
        m = IMG.match(ln)
        if m:
            p = os.path.join(ROOT, m.group(2))
            if os.path.exists(p):
                out.append(f'<figure class="fig"><img src="{data_uri(p)}" alt="{html.escape(m.group(1))}">'
                           f'<figcaption>{html.escape(m.group(1)[0].upper() + m.group(1)[1:])} · '
                           f'<code>{html.escape(os.path.basename(p))}</code>, drawn by Cell 16.</figcaption></figure>')
            after(m.group(2))
            i += 1
            continue
        if ln.startswith("```"):
            block = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                block.append(lines[i])
                i += 1
            out.append('<pre class="fence">' + html.escape("\n".join(block)) + "</pre>")
            i += 1
            continue
        if ln.startswith("|") and i + 1 < len(lines) and set(lines[i + 1].replace("|", "").strip()) <= set("-: "):
            head = [c.strip() for c in ln.strip().strip("|").split("|")]
            i += 2
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            t = ['<div class="tw"><table>']
            if any(head):
                t.append("<thead><tr>" + "".join(f"<th>{inline(c)}</th>" for c in head) + "</tr></thead>")
            t.append("<tbody>")
            for r in rows:
                t.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>")
            t.append("</tbody></table></div>")
            out.append("".join(t))
            continue
        m = re.match(r"^(#{1,6})\s+(.*)$", ln)
        if m:
            lvl = min(len(m.group(1)) + 1, 6)
            out.append(f"<h{lvl}>{inline(m.group(2))}</h{lvl}>")
            i += 1
            continue
        if ln.startswith(">"):
            quote = []
            while i < len(lines) and lines[i].startswith(">"):
                quote.append(lines[i].lstrip(">").strip())
                i += 1
            out.append("<blockquote>" + inline(" ".join(quote)) + "</blockquote>")
            continue
        if LIST.match(ln):
            items, ordered = [], bool(re.match(r"^\s*\d+\.\s+", ln))
            while i < len(lines):
                if LIST.match(lines[i]):
                    items.append(LIST.sub("", lines[i], count=1))
                elif lines[i].startswith("  ") and lines[i].strip() and items:
                    items[-1] += " " + lines[i].strip()
                else:
                    break
                i += 1
            tag = "ol" if ordered else "ul"
            out.append(f"<{tag}>" + "".join(f"<li>{inline(x)}</li>" for x in items) + f"</{tag}>")
            after(" ".join(items))
            continue
        para = []
        while i < len(lines) and lines[i].strip() and (not para or not _starts_block(lines[i])):
            para.append(lines[i].strip())
            i += 1
        text = " ".join(para)
        out.append("<p>" + inline(text) + "</p>")
        after(text)
    return "\n".join(out)


def cell_output(cell):
    parts = []
    for o in cell.get("outputs", []):
        if o.get("output_type") == "stream":
            parts.append("".join(o.get("text", [])))
        elif o.get("output_type") in ("execute_result", "display_data"):
            d = o.get("data", {})
            if "text/plain" in d and "image/png" not in d:
                parts.append("".join(d["text/plain"]))
        elif o.get("output_type") == "error":
            parts.append("\n".join(o.get("traceback", [])))
    return ANSI.sub("", "".join(parts)).rstrip()


def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:48]


def readme_parts():
    text = open(os.path.join(ROOT, "README.md"), encoding="utf-8").read()
    intro, *blocks = re.split(r"^## ", text, flags=re.M)
    brief = " ".join(l.lstrip(">").strip() for l in intro.split("\n") if l.startswith(">"))
    chapters = []
    for block in blocks:
        title, _, body = block.partition("\n")
        if not title.strip().startswith(SKIP_CHAPTERS):
            chapters.append((title.strip(), body))
    return brief, chapters


# ------------------------------------------------------------------ diagrams, from summary.json
def memory_diagram(R):
    pr = R["probe"]
    ordi, rev = pr["mib_per_layer_ordinary"], pr["mib_per_layer_midpoint"]
    top = max(pr["depth"], key=int)
    mo, mr = pr["depth"][top]["ordinary"] / 2**20, pr["depth"][top]["midpoint"] / 2**20
    layers = [("layer 16", 58), ("layer 3", 132), ("layer 2", 178), ("layer 1", 224)]
    g = []

    def panel(x0, title, reversible):
        g.append(f'<text x="{x0 + 10}" y="24" font-size="14" font-weight="600" fill="var(--ink)">{title}</text>')
        g.append(f'<text x="{x0 + 10}" y="42" font-size="11.5" fill="var(--ink-3)">'
                 f'{"the forward pass keeps only the top; the backward pass rebuilds" if reversible else "the forward pass keeps what every layer computed"}</text>')
        g.append(f'<text x="{x0 + 90}" y="120" font-size="16" fill="var(--ink-3)" text-anchor="middle">⋮</text>')
        for name, y in layers:
            live = reversible and name == "layer 2"
            fill, stroke = ("var(--accent-soft)", "var(--accent)") if live else ("var(--surface-2)", "var(--line-2)")
            g.append(f'<rect x="{x0 + 10}" y="{y}" width="160" height="36" rx="6" fill="{fill}" stroke="{stroke}"/>')
            g.append(f'<text x="{x0 + 90}" y="{y + 23}" font-size="12.5" text-anchor="middle" fill="var(--ink)">{name}</text>')
            cx = x0 + 182
            if not reversible:
                g.append(f'<rect x="{cx}" y="{y + 4}" width="170" height="28" rx="5" fill="var(--bad-soft)" stroke="var(--bad-line)"/>')
                g.append(f'<text x="{cx + 85}" y="{y + 22}" font-size="11.5" text-anchor="middle" fill="var(--bad)">working numbers kept</text>')
            elif name == "layer 16":
                g.append(f'<rect x="{cx}" y="{y + 4}" width="170" height="28" rx="5" fill="var(--good-soft)" stroke="var(--good-line)"/>')
                g.append(f'<text x="{cx + 85}" y="{y + 22}" font-size="11.5" text-anchor="middle" fill="var(--good)">top two states kept</text>')
            elif live:
                g.append(f'<rect x="{cx}" y="{y + 4}" width="170" height="28" rx="5" fill="var(--accent-soft)" stroke="var(--accent)"/>')
                g.append(f'<text x="{cx + 85}" y="{y + 22}" font-size="11.5" text-anchor="middle" fill="var(--accent-ink)">rebuilt, used, freed</text>')
            else:
                g.append(f'<rect x="{cx}" y="{y + 4}" width="170" height="28" rx="5" fill="none" stroke="var(--line-2)" stroke-dasharray="4 4"/>')
                g.append(f'<text x="{cx + 85}" y="{y + 22}" font-size="11.5" text-anchor="middle" fill="var(--ink-3)">nothing kept</text>')
        if reversible:
            g.append(f'<line x1="{x0 + 370}" y1="64" x2="{x0 + 370}" y2="250" stroke="var(--accent)" stroke-width="2" marker-end="url(#dn)"/>')
            g.append(f'<text x="{x0 + 10}" y="282" font-size="12" fill="var(--ink-2)">backward walks down: '
                     f'<tspan font-family="var(--mono)">previous = next − 2h·f(current)</tspan></text>')
        else:
            g.append(f'<text x="{x0 + 10}" y="282" font-size="12" fill="var(--ink-2)">backward reads the kept numbers, top to bottom</text>')
        per, at = (rev, mr) if reversible else (ordi, mo)
        col = "var(--good)" if reversible else "var(--bad)"
        g.append(f'<text x="{x0 + 10}" y="310" font-size="14" font-weight="600" fill="{col}">+{per:.1f} MiB for each extra layer</text>')
        g.append(f'<text x="{x0 + 10}" y="330" font-size="11.5" fill="var(--ink-3)">{top} layers at batch 32: {at:,.0f} MiB peak</text>')

    panel(0, "Ordinary stack", False)
    panel(400, "Reversible stack", True)
    svg = ('<svg viewBox="0 0 790 344" role="img" aria-label="Ordinary stack keeps every layer\'s working numbers; '
           'the reversible stack keeps only the top two states and rebuilds one layer at a time">'
           '<defs><marker id="dn" viewBox="0 0 10 10" refX="5" refY="9" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
           '<path d="M0 0 L10 0 L5 10 z" fill="var(--accent)"/></marker></defs>'
           '<line x1="392" y1="10" x2="392" y2="334" stroke="var(--line)"/>' + "".join(g) + "</svg>")
    return (f'<figure class="diagram">{svg}<figcaption>Where the memory goes. The per-layer numbers are '
            f'measured by the memory probe (Cell 12), from the slope of peak memory against depth. The '
            f'reversible stack still pays for each layer\'s weights and Adam state, which is the {rev:.1f} MiB.'
            f'</figcaption></figure>')


def speed_diagram(R):
    res = {r["run"]: r for r in R["results"]}
    slow = {k: 1 - res[k]["tokens_per_second"] / res["A"]["tokens_per_second"] for k in "BC"}
    u, x0 = 130, 150
    rows = [("ordinary", 40, [("forward", 1, "var(--accent-soft)", "var(--accent)"),
                              ("backward ≈ 2 × forward", 2, "var(--surface-2)", "var(--line-2)")]),
            ("reversible", 104, [("forward", 1, "var(--accent-soft)", "var(--accent)"),
                                 ("forward again, to rebuild", 1, "var(--warn-soft)", "var(--warn-line)"),
                                 ("backward ≈ 2 × forward", 2, "var(--surface-2)", "var(--line-2)")])]
    g = []
    for name, y, parts in rows:
        g.append(f'<text x="10" y="{y + 23}" font-size="13" font-weight="600" fill="var(--ink)">{name}</text>')
        x = x0
        for label, n, fill, stroke in parts:
            g.append(f'<rect x="{x}" y="{y}" width="{u * n - 4}" height="36" rx="5" fill="{fill}" stroke="{stroke}"/>')
            g.append(f'<text x="{x + (u * n - 4) / 2}" y="{y + 23}" font-size="11.5" text-anchor="middle" fill="var(--ink)">{label}</text>')
            x += u * n
        units = sum(p[1] for p in parts)
        g.append(f'<text x="{x + 6}" y="{y + 23}" font-size="12" fill="var(--ink-3)">{units} units</text>')
    svg = ('<svg viewBox="0 0 760 184" role="img" aria-label="Work per training step: ordinary three units, reversible four units">'
           + "".join(g)
           + f'<text x="10" y="170" font-size="12.5" fill="var(--ink-2)">Rule of thumb: 3 units of work become 4, so '
             f'about 25% fewer tokens per second. Measured: midpoint {slow["B"]:.0%}, two-stream {slow["C"]:.0%}.</text></svg>')
    return (f'<figure class="diagram">{svg}<figcaption>Why reversibility costs speed. Each layer runs one '
            f'extra forward pass during the backward pass, to rebuild its input. The "backward ≈ 2 × forward" '
            f'ratio is the usual rule of thumb, not a measurement from this run.</figcaption></figure>')


def facts(R):
    res = {r["run"]: r for r in R["results"]}
    A, B, C, D = (res[k] for k in "ABCD")
    pr = R["probe"]
    lb = pr["largest_batch"]
    top = max(pr["depth"], key=int)
    better = "midpoint" if R["variant"]["winner"] == "B" else "two-stream"
    return [
        ("memory at the same batch", "peak memory at batch 32, ordinary → midpoint",
         f"{A['peak_mib']:,.0f} → {B['peak_mib']:,.0f} MiB",
         f"{A['peak_mib'] / B['peak_mib']:.1f}× less; two-stream {C['peak_mib']:,.0f} MiB"),
        ("memory per extra layer", "ordinary vs reversible, from the depth probe",
         f"{pr['mib_per_layer_ordinary']:.1f} vs {pr['mib_per_layer_midpoint']:.1f} MiB",
         f"at {top} layers: {pr['depth'][top]['ordinary'] / 2**20:,.0f} vs {pr['depth'][top]['midpoint'] / 2**20:,.0f} MiB"),
        ("largest batch that fits", "ordinary · midpoint · two-stream, on the T4",
         f"{lb['ordinary']} · {lb['midpoint']} · {lb['two-stream']}",
         f"{pr['ratio_midpoint']:.2f}× and {pr['ratio_two-stream']:.2f}× the ordinary model"),
        ("the speed cost", "fewer tokens per second than ordinary",
         f"{1 - B['tokens_per_second'] / A['tokens_per_second']:.0%} · {1 - C['tokens_per_second'] / A['tokens_per_second']:.0%}",
         f"midpoint · two-stream; ordinary {A['tokens_per_second']:,.0f} tokens/s"),
        ("which variant trained better", "final validation loss at batch 32",
         f"{better} {min(B['final_val'], C['final_val']):.4f}",
         f"the other {max(B['final_val'], C['final_val']):.4f}; ordinary {A['final_val']:.4f}"),
        ("the big batch (run D)", f"batch {D['batch']}, the same 50M tokens",
         f"{D['final_val']:.4f}",
         f"{D['tokens_per_second']:,.0f} tokens/s, no faster; {A['steps'] / D['steps']:.1f}× fewer steps"),
    ]


def build(nb_path):
    nb = json.load(open(nb_path, encoding="utf-8"))
    R = json.load(open(os.path.join(ROOT, "outputs", "summary.json"), encoding="utf-8"))
    meta, acc = R["meta"], R["acceptance"]
    brief, chapters = readme_parts()

    wall, wall_exact = "", None
    z = os.path.join(ROOT, "outputs", "era5s13_outputs_full_v1.zip")
    if os.path.exists(z):
        m1 = json.loads(zipfile.ZipFile(z).read("summary.json"))["meta"]
        f = "%Y-%m-%d %H:%M:%S"
        wall_exact = (datetime.strptime(m1['finished'], f) - datetime.strptime(m1['started'], f)).total_seconds() / 60
        wall = f"{wall_exact:.0f} min"

    cell4 = next((cell_output(c) for c in nb["cells"] if c["cell_type"] == "code"
                  and "build_data()" in "".join(c["source"])), "")
    loop_html = loop_section(R, ROOT, wall_exact, cell4)

    hooks = {"outputs/speed_memory.png": speed_diagram(R),
             "Only one layer's working numbers exist": memory_diagram(R)}

    # ---- the write-up: README chapters, in README order
    learn = []
    for title, body in chapters:
        learn.append(f"""
<section class="chapter" id="{slug(title)}">
  <div class="kicker">{html.escape(KICKER.get(title, ""))}</div>
  <div>
    <h2>{inline(title)}</h2>
    <div class="prose">{md_to_html(body, hooks)}</div>
  </div>
</section>""")

    # ---- cells
    pairs, pending, seen_intro = [], None, False
    for c in nb["cells"]:
        if c["cell_type"] == "markdown":
            src = "".join(c["source"])
            if not seen_intro and not src.lstrip().startswith("## "):
                seen_intro = True
            else:
                pending = src
        elif c["cell_type"] == "code":
            pairs.append((pending, "".join(c["source"]), cell_output(c)))
            pending = None

    sections, toc = [], []
    for idx, (md, code, out) in enumerate(pairs, start=1):
        md = md or ""
        m = re.search(r"^##\s+Cell\s+([0-9a-z]+)\s+—\s+(.*)$", md, re.M)
        num = m.group(1) if m else str(idx)
        title = m.group(2) if m else f"Cell {idx}"
        body_md = re.sub(r"^##\s+Cell.*$", "", md, count=1, flags=re.M)
        g = re.match(r"^gate\s+(\S+?):\s*(.*)$", title, re.I)
        gate = g.group(1) if g else None
        clean = g.group(2) if g else title
        clean = clean[0].upper() + clean[1:]
        anchor = f"cell-{num}"
        toc.append((anchor, num, clean, gate))

        chips = []
        if gate:
            chips.append(f'<span class="chip gate">gate {html.escape(gate)}</span>')
        if not out.strip():
            chips.append('<span class="chip quiet">no output</span>')

        sections.append(f"""
<section class="cell" id="{anchor}">
  <div class="rail">
    <div class="cellno">{html.escape(num)}</div>
    <div class="chips">{''.join(chips)}</div>
  </div>
  <div class="main">
    <h2>{inline(clean)}</h2>
    <div class="prose">{md_to_html(body_md)}</div>
    <details class="src">
      <summary><span>source</span><span class="loc">{len(code.splitlines())} lines</span></summary>
      <pre class="code">{highlight(code)}</pre>
    </details>
    <div class="outwrap">
      <div class="outlabel">what it printed</div>
      <pre class="out">{html.escape(out) if out.strip() else '(no output)'}</pre>
    </div>
  </div>
</section>""")

    factcards = "".join(
        f'<div class="fact"><div class="ft">{html.escape(t)}</div><div class="fq">{html.escape(q)}</div>'
        f'<div class="fv">{html.escape(v)}</div><div class="fn">{html.escape(n)}</div></div>'
        for t, q, v, n in facts(R))
    toc_html = "".join(
        f'<a class="tocrow" href="#{a}"><span class="tocno">{html.escape(n)}</span>'
        f'<span class="toctitle">{inline(t)}</span>'
        f'<span class="toctags">{f"<b>gate {html.escape(gt)}</b>" if gt else ""}</span></a>'
        for a, n, t, gt in toc)

    css = open(os.path.join(HERE, "walkthrough.css"), encoding="utf-8").read()
    page = f"""<title>The Reversible Stack</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;0,6..72,600;1,6..72,400&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>{css}</style>

<header class="hero">
  <div class="wrap">
    <p class="eyebrow">ERA V5 · Session 13 · Distributed training II: model and pipeline parallel</p>
    <h1>The Reversible<br>Stack</h1>
    <p class="brief">{inline(brief)}</p>
    <p class="lede">One 22.5M-parameter model trained four times on the same 50M tokens: an ordinary
    stack, two reversible stacks that rebuild each layer instead of storing it, and a reversible
    stack at a much larger batch. The page opens with the headline numbers, then the full loop
    from web pages to results, then the write-up, and last every cell with the transcript it
    printed on Colab.</p>
    <dl class="runmeta">
      <div><dt>gpu</dt><dd>Colab {html.escape(meta['gpu'])}</dd></div>
      <div><dt>torch</dt><dd>{html.escape(meta['torch'])}</dd></div>
      <div><dt>model</dt><dd>{R['config']['n_params'] / 1e6:.1f}M params</dd></div>
      {f'<div><dt>full run</dt><dd>{wall}</dd></div>' if wall else ''}
      <div><dt>acceptance</dt><dd class="pass">{acc['passed']}/{acc['total']}</dd></div>
    </dl>
  </div>
</header>

<main class="wrap">
  <section class="facts">
    <h2 class="secttl">What the run measured</h2>
    <div class="factgrid six">{factcards}</div>
  </section>

  {loop_html}

  <section class="learn">
    <h2 class="secttl">The write-up — from the README, generated from summary.json</h2>
    {''.join(learn)}
  </section>

  <section class="toc">
    <h2 class="secttl">Every cell</h2>
    <nav>{toc_html}</nav>
  </section>

  <div class="cells">{''.join(sections)}</div>
</main>

<footer class="foot">
  <div class="wrap">
    <p>Built from <code>README.md</code>, <code>outputs/summary.json</code> and
    <code>{html.escape(os.path.basename(nb_path))}</code> by <code>tools/make_walkthrough.py</code>.
    The write-up is the README's own text, whose numbers come from <code>summary.json</code> via
    <code>tools/make_readme.py</code>; the two diagrams take their numbers from the same file; the
    transcripts are the executed notebook's own output. Nothing on this page was typed by hand.</p>
  </div>
</footer>
"""
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(page)
    return OUT, len(pairs), len(chapters)


if __name__ == "__main__":
    nb = sys.argv[1] if len(sys.argv) > 1 else newest_executed()
    path, n, k = build(nb)
    print(f"{os.path.basename(nb)} -> {os.path.basename(path)}")
    print(f"  {k} README chapters · {n} code cells · {os.path.getsize(path) / 1e6:.2f} MB")

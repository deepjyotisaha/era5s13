"""The "full loop" section of the report: how the experiment runs, from web text to the results
table, drawn as diagrams. Every number is computed from summary.json, the run logs, or the
executed notebook's own output; the few rules restated here (window cutting, step count,
schedule, probe search) are the notebook's own formulas, re-evaluated on the run's settings.
Used by tools/make_walkthrough.py."""
import html
import json
import math
import os
import re

MiB = 2**20


# ------------------------------------------------------------------ small SVG helpers
def _t(x, y, s, size=12, weight=400, fill="var(--ink)", anchor="start", mono=False, italic=False):
    style = "font-family:var(--mono);" if mono else ""
    style += "font-style:italic;" if italic else ""
    st = f' style="{style}"' if style else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" fill="{fill}" '
            f'text-anchor="{anchor}"{st}>{html.escape(s)}</text>')


def _box(x, y, w, h, fill="var(--surface-2)", stroke="var(--line-2)", dash=False, rx=7):
    d = ' stroke-dasharray="5 4"' if dash else ""
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}" stroke="{stroke}"{d}/>'


def _arrow(x1, y1, x2, y2, mid, color="var(--ink-3)"):
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" '
            f'stroke-width="1.6" marker-end="url(#{mid})"/>')


def _marker(mid, color="var(--ink-3)"):
    return (f'<defs><marker id="{mid}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
            f'markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="{color}"/>'
            f'</marker></defs>')


def _card(x, y, w, h, title, sub, lines, fill="var(--surface-2)", stroke="var(--line-2)"):
    g = [_box(x, y, w, h, fill, stroke), _t(x + 12, y + 22, title, 13.5, 600)]
    if sub:
        g.append(_t(x + 12, y + 39, sub, 11, fill="var(--ink-3)", mono=True))
    for k, ln in enumerate(lines):
        g.append(_t(x + 12, y + (58 if sub else 44) + 16 * k, ln, 11.5, fill="var(--ink-2)"))
    return "".join(g)


def _svg(w, h, label, body, caption):
    return (f'<figure class="diagram"><svg viewBox="0 0 {w} {h}" role="img" aria-label="{html.escape(label)}">'
            f'{body}</svg><figcaption>{caption}</figcaption></figure>')


def _table(head, rows):
    t = ['<div class="tw"><table><thead><tr>'] + [f"<th>{h}</th>" for h in head] + ["</tr></thead><tbody>"]
    for r in rows:
        t.append("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>")
    return "".join(t) + "</tbody></table></div>"


# ------------------------------------------------------------------ the notebook's own rules
def lr_at(step, steps, lr, warmup_frac, min_frac):
    warm = max(10, int(warmup_frac * steps))
    if step < warm:
        return lr * (step + 1) / warm
    t = (step - warm) / max(1, steps - warm)
    return lr * (min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * t)))


def probe_tries(largest, cap):
    """Cell 12's search, replayed with the result it found: every try at or below the largest
    batch fitted and every try above it ran out of memory, so the sequence of tries is fixed."""
    tries, lo, hi = [], 0, 8
    while hi <= cap:
        ok = hi <= largest
        tries.append((hi, ok))
        if not ok:
            break
        lo, hi = hi, hi * 2
    hi = min(hi, cap + 1)
    while hi - lo > 8:
        mid = (lo + hi) // 2 // 8 * 8
        if mid <= lo:
            break
        ok = mid <= largest
        tries.append((mid, ok))
        lo, hi = (mid, hi) if ok else (lo, mid)
    assert lo == largest, (lo, largest)
    return tries


# ------------------------------------------------------------------ the section
def loop_section(R, root, wall_min, cell4_out):
    cfg, mc, tc, dc = R["config"], R["config"]["model"], R["config"]["train"], R["config"]["data"]
    res = {r["run"]: r for r in R["results"]}
    A, D = res["A"], res["D"]
    logs = os.path.join(root, "outputs", "logs")
    L = {f[:-5]: json.load(open(os.path.join(logs, f), encoding="utf-8"))
         for f in os.listdir(logs) if f.endswith(".json")}
    sweep_logs = {k: v for k, v in L.items() if k.startswith("sweep_")}
    runlog = {k[0]: v for k, v in L.items() if k[:2] in ("A_", "B_", "C_", "D_")}

    seq, bA, bD = mc["seq"], tc["batch"], D["batch"]
    tpsA, tpsD = bA * seq, bD * seq
    stepsA, stepsD = tc["tokens"] // tpsA, tc["tokens"] // tpsD
    assert stepsA == A["steps"] and stepsD == D["steps"]
    n_win = (R["data"]["train_tokens"] - 1) // seq
    winA, winD = stepsA * bA, stepsD * bD
    warmA = max(10, int(tc["warmup_frac"] * stepsA))
    warmD = max(10, int(tc["warmup_frac"] * stepsD))
    evA = max(1, tc["eval_every_tokens"] // tpsA)
    evD = max(1, tc["eval_every_tokens"] // tpsD)
    val_tok = tc["eval_batches"] * bA * seq
    cpt = R["data"]["chars_per_token"]
    tmin = {k: runlog[k]["train_seconds"] / 60 for k in "ABCD"}
    sweep_min = sum(v["train_seconds"] for v in sweep_logs.values()) / 60
    other_min = wall_min - sum(tmin.values()) - sweep_min if wall_min else None
    sec_step = {k: runlog[k]["train_seconds"] / runlog[k]["steps"] for k in "ABCD"}
    m = re.search(r"round trip\s*:\s*(\d+) tokens -> '([^']*)'", cell4_out or "")
    rt_n, rt_s = (int(m.group(1)), m.group(2)) if m else (None, None)
    score_full = lambda b: b * seq * mc["vocab"] * 4
    score_slice = tc["loss_chunk"] * mc["vocab"] * 4
    n_slices = lambda b: math.ceil(b * seq / tc["loss_chunk"])

    ch = []

    def chapter(kicker, anchor, title, body):
        ch.append(f'<section class="chapter" id="{anchor}"><div class="kicker">{kicker}</div><div>'
                  f'<h2>{title}</h2><div class="prose">{body}</div></div></section>')

    # ---- 1 · the whole notebook
    mk = _marker("lo1")
    X = [10, 215, 420, 625]
    W, H = 185, 84
    y1, y2 = 36, 152
    stages = [
        ("Data", "Cell 4", [f"{R['data']['train_tokens'] / 1e6:.1f}M train + {R['data']['val_tokens'] / 1e6:.0f}M val tokens",
                            "built once, cached on Drive"], "var(--accent-soft)", "var(--accent)"),
        ("Safety gates", "Cells 7–9", ["gradients, planted bug, drift", "the notebook stops on a failure"],
         "var(--bad-soft)", "var(--bad-line)"),
        ("Step-size sweep", "Cell 11", [f"{len(sweep_logs)} short runs × {cfg['sweep']['steps']} steps",
                                        f"{sweep_min:.0f} train-min; picks h and start"], "var(--surface-2)", "var(--line-2)"),
        ("Memory probe", "Cell 12", ["largest batch for each model", "and memory against depth"],
         "var(--surface-2)", "var(--line-2)"),
        ("Runs A, B, C", "Cell 13", [f"{stepsA:,} steps each, batch {bA}",
                                     f"{tmin['A']:.1f} · {tmin['B']:.1f} · {tmin['C']:.1f} train-min"],
         "var(--good-soft)", "var(--good-line)"),
        ("Run D", "Cell 14", [f"{stepsD:,} steps, batch {bD}", f"{tmin['D']:.1f} train-min"],
         "var(--good-soft)", "var(--good-line)"),
        ("Results", "Cells 15–17", ["table, figures, 12 checks", "summary.json and the zip"],
         "var(--surface-2)", "var(--line-2)"),
    ]
    pos = [(X[0], y1), (X[1], y1), (X[2], y1), (X[3], y1), (X[3], y2), (X[2], y2), (X[1], y2)]
    g = [mk]
    for (t, c, lines, f, s), (x, y) in zip(stages, pos):
        g.append(_card(x, y, W, H, t, c, lines, f, s))
    for i in range(3):
        g.append(_arrow(X[i] + W + 2, y1 + H / 2, X[i + 1] - 3, y1 + H / 2, "lo1"))
    g.append(_arrow(X[3] + W / 2, y1 + H + 2, X[3] + W / 2, y2 - 3, "lo1"))
    for i in (3, 2):
        g.append(_arrow(X[i] - 2, y2 + H / 2, X[i - 1] + W + 3, y2 + H / 2, "lo1"))
    if wall_min:
        g.append(_box(X[0], y2, W, H, "none", "var(--line-2)", dash=True))
        g.append(_t(X[0] + 12, y2 + 22, f"{wall_min:.0f} min end to end", 13.5, 600))
        g.append(_t(X[0] + 12, y2 + 42, f"training runs {sum(tmin.values()):.0f} min", 11.5, fill="var(--ink-2)"))
        g.append(_t(X[0] + 12, y2 + 58, f"sweep {sweep_min:.0f} min", 11.5, fill="var(--ink-2)"))
        g.append(_t(X[0] + 12, y2 + 74, f"everything else {other_min:.0f} min", 11.5, fill="var(--ink-2)"))
    g.append(_t(10, 22, "The notebook, in the order it runs", 12, 600, "var(--ink-2)"))
    chapter("1 · overview", "loop-overview", "The whole notebook, in the order it runs",
            "<p>Seven stages run top to bottom. Cells 1–3, 5, 6 and 10 are not on the map: they set things "
            "up (settings, Drive storage, the model, the training loop) and are used by the stages below. "
            "Train-minutes count only the training steps; validation passes, loading and saving are in "
            "\"everything else\".</p>"
            + _svg(820, 250, "The seven stages of the notebook", "".join(g),
                   "Green boxes are the four runs the brief asks for. Every stage saves its result to Drive, "
                   "so a disconnect never repeats finished work."))

    # ---- 2 · data
    g = [_marker("lo2"), _t(10, 22, "FineWeb sample-10BT, streamed from Hugging Face, one web page after another (not to scale)",
                            12, 600, "var(--ink-2)")]
    g += [_box(10, 36, 160, 36, "var(--accent-soft)", "var(--accent)", rx=5),
          _t(90, 59, "validation pages", 12, 600, anchor="middle"),
          _box(170, 36, 350, 36, "var(--warn-soft)", "var(--warn-line)", rx=5),
          _t(345, 59, f"next {dc['tokenizer_chars'] / 1e6:.0f}M characters: tokenizer pages", 12, 600, anchor="middle"),
          _box(520, 36, 290, 36, "var(--surface-2)", "var(--line-2)", rx=5),
          _t(665, 59, "more pages …", 12, 600, anchor="middle")]
    g += [f'<path d="M172 84 L172 92 L808 92 L808 84" fill="none" stroke="var(--good)" stroke-width="1.4"/>',
          _t(490, 108, "the training text starts here and runs on", 11.5, fill="var(--good)", anchor="middle")]
    yb = 136
    g.append(_card(10, yb, 220, 76, "val.bin", "1 file",
                   [f"{R['data']['val_tokens']:,} tokens", f"{R['data']['val_tokens'] * 2 / 1e6:.0f} MB, never trained on"],
                   "var(--accent-soft)", "var(--accent)"))
    g.append(_card(262, yb, 246, 76, "the tokenizer", "byte-level BPE",
                   [f"learns {R['data']['vocab_size']:,} word pieces", "from the tokenizer pages"],
                   "var(--warn-soft)", "var(--warn-line)"))
    g.append(_card(540, yb, 270, 76, "train.bin", "1 file",
                   [f"{R['data']['train_tokens']:,} tokens", f"{R['data']['train_tokens'] * 2 / 1e6:.0f} MB of 2-byte ids"],
                   "var(--good-soft)", "var(--good-line)"))
    g += [_arrow(90, 74, 110, yb - 3, "lo2"), _arrow(345, 74, 375, yb - 3, "lo2"),
          _arrow(675, 110, 675, yb - 3, "lo2"),
          _arrow(262, yb + 40, 233, yb + 40, "lo2"), _arrow(508, yb + 40, 537, yb + 40, "lo2"),
          _t(385, yb + 98, "the tokenizer turns both piles of text into ids", 11.5, fill="var(--ink-3)", anchor="middle")]
    if rt_n:
        g += [_t(10, 262, "example:", 11.5, 600, "var(--ink-2)"),
              _t(72, 262, f"“{rt_s}”", 11.5, fill="var(--ink-2)", italic=True),
              _t(10, 280, f"= {len(rt_s)} characters → {rt_n} tokens, and decoding the ids gives the sentence back exactly",
                 11.5, fill="var(--ink-3)")]
    chapter("2 · data", "loop-data", "From web pages to two files of token ids",
            f"<p>The notebook reads FineWeb pages in order. The first pages go to <strong>validation</strong> and "
            f"are never trained on. The next {dc['tokenizer_chars'] / 1e6:.0f}M characters teach the "
            f"<strong>tokenizer</strong> its {R['data']['vocab_size']:,} word pieces, and those same pages are also the "
            f"start of the <strong>training text</strong>, which keeps going until it has "
            f"{R['data']['train_tokens'] / 1e6:.1f}M tokens. On this text one token is {cpt:.2f} characters on average, "
            f"so the training file holds about {R['data']['train_tokens'] * cpt / 1e6:.0f}M characters of web text. "
            f"Every page ends with a special end-of-text token, so the model can tell where one page stops.</p>"
            + _svg(820, 292, "FineWeb pages split into validation, tokenizer and training text", "".join(g),
                   "Each id is stored in 2 bytes, because 8,192 ids fit easily under the 65,536 that 2 bytes can hold. "
                   "The files are built once and reused by every run and every re-run."))

    # ---- 3 · windows
    g = [_marker("lo3"), _t(10, 22, "train.bin, cut into windows (not to scale)", 12, 600, "var(--ink-2)")]
    for k, (x0, x1) in enumerate([(150, 404), (400, 654), (650, 804)]):
        lab = f"window {k + 1}" if k < 2 else "window 3 …"
        g.append(_box(x0, 36 + (k % 2) * 6, x1 - x0, 24, ["var(--accent-soft)", "var(--good-soft)", "var(--surface-2)"][k],
                      ["var(--accent)", "var(--good-line)", "var(--line-2)"][k], rx=4))
        g.append(_t((x0 + x1) / 2, 53 + (k % 2) * 6, lab, 11.5, 600, anchor="middle"))
    g += [_t(10, 53, f"{n_win:,} windows", 12, 600),
          _t(150, 84, f"each window is {seq + 1} tokens; the next one starts {seq} tokens later, so neighbours share one token",
             11.5, fill="var(--ink-3)")]
    toy = ["The", "cat", "sat", "on", "the", "mat"]
    x0, bw, gap = 150, 72, 8
    g += [_t(10, 124, "a window", 12, 600), _t(10, 139, f"(6 here, {seq + 1} real)", 11, fill="var(--ink-3)"),
          _t(10, 180, "input x", 12, 600, "var(--accent-ink)"), _t(10, 195, f"(5 here, {seq} real)", 11, fill="var(--ink-3)"),
          _t(10, 244, "target y", 12, 600, "var(--good)"), _t(10, 259, "(the next token)", 11, fill="var(--ink-3)")]
    for i, w in enumerate(toy):
        x = x0 + i * (bw + gap)
        g += [_box(x, 108, bw, 32, "var(--surface-2)", "var(--line-2)", rx=5), _t(x + bw / 2, 129, w, 12.5, anchor="middle")]
        if i < 5:
            g += [_box(x, 164, bw, 32, "var(--accent-soft)", "var(--accent)", rx=5),
                  _t(x + bw / 2, 185, w, 12.5, anchor="middle")]
        if i >= 1:
            xt = x0 + (i - 1) * (bw + gap)
            g += [_box(xt, 228, bw, 32, "var(--good-soft)", "var(--good-line)", rx=5),
                  _t(xt + bw / 2, 249, w, 12.5, anchor="middle"),
                  _arrow(xt + bw / 2, 198, xt + bw / 2, 225, "lo3")]
    g += [_t(640, 180, "every position guesses", 11.5, fill="var(--ink-2)"),
          _t(640, 196, "the token after it", 11.5, fill="var(--ink-2)"),
          _t(640, 228, f"one window = {seq} guesses", 11.5, 600, "var(--ink)"),
          _t(640, 244, "to learn from", 11.5, fill="var(--ink-2)")]
    chapter("3 · windows", "loop-windows", "Cutting the token file into windows",
            f"<p>The training file is one long line of {R['data']['train_tokens']:,} ids. The notebook cuts it into "
            f"<strong>{n_win:,} windows</strong> of {seq + 1} tokens. The model reads the first {seq} tokens of a window "
            f"(the input) and, at every position, guesses the token that comes next (the target). So one window is "
            f"{seq} small exercises, not one. The attention inside the model only looks backwards, so the guess at "
            f"position 5 sees positions 1–5 and never the answer.</p>"
            f"<p>The windows are then shuffled <strong>once</strong>, with a fixed seed ({R['meta']['seed']}). That fixed "
            f"order is what makes the runs comparable: every run reads the same windows in the same order.</p>"
            + _svg(820, 272, "Windows cut from the token file, and one toy window split into input and target", "".join(g),
                   "The toy uses whole words to keep it readable; real tokens are often pieces of words."))

    # ---- 4 · batches and steps
    g = [_marker("lo4")]
    full0, full1 = 150, 810
    g += [_t(10, 22, "the shuffled order", 12, 600), _t(10, 37, f"{n_win:,} windows", 11, fill="var(--ink-3)"),
          _box(full0, 14, full1 - full0, 22, "var(--surface-2)", "var(--line-2)", rx=4)]
    xa = full0 + (full1 - full0) * winA / n_win
    g += [f'<rect x="{xa:.1f}" y="14" width="{full1 - xa:.1f}" height="22" fill="var(--bad-soft)" stroke="var(--bad-line)"/>',
          _t(full1, 52, f"last {n_win - winA:,} windows never read by runs A–C", 11, fill="var(--bad)", anchor="end"),
          f'<rect x="{full0}" y="14" width="4" height="22" fill="var(--accent)"/>',
          f'<path d="M{full0} 36 L{full0} 72 M{full0 + 4} 36 L{full1} 72" stroke="var(--accent)" stroke-dasharray="3 3" fill="none"/>',
          _t(full0 + 12, 52, "zoom on the first 480", 11, fill="var(--accent-ink)")]
    px = (full1 - full0) / 480
    g += [_t(10, 96, f"runs A–C", 12, 600), _t(10, 111, f"batch {bA}", 11, fill="var(--ink-3)")]
    for s in range(480 // bA):
        x = full0 + s * bA * px
        g += [_box(x + 1, 78, bA * px - 2, 34, "var(--good-soft)", "var(--good-line)", rx=3),
              _t(x + bA * px / 2, 100, str(s + 1), 11, anchor="middle")]
    g += [_t(10, 146, "run D", 12, 600), _t(10, 161, f"batch {bD}", 11, fill="var(--ink-3)")]
    for s in range(480 // bD):
        x = full0 + s * bD * px
        g += [_box(x + 1, 128, bD * px - 2, 34, "var(--warn-soft)", "var(--warn-line)", rx=3),
              _t(x + bD * px / 2, 150, f"step {s + 1} · windows {s * bD + 1}–{(s + 1) * bD}", 11.5, anchor="middle")]
    g.append(_t(10, 190, f"Numbers in the green boxes are step numbers. Run D's step 1 reads exactly the windows that runs A–C read "
                         f"in their steps 1 to {bD / bA:g}.", 11.5, fill="var(--ink-2)"))
    rows = [[f"A, B, C", f"{bA}", f"{bA} × {seq} = <strong>{tpsA:,}</strong>", f"{tc['tokens']:,} ÷ {tpsA:,} → <strong>{stepsA:,}</strong>",
             f"{stepsA * tpsA:,}", f"{winA:,} of {n_win:,}", f"{sec_step['A']:.2f} s (A) · {sec_step['B']:.2f} s (B)"],
            [f"D", f"{bD}", f"{bD} × {seq} = <strong>{tpsD:,}</strong>", f"{tc['tokens']:,} ÷ {tpsD:,} → <strong>{stepsD:,}</strong>",
             f"{stepsD * tpsD:,}", f"{winD:,} of {n_win:,}", f"{sec_step['D']:.2f} s"]]
    chapter("4 · batches", "loop-batches", "Batches and steps: how 50M tokens become 3,051 steps",
            f"<p>A <strong>step</strong> is one update of the weights. Each step reads the next <strong>batch</strong> of "
            f"windows from the shuffled order: {bA} windows for runs A–C, {bD} for run D. The number of steps is the "
            f"token budget divided by the tokens in one batch, rounded down. Each window is read at most once, so every "
            f"run makes <strong>one pass</strong> over the data and never sees the same text twice.</p>"
            + _svg(820, 200, "The shuffled window order and how batches of 32 and 240 read it", "".join(g),
                   "Both runs read the same windows in the same order; only the grouping into steps differs. "
                   "A bigger batch gives fewer, larger steps for the same tokens.")
            + _table(["runs", "windows per step", "tokens per step", "steps", "tokens trained", "windows read", "time per step"], rows)
            + f"<p>The budget does not divide evenly, so each run trains on slightly under 50M tokens. Run D takes "
              f"{stepsA / stepsD:.1f}× fewer steps than run A. It moves the weights only {stepsD} times, which is why its "
              f"loss ends much higher even though it reads nearly the same text.</p>")

    # ---- 5 · one step
    g = [_marker("lo5"), _t(10, 22, "forward pass: from ids to one loss number", 12, 600, "var(--ink-2)")]
    Xs = [10, 173, 336, 499, 662]
    bw5 = 148
    fwd = [("token ids", [f"{bA} × {seq} whole numbers", f"read from {bA} windows"], "var(--surface-2)", "var(--line-2)"),
           ("embedding", [f"each id → {mc['d']} numbers", f"grid {bA}×{seq}×{mc['d']}"], "var(--accent-soft)", "var(--accent)"),
           (f"{mc['n_layer']} blocks", ["attention + MLP each", "stacked one of 3 ways"], "var(--accent-soft)", "var(--accent)"),
           ("scores", [f"vs all {mc['vocab']:,} tokens", f"{n_slices(bA)} slices of {tc['loss_chunk']:,} positions"], "var(--warn-soft)", "var(--warn-line)"),
           ("loss", ["average surprise", f"one number, e.g. {A['final_val']:.2f}"], "var(--good-soft)", "var(--good-line)")]
    for x, (t, lines, f, s) in zip(Xs, fwd):
        g.append(_card(x, 34, bw5, 78, t, None, lines, f, s))
    for i in range(4):
        g.append(_arrow(Xs[i] + bw5 + 1, 73, Xs[i + 1] - 2, 73, "lo5"))
    g.append(_t(415, 170, "backward pass and update", 12, 600, "var(--ink-2)", anchor="middle"))
    Xb = [10, 213, 416, 620]
    bwb = 190
    back = [("next step", ["clear the gradients", f"read the next {bA} windows"], "var(--surface-2)", "var(--line-2)"),
            ("AdamW update", ["learning rate from the schedule", f"weight decay {tc['weight_decay']} on matrices"], "var(--good-soft)", "var(--good-line)"),
            ("unscale + clip", ["undo the fp16 loss scaling", f"cap gradient size at {tc['grad_clip']}"], "var(--surface-2)", "var(--line-2)"),
            ("backward", [f"gradients for {cfg['n_params'] / 1e6:.1f}M weights", "reversible: rebuild each layer"], "var(--bad-soft)", "var(--bad-line)")]
    for x, (t, lines, f, s) in zip(Xb, back):
        g.append(_card(x, 180, bwb, 78, t, None, lines, f, s))
    for i in (3, 2, 1):
        g.append(_arrow(Xb[i] - 1, 219, Xb[i - 1] + bwb + 2, 219, "lo5"))
    g.append(_arrow(Xs[4] + bw5 / 2, 114, Xb[3] + bwb / 2, 177, "lo5"))
    g.append(_arrow(Xb[0] + 60, 178, Xs[0] + 60, 115, "lo5"))
    g.append(_t(Xb[0] + 72, 150, f"repeat, {stepsA:,} times", 11.5, 600, "var(--ink-2)"))
    rows = [[f"A–C (batch {bA})", f"{bA * seq:,}", f"{score_full(bA) / MiB:,.0f} MiB", f"{n_slices(bA)}", f"{score_slice / MiB:,.0f} MiB"],
            [f"D (batch {bD})", f"{bD * seq:,}", f"{score_full(bD) / MiB:,.0f} MiB", f"{n_slices(bD)}", f"{score_slice / MiB:,.0f} MiB"]]
    chapter("5 · one step", "loop-step", "Inside one training step",
            f"<p><strong>Forward:</strong> each id becomes {mc['d']} numbers (its embedding, plus one for its position). "
            f"That grid flows through the {mc['n_layer']} blocks. At the end every position scores all "
            f"{mc['vocab']:,} possible next tokens, and the loss measures how surprised the model was by the real next "
            f"token, averaged over all {bA * seq:,} positions. <strong>Backward:</strong> PyTorch works out how each of "
            f"the {cfg['n_params'] / 1e6:.1f}M weights should change to lower that loss. This is the only place the "
            f"reversible runs differ: they rebuild each layer's input instead of reading it from memory. "
            f"<strong>Update:</strong> AdamW moves every weight a small amount, and the next batch comes in.</p>"
            + _svg(820, 270, "One training step: forward pass, backward pass, and the weight update", "".join(g),
                   "The matrix work runs in 16-bit (fp16) for speed. The loss is multiplied by a large factor before the "
                   "backward pass so small gradients do not round to zero, and divided back before the update.")
            + "<p><strong>Why the scores are computed in slices.</strong> A table of scores for every position against "
              "every token would be large, and larger still for run D. The notebook computes it one slice at a time "
              "and recomputes each slice in the backward pass, so only one slice exists at any moment. Every run does "
              "this, so it does not tilt the comparison.</p>"
            + _table(["runs", "positions per step", "full score table (fp32)", "slices", "what exists at once"], rows))

    # ---- 6 · one run
    x0, x1 = 70, 790
    sx = lambda s: x0 + (x1 - x0) * s / stepsA
    lr0 = tc["lr"]
    ly0, ly1 = 36, 118
    lyv = lambda v: ly1 - (ly1 - ly0) * v / lr0
    pts = " ".join(f"{sx(s):.1f},{lyv(lr_at(s, stepsA, lr0, tc['warmup_frac'], tc['min_lr_frac'])):.1f}"
                   for s in list(range(0, stepsA, 6)) + [stepsA - 1])
    vals = runlog["A"]["val"]
    vmin, vmax = math.floor(min(p["loss"] for p in vals) * 2) / 2, math.ceil(max(p["loss"] for p in vals) * 2) / 2
    vy0, vy1 = 170, 252
    vy = lambda v: vy1 - (vy1 - vy0) * (v - vmin) / (vmax - vmin)
    lrend = lr0 * tc["min_lr_frac"]
    vpts = " ".join("%.1f,%.1f" % (sx(p["step"]), vy(p["loss"])) for p in vals)
    g = [_t(x0, 22, "learning rate", 12, 600, "var(--ink-2)"),
         f'<line x1="{x0}" y1="{ly1}" x2="{x1}" y2="{ly1}" stroke="var(--line-2)"/>',
         _t(x0 - 6, lyv(lr0) + 4, f"{lr0:.0e}", 10.5, fill="var(--ink-3)", anchor="end", mono=True),
         _t(x0 - 6, lyv(lrend) + 4, f"{lrend:.0e}", 10.5, fill="var(--ink-3)", anchor="end", mono=True),
         f'<line x1="{x0}" y1="{lyv(lrend):.1f}" x2="{x1}" y2="{lyv(lrend):.1f}" stroke="var(--line)" stroke-dasharray="3 3"/>',
         f'<polyline points="{pts}" fill="none" stroke="var(--accent)" stroke-width="2"/>',
         f'<line x1="{sx(warmA):.1f}" y1="{ly0 - 4}" x2="{sx(warmA):.1f}" y2="{ly1}" stroke="var(--ink-3)" stroke-dasharray="3 3"/>',
         _t(sx(warmA) + 6, ly0 + 6, f"warm-up ends at step {warmA}", 11, fill="var(--ink-2)"),
         _t(x1, lyv(lrend) - 8, f"cosine glide down to {tc['min_lr_frac']:.0%} of the peak", 11, fill="var(--ink-2)", anchor="end"),
         _t(x0, 156, f"validation loss, run A, measured every {evA} steps", 12, 600, "var(--ink-2)"),
         f'<line x1="{x0}" y1="{vy1}" x2="{x1}" y2="{vy1}" stroke="var(--line-2)"/>',
         _t(x0 - 6, vy(vmax) + 4, f"{vmax:.1f}", 10.5, fill="var(--ink-3)", anchor="end", mono=True),
         _t(x0 - 6, vy(vmin) + 4, f"{vmin:.1f}", 10.5, fill="var(--ink-3)", anchor="end", mono=True),
         f'<polyline points="{vpts}" fill="none" stroke="var(--good)" stroke-width="1.6"/>']
    for p in vals:
        g.append(f'<circle cx="{sx(p["step"]):.1f}" cy="{vy(p["loss"]):.1f}" r="3.2" fill="var(--good)"/>')
    g += [_t(sx(vals[0]["step"]) + 8, vy(vals[0]["loss"]) + 4, f"{vals[0]['loss']:.2f} at step {vals[0]['step']}", 11, fill="var(--ink-2)"),
          _t(sx(vals[-1]["step"]) - 2, vy(vals[-1]["loss"]) - 10, f"{vals[-1]['loss']:.2f} at step {vals[-1]['step']:,}", 11,
             fill="var(--ink-2)", anchor="end")]
    for s in range(0, stepsA + 1, 500):
        g += [f'<line x1="{sx(s):.1f}" y1="{vy1}" x2="{sx(s):.1f}" y2="{vy1 + 5}" stroke="var(--ink-3)"/>',
              _t(sx(s), vy1 + 18, f"{s:,}", 10.5, fill="var(--ink-3)", anchor="middle", mono=True)]
    g.append(_t((x0 + x1) / 2, vy1 + 36, "training step, run A", 11.5, fill="var(--ink-2)", anchor="middle"))
    items = [
        f"<strong>Every step</strong> sets the learning rate from the schedule above: it climbs for the first "
        f"{tc['warmup_frac']:.1%} of steps ({warmA} for runs A–C, {warmD} for run D), then glides down along a cosine "
        f"to {tc['min_lr_frac']:.0%} of its peak. The peak is {lr0:g} for runs A–C and {D_lr(R):.2e} for run D.",
        f"<strong>Every {tc['log_every']} steps</strong> the average training loss is written to the log.",
        f"<strong>Every {tc['eval_every_tokens'] / 1e6:g}M tokens</strong> ({evA} steps at batch {bA}, {evD} at batch {bD}) the "
        f"model is scored on the same {val_tok:,} validation tokens (the first {tc['eval_batches'] * bA} validation windows, in a "
        f"fixed order). That gives {len(vals)} points per run, the dots above, and the last one is the \"final val loss\".",
        f"<strong>The speed clock</strong> starts at step 10, so start-up is not counted, and pauses during every "
        f"validation pass. Tokens per second = tokens trained while the clock ran ÷ seconds on the clock.",
        f"<strong>Every {tc['ckpt_minutes']:g} minutes</strong> weights, optimizer state and the log are saved to Drive. "
        f"If Colab disconnects, the run resumes from there. When the run finishes, its log is saved and the run is "
        f"never trained again: a re-run replays the log instead.",
    ]
    chapter("6 · one run", "loop-run", "One run from first step to last",
            _svg(820, 300, "Learning-rate schedule and validation loss over run A", "".join(g),
                 f"The schedule is the notebook's own formula, evaluated for {stepsA:,} steps; the dots are run A's "
                 f"logged validation losses.")
            + "<ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>")

    # ---- 7 · before the long runs
    sv = R["sweep"]["val"]
    best = R["sweep"]["best"]
    side = R["sweep"]["side_check"]
    rows = []
    for stack, name, start in (("midpoint", "midpoint", "Euler"), ("twostream", "two-stream", "plain")):
        for h, v in sv[stack].items():
            pick = " ← chosen" if float(h) == float(best[stack]) else ""
            rows.append([name, h, start, f"{v:.4f}{pick}"])
    rows.append(["midpoint", str(best["midpoint"]), "plain (side check)",
                 f"{side['none']:.4f} ← used for run B" if side["run_b_start"] == "none" else f"{side['none']:.4f}"])
    lb = R["probe"]["largest_batch"]
    cap = cfg["probe"]["batch_cap"]
    g = [_t(10, 20, "each chip is one try of two training steps at that batch size", 12, 600, "var(--ink-2)"),
         _box(470, 8, 14, 14, "var(--good-soft)", "var(--good-line)", rx=3), _t(490, 20, "fits", 11.5, fill="var(--ink-2)"),
         _box(530, 8, 14, 14, "var(--bad-soft)", "var(--bad-line)", rx=3), _t(550, 20, "out of memory", 11.5, fill="var(--ink-2)"),
         _box(650, 8, 14, 14, "var(--good-soft)", "var(--accent)", rx=3), _t(670, 20, "the answer", 11.5, fill="var(--ink-2)")]
    for r, name in enumerate(("ordinary", "midpoint", "two-stream")):
        y = 40 + r * 52
        g += [_t(10, y + 18, name, 12.5, 600), _t(10, y + 33, f"largest: {lb[name]}", 11, fill="var(--ink-3)")]
        for k, (b, ok) in enumerate(probe_tries(lb[name], cap)):
            x = 150 + k * 46
            ans = ok and b == lb[name]
            g.append(f'<rect x="{x}" y="{y}" width="42" height="30" rx="4" fill="var(--{"good" if ok else "bad"}-soft)" '
                     f'stroke="{"var(--accent)" if ans else ("var(--good-line)" if ok else "var(--bad-line)")}" '
                     f'stroke-width="{2.2 if ans else 1}"/>')
            g.append(_t(x + 21, y + 20, str(b), 11, 600 if ans else 400, "var(--good)" if ok else "var(--bad)", "middle", mono=True))
    chapter("7 · before the long runs", "loop-prep", "Two cheap stages that set up the long runs",
            f"<p><strong>The step-size sweep.</strong> The reversible rules have a step size <em>h</em>, and it has to be "
            f"picked. {len(sweep_logs)} short runs of {cfg['sweep']['steps']} steps each "
            f"({cfg['sweep']['steps'] * tpsA / 1e6:.1f}M tokens, about {sweep_min / len(sweep_logs):.0f} minutes each) try "
            f"the candidates. The lowest validation loss wins. One extra run checks whether midpoint trains better from "
            f"the plain start than from Lightning LM's Euler start.</p>"
            + _table(["stack", "h", "start", "val loss after 300 steps"], rows)
            + f"<p><strong>The memory probe.</strong> It trains nothing. It builds each model and tries two training "
              f"steps on random tokens at a batch size, starting at 8 and doubling until the GPU runs out of memory. Then "
              f"it halves the gap between the last fit and the first failure until the gap is 8. Two steps, not one, "
              f"because Adam creates its running averages at the end of the first step, so only the second step shows "
              f"the real peak. The same probe then measures one batch-{cfg['probe']['depth_batch']} step at "
              f"{', '.join(str(d) for d in cfg['probe']['depths'])} layers, which gives the memory-per-layer numbers.</p>"
            + _svg(820, 196, "The memory probe's search for the largest batch, for each model", "".join(g),
                   "The tries are replayed from the notebook's search rule and the largest batch it reported: every size up "
                   "to the answer fits and every size above it fails, so the sequence is fixed. Run D then used "
                   f"{bD}, not the largest {lb[R['run_d']['variant']]}: at the largest it would have had only about "
                   f"{tc['tokens'] // (lb[R['run_d']['variant']] * seq)} steps."))

    return ('<section class="learn loop" id="the-full-loop"><h2 class="secttl">The full loop — from web pages '
            'to the results table</h2>' + "".join(ch) + "</section>")


def D_lr(R):
    return R["run_d"]["lr"]

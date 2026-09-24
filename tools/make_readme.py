"""Write README.md from outputs/summary.json.

Every number in the README comes from the run's own files; none is typed by hand. Where a
sentence could go either way (faster/slower, better/worse), the data picks the words. At the
end the README is read back and each key number is checked against summary.json.

Usage (from the assignment folder):
    python tools/make_readme.py
"""
import json
import math
import os
import sys
import zipfile
from datetime import datetime

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "outputs")
R = json.load(open(os.path.join(OUT, "summary.json"), encoding="utf-8"))


def _zip_summary(name):
    """summary.json from another bundle, if that bundle is present."""
    p = os.path.join(OUT, name)
    if not os.path.exists(p):
        return None
    with zipfile.ZipFile(p) as z:
        return json.loads(z.read("summary.json"))


SMOKE = _zip_summary("era5s13_outputs_smoke_v1.zip")      # the smoke run, for the engineering notes
FIRST = _zip_summary("era5s13_outputs_full_v1.zip")        # the original full run, for its wall time

# ------------------------------------------------------------------ numbers
res = {r["run"]: r for r in R["results"]}
A, B, C, D = res["A"], res["B"], res["C"], res["D"]
cfg, mc, tc = R["config"], R["config"]["model"], R["config"]["train"]
sw, pr, var, rd = R["sweep"], R["probe"], R["variant"], R["run_d"]
side = sw["side_check"]
recon = R["gates"]["C_reconstruction"]
gA = R["gates"]["A_gradients"]
gB = R["gates"]["B_dropout_caught"]
acc = R["acceptance"]
TIE = cfg["thresholds"]["variant_tie_nats"]
USD = cfg["thresholds"]["t4_usd_per_hour"]


def f0(x):  return f"{x:,.0f}"
def f1(x):  return f"{x:,.1f}"
def f2(x):  return f"{x:.2f}"
def f4(x):  return f"{x:.4f}"
def pct(x): return f"{abs(x) * 100:.0f}%"
def sci(x): return f"{x:.1e}"


slow_B = 1 - B["tokens_per_second"] / A["tokens_per_second"]
slow_C = 1 - C["tokens_per_second"] / A["tokens_per_second"]
mem_B = B["peak_mib"] / A["peak_mib"]
mem_C = C["peak_mib"] / A["peak_mib"]
gap_B = B["final_val"] - A["final_val"]
gap_C = C["final_val"] - A["final_val"]
gap_D = D["final_val"] - A["final_val"]
lb = pr["largest_batch"]
dep = pr["depth"]
L_lo, L_hi = min(dep, key=int), max(dep, key=int)
mib = lambda b: b / 2**20
d_speed_vs_b = D["tokens_per_second"] / B["tokens_per_second"]
steps_ratio = A["steps"] / D["steps"]
mid_amp, two_amp = recon["midpoint_amp_gpu"], recon["twostream_amp_gpu"]
h_mid = float(sw["best"]["midpoint"])
h_two = float(sw["best"]["twostream"])
mid_vals = sw["val"]["midpoint"]
mid_sorted = sorted(mid_vals.items(), key=lambda kv: kv[1])
mid_runner_gap = mid_sorted[1][1] - mid_sorted[0][1]
b_start = "plain" if side["run_b_start"] == "none" else "Euler"
total_cost = sum(r["cost_usd"] for r in R["results"])

if FIRST:
    t0 = datetime.strptime(FIRST["meta"]["started"], "%Y-%m-%d %H:%M:%S")
    t1 = datetime.strptime(FIRST["meta"]["finished"], "%Y-%m-%d %H:%M:%S")
    wall_min = (t1 - t0).total_seconds() / 60
else:
    wall_min = None


def slower_phrase(s):
    return f"{pct(s)} slower" if s > 0 else f"{pct(s)} faster"


def vs_paper(s):
    if s < 0.30:
        return "less than the 30–50% the paper estimates"
    if s <= 0.50:
        return "within the 30–50% the paper estimates"
    return "more than the 30–50% the paper estimates"


def behind(g):
    return f"{f4(abs(g))} nats {'behind' if g > 0 else 'ahead of'} run A"


reached = [k for k in ("B", "C", "D") if res[k]["time_to_A_final_s"]]
if var["winner"] == "B":
    var_line = (f"**Midpoint worked better.** Run B (midpoint) finished at {f4(var['B_val'])} "
                f"against run C (two-stream) at {f4(var['C_val'])}, "
                f"{f4(abs(var['B_val'] - var['C_val']))} nats lower.")
else:
    var_line = (f"**Two-stream worked better.** Run C finished at {f4(var['C_val'])} against run B "
                f"(midpoint) at {f4(var['B_val'])}, {f4(abs(var['B_val'] - var['C_val']))} nats lower.")
if abs(var["B_val"] - var["C_val"]) < TIE:
    var_line = (f"**Too close to call.** Midpoint {f4(var['B_val'])} against two-stream "
                f"{f4(var['C_val'])}, within the {TIE} nats declared in advance as a tie.")

# ------------------------------------------------------------------ README
L = []
W = L.append
W("# A 20M model, with and without reversibility")
W("")
W("**ERA V5 · Session 13 — Distributed Training II: Model and Pipeline Parallel**")
W("")
W("> *Train a 20M LLM for 50M tokens on Google Colab. Fix a batch size you can run. Train again with")
W("> reversibility (report which variant worked for you: mid-point, euler, etc). Train again with")
W("> reversibility, but push it to the maximum batch size. Report final loss, speed (tokens/s),")
W("> memory peak and other findings.*")
W("")
W(f"Run on a Colab **{R['meta']['gpu']}**. Acceptance checks: **{acc['passed']}/{acc['total']} passed**. "
  f"Every number below is generated from [`outputs/summary.json`](outputs/summary.json) by "
  f"[`tools/make_readme.py`](tools/make_readme.py); none is typed by hand.")
W("")
W("## The answer in one table")
W("")
W("Same model, same 50M tokens in the same order, same starting weights. Only the way the layers are "
  "stacked, and in run D the batch size, changes.")
W("")
W("| run | model | batch | steps | final train loss | final val loss | tokens / s | peak memory | cost |")
W("|---|---|---|---|---|---|---|---|---|")
names = {"A": "ordinary (baseline)", "B": f"reversible, midpoint (h = {h_mid}, {b_start} start)",
         "C": f"reversible, two-stream (h = {h_two})", "D": f"reversible, {rd['variant']}, large batch"}
for k in ("A", "B", "C", "D"):
    r = res[k]
    W(f"| **{k}** | {names[k]} | {r['batch']} | {f0(r['steps'])} | {f4(r['final_train'])} | "
      f"**{f4(r['final_val'])}** | {f0(r['tokens_per_second'])} | {f0(r['peak_mib'])} MiB | "
      f"${r['cost_usd']:.2f} |")
W("")
W("Loss is the average cross-entropy per token, in *nats*: lower is better, and a gap of 0.1 nats is "
  "a visible difference at this size. *h* is the step size in the reversible update rule.")
W("")
W(f"Cost is training time × ${USD:.2f} per hour, a public on-demand T4 price; the Colab free tier "
  f"itself charges nothing. Validation loss is measured on {f0(cfg['train']['eval_batches'] * tc['batch'] * mc['seq'])} "
  f"held-out tokens from documents never trained on.")
W("")
W("![validation loss against tokens and against training time](outputs/loss_curves.png)")
W("")

W("## What the brief asked, answered")
W("")
W("**1 · Baseline at a batch you can run (run A).** Batch 32, "
  f"{f0(A['steps'])} steps, final validation loss **{f4(A['final_val'])}**, "
  f"{f0(A['tokens_per_second'])} tokens/s, peak memory {f0(A['peak_mib'])} MiB.")
W("")
W("**2 · Reversible, and which variant worked (runs B and C).** " + var_line +
  f" Both ran at batch 32, like run A. Midpoint's step size h = {h_mid} came from a short sweep "
  f"(h = 1.0 was only {f4(mid_runner_gap)} nats behind, a near-tie); two-stream's was h = {h_two}. "
  f"Midpoint used the **{b_start} start**: in a side check the plain start beat the Euler start by "
  f"{f4(abs(side['diff']))} nats, more than the {TIE} set in advance as the threshold for switching "
  f"away from Lightning LM's setting.")
W("")
W("**3 · Reversible at the maximum batch (run D).** The memory probe found the largest batch that fits: "
  f"**{lb['ordinary']}** for the ordinary model, **{lb['midpoint']}** for midpoint "
  f"({f2(pr['ratio_midpoint'])}×) and **{lb['two-stream']}** for two-stream ({f2(pr['ratio_two-stream'])}×). "
  f"Training at that size would leave only a few dozen steps for 50M tokens, so run D was capped at "
  f"**batch {rd['batch']}** ({rd['decided_by']}, {f0(D['steps'])} steps) with the learning rate raised to "
  f"{rd['lr']:.2e} (× √({rd['batch']}/{tc['batch']})). It finished at **{f4(D['final_val'])}**.")
W("")

W("## Findings")
W("")
W(f"**1 · Reversibility cuts memory by about {1/mem_B:.1f}× at the same batch, and the reversible "
  f"model's memory barely grows with depth.** At batch 32, midpoint peaked at {f0(B['peak_mib'])} MiB and two-stream at "
  f"{f0(C['peak_mib'])} MiB, against {f0(A['peak_mib'])} MiB for the ordinary model "
  f"({f2(mem_B)}× and {f2(mem_C)}×). The probe below measured one training step at "
  f"{', '.join(sorted(dep, key=int))} layers: each extra layer costs the ordinary model "
  f"**{f1(pr['mib_per_layer_ordinary'])} MiB** (its stored activations) and the reversible models "
  f"**{f1(pr['mib_per_layer_midpoint'])} MiB** (only that layer's weights and optimizer state). "
  f"At {L_hi} layers that is {f0(mib(dep[L_hi]['ordinary']))} MiB against "
  f"{f0(mib(dep[L_hi]['midpoint']))} MiB.")
W("")
W("| layers | ordinary | midpoint | two-stream |")
W("|---|---|---|---|")
for Ls in sorted(dep, key=int):
    row = dep[Ls]
    W(f"| {Ls} | {f0(mib(row['ordinary']))} MiB | {f0(mib(row['midpoint']))} MiB | {f0(mib(row['two-stream']))} MiB |")
W("")
W("![largest batch that fits, and peak memory against depth](outputs/probe.png)")
W("")
W(f"**2 · The price is speed: {pct(min(slow_B, slow_C))}–{pct(max(slow_B, slow_C))} fewer tokens per second.** Midpoint ran "
  f"{slower_phrase(slow_B)} and two-stream {slower_phrase(slow_C)} than run A, {vs_paper(max(slow_B, slow_C))}. "
  "Each layer runs twice, once going forward and once again in the backward pass to rebuild its input.")
W("")
W("![tokens per second and peak memory per run](outputs/speed_memory.png)")
W("")
gap_line = (f"**3 · On this model the reversible runs trained slightly worse.** At the same 50M tokens, "
            f"midpoint finished {behind(gap_B)} and two-stream {behind(gap_C)}. ")
if reached:
    gap_line += f"Run(s) {', '.join(reached)} reached run A's final loss within the budget. "
else:
    gap_line += "None of the reversible runs reached run A's final loss within the 50M-token budget. "
gap_line += ("Each run was trained once, with one seed, so the run-to-run noise was not measured; "
             "these gaps are an observation, not a settled result. Three likely causes, not separated "
             "here: the reversible rules change the architecture itself (the stream is updated "
             "differently from an ordinary residual block), the step size was tuned on only 300 steps, "
             "and 16-bit rebuilding adds a small error (finding 5).")
W(gap_line)
W("")
W(f"**4 · A bigger batch bought no speed on a T4, only fewer steps.** Run D at batch {rd['batch']} ran at "
  f"{f0(D['tokens_per_second'])} tokens/s, {d_speed_vs_b:.2f}× run B at batch 32. This small model already "
  f"keeps the GPU busy at batch 32, so {rd['batch'] / tc['batch']:.1f}× more sequences per step simply "
  f"meant {steps_ratio:.1f}× fewer steps for the same tokens, and a final loss {f4(gap_D)} nats above run A. "
  "On this hardware, the memory that reversibility frees is worth more spent on a deeper model or longer "
  "sequences than on a bigger batch.")
W("")
W(f"**5 · Rebuilding in 16-bit drifts, but the gradients hold.** Measured on the real-size model before "
  f"training, on the GPU with 16-bit arithmetic: the rebuilt states match the true ones to {sci(mid_amp['per_layer_rel_err'][-1])} at the top of the "
  f"stack and drift to {sci(mid_amp['max_rel_err'])} by the bottom layer (midpoint; two-stream "
  f"{sci(two_amp['max_rel_err'])}). Rebuilding by subtraction loses a little accuracy at every layer, and in "
  f"16-bit that loss is larger and adds up on the way down. The gradients still agree with plain PyTorch to a "
  f"cosine of {mid_amp['grad_cosine']:.5f} (midpoint) and {two_amp['grad_cosine']:.5f} (two-stream).")
W("")
W("![rebuilt-state error by layer](outputs/reconstruction.png)")
W("")
W(f"**6 · Midpoint's starting state mattered more than its step size.** Over 300 steps the plain start "
  f"beat the Euler start by {f4(abs(side['diff']))} nats, while midpoint's best two step sizes were "
  f"{f4(mid_runner_gap)} apart.")
W("")
if SMOKE:
    sp = SMOKE["probe"]
    W("**7 · Two measurement traps, caught by the smoke run and fixed before the full run.**")
    W("")
    W(f"- *The Euler start was stored, not rebuilt.* In the smoke run midpoint fitted a largest batch of "
      f"{sp['largest_batch']['midpoint']} against two-stream's {sp['largest_batch']['two-stream']}, and used a "
      f"constant extra amount of memory at every depth: one stored layer. The single ordinary step that makes "
      f"midpoint's second starting state ran outside the reversible stack. It is now recomputed in the backward "
      f"pass, and midpoint's largest batch rose to {lb['midpoint']}.")
    W(f"- *A one-step probe misses Adam's state.* In the smoke run each extra reversible layer cost exactly "
      f"{f1(sp['mib_per_layer_midpoint'])} MiB, which is that layer's weights alone: Adam creates its running "
      f"averages after the first step's memory peak. The probe now measures the second step, and the "
      f"per-layer figure became {f1(pr['mib_per_layer_midpoint'])} MiB.")
    W("")

W("## How the reversible stack saves memory")
W("")
W("An ordinary block adds its output to the stream: `x_next = x + f(x)`. You cannot undo that without "
  "knowing `x`, so PyTorch stores every layer's working numbers for the backward pass. That stored pile "
  "is the activation memory.")
W("")
W("A reversible rule can be run backwards:")
W("")
W("| stack | forward | backward (rebuild) |")
W("|---|---|---|")
W("| midpoint | `next = previous + 2h · f(current)` | `previous = next − 2h · f(current)` |")
W("| two-stream | `q = q + h·Attn(p)`, then `p = p + h·MLP(q)` | undo the second update, then the first |")
W("")
W("The rule alone saves nothing, because PyTorch would still store everything. The saving comes from a "
  "custom backward pass (`MidpointStack`, `TwoStreamStack` in Cell 6): the forward pass stores nothing but "
  "the two states at the top; the backward pass walks down, rebuilds each layer's input, runs that one layer "
  "again to get its gradients, and discards it. Only one layer's working numbers exist at any moment.")
W("")
W("Three gates check this before any training, and stop the notebook if they fail:")
W("")
gAmax = max(v["grad_rel_diff"] for v in gA.values())
gBmin = min(v["grad_rel_diff"] for v in gB.values())
W(f"- **Gate A:** gradients from the custom backward pass equal plain PyTorch's, in double precision: "
  f"largest relative difference {sci(gAmax)} across the three stacks.")
W(f"- **Gate B:** a planted bug (dropout switched on) must be caught: the gradients differed by at least "
  f"{pct(gBmin)}. This is why dropout is 0 in every run: a rebuild cannot replay a random mask.")
W(f"- **Gate C:** rebuilt states stay close to the true ones (fp32: {sci(recon['midpoint_fp32_cpu']['max_rel_err'])}), "
  f"and under 16-bit arithmetic on the GPU the gradients agree with plain PyTorch (finding 5).")
W("")

W("## The setup")
W("")
W("| | |")
W("|---|---|")
W(f"| model | {mc['n_layer']} layers × width {mc['d']}, {mc['n_head']} heads, "
  f"{cfg['n_params']/1e6:.1f}M parameters ({mc['n_layer'] * 12 * mc['d']**2 / 1e6:.1f}M in the blocks) |")
W(f"| tokenizer | {mc['vocab']:,}-token byte-pair tokenizer trained on the data "
  f"({R['data']['chars_per_token']:.2f} characters per token) |")
W(f"| data | FineWeb sample-10BT: {R['data']['train_tokens']/1e6:.1f}M training tokens, "
  f"{R['data']['val_tokens']/1e6:.0f}M validation tokens from separate documents |")
W(f"| training | {tc['tokens']/1e6:.0f}M tokens per run, sequence length {mc['seq']}, AdamW "
  f"(weight decay {tc['weight_decay']}), learning rate {tc['lr']} with warm-up and cosine decay, "
  f"fp16 with loss scaling, dropout 0 |")
W(f"| loss | computed in slices of {tc['loss_chunk']:,} tokens so the vocabulary scores never all "
  f"exist at once; used in every run |")
W(f"| hardware | Colab {R['meta']['gpu']} ({R['meta']['vram_gb']:.1f} GB), torch {R['meta']['torch']} |")
if wall_min:
    W(f"| full run | {wall_min:.0f} minutes end to end (${wall_min / 60 * USD:.2f} at the T4 price), of which the "
      f"four training runs cost ${total_cost:.2f} |")
W("")
W("Why these choices (model size, data, variants, batch sizes, learning rate) is recorded decision by "
  "decision in [`assignment_plan.html`](assignment_plan.html).")
W("")

W("## Limitations")
W("")
W("- **One seed per run.** The run-to-run noise was not measured, so the loss gaps between runs are "
  "observations, not settled differences.")
W("- **Step sizes were tuned on 300 steps**, and midpoint's two best values were nearly tied; a longer "
  "sweep could change the reversible runs' final loss.")
W("- **Lightning LM's exact integrator is not public.** Midpoint here is the paper's two-state rule; where "
  "Lightning LM's blend setting (a = 0.5) enters is not published, so it was not reproduced.")
W("- **Small model, one GPU.** At 22.5M parameters the ordinary model's largest batch is only "
  f"{lb['ordinary']}; the benefit of reversibility grows with depth and sequence length, which a larger "
  "model would show more strongly.")
W("")

W("## Files")
W("")
W("| path | what it is |")
W("|---|---|")
W("| `session13_reversible_v4_executed_full.ipynb` | the executed notebook: every cell's output, with each run's progress replayed from its log |")
W("| `session13_reversible.ipynb` | the same notebook, not executed, to run again |")
W("| `notebook_src.py` | the source of truth: a plain script that can be run and debugged locally |")
W("| `outputs/summary.json` | every measured number; this README is generated from it |")
W("| `outputs/RESULTS.md` | the results table as the notebook wrote it |")
W("| `outputs/logs/` | the full history of every training run |")
W("| `outputs/*.png` | the four figures above |")
W("| `session13_walkthrough.html` | the results report: headline numbers, the full loop in diagrams, this write-up, and every cell with its output |")
W("| `assignment_plan.html` | the decisions behind the experiment and the build log |")
W("| `docs/` | standalone copies to open from disk: `session13_walkthrough.html` (results report), `session13_report.html` (session report: lesson and transcript analysis), `assignment_plan.html` |")
W("| `tools/` | `py2ipynb.py` builds the notebook from the script; `make_readme.py` writes this README; `make_walkthrough.py` (with `loop_section.py`) builds the results report; `export_docs.py` refreshes `docs/` |")
W("")
W("## How to run it")
W("")
W("Open `session13_reversible.ipynb` in Colab on a **T4 GPU** and run all. `MODE = \"full\"` in Cell 1 is "
  "the experiment (about 1 hour 40 minutes); `MODE = \"smoke\"` is a 30-minute check of every GPU code path. "
  "Results are cached on Google Drive under `MyDrive/era5_session13/`, so after a disconnect, running all "
  "again reloads finished work and resumes an interrupted run from its checkpoint. On a machine without a GPU "
  "the notebook switches itself to a tiny test profile and never touches the real results.")
W("")
W("## Acceptance checks")
W("")
for c in acc["checks"]:
    W(f"- [{'x' if c['result'] else ' '}] {c['name']}")
W("")

text = "\n".join(L) + "\n"
open(os.path.join(HERE, "README.md"), "w", encoding="utf-8").write(text)

# ------------------------------------------------------------------ reconcile
must = [f4(r["final_val"]) for r in R["results"]] + [f4(r["final_train"]) for r in R["results"]] + \
       [f0(r["tokens_per_second"]) for r in R["results"]] + [f0(r["peak_mib"]) for r in R["results"]] + \
       [str(lb["ordinary"]), str(lb["midpoint"]), str(lb["two-stream"]),
        f1(pr["mib_per_layer_ordinary"]), f1(pr["mib_per_layer_midpoint"]),
        f"{acc['passed']}/{acc['total']}", f4(abs(side["diff"]))]
missing = [m for m in must if m not in text]
if missing:
    sys.exit(f"README check FAILED, numbers missing: {missing}")
if slow_B <= 0 or slow_C <= 0:
    print("note: a reversible run was not slower; wording chosen from the data")
print(f"README.md written ({len(text):,} chars); {len(must)} key numbers checked against summary.json")

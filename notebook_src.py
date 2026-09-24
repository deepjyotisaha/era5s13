# %% [markdown]
# # Session 13 — A 20M model, with and without reversibility
#
# **The brief.** Train a 20M LLM for 50M tokens with a batch size you can run. Train it again
# with reversibility, and report which variant worked. Train again with reversibility at the
# largest batch that fits. Report final loss, speed (tokens/s), peak memory and other findings.
#
# **The decisions behind this notebook** are recorded, with their reasons, in
# `assignment_plan.html`. In one line each:
#
# | decision | choice |
# |---|---|
# | model | 16 layers × width 320, 8K-token tokenizer trained on the data (22.3M parameters) |
# | data | FineWeb sample-10BT, 50M training tokens, identical order for every run |
# | variants | run B midpoint, run C two-stream (symplectic Euler); Euler-start side check |
# | batches | 32 for runs A–C; run D capped so it keeps about 400 steps; true maximum probed separately |
# | learning rate | scaled by the square root of the batch increase for run D |
# | extras | memory-vs-depth probe, cost per run |
# | fixed | Colab T4, fp16, sequence length 512, dropout 0 everywhere |
#
# ### How it is built
#
# ```
# CELL 1     environment, profile (full on a GPU, reduced on a CPU)
# CELL 2     configuration and thresholds, declared before anything is measured
# CELL 3     storage on Drive, the stage cache, run cache and checkpoints
# CELL 4     data: stream FineWeb, train the tokenizer, tokenise once
# CELL 5     the model: ordinary, midpoint and two-stream stacks
# CELL 6     the reversible stacks with their own backward pass
# CELL 7     gradients match plain autograd, in fp64 ───────── ⛔ GATE A
# CELL 8     dropout breaks the rebuild (planted bug) ──────── ⛔ GATE B
# CELL 9     how far rebuilt states drift ──────────────────── ⛔ GATE C
# CELL 10    the training loop
# CELL 11    step-size sweep and the Euler-start side check
# CELL 12    memory probe: largest batch, and memory vs depth
# CELL 13    runs A, B, C at batch 32; pick the better variant
# CELL 14    run D at the capped large batch
# CELL 15    results table, time to target, cost
# CELL 16    figures
# CELL 17    acceptance checks, summary.json, RESULTS.md, bundle
# ```
#
# ### Running this
#
# Runtime → Change runtime type → **T4 GPU**, then Runtime → Run all. Everything needed for the
# write-up lands in `outputs/` on your Drive and is zipped by the last cell. If Colab
# disconnects, run all again: finished stages and runs reload from the cache, and an
# interrupted run resumes from its last checkpoint.

# %% [markdown]
# ## Cell 1 — environment
#
# ### What this block does
# Imports, fixes the seed, and decides the **profile**: `full` on a GPU (the real experiment) or
# `cpu`, a tiny version of everything used to test the notebook on a laptop.
#
# ### What you should see
# A version box, the GPU name, and `profile: full`. If it says `cpu`, you are not on a GPU
# runtime and none of the numbers are the real experiment.

# %%
import sys, subprocess

def _pip(mod, pkg=None):
    """Install only if the import actually fails - Colab already has most of this."""
    try:
        __import__(mod)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg or mod], check=False)

_pip("tokenizers")
_pip("datasets")

import os, math, time, json, zipfile, random, platform, hashlib, inspect, io, gc, copy
from contextlib import redirect_stdout, nullcontext
from dataclasses import dataclass, asdict, replace

try:
    sys.stdout.reconfigure(encoding="utf-8")   # Windows consoles default to cp1252
except Exception:
    pass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NOTEBOOK_VERSION = "v1"
SEED = 1337
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

# ┌──────────────────────────────────────────────────────────────────────────────────────┐
# │ MODE: "smoke" first. It uses the real model and the real data but only ~2M tokens per   │
# │ run, so every GPU code path is exercised in about 30 minutes (mostly one-time data      │
# │ preparation, saved to Drive). When it ends with all                                     │
# │ checks passing, change this to "full" and run all again. The data is reused.            │
# └──────────────────────────────────────────────────────────────────────────────────────┘
MODE = "full"     # the smoke run passed on 23 Sep; "full" is the experiment

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROFILE = os.environ.get("ERA5_PROFILE") or (MODE if DEVICE == "cuda" else "cpu")
AMP = DEVICE == "cuda"                  # fp16 autocast + loss scaling (a T4 has no bf16)
AMP_DTYPE = torch.float16

if DEVICE == "cuda":
    _p = torch.cuda.get_device_properties(0)
    GPU_NAME, VRAM_GB = _p.name, _p.total_memory / 1e9
else:
    GPU_NAME, VRAM_GB = None, 0.0

_BANNER = [f"NOTEBOOK {NOTEBOOK_VERSION}  -  Session 13, reversible 20M model",
           f"profile: {PROFILE}   device: {DEVICE}" + (f"   gpu: {GPU_NAME}" if GPU_NAME else "")]
if PROFILE == "cpu":
    _BANNER += ["", "!! REDUCED PROFILE: tiny model, tiny data, GPU-only cells skipped.",
                "!! Use it to test the notebook. None of these numbers are the experiment."]
elif PROFILE == "smoke":
    _BANNER += ["", "SMOKE TEST: real model and data, ~2M tokens per run.",
                "Checks every GPU code path. Set MODE = \"full\" afterwards for the experiment."]
_w = max(len(x) for x in _BANNER) + 4
print("+" + "-" * _w + "+")
for _line in _BANNER:
    print("|  " + _line.ljust(_w - 2) + "|")
print("+" + "-" * _w + "+\n")
print(f"python : {platform.python_version()}")
print(f"torch  : {torch.__version__}")
if GPU_NAME:
    print(f"gpu    : {GPU_NAME}, {VRAM_GB:.1f} GB")

R = {"meta": {"notebook_version": NOTEBOOK_VERSION, "torch": torch.__version__,
              "python": platform.python_version(), "seed": SEED, "device": DEVICE,
              "profile": PROFILE, "gpu": GPU_NAME, "vram_gb": VRAM_GB,
              "started": time.strftime("%Y-%m-%d %H:%M:%S")},
     "gates": {}}

# %% [markdown]
# ## Cell 2 — configuration and thresholds
#
# ### What this block does
# Every number that shapes the experiment, in one place, printed. The thresholds that decide a
# pass or a fail are written down **before** anything is measured, so they cannot be tuned to
# fit the result afterwards.
#
# ### Why these values
# The model is option D from the plan: 16 layers of width 320, about 12 × 320² = 1.23M weights
# per layer, plus a 8,192-token embedding shared with the output layer. The learning rate and
# warm-up are ordinary choices for a model this size; they are the same for every run except
# run D, whose learning rate is scaled by the square root of its batch increase.

# %%
@dataclass(frozen=True)
class ModelCfg:
    vocab: int = 8192
    d: int = 320            # width: how many numbers carry one token between layers
    n_layer: int = 16       # depth
    n_head: int = 5         # 320 = 5 heads x 64
    seq: int = 512
    mlp_mult: int = 4       # feed-forward middle = 4 x width = 1,280

@dataclass(frozen=True)
class TrainCfg:
    tokens: int = 50_000_000
    batch: int = 32
    lr: float = 1e-3
    min_lr_frac: float = 0.1
    warmup_frac: float = 0.065      # about 200 steps at batch 32
    weight_decay: float = 0.1
    beta2: float = 0.95
    grad_clip: float = 1.0
    eval_every_tokens: int = 2_500_000
    eval_batches: int = 20          # 20 x 32 sequences of 512 = 327k validation tokens
    log_every: int = 25
    loss_chunk: int = 8192          # tokens per slice of the chunked loss
    ckpt_minutes: float = 8.0
    min_steps_run_d: int = 400

@dataclass(frozen=True)
class DataCfg:
    train_tokens: int = 50_500_000
    val_tokens: int = 1_000_000
    tokenizer_chars: int = 40_000_000
    fineweb_name: str = "sample-10BT"

@dataclass(frozen=True)
class SweepCfg:
    midpoint_h: tuple = (0.25, 0.5, 1.0)
    twostream_h: tuple = (0.5, 1.0)
    steps: int = 300

@dataclass(frozen=True)
class ProbeCfg:
    depths: tuple = (8, 16, 24, 32)
    depth_batch: int = 32
    batch_cap: int = 8192
    safety: float = 0.9             # train at no more than 90% of the largest batch that fits

@dataclass(frozen=True)
class Thresholds:
    grad_rel_fp64: float = 1e-9     # Gate A: reversible vs plain autograd, fp64
    dropout_caught: float = 1e-3    # Gate B: the planted dropout bug must exceed this
    recon_rel_fp32: float = 1e-4    # Gate C: rebuilt states vs true states, fp32
    grad_cos_amp: float = 0.999     # Gate C (GPU): gradient agreement under fp16 autocast
    variant_tie_nats: float = 0.02  # below this, B vs C is reported as too close to call
    t4_usd_per_hour: float = 0.35   # public on-demand T4 price, used only for the cost table

MC, TC, DC, SC, PC, TH = ModelCfg(), TrainCfg(), DataCfg(), SweepCfg(), ProbeCfg(), Thresholds()
if PROFILE == "cpu":
    MC = replace(MC, vocab=1024, d=64, n_layer=4, n_head=4, seq=128)
    TC = replace(TC, tokens=400_000, batch=8, eval_every_tokens=100_000, eval_batches=4,
                 log_every=10, loss_chunk=512, min_steps_run_d=100)
    DC = replace(DC, train_tokens=450_000, val_tokens=40_000, tokenizer_chars=1_000_000)
    SC = replace(SC, midpoint_h=(0.5, 1.0), twostream_h=(1.0,), steps=40)
    PC = replace(PC, depths=(2, 4, 8), depth_batch=4)
elif PROFILE == "smoke":
    TC = replace(TC, tokens=2_000_000, eval_every_tokens=500_000, eval_batches=5,
                 min_steps_run_d=40)
    SC = replace(SC, steps=60)
    PC = replace(PC, depths=(8, 16))
TC = replace(TC, ckpt_minutes=float(os.environ.get("ERA5_CKPT_MIN", TC.ckpt_minutes)))

def n_params(mc=MC):
    per_layer = 12 * mc.d * mc.d + 4 * mc.d          # matrices + LayerNorm scales and shifts
    return mc.n_layer * per_layer + mc.vocab * mc.d + mc.seq * mc.d + 2 * mc.d

print(f"model   : {MC.n_layer} layers x width {MC.d}, {MC.n_head} heads, vocab {MC.vocab:,}, "
      f"sequence {MC.seq}")
print(f"          about {n_params()/1e6:.1f}M parameters "
      f"({MC.n_layer*12*MC.d*MC.d/1e6:.1f}M in the blocks)")
print(f"training: {TC.tokens/1e6:.1f}M tokens, batch {TC.batch} "
      f"-> {TC.tokens // (TC.batch*MC.seq):,} steps, lr {TC.lr}, weight decay {TC.weight_decay}")
print(f"sweep   : midpoint h {SC.midpoint_h}, two-stream h {SC.twostream_h}, {SC.steps} steps each")
print(f"thresholds: {asdict(TH)}")
R["config"] = {"model": asdict(MC), "train": asdict(TC), "data": asdict(DC),
               "sweep": asdict(SC), "probe": asdict(PC), "thresholds": asdict(TH),
               "n_params": n_params()}

# %% [markdown]
# ## Cell 3 — storage and caching
#
# ### What this block does
# Mounts Google Drive and sets up two folders. **outputs/** holds what you download: logs,
# plots, `summary.json`. **cache/** holds machinery: the tokenised data, finished stages,
# finished runs and mid-run checkpoints. Deleting the cache only means recomputing.
#
# ### How it works
# A finished stage or run is saved together with a hash of the configuration **and of the code
# that produced it**. Change a hyperparameter or edit a function, and the old result is
# recomputed instead of silently replayed. A cached stage also replays what it printed, so the
# executed notebook reads the same after a reconnect.

# %%
RESET_CACHE = False        # True forces every stage and every run to recompute

try:
    from google.colab import drive as _colab_drive
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

DRIVE_OK = False
if IN_COLAB:
    try:
        _colab_drive.mount("/content/drive")
        ROOT, DRIVE_OK = "/content/drive/MyDrive/era5_session13", True
    except Exception as e:
        ROOT = "/content/era5_session13"
        print("!" * 74)
        print(f"!! DRIVE MOUNT FAILED ({type(e).__name__}). Falling back to {ROOT}, which")
        print("!! dies with the runtime. A disconnect will lose finished runs.")
        print("!" * 74)
    OUT_DIR, CACHE_DIR = os.path.join(ROOT, "outputs"), os.path.join(ROOT, "cache")
else:
    ROOT, OUT_DIR, CACHE_DIR = ".", "outputs", "checkpoints"
# Keyed by the data settings, not the profile, so the smoke run's tokenised data is reused.
DATA_DIR = os.path.join(CACHE_DIR, "data_" + hashlib.sha1(
    json.dumps({"d": asdict(DC), "v": MC.vocab}, sort_keys=True).encode()).hexdigest()[:8])
if PROFILE != "full":                     # smoke and CPU results never mix with the real ones
    OUT_DIR = os.path.join(OUT_DIR, PROFILE)
LOG_DIR = os.path.join(OUT_DIR, "logs")
for _d in (OUT_DIR, CACHE_DIR, DATA_DIR, LOG_DIR):
    os.makedirs(_d, exist_ok=True)

def out(name):   return os.path.join(OUT_DIR, name)
def cache(name): return os.path.join(CACHE_DIR, name)

def _h(obj):
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:10]

RUN_KEY = _h({"model": asdict(MC), "train": {k: v for k, v in asdict(TC).items() if k != "ckpt_minutes"}, "data": asdict(DC), "seed": SEED,
              "device": DEVICE, "profile": PROFILE, "v": NOTEBOOK_VERSION})
STAGE_KEY = _h({"run": RUN_KEY, "sweep": asdict(SC), "probe": asdict(PC), "th": asdict(TH)})

def _src_hash(*objs):
    """Hash of the SOURCE of the functions and classes a cached result depends on."""
    parts = []
    for o in objs:
        try:
            parts.append(inspect.getsource(o))
        except (OSError, TypeError):
            parts.append("")
    return hashlib.sha1("".join(parts).encode()).hexdigest()[:10]

def _atomic_save(obj, path, as_json=False):
    tmp = path + ".tmp"
    if as_json:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=1, default=float)
    else:
        torch.save(obj, tmp)
    os.replace(tmp, path)

class _Tee(io.StringIO):
    """Capture printed output while still showing it live."""
    def __init__(self, mirror):
        super().__init__(); self._mirror = mirror
    def write(self, s):
        self._mirror.write(s); return super().write(s)

def stage(key, fn, deps=()):
    """Run fn() once per configuration and code version; cache its value and its transcript."""
    path = cache(f"stage_{key}.json")
    fh = _src_hash(fn, *deps)
    if not RESET_CACHE and os.path.exists(path):
        try:
            blob = json.load(open(path, encoding="utf-8"))
            if blob.get("key") == STAGE_KEY and blob.get("src") == fh:
                print(f"[cached] stage '{key}' - replayed, not recomputed")
                print(blob.get("stdout", ""), end="")
                return blob["value"]
            print(f"[stale] stage '{key}' - config or code changed, recomputing")
        except Exception as e:
            print(f"[cache miss] stage '{key}': {type(e).__name__}, recomputing")
    tee = _Tee(sys.stdout)
    with redirect_stdout(tee):
        value = fn()
    _atomic_save({"key": STAGE_KEY, "src": fh, "value": value, "stdout": tee.getvalue()},
                 path, as_json=True)
    return value

print(f"in colab   : {IN_COLAB}  (drive mounted: {DRIVE_OK})")
print(f"outputs    : {OUT_DIR}   <- download this folder")
print(f"cache      : {CACHE_DIR}   <- safe to delete; forces recomputation")
print(f"run key    : {RUN_KEY}    stage key: {STAGE_KEY}")

# %% [markdown]
# ## Cell 4 — data: FineWeb, a tokenizer of our own, tokenised once
#
# ### What this block does
# Streams FineWeb (sample-10BT) from Hugging Face, trains an 8,192-token byte-pair tokenizer on
# the start of it, and turns the text into two flat files of token ids: 50.5M training tokens
# and 1M validation tokens from **separate documents**. Everything is saved to Drive, so this
# runs once.
#
# ### How every run sees the same data
# The training file is cut into windows of 513 tokens (512 inputs plus the next token as the
# target). The order of the windows is shuffled once with a fixed seed. Run A's step 1 reads
# windows 1–32 of that order; run D, with a bigger batch, reads windows 1–240 in its step 1. So
# every run reads the same windows in the same order and only the grouping differs.
#
# ### What you should see
# The tokenizer's vocabulary size, the token counts, a sample sentence round-tripped through
# the tokenizer, and the average number of characters per token (about 4 for English web text).

# %%
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

EOT = "<|eot|>"

def _fineweb_texts():
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb", name=DC.fineweb_name, split="train",
                      streaming=True)
    for row in ds:
        yield row["text"]

def _fallback_texts():
    """Only used by the CPU profile when the FineWeb stream is unreachable."""
    import urllib.request
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    path = os.path.join(DATA_DIR, "tinyshakespeare.txt")
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)
    text = open(path, encoding="utf-8").read()
    for _ in range(20):                      # small corpus: repeat it (CPU profile only)
        for block in text.split("\n\n"):
            if len(block) > 64:
                yield block

def build_data():
    meta_path = os.path.join(DATA_DIR, "meta.json")
    want = {"data": asdict(DC), "vocab": MC.vocab}
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path, encoding="utf-8"))
        if meta.get("want") == want:
            print(f"[cached] tokenised data found in {DATA_DIR}")
            return meta
    t0 = time.time()
    source = "fineweb"
    try:
        stream = _fineweb_texts()
        first = next(stream)
    except Exception as e:
        if PROFILE == "full":
            raise
        print(f"FineWeb unreachable ({type(e).__name__}); CPU profile falls back to Tiny Shakespeare")
        source, stream = "tinyshakespeare", _fallback_texts()
        first = next(stream)

    def docs():
        yield first
        yield from stream

    it = docs()
    # 1) validation documents first, kept aside as raw text (6 chars per token is generous)
    val_texts, n = [], 0
    for t in it:
        val_texts.append(t); n += len(t)
        if n >= DC.val_tokens * 6:
            break
    # 2) the tokenizer is trained on the next documents, which also start the training set
    tok_texts, n = [], 0
    for t in it:
        tok_texts.append(t); n += len(t)
        if n >= DC.tokenizer_chars:
            break
    print(f"source {source}: {len(val_texts):,} validation docs, {len(tok_texts):,} docs "
          f"({sum(len(x) for x in tok_texts)/1e6:.1f}M chars) for the tokenizer")

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=MC.vocab, special_tokens=[EOT], min_frequency=2,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(tok_texts, trainer=trainer)
    tok.save(os.path.join(DATA_DIR, "tokenizer.json"))
    eot = tok.token_to_id(EOT)
    print(f"tokenizer trained: vocabulary {tok.get_vocab_size():,} in {time.time()-t0:.0f}s")

    def encode_to(texts_iter, limit, path):
        chunks, n, chars, batch = [], 0, 0, []
        def flush(batch):
            nonlocal n, chars
            enc = tok.encode_batch(batch)
            for e, t in zip(enc, batch):
                ids = np.asarray(e.ids + [eot], dtype=np.uint16)
                chunks.append(ids); n += len(ids); chars += len(t)
        for t in texts_iter:
            batch.append(t)
            if len(batch) == 512:
                flush(batch); batch = []
                if n >= limit:
                    break
        if batch and n < limit:
            flush(batch)
        arr = np.concatenate(chunks)[:limit]
        if len(arr) < limit:
            raise RuntimeError(f"only {len(arr):,} tokens available for {path}, wanted {limit:,}")
        arr.tofile(path)
        return len(arr), chars / max(n, 1)

    n_val, cpt_val = encode_to(iter(val_texts), DC.val_tokens, os.path.join(DATA_DIR, "val.bin"))
    def train_iter():
        yield from tok_texts
        yield from it
    n_tr, cpt = encode_to(train_iter(), DC.train_tokens, os.path.join(DATA_DIR, "train.bin"))
    meta = {"want": want, "source": source, "train_tokens": n_tr, "val_tokens": n_val,
            "chars_per_token": cpt, "eot_id": eot, "vocab_size": tok.get_vocab_size(),
            "seconds": time.time() - t0}
    _atomic_save(meta, meta_path, as_json=True)
    return meta

DATA_META = build_data()
TOK = Tokenizer.from_file(os.path.join(DATA_DIR, "tokenizer.json"))
TRAIN = np.memmap(os.path.join(DATA_DIR, "train.bin"), dtype=np.uint16, mode="r")
VAL = np.memmap(os.path.join(DATA_DIR, "val.bin"), dtype=np.uint16, mode="r")
assert DATA_META["vocab_size"] <= MC.vocab

class Windows:
    """Fixed windows of seq+1 tokens in a fixed shuffled order. Window k is always the same."""
    def __init__(self, data, seq, shuffle, seed=SEED):
        self.data, self.seq = data, seq
        n = (len(data) - 1) // seq
        self.order = np.random.default_rng(seed).permutation(n) if shuffle else np.arange(n)
    def __len__(self):
        return len(self.order)
    def get(self, start, count, device=DEVICE):
        idx = self.order[start:start + count]
        x = np.stack([self.data[i * self.seq: i * self.seq + self.seq + 1] for i in idx]).astype(np.int64)
        x = torch.from_numpy(x)
        if device == "cuda":
            x = x.pin_memory().to(device, non_blocking=True)
        return x[:, :-1], x[:, 1:]

TRAIN_W = Windows(TRAIN, MC.seq, shuffle=True)
VAL_W = Windows(VAL, MC.seq, shuffle=False)
_sample = "Reversible networks rebuild their activations instead of storing them."
_enc = TOK.encode(_sample)
print(f"source           : {DATA_META['source']}")
print(f"training tokens  : {len(TRAIN):,}  ({len(TRAIN_W):,} windows of {MC.seq}+1)")
print(f"validation tokens: {len(VAL):,}  (separate documents)")
print(f"chars per token  : {DATA_META['chars_per_token']:.2f}")
print(f"round trip       : {len(_enc.ids)} tokens -> {TOK.decode(_enc.ids)!r}")
R["data"] = {k: DATA_META[k] for k in ("source", "train_tokens", "val_tokens", "chars_per_token", "vocab_size")}

# %% [markdown]
# ## Cell 5 — the model: one block, three ways to stack it
#
# ### What this block does
# Defines one transformer block and three ways to stack 16 of them. They share **the same
# weights at initialisation** (same seed, same shapes), so any difference between runs comes
# from how the stack is wired, nothing else.
#
# ### The three stacks, in plain words
# Write `x` for the stream of numbers that flows from layer to layer, and `f(x)` for what one
# block adds to it (attention, then the feed-forward part).
#
# | stack | rule per layer | can it run backwards? |
# |---|---|---|
# | **ordinary** (run A) | `x_next = x + f(x)` | no: undoing it needs f(x), which needs x |
# | **midpoint** (run B) | `next = previous + 2h · f(current)` | yes: `previous = next − 2h · f(current)` |
# | **two-stream** (run C) | `q = q + h·Attn(p)`, then `p = p + h·MLP(q)` | yes: undo the second, then the first |
#
# Midpoint needs two starting states. `bootstrap="euler"` makes the second one with one
# ordinary step (the Lightning LM setting); `"none"` just copies the first.
#
# ### Two details that matter for memory
# - The stream `x` is kept in **fp32**; only the block's matrix work runs in fp16. The rebuild
#   subtracts numbers, and fp16 subtraction would lose too much accuracy across 16 layers.
# - The loss is computed in **slices of 8,192 tokens**, each recomputed during the backward
#   pass. Without this, the scores over the 8,192-token vocabulary for a batch of 240 would
#   take several GB and swamp the memory being measured.

# %%
class Attention(nn.Module):
    def __init__(self, mc):
        super().__init__()
        self.n_head = mc.n_head
        self.qkv = nn.Linear(mc.d, 3 * mc.d, bias=False)
        self.proj = nn.Linear(mc.d, mc.d, bias=False)
    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(B, T, C))

class MLP(nn.Module):
    def __init__(self, mc):
        super().__init__()
        self.up = nn.Linear(mc.d, mc.mlp_mult * mc.d, bias=False)
        self.down = nn.Linear(mc.mlp_mult * mc.d, mc.d, bias=False)
    def forward(self, x):
        return self.down(F.gelu(self.up(x)))

class Block(nn.Module):
    def __init__(self, mc):
        super().__init__()
        self.ln1, self.attn = nn.LayerNorm(mc.d), Attention(mc)
        self.ln2, self.mlp = nn.LayerNorm(mc.d), MLP(mc)
        self.drop_p = 0.0                         # only Gate B's planted bug sets this
    def _drop(self, t):
        return F.dropout(t, self.drop_p, self.training) if self.drop_p > 0 else t
    def attn_delta(self, x):
        return self._drop(self.attn(self.ln1(x))).to(x.dtype)   # back to the fp32 stream
    def mlp_delta(self, x):
        return self._drop(self.mlp(self.ln2(x))).to(x.dtype)
    def delta(self, x):
        """f(x): everything one ordinary pre-norm block adds to the stream."""
        a = self.attn_delta(x)
        return a + self.mlp_delta(x + a)

def _ce_slice(h, w, y):
    logits = h @ w.t()
    if logits.dtype in (torch.float16, torch.bfloat16):
        logits = logits.float()
    return F.cross_entropy(logits, y, reduction="sum")

def chunked_ce(h, w, y, chunk):
    """Mean cross-entropy, computed slice by slice and recomputed in the backward pass, so the
    full (tokens x vocabulary) score table never exists at once."""
    h, y = h.reshape(-1, h.size(-1)), y.reshape(-1)
    total = h.new_zeros((), dtype=torch.float64 if h.dtype == torch.float64 else torch.float32)
    for i in range(0, h.size(0), chunk):
        if torch.is_grad_enabled():
            total = total + checkpoint(_ce_slice, h[i:i + chunk], w, y[i:i + chunk],
                                       use_reentrant=False)
        else:
            total = total + _ce_slice(h[i:i + chunk], w, y[i:i + chunk])
    return total / y.numel()

# %% [markdown]
# ## Cell 6 — the reversible stacks and their own backward pass
#
# ### What this block does
# This is where the memory saving actually happens. Each reversible stack is a
# `torch.autograd.Function` with a hand-written backward pass.
#
# ### How it works
# **Forward:** run every layer with graph-building switched off, so nothing is stored, and keep
# only the two states at the top of the stack.
#
# **Backward:** walk down from the top layer. For each layer, rebuild its input from the states
# above it using the rule run backwards, run that one layer again with gradients switched on,
# take the gradients, and discard that layer's working numbers before moving to the next. At
# any moment only one layer's working numbers exist.
#
# The recomputation of `f` is used twice: once to rebuild the earlier state and once to get the
# gradients, so each layer runs twice in total (once forward, once backward) — the source of the
# expected 30–50% slowdown.
#
# ### Inputs / Outputs
# In: the two starting states, the step size, the blocks. Out: the two final states. An
# optional `recorder` list collects every rebuilt state, which Gate C compares to the truth.

# %%
def _params(blocks):
    return [p for b in blocks for p in b.parameters()]

class MidpointStack(torch.autograd.Function):
    """(a, b) -> (b, a + 2h f(b)) per layer. Inverse: a = new_b - 2h f(new_a)."""

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, a, b, h, blocks, recorder, *params):
        with torch.no_grad():
            for blk in blocks:
                a, b = b, a + 2.0 * h * blk.delta(b)
        ctx.h, ctx.blocks, ctx.recorder = h, blocks, recorder
        ctx.save_for_backward(a, b)
        return a, b

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, ga, gb):
        a, b = ctx.saved_tensors
        h, blocks = ctx.h, ctx.blocks
        ga = torch.zeros_like(a) if ga is None else ga
        gb = torch.zeros_like(b) if gb is None else gb
        per_block = []
        for blk in reversed(blocks):
            ps = list(blk.parameters())
            with torch.enable_grad():
                cur = a.detach().requires_grad_(True)          # b_l = a_{l+1}
                y = blk.delta(cur)
            prev = b - 2.0 * h * y.detach()                     # a_l = b_{l+1} - 2h f(b_l)
            grads = torch.autograd.grad(y, [cur] + ps, 2.0 * h * gb, allow_unused=True)
            ga, gb = gb, ga + grads[0]
            per_block.append(grads[1:])
            if ctx.recorder is not None:
                ctx.recorder.append((prev.detach(), cur.detach()))
            a, b = prev, cur.detach()
        pg = [g for gs in reversed(per_block) for g in gs]
        return (ga, gb, None, None, None, *pg)

class TwoStreamStack(torch.autograd.Function):
    """(q, p) -> q' = q + h Attn(p); p' = p + h MLP(q'). Inverse: undo p first, then q."""

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, q, p, h, blocks, recorder, *params):
        with torch.no_grad():
            for blk in blocks:
                q = q + h * blk.attn_delta(p)
                p = p + h * blk.mlp_delta(q)
        ctx.h, ctx.blocks, ctx.recorder = h, blocks, recorder
        ctx.save_for_backward(q, p)
        return q, p

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, gq, gp):
        q, p = ctx.saved_tensors
        h, blocks = ctx.h, ctx.blocks
        gq = torch.zeros_like(q) if gq is None else gq
        gp = torch.zeros_like(p) if gp is None else gp
        per_block = []
        for blk in reversed(blocks):
            mps, aps = list(blk.ln2.parameters()) + list(blk.mlp.parameters()), \
                       list(blk.ln1.parameters()) + list(blk.attn.parameters())
            with torch.enable_grad():
                qn = q.detach().requires_grad_(True)
                m = blk.mlp_delta(qn)
            p_prev = p - h * m.detach()
            with torch.enable_grad():
                pp = p_prev.detach().requires_grad_(True)
                at = blk.attn_delta(pp)
            q_prev = q - h * at.detach()
            gm = torch.autograd.grad(m, [qn] + mps, h * gp, allow_unused=True)
            gq_total = gq + gm[0]
            gat = torch.autograd.grad(at, [pp] + aps, h * gq_total, allow_unused=True)
            gq, gp = gq_total, gp + gat[0]
            # order must match blk.parameters(): ln1, attn, ln2, mlp
            per_block.append(tuple(gat[1:]) + tuple(gm[1:]))
            if ctx.recorder is not None:
                ctx.recorder.append((q_prev.detach(), p_prev.detach()))
            q, p = q_prev, p_prev
        pg = [g for gs in reversed(per_block) for g in gs]
        return (gq, gp, None, None, None, *pg)

class GPT(nn.Module):
    """One set of weights; `scheme` decides how the blocks are stacked, `reversible` decides
    whether the reversible stacks use the custom backward pass (True) or plain autograd (False,
    used only by the gates as the reference)."""

    def __init__(self, mc, scheme="standard", h=1.0, bootstrap="euler", reversible=True, n_layer=None):
        super().__init__()
        self.mc, self.scheme, self.h, self.bootstrap, self.reversible = mc, scheme, h, bootstrap, reversible
        L = n_layer or mc.n_layer
        self.wte = nn.Embedding(mc.vocab, mc.d)
        self.wpe = nn.Embedding(mc.seq, mc.d)
        self.blocks = nn.ModuleList(Block(mc) for _ in range(L))
        self.ln_f = nn.LayerNorm(mc.d)
        self.recorder = None
        self.apply(self._init)
        for b in self.blocks:        # GPT-2 style: shrink the layers that write into the stream
            for w in (b.attn.proj.weight, b.mlp.down.weight):
                nn.init.normal_(w, std=0.02 / math.sqrt(2 * L))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def states(self, idx):
        """Run the stack. Returns the final state and, in reference mode, every state."""
        T = idx.size(1)
        x = (self.wte(idx) + self.wpe(torch.arange(T, device=idx.device))).to(self.wte.weight.dtype)
        trace = []
        if self.scheme == "standard":
            for blk in self.blocks:
                x = x + blk.delta(x)
            return x, trace
        if self.scheme == "midpoint":
            a = x
            if self.bootstrap != "euler":
                b = x
            elif self.reversible and torch.is_grad_enabled():
                # The Euler start is one ordinary step outside the reversible stack. Stored the
                # normal way it would keep a whole layer's working numbers (the smoke run measured
                # +162.9 MiB at batch 32), so it is recomputed in the backward pass instead.
                b = x + self.h * checkpoint(self.blocks[0].delta, x, use_reentrant=False)
            else:
                b = x + self.h * self.blocks[0].delta(x)
            if self.reversible:
                a, b = MidpointStack.apply(a, b, self.h, self.blocks, self.recorder, *_params(self.blocks))
            else:
                for blk in self.blocks:
                    trace.append((a, b))
                    a, b = b, a + 2.0 * self.h * blk.delta(b)
            return b, trace
        if self.scheme == "twostream":
            q = p = x
            if self.reversible:
                q, p = TwoStreamStack.apply(q, p, self.h, self.blocks, self.recorder, *_params(self.blocks))
            else:
                for blk in self.blocks:
                    trace.append((q, p))
                    q = q + self.h * blk.attn_delta(p)
                    p = p + self.h * blk.mlp_delta(q)
            return 0.5 * (q + p), trace
        raise ValueError(self.scheme)

    def forward(self, idx, targets, chunk=None):
        x, _ = self.states(idx)
        return chunked_ce(self.ln_f(x), self.wte.weight, targets, chunk or TC.loss_chunk)

def make_model(scheme, h=1.0, bootstrap="euler", reversible=True, n_layer=None, mc=None):
    torch.manual_seed(SEED)                      # identical starting weights for every stack
    return GPT(mc or MC, scheme, h, bootstrap, reversible, n_layer).to(DEVICE)

def state_bytes(model):
    """fp32 weights + fp32 gradients + two fp32 Adam moments = 16 bytes per parameter."""
    return 16 * sum(p.numel() for p in model.parameters())

_m = make_model("standard")
print(f"parameters: {sum(p.numel() for p in _m.parameters()):,} "
      f"(training state {state_bytes(_m)/2**20:.0f} MiB)")
del _m

# %% [markdown]
# ## Cell 7 — Gate A: the custom backward pass gives the right gradients
#
# ### What this block does
# Builds a small version of each reversible stack in **double precision on the CPU**, runs it
# twice from the same weights and the same batch — once with ordinary autograd storing
# everything, once with our custom backward pass — and compares every gradient.
#
# ### Why it matters
# A wrong custom backward pass still trains; the loss just ends up somewhere slightly wrong.
# Nothing else in the notebook would notice. In double precision the two must agree to about
# 12 decimal places, so any real mistake shows up as a huge difference.
#
# ### What you should see
# Three lines, one per stack, each with a relative difference around 1e-15 and `PASS`.

# %%
class GateError(AssertionError):
    pass

def gate(ok, msg):
    print(f"  [{'PASS' if ok else 'FAIL'}] {msg}")
    if not ok:
        raise GateError(msg)

TINY = ModelCfg(vocab=64, d=32, n_layer=4, n_head=2, seq=16)

def _grad_compare(scheme, bootstrap, h=0.5, drop_p=0.0, dtype=torch.float64, device="cpu", tiny=TINY):
    torch.manual_seed(SEED)
    ref = GPT(tiny, scheme, h, bootstrap, reversible=False).to(device, dtype)
    rev = copy.deepcopy(ref); rev.reversible = True
    for m in (ref, rev):
        for b in m.blocks:
            b.drop_p = drop_p
        m.train()
    g = torch.Generator().manual_seed(7)
    idx = torch.randint(0, tiny.vocab, (3, tiny.seq), generator=g).to(device)
    tgt = torch.randint(0, tiny.vocab, (3, tiny.seq), generator=g).to(device)
    out = []
    for m in (ref, rev):
        torch.manual_seed(99)                    # same dropout draws in both forward passes
        loss = m(idx, tgt, chunk=17)
        loss.backward()
        out.append((loss.item(), torch.cat([p.grad.reshape(-1) for p in m.parameters()])))
    (l1, g1), (l2, g2) = out
    rel = ((g1 - g2).abs().max() / g1.abs().max()).item()
    return {"loss_ref": l1, "loss_rev": l2, "grad_rel_diff": rel}

def _gate_a():
    res = {}
    for scheme, boot in [("midpoint", "euler"), ("midpoint", "none"), ("twostream", "none")]:
        r = _grad_compare(scheme, boot)
        res[f"{scheme}/{boot}"] = r
        gate(r["grad_rel_diff"] < TH.grad_rel_fp64 and abs(r["loss_ref"] - r["loss_rev"]) < 1e-12,
             f"{scheme:9s} start={boot:5s}: gradient rel. diff {r['grad_rel_diff']:.1e} "
             f"(< {TH.grad_rel_fp64:.0e}), loss {r['loss_rev']:.6f}")
    return res

R["gates"]["A_gradients"] = stage("gate_a", _gate_a,
                                  deps=(_grad_compare, GPT, Block, MidpointStack, TwoStreamStack, chunked_ce))

# %% [markdown]
# ## Cell 8 — Gate B: dropout breaks the rebuild (a planted bug)
#
# ### What this block does
# Repeats Gate A with dropout switched on (10%). The ordinary model and the reversible model
# get the same random dropout pattern in their forward passes, but the reversible model's
# backward pass **runs each layer again and draws a new pattern**. The gradients should now
# disagree, and this gate passes only if that disagreement is large.
#
# ### Why it matters
# This is the reason dropout is 0 in every run (see the plan). A gate that has never been seen
# to fail proves little; this shows the Gate A comparison would catch this exact mistake.
#
# ### What you should see
# Relative differences far above the threshold for both stacks, and `PASS` (bug caught).

# %%
def _gate_b():
    res = {}
    for scheme, boot in [("midpoint", "euler"), ("twostream", "none")]:
        r = _grad_compare(scheme, boot, drop_p=0.1)
        res[scheme] = r
        gate(r["grad_rel_diff"] > TH.dropout_caught,
             f"{scheme:9s} with dropout 0.1: gradient rel. diff {r['grad_rel_diff']:.2e} "
             f"(> {TH.dropout_caught:.0e}), bug caught")
    return res

R["gates"]["B_dropout_caught"] = stage("gate_b", _gate_b, deps=(_grad_compare, GPT, Block))

# %% [markdown]
# ## Cell 9 — Gate C: how far rebuilt states drift
#
# ### What this block does
# Runs each reversible stack forward while recording every true state, then runs the custom
# backward pass while recording every **rebuilt** state, and compares them layer by layer. On
# the CPU this is done in fp32. On a GPU it is also done at the real size under fp16 autocast,
# and the gradients are compared with plain autograd's (cosine similarity).
#
# ### Why it matters
# Rebuilding by subtraction loses a little accuracy at every layer. If that loss grows layer
# after layer, the gradients of the bottom layers are wrong even though Gate A passed in fp64.
#
# ### What you should see
# The worst relative error per stack (small, well under the threshold) and, on a GPU, gradient
# cosine similarities very close to 1.

# %%
def _recon_errors(scheme, bootstrap, h, mc, dtype, device, batch, amp=False):
    torch.manual_seed(SEED)
    ref = GPT(mc, scheme, h, bootstrap, reversible=False).to(device, dtype)
    rev = copy.deepcopy(ref); rev.reversible = True
    g = torch.Generator().manual_seed(11)
    idx = torch.randint(0, mc.vocab, (batch, mc.seq), generator=g).to(device)
    tgt = torch.randint(0, mc.vocab, (batch, mc.seq), generator=g).to(device)
    ctx = torch.autocast("cuda", dtype=AMP_DTYPE) if amp else nullcontext()
    with ctx:
        with torch.no_grad():
            _, trace = ref.states(idx)
        loss_ref = ref(idx, tgt)
    loss_ref.backward()
    rev.recorder = []
    with ctx:
        loss_rev = rev(idx, tgt)
    loss_rev.backward()
    rebuilt = list(reversed(rev.recorder))       # recorder runs top-down; flip to bottom-up
    errs = []
    for (t0, t1), (r0, r1) in zip(trace, rebuilt):
        num = max((t0 - r0).abs().max().item(), (t1 - r1).abs().max().item())
        den = max(t0.abs().max().item(), t1.abs().max().item(), 1e-30)
        errs.append(num / den)
    g1 = torch.cat([p.grad.reshape(-1).float() for p in ref.parameters()])
    g2 = torch.cat([p.grad.reshape(-1).float() for p in rev.parameters()])
    cos = F.cosine_similarity(g1.double(), g2.double(), dim=0).item()
    return {"per_layer_rel_err": errs, "max_rel_err": max(errs), "grad_cosine": cos}

def _gate_c():
    res = {}
    small = ModelCfg(vocab=128, d=64, n_layer=8, n_head=4, seq=64)
    for scheme, boot, h in [("midpoint", "euler", 0.5), ("twostream", "none", 1.0)]:
        r = _recon_errors(scheme, boot, h, small, torch.float32, "cpu", batch=4)
        res[f"{scheme}_fp32_cpu"] = r
        gate(r["max_rel_err"] < TH.recon_rel_fp32,
             f"{scheme:9s} fp32, 8 layers: worst rebuilt-state error {r['max_rel_err']:.1e} "
             f"(< {TH.recon_rel_fp32:.0e}); gradient cosine {r['grad_cosine']:.6f}")
    if DEVICE == "cuda":
        for scheme, boot, h in [("midpoint", "euler", 0.5), ("twostream", "none", 1.0)]:
            r = _recon_errors(scheme, boot, h, MC, torch.float32, "cuda", batch=4, amp=True)
            res[f"{scheme}_amp_gpu"] = r
            print(f"  {scheme:9s} fp16 autocast, real size: worst rebuilt-state error "
                  f"{r['max_rel_err']:.1e}, error at layer 1 vs top: "
                  f"{r['per_layer_rel_err'][0]:.1e} vs {r['per_layer_rel_err'][-1]:.1e}")
            gate(r["grad_cosine"] > TH.grad_cos_amp,
                 f"{scheme:9s} fp16 autocast: gradient cosine vs plain autograd "
                 f"{r['grad_cosine']:.6f} (> {TH.grad_cos_amp})")
    else:
        print("  [SKIP] fp16 autocast check at real size needs a GPU")
    return res

R["gates"]["C_reconstruction"] = stage("gate_c", _gate_c, deps=(_recon_errors, GPT, Block,
                                                                  MidpointStack, TwoStreamStack))

# %% [markdown]
# ## Cell 10 — the training loop
#
# ### What this block does
# One function trains one configuration and returns its history. Every run uses it.
#
# ### How it works
# - AdamW, weight decay 0.1 on the weight matrices only, gradient clipping at 1.0.
# - Learning rate: linear warm-up for the first 6.5% of steps, then cosine down to 10%.
# - fp16 autocast with loss scaling on the GPU.
# - Validation loss on the same 327k held-out tokens every 2.5M training tokens.
# - **Speed** is measured on the GPU clock over the training steps only: the first 10 steps
#   (warm-up of the kernels) and every validation pass are excluded.
# - **Peak memory** is `torch.cuda.max_memory_allocated()` over the whole run.
# - A checkpoint (weights, optimizer moments, loss scaler, history) is saved every 8 minutes;
#   after a disconnect the run resumes from it. A finished run is cached and never retrained.
#
# ### Inputs / Outputs
# In: a key, the stack, the step size, the start, batch, learning rate, token budget.
# Out: a history dict, also written to `outputs/logs/<key>.json`.

# %%
def make_opt(model, lr):
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    groups = [{"params": decay, "weight_decay": TC.weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    kw = {"fused": True} if DEVICE == "cuda" else {}
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, TC.beta2), eps=1e-8, **kw)

def lr_at(step, steps, lr):
    warm = max(10, int(TC.warmup_frac * steps))
    if step < warm:
        return lr * (step + 1) / warm
    t = (step - warm) / max(1, steps - warm)
    return lr * (TC.min_lr_frac + (1 - TC.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * t)))

def _sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()

@torch.no_grad()
def evaluate(model):
    model.eval()
    ctx = torch.autocast("cuda", dtype=AMP_DTYPE) if AMP else nullcontext()
    tot = 0.0
    for i in range(TC.eval_batches):
        x, y = VAL_W.get(i * TC.batch, TC.batch)
        with ctx:
            tot += model(x, y).item()
    model.train()
    return tot / TC.eval_batches

def _ckpt_path(key): return cache(f"ckpt_{RUN_KEY}_{key}.pt")
def _run_path(key):  return cache(f"run_{RUN_KEY}_{key}.json")

def train_run(key, scheme, h=1.0, bootstrap="euler", batch=None, lr=None, tokens=None, steps=None):
    batch, lr = batch or TC.batch, lr or TC.lr
    steps = steps or (tokens or TC.tokens) // (batch * MC.seq)
    if steps * batch > len(TRAIN_W):
        raise ValueError(f"{key}: needs {steps*batch:,} windows, only {len(TRAIN_W):,} exist")
    model = make_model(scheme, h, bootstrap, reversible=True)
    opt = make_opt(model, lr)
    scaler = torch.amp.GradScaler("cuda", enabled=AMP)
    ctx = torch.autocast("cuda", dtype=AMP_DTYPE) if AMP else nullcontext()
    eval_every = max(1, TC.eval_every_tokens // (batch * MC.seq))
    hist = {"key": key, "scheme": scheme, "h": h, "bootstrap": bootstrap, "batch": batch,
            "lr": lr, "steps": steps, "tokens_per_step": batch * MC.seq,
            "train": [], "val": [], "train_seconds": 0.0, "timed_tokens": 0}
    start = 0
    ck = _ckpt_path(key)
    if not RESET_CACHE and os.path.exists(ck):
        try:
            blob = torch.load(ck, map_location=DEVICE, weights_only=False)
            model.load_state_dict(blob["model"]); opt.load_state_dict(blob["opt"])
            scaler.load_state_dict(blob["scaler"]); hist = blob["hist"]; start = blob["step"] + 1
            print(f"    [resume] '{key}' from step {start:,} of {steps:,}")
        except Exception as e:
            print(f"    [resume failed] '{key}': {type(e).__name__}, starting over")
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model.train()
    last_ck, win_t0, win_tok, run_loss = time.time(), None, 0, []
    for step in range(start, steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, steps, lr)
        x, y = TRAIN_W.get(step * batch, batch)
        if step == start + 10:
            _sync(); win_t0, win_tok = time.time(), 0
        with ctx:
            loss = model(x, y)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), TC.grad_clip)
        scaler.step(opt); scaler.update()
        opt.zero_grad(set_to_none=True)
        run_loss.append(loss.item())
        if win_t0 is not None:
            win_tok += batch * MC.seq
        last = step == steps - 1
        is_eval = (step + 1) % eval_every == 0 or last
        # Record the training loss at every log step AND every evaluation. With a large batch
        # an evaluation can come before the first log step (run D: eval at step 10 or 20,
        # log every 25), and the evaluation line needs a training loss to print.
        if ((step + 1) % TC.log_every == 0 or is_eval) and run_loss:
            hist["train"].append({"step": step + 1, "tokens": (step + 1) * batch * MC.seq,
                                  "loss": float(np.mean(run_loss))})
            run_loss = []
        if is_eval:
            if win_t0 is not None:
                _sync(); hist["train_seconds"] += time.time() - win_t0; hist["timed_tokens"] += win_tok
            v = evaluate(model)
            hist["val"].append({"step": step + 1, "tokens": (step + 1) * batch * MC.seq, "loss": v,
                                "train_seconds": hist["train_seconds"]})
            tps = hist["timed_tokens"] / hist["train_seconds"] if hist["train_seconds"] else float("nan")
            print(f"    {key:<22} step {step+1:>6,}/{steps:,}  tokens {(step+1)*batch*MC.seq/1e6:6.1f}M  "
                  f"train {(hist['train'][-1]['loss'] if hist['train'] else float('nan')):.4f}  "
                  f"val {v:.4f}  {tps:,.0f} tok/s")
            if win_t0 is not None:
                _sync(); win_t0, win_tok = time.time(), 0
        if time.time() - last_ck > TC.ckpt_minutes * 60 and not last:
            if win_t0 is not None:
                _sync(); hist["train_seconds"] += time.time() - win_t0; hist["timed_tokens"] += win_tok
                win_t0, win_tok = time.time(), 0
            _atomic_save({"model": model.state_dict(), "opt": opt.state_dict(),
                          "scaler": scaler.state_dict(), "hist": hist, "step": step}, ck)
            last_ck = time.time()
    hist["tokens_per_second"] = hist["timed_tokens"] / hist["train_seconds"] if hist["train_seconds"] else None
    hist["peak_mem_bytes"] = torch.cuda.max_memory_allocated() if DEVICE == "cuda" else None
    hist["state_bytes"] = state_bytes(model)
    hist["final_val"] = hist["val"][-1]["loss"]
    hist["final_train"] = hist["train"][-1]["loss"]
    del model, opt
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return hist

def _replay(hist):
    """Print a reloaded run's progress from its saved log, so the executed notebook shows
    every run's history even when nothing was retrained. Each line is one evaluation.
    Speed per line is not stored, so these lines show cumulative training minutes instead;
    the run's measured tokens/s and peak memory are printed at the end."""
    key, tr = hist["key"], {p["step"]: p["loss"] for p in hist["train"]}
    print(f"    [cached run] '{key}' reloaded, not retrained; progress replayed from its saved log:")
    for p in hist["val"]:
        earlier = [s for s in tr if s <= p["step"]]
        t = tr[max(earlier)] if earlier else float("nan")
        print(f"    {key:<22} step {p['step']:>6,}/{hist['steps']:,}  tokens {p['tokens']/1e6:6.1f}M  "
              f"train {t:.4f}  val {p['loss']:.4f}  {p['train_seconds']/60:5.1f} train-min")
    tps, pk = hist.get("tokens_per_second"), hist.get("peak_mem_bytes")
    print(f"    {key:<22} measured: "
          + (f"{tps:,.0f} tokens/s" if tps else "speed not timed")
          + (f", peak memory {pk/2**20:,.0f} MiB" if pk else ""))

def cached_run(key, **kw):
    """Train once per configuration and code version; a finished run reloads instantly."""
    path = _run_path(key)
    fh = _src_hash(train_run, GPT, Block, MidpointStack, TwoStreamStack, chunked_ce, lr_at, make_opt)
    want = {"src": fh, "kw": kw}
    if not RESET_CACHE and os.path.exists(path):
        blob = json.load(open(path, encoding="utf-8"))
        if blob.get("want") == json.loads(json.dumps(want, default=str)):
            _replay(blob["hist"])
            return blob["hist"]
    hist = train_run(key, **kw)
    _atomic_save({"want": want, "hist": hist}, path, as_json=True)
    _atomic_save(hist, os.path.join(LOG_DIR, f"{key}.json"), as_json=True)
    if os.path.exists(_ckpt_path(key)):
        os.remove(_ckpt_path(key))
    return hist

print(f"eval every {TC.eval_every_tokens/1e6:.1f}M tokens on {TC.eval_batches*TC.batch*MC.seq:,} "
      f"validation tokens; checkpoint every {TC.ckpt_minutes:g} min")

# %% [markdown]
# ## Cell 11 — step-size sweep, and the side check that picks midpoint's start
#
# ### What this block does
# Short runs (300 steps at batch 32, about 4.9M tokens each) to choose the step size `h` for
# each reversible stack, then one more short run for the side check: midpoint with the plain
# start instead of the Euler start.
#
# ### Why
# The Lightning LM library and the exact place its `a = 0.5` enters are not public, so the step
# size is the one setting we tune ourselves (see the plan, decision 3). It is chosen on the
# same short budget for both stacks, and the full runs use the winners.
#
# ### The rule for midpoint's start (decided 23 Sep, before seeing the full-run result)
# Run B uses the Euler start (Lightning LM's setting) **unless** the plain start is better by
# more than 0.02 nats in this side check. A near-tie keeps the instructor's setting. The Euler
# start no longer costs memory (Cell 6 recomputes it), so this choice affects only training.
#
# ### What you should see
# A small table of validation losses, the chosen `h` for each stack, and one sentence naming
# the start run B will use and why, printed from the numbers.

# %%
def _sweep():
    res = {"midpoint": {}, "twostream": {}}
    for h in SC.midpoint_h:
        res["midpoint"][str(h)] = cached_run(f"sweep_mid_h{h}", scheme="midpoint", h=h,
                                             bootstrap="euler", steps=SC.steps)["final_val"]
    for h in SC.twostream_h:
        res["twostream"][str(h)] = cached_run(f"sweep_two_h{h}", scheme="twostream", h=h,
                                              bootstrap="none", steps=SC.steps)["final_val"]
    best = {s: float(min(v, key=v.get)) for s, v in res.items()}
    side = cached_run(f"sweep_mid_h{best['midpoint']}_nostart", scheme="midpoint",
                      h=best["midpoint"], bootstrap="none", steps=SC.steps)["final_val"]
    euler = res["midpoint"][str(best["midpoint"])]
    print(f"\n  {'stack':<11}{'h':>6}{'val loss':>11}")
    for s, v in res.items():
        for h, l in v.items():
            print(f"  {s:<11}{h:>6}{l:>11.4f}{'  <- chosen' if float(h) == best[s] else ''}")
    diff = side - euler
    verdict = ("too close to matter" if abs(diff) < TH.variant_tie_nats else
               ("the Euler start is better" if diff > 0 else "the plain start is better"))
    print(f"\n  side check, midpoint h={best['midpoint']}: Euler start {euler:.4f} vs plain start "
          f"{side:.4f} -> difference {diff:+.4f} nats: {verdict}")
    start = "none" if diff < -TH.variant_tie_nats else "euler"
    why = (f"the plain start was better by {-diff:.4f} nats (more than {TH.variant_tie_nats})"
           if start == "none" else
           f"the plain start was not better by more than {TH.variant_tie_nats} nats, so the "
           f"Lightning LM setting is kept")
    print(f"  run B will use the {'plain' if start == 'none' else 'Euler'} start: {why}")
    return {"val": res, "best": best, "start_mid": start,
            "side_check": {"euler": euler, "none": side, "diff": diff, "verdict": verdict,
                           "run_b_start": start, "why": why}}

R["sweep"] = stage("sweep", _sweep, deps=(train_run,))
H_MID, H_TWO = R["sweep"]["best"]["midpoint"], R["sweep"]["best"]["twostream"]
START_MID = R["sweep"]["start_mid"]

# %% [markdown]
# ## Cell 12 — memory probe: the largest batch, and memory against depth
#
# ### What this block does
# Two measurements, no training:
# 1. **Largest batch.** For each stack, find the largest batch for which one full training step
#    (forward, backward, optimizer step) fits in GPU memory, by doubling then halving the gap.
#    The ratio reversible ÷ ordinary is this assignment's version of the paper's 9.88×.
# 2. **Memory against depth.** At batch 32, one step at 8, 16, 24 and 32 layers, recording peak
#    memory. The ordinary stack should climb with every layer; the reversible ones should not.
#
# ### What you should see
# The largest batch per stack and the ratio, and a depth table. On a CPU this cell is skipped:
# the CPU has no peak-memory counter worth reporting.

# %%
def _try_step(scheme, h, bootstrap, batch, n_layer=None):
    """Peak bytes of a training step at this batch, or None if it does not fit.

    Two steps are taken and the peak is measured over both. Adam creates its two running
    averages at the END of the first step, after that step's memory peak, so a one-step probe
    misses them (the smoke run showed exactly one layer's weights, 4.7 MiB, per added layer).
    In real training they already exist at every peak, so the second step is the honest one."""
    model = opt = x = y = loss = None
    try:
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        model = make_model(scheme, h, bootstrap, n_layer=n_layer)
        opt = make_opt(model, TC.lr)
        scaler = torch.amp.GradScaler("cuda", enabled=AMP)
        x = torch.randint(0, MC.vocab, (batch, MC.seq), device=DEVICE)
        y = torch.randint(0, MC.vocab, (batch, MC.seq), device=DEVICE)
        for _ in range(2):
            with torch.autocast("cuda", dtype=AMP_DTYPE):
                loss = model(x, y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            opt.zero_grad(set_to_none=True)
            loss = None
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated()
    except torch.OutOfMemoryError:
        return None
    finally:
        del model, opt, x, y, loss
        gc.collect(); torch.cuda.empty_cache()

def _largest_batch(scheme, h, bootstrap):
    lo, hi = 0, 8
    while hi <= PC.batch_cap and _try_step(scheme, h, bootstrap, hi) is not None:
        lo, hi = hi, hi * 2
    hi = min(hi, PC.batch_cap + 1)
    while hi - lo > 8:
        mid = (lo + hi) // 2 // 8 * 8
        if mid <= lo:
            break
        if _try_step(scheme, h, bootstrap, mid) is not None:
            lo = mid
        else:
            hi = mid
    return lo

def _probe():
    if DEVICE != "cuda":
        print("  [SKIP] needs a GPU")
        return {"skipped": True}
    res = {"largest_batch": {}, "depth": {}}
    for name, scheme, h, boot in [("ordinary", "standard", 1.0, "none"),
                                  ("midpoint", "midpoint", H_MID, START_MID),
                                  ("two-stream", "twostream", H_TWO, "none")]:
        b = _largest_batch(scheme, h, boot)
        res["largest_batch"][name] = b
        print(f"  largest batch, {name:<10}: {b:>5}  ({b*MC.seq/1e3:,.0f}k tokens per step)")
    base = res["largest_batch"]["ordinary"]
    for name in ("midpoint", "two-stream"):
        r = res["largest_batch"][name] / base if base else float("nan")
        res[f"ratio_{name}"] = r
        print(f"  {name} fits {r:.2f}x the ordinary model's largest batch")
    print(f"\n  peak memory for one step at batch {PC.depth_batch} (MiB)")
    print(f"  {'layers':>8}" + "".join(f"{n:>13}" for n in ("ordinary", "midpoint", "two-stream")))
    for L in PC.depths:
        row = {}
        for name, scheme, h, boot in [("ordinary", "standard", 1.0, "none"),
                                      ("midpoint", "midpoint", H_MID, START_MID),
                                      ("two-stream", "twostream", H_TWO, "none")]:
            pk = _try_step(scheme, h, boot, PC.depth_batch, n_layer=L)
            row[name] = pk
        res["depth"][str(L)] = row
        print(f"  {L:>8}" + "".join(f"{(v or float('nan'))/2**20:>13,.0f}" for v in row.values()))
    d = res["depth"]
    lo_L, hi_L = str(PC.depths[0]), str(PC.depths[-1])
    for name in ("ordinary", "midpoint", "two-stream"):
        if d[lo_L][name] and d[hi_L][name]:
            per_layer = (d[hi_L][name] - d[lo_L][name]) / (PC.depths[-1] - PC.depths[0])
            res[f"mib_per_layer_{name}"] = per_layer / 2**20
            print(f"  {name:<10} grows {per_layer/2**20:6.1f} MiB per extra layer")
    return res

R["probe"] = stage("probe", _probe, deps=(_try_step, _largest_batch, GPT, MidpointStack, TwoStreamStack))

# %% [markdown]
# ## Cell 13 — runs A, B and C at batch 32
#
# ### What this block does
# The three full runs, 50M tokens each, same data order, same starting weights:
# **A** ordinary, **B** midpoint (start chosen in Cell 11), **C** two-stream. Then it picks the better
# reversible variant by final validation loss, and prints the verdict from the numbers.
#
# ### What you should see
# Progress lines every 2.5M tokens, then a small table and one sentence naming the better
# variant, or saying the two are too close to call.

# %%
RUNS = {}
RUNS["A"] = cached_run(f"A_ordinary_b{TC.batch}", scheme="standard", h=1.0, bootstrap="none")
RUNS["B"] = cached_run(f"B_midpoint_b{TC.batch}", scheme="midpoint", h=H_MID, bootstrap=START_MID)
RUNS["C"] = cached_run(f"C_twostream_b{TC.batch}", scheme="twostream", h=H_TWO, bootstrap="none")

_b, _c = RUNS["B"]["final_val"], RUNS["C"]["final_val"]
if abs(_b - _c) < TH.variant_tie_nats:
    WINNER = "B"
    _verdict = (f"midpoint {_b:.4f} vs two-stream {_c:.4f}: within {TH.variant_tie_nats} nats, "
                f"too close to call; run D uses midpoint, the lesson's default")
else:
    WINNER = "B" if _b < _c else "C"
    _verdict = (f"{'midpoint' if WINNER == 'B' else 'two-stream'} trained better: "
                f"{min(_b, _c):.4f} vs {max(_b, _c):.4f} nats")
print(f"\n  variant verdict: {_verdict}")
R["variant"] = {"B_val": _b, "C_val": _c, "winner": WINNER, "verdict": _verdict}

# %% [markdown]
# ## Cell 14 — run D: the better variant at a much larger batch
#
# ### What this block does
# Sets run D's batch to the smaller of two limits, then trains it:
# - **memory limit:** 90% of the largest batch that fitted in Cell 12, rounded down to a
#   multiple of 8;
# - **step limit:** the largest batch that still leaves at least 400 steps for 50M tokens.
#
# The learning rate is raised by the square root of the batch increase (decision 4c).
#
# ### What you should see
# Which limit decided the batch, the batch and learning rate, then the run's progress lines.

# %%
_win = RUNS[WINNER]
_scheme = _win["scheme"]
_name = "midpoint" if _scheme == "midpoint" else "two-stream"
step_limit = TC.tokens // (MC.seq * TC.min_steps_run_d) // 8 * 8
if R["probe"].get("skipped"):
    mem_limit, _why = None, "no GPU probe: step limit only"
    BATCH_D = step_limit
else:
    mem_limit = int(PC.safety * R["probe"]["largest_batch"][_name]) // 8 * 8
    BATCH_D = min(step_limit, mem_limit)
    _why = "step limit" if BATCH_D == step_limit else "memory limit"
LR_D = TC.lr * math.sqrt(BATCH_D / TC.batch)
print(f"  step limit  : {step_limit} (keeps >= {TC.min_steps_run_d} steps)")
print(f"  memory limit: {mem_limit}")
print(f"  run D batch : {BATCH_D}, decided by the {_why}; "
      f"{TC.tokens // (BATCH_D * MC.seq):,} steps; learning rate {LR_D:.2e} "
      f"(= {TC.lr} x sqrt({BATCH_D}/{TC.batch}))")
RUNS["D"] = cached_run(f"D_{_name}_b{BATCH_D}", scheme=_scheme, h=_win["h"],
                       bootstrap=_win["bootstrap"], batch=BATCH_D, lr=LR_D)
R["run_d"] = {"batch": BATCH_D, "lr": LR_D, "step_limit": step_limit, "memory_limit": mem_limit,
              "decided_by": _why, "variant": _name}

# %% [markdown]
# ## Cell 15 — results, time to target, cost
#
# ### What this block does
# One table with everything the brief asks for, per run: final training and validation loss,
# tokens per second, peak memory. Plus two derived numbers:
# - **time to target:** training seconds until the validation loss first reaches run A's final
#   validation loss (linear interpolation between evaluations), the fair speed comparison when
#   batch sizes differ;
# - **cost:** training hours × a public T4 price, clearly an estimate.
#
# ### What you should see
# The table, then sentences generated from the numbers (never written in advance).

# %%
def time_to(hist, target):
    prev = None
    for p in hist["val"]:
        if p["loss"] <= target:
            if prev is None:
                return p["train_seconds"]
            f = (prev["loss"] - target) / max(prev["loss"] - p["loss"], 1e-12)
            return prev["train_seconds"] + f * (p["train_seconds"] - prev["train_seconds"])
        prev = p
    return None

TARGET = RUNS["A"]["final_val"]
rows = []
for k in ("A", "B", "C", "D"):
    hst = RUNS[k]
    tts = time_to(hst, TARGET)
    hours = hst["train_seconds"] / 3600
    rows.append({"run": k, "stack": hst["scheme"], "batch": hst["batch"], "steps": hst["steps"],
                 "final_train": hst["final_train"], "final_val": hst["final_val"],
                 "tokens_per_second": hst["tokens_per_second"],
                 "peak_mib": (hst["peak_mem_bytes"] or 0) / 2**20 if hst["peak_mem_bytes"] else None,
                 "train_seconds": hst["train_seconds"],
                 "time_to_A_final_s": tts,
                 "cost_usd": hours * TH.t4_usd_per_hour,
                 "cost_to_target_usd": tts / 3600 * TH.t4_usd_per_hour if tts else None})
R["results"] = rows

print(f"  {'run':<4}{'stack':<11}{'batch':>6}{'steps':>7}{'train':>9}{'val':>9}{'tok/s':>10}"
      f"{'peak MiB':>10}{'to A final':>12}{'cost $':>8}")
for r in rows:
    print(f"  {r['run']:<4}{r['stack']:<11}{r['batch']:>6}{r['steps']:>7,}{r['final_train']:>9.4f}"
          f"{r['final_val']:>9.4f}{(r['tokens_per_second'] or float('nan')):>10,.0f}"
          f"{(r['peak_mib'] or float('nan')):>10,.0f}"
          f"{(format(r['time_to_A_final_s'], ',.0f') + 's') if r['time_to_A_final_s'] else 'never':>12}"
          f"{r['cost_usd']:>8.3f}")

A_, B_, D_ = rows[0], rows[1], rows[3]
if A_["tokens_per_second"] and B_["tokens_per_second"]:
    slow = 1 - B_["tokens_per_second"] / A_["tokens_per_second"]
    print(f"\n  At batch {TC.batch}, midpoint runs at {B_['tokens_per_second']:,.0f} tokens/s against "
          f"{A_['tokens_per_second']:,.0f}: {abs(slow)*100:.0f}% {'slower' if slow > 0 else 'faster'}.")
if A_["peak_mib"] and B_["peak_mib"]:
    print(f"  Peak memory at batch {TC.batch}: {B_['peak_mib']:,.0f} MiB for midpoint vs "
          f"{A_['peak_mib']:,.0f} MiB ordinary ({B_['peak_mib']/A_['peak_mib']:.2f}x).")
if A_["tokens_per_second"] and D_["tokens_per_second"]:
    r_ = D_["tokens_per_second"] / A_["tokens_per_second"]
    print(f"  Run D at batch {D_['batch']} runs at {D_['tokens_per_second']:,.0f} tokens/s, "
          f"{r_:.2f}x run A; its final validation loss is {D_['final_val']:.4f} against A's "
          f"{A_['final_val']:.4f} ({D_['final_val']-A_['final_val']:+.4f}).")
for r in rows[1:]:
    if r["time_to_A_final_s"] is None:
        print(f"  Run {r['run']} never reached run A's final validation loss.")

# %% [markdown]
# ## Cell 16 — figures
#
# ### What this block does
# Draws every figure from the results dictionary and saves it to `outputs/`, so they can be
# redrawn later without retraining.

# %%
plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False})
COL = {"A": "#b1521f", "B": "#1f6f8b", "C": "#7a4b9e", "D": "#2c6b45"}
LAB = {"A": "A ordinary, b32", "B": "B midpoint, b32", "C": "C two-stream, b32",
       "D": f"D {_name}, b{BATCH_D}"}

def _save(fig, name):
    fig.tight_layout()
    fig.savefig(out(name), bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}")

fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
for k, hst in RUNS.items():
    ax[0].plot([p["tokens"]/1e6 for p in hst["val"]], [p["loss"] for p in hst["val"]], color=COL[k], label=LAB[k])
    ax[1].plot([p["train_seconds"]/60 for p in hst["val"]], [p["loss"] for p in hst["val"]], color=COL[k], label=LAB[k])
ax[0].set_xlabel("training tokens (millions)"); ax[1].set_xlabel("training minutes")
for a in ax:
    a.set_ylabel("validation loss"); a.axhline(TARGET, color="#9a968e", lw=0.8, ls="--")
ax[0].legend(frameon=False)
_save(fig, "loss_curves.png")

fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
ks = list(RUNS)
ax[0].bar([LAB[k] for k in ks], [RUNS[k]["tokens_per_second"] or 0 for k in ks], color=[COL[k] for k in ks])
ax[0].set_ylabel("tokens per second")
ax[1].bar([LAB[k] for k in ks], [(RUNS[k]["peak_mem_bytes"] or 0)/2**20 for k in ks], color=[COL[k] for k in ks])
ax[1].set_ylabel("peak memory (MiB)")
for a in ax:
    a.tick_params(axis="x", rotation=20)
_save(fig, "speed_memory.png")

if not R["probe"].get("skipped"):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    lb = R["probe"]["largest_batch"]
    ax[0].bar(list(lb), list(lb.values()), color=["#b1521f", "#1f6f8b", "#7a4b9e"])
    ax[0].set_ylabel("largest batch that fits")
    for name, c in zip(("ordinary", "midpoint", "two-stream"), ("#b1521f", "#1f6f8b", "#7a4b9e")):
        Ls = [int(L) for L in R["probe"]["depth"]]
        ax[1].plot(Ls, [(R["probe"]["depth"][str(L)][name] or float("nan"))/2**20 for L in Ls],
                   marker="o", color=c, label=name)
    ax[1].set_xlabel("layers"); ax[1].set_ylabel(f"peak MiB, batch {PC.depth_batch}")
    ax[1].legend(frameon=False)
    _save(fig, "probe.png")

fig, ax = plt.subplots(figsize=(5, 3.2))
for key, c in (("midpoint_fp32_cpu", "#1f6f8b"), ("twostream_fp32_cpu", "#7a4b9e"),
               ("midpoint_amp_gpu", "#1f6f8b"), ("twostream_amp_gpu", "#7a4b9e")):
    r = R["gates"]["C_reconstruction"].get(key)
    if r:
        ax.semilogy(range(1, len(r["per_layer_rel_err"]) + 1), np.maximum(r["per_layer_rel_err"], 1e-17),
                    color=c, ls="-" if "gpu" in key else ":", marker="o", label=key.replace("_", " "))
ax.set_xlabel("layer"); ax.set_ylabel("rebuilt-state relative error"); ax.legend(frameon=False, fontsize=7)
_save(fig, "reconstruction.png")

# %% [markdown]
# ## Cell 17 — acceptance checks, summary.json, RESULTS.md, bundle
#
# ### What this block does
# Collects every check into one list and prints `N/N`, writes `summary.json` (everything the
# README is generated from) and `RESULTS.md`, and zips `outputs/` into `era5s13_outputs.zip`.
# A check that could not run on this machine is counted as **skipped**, never as passed.

# %%
G = R["gates"]
_gpu = DEVICE == "cuda"
CHECKS = [
    ("A. midpoint (Euler start) gradients == plain autograd, fp64", True),
    ("A. midpoint (plain start) gradients == plain autograd, fp64", True),
    ("A. two-stream gradients == plain autograd, fp64", True),
    ("B. planted dropout bug caught, both stacks", True),
    ("C. rebuilt states within threshold, fp32", True),
    ("C. gradient cosine vs autograd under fp16 autocast", True if _gpu else None),
    ("data: 50M-token budget fits the tokenised file", len(TRAIN_W) >= TC.tokens // MC.seq),
    ("runs: every run finished its step budget", all(RUNS[k]["train"][-1]["step"] == RUNS[k]["steps"] for k in RUNS)),
    ("runs: every run's validation loss fell", all(RUNS[k]["val"][-1]["loss"] < RUNS[k]["val"][0]["loss"] for k in RUNS)),
    ("probe: reversible largest batch > ordinary", None if R["probe"].get("skipped") else
        min(R["probe"]["largest_batch"]["midpoint"], R["probe"]["largest_batch"]["two-stream"]) >
        R["probe"]["largest_batch"]["ordinary"]),
    ("probe: reversible memory grows less per layer than ordinary", None if R["probe"].get("skipped") else
        R["probe"].get("mib_per_layer_midpoint", 1e9) < R["probe"].get("mib_per_layer_ordinary", 0)),
    (f"run D: batch larger than {TC.batch}", BATCH_D > TC.batch),
]
print("ACCEPTANCE CHECKS")
for name, ok in CHECKS:
    print(f"  [{'PASS' if ok else ('SKIP' if ok is None else 'FAIL')}] {name}")
_ran = [ok for _, ok in CHECKS if ok is not None]
_npass, _nskip = sum(1 for ok in _ran if ok), sum(1 for _, ok in CHECKS if ok is None)
print(f"\nACCEPTANCE: {_npass}/{len(_ran)} passed" + (f", {_nskip} skipped" if _nskip else ""))
R["acceptance"] = {"passed": _npass, "total": len(_ran), "skipped": _nskip,
                   "checks": [{"name": n, "result": None if ok is None else bool(ok)} for n, ok in CHECKS]}
R["runs"] = RUNS
R["meta"]["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
_atomic_save(R, out("summary.json"), as_json=True)

L_ = [f"# Session 13 results — notebook {NOTEBOOK_VERSION}, profile {PROFILE}\n",
      f"Model {MC.n_layer} × {MC.d}, vocab {MC.vocab:,}, {n_params()/1e6:.1f}M parameters. "
      f"Device {DEVICE}{' (' + GPU_NAME + ')' if GPU_NAME else ''}.\n",
      f"Acceptance: **{_npass}/{len(_ran)}**" + (f" ({_nskip} skipped)" if _nskip else "") + "\n",
      "| run | stack | batch | steps | final train | final val | tokens/s | peak MiB | cost $ |",
      "|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    L_.append(f"| {r['run']} | {r['stack']} | {r['batch']} | {r['steps']:,} | {r['final_train']:.4f} | "
              f"{r['final_val']:.4f} | {(r['tokens_per_second'] or float('nan')):,.0f} | "
              f"{(r['peak_mib'] or float('nan')):,.0f} | {r['cost_usd']:.3f} |")
L_.append(f"\nVariant verdict: {R['variant']['verdict']}\n")
L_.append(f"Run D batch {BATCH_D} ({R['run_d']['decided_by']}), learning rate {LR_D:.2e}.\n")
if not R["probe"].get("skipped"):
    L_.append("Largest batch: " + ", ".join(f"{k} {v}" for k, v in R["probe"]["largest_batch"].items()) + "\n")
L_.append("\n## Acceptance\n")
for c in R["acceptance"]["checks"]:
    L_.append(f"- [{'PASS' if c['result'] else ('SKIP' if c['result'] is None else 'FAIL')}] {c['name']}")
with open(out("RESULTS.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(L_) + "\n")

_zip = out("era5s13_outputs.zip")
with zipfile.ZipFile(_zip, "w", zipfile.ZIP_DEFLATED) as z:
    for root, _dirs, files in os.walk(OUT_DIR):
        for name in files:
            if name.endswith((".png", ".json", ".md")):
                p = os.path.join(root, name)
                z.write(p, arcname=os.path.relpath(p, OUT_DIR))
print(f"\nbundle: {_zip}  ({os.path.getsize(_zip)/1024:.0f} KB)")
print("Download it together with the executed .ipynb.")

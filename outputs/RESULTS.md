# Session 13 results — notebook v1, profile full

Model 16 × 320, vocab 8,192, 22.5M parameters. Device cuda (Tesla T4).

Acceptance: **12/12**

| run | stack | batch | steps | final train | final val | tokens/s | peak MiB | cost $ |
|---|---|---|---|---|---|---|---|---|
| A | standard | 32 | 3,051 | 4.2021 | 4.1145 | 53,446 | 4,051 | 0.091 |
| B | midpoint | 32 | 3,051 | 4.2875 | 4.2042 | 41,109 | 1,144 | 0.118 |
| C | twostream | 32 | 3,051 | 4.3245 | 4.2515 | 41,499 | 1,165 | 0.117 |
| D | midpoint | 240 | 406 | 5.8336 | 5.8695 | 41,975 | 3,774 | 0.113 |

Variant verdict: midpoint trained better: 4.2042 vs 4.2515 nats

Run D batch 240 (step limit), learning rate 2.74e-03.

Largest batch: ordinary 144, midpoint 896, two-stream 912


## Acceptance

- [PASS] A. midpoint (Euler start) gradients == plain autograd, fp64
- [PASS] A. midpoint (plain start) gradients == plain autograd, fp64
- [PASS] A. two-stream gradients == plain autograd, fp64
- [PASS] B. planted dropout bug caught, both stacks
- [PASS] C. rebuilt states within threshold, fp32
- [PASS] C. gradient cosine vs autograd under fp16 autocast
- [PASS] data: 50M-token budget fits the tokenised file
- [PASS] runs: every run finished its step budget
- [PASS] runs: every run's validation loss fell
- [PASS] probe: reversible largest batch > ordinary
- [PASS] probe: reversible memory grows less per layer than ordinary
- [PASS] run D: batch larger than 32

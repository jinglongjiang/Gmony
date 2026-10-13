# KDA-VL Parent Reconstruction and Drop-in Freeze

## Decision

The engineering deliverable is a canonical scene-level KDA replacement, not a new
navigation mechanism or a performance result. Formal IL/RL and paper evaluation are
NOT_RUN. No shixu implementation was imported or copied. Original dormant optional
branches remain inherited; the frozen route is train.algo=sarl with scalar MC value.

The unique mapping from an available checkpoint to the submitted paper's final
checkpoint remains UNVERIFIED. Consequently this is not a completed paper reproduction
and not permission to compare old published Mamba numbers with newly trained KDA.

## Two immutable references

| Reference | Asset | What is established |
|---|---|---|
| Original source | original backup CrowdNav | source/checkpoint numerical reference |
| Git Legacy | tag mamba-vl-legacy-20261006, commit 0c4ca45 | restored source frozen before edits |
| VL-v2 | current drop-in branch, final commit/tag | common corrected contract for future retraining |

Checkpoint used for numerical parity: rl_model_ep10000_T24.pth from the original
backup's crowd_nav/runs/mamba_vl directory. SHA256:
a2b5ac50c8362134e4c8b19e7b692cae4f394e70294f9cb1ed4bfb52d9675a65.
It has episode=10000, algo=sarl and a strict-loading, four-layer Mamba state dictionary.
Its limited metadata does not uniquely identify the paper's final model or runtime.

The fixed three-step fixture saves tokens, actor-encoder inputs, scene features,
Mamba outputs, scalar values, all 80 raw/final scores, rewards, clearances, grid
commands, prior executed commands and final executed commands.
Original versus tagged source and tagged repeat versus repeat are bitwise identical
in the installed verification runtime. This is checkpoint/source numerical parity;
historical dependency parity and 3000-episode paper benchmark reproduction are separate
UNVERIFIED / NOT_RUN items. Config and environment hashes are stored in JSON.
The loader removes only the torch.compile `_orig_mod.` key prefix. The parity
execution is eager; it does not establish historical compiled-runtime parity.

## Contract findings and decisions

| Finding | Executable evidence | Decision |
|---|---|---|
| Native packed robot order is radius,gx,gy,v_pref, but converters interpreted radius,v_pref,gx,gy | sentinel goal distance 4.560976 instead of 6.964194; preferred speed 3.0 instead of 1.2 | correct both converters and preferred-speed index in successor helper |
| Human tokens already contain relative x/y, spatial relations subtract robot x/y again | token (3.25,1.5), robot (1.25,-2), derived relation (2,3.5) | correct interpretation only, keep spatial layers/shapes/parameters unchanged |
| Human selection is first-five truncation, TTC sorting, then spatial distance sorting | synthetic 5/10/20-person states exclude a close human outside the first five | document actual behavior; do not change selection algorithm |
| Lookahead scores grid command, execution smooths afterward | checkpoint fixture with a known prior executed command records the mismatch | map all candidates to their executable commands before successor/reward/clearance/value/filter/risk; no second smoothing |

These are common engineering corrections, not scientific contributions.
Reward coefficients, margin .2, risk weight .8, smoothing .3, all-unsafe fallback,
action support, epsilon exploration and MC-return definitions are otherwise unchanged.
All three models must be retrained under the same corrected contract.

## Official KDA integration

The official Moonshot repository delegates its open KDA kernels to FLA:
https://github.com/MoonshotAI/Kimi-Linear
https://github.com/fla-org/flash-linear-attention

Pinned repositories:
- Kimi-Linear: 8c1d85eb6b5f8fcefb15758691b0ce50b0827ce3.
- flash-linear-attention: 9f38d24980c46d46bd38614e743cdacd21906578.

The adapter calls the unmodified full FLA KimiDeltaAttention layer, not a handwritten
delta recurrence or the entire Kimi language model. Four layers, width 256, heads=4,
head_dim=64, expand_v=1, short_conv=True, conv_size=4. Native q/k/v projections,
per-channel decay, beta update, short convolutions and gated output projection remain
inside the official layer. Existing residual/post-LayerNorm wrapping is inherited from
the parent. Every window starts with no cache and produces no persistent cache.

Public interface: temporal_backbone(x, mask=None), [B,24,256] -> [B,24,256].
Value head: unchanged Linear(256,1) on the final output. There is no actor memory,
forecast output, auxiliary loss, external gate, new action, duration or planner.

## Runtime and numerical validation

Local host GLIBC=2.31 cannot load the tested official Mamba wheels. The verification
uses the already-local Ubuntu 22.04 CUDA image, with host NVIDIA driver and the isolated
Python environment mounted inside it. Host Python/environments were not upgraded.
No remote server or Kimi model weights were downloaded.

Runtime: Python 3.11.17, torch 2.7.1+cu126, Triton 3.3.1, official Mamba-1 operator
from mamba-ssm 2.2.5, FLA 0.6.0 at the pinned source commit, RTX3060.
The image's toolkit banner says CUDA11.8; torch's actual runtime is CUDA12.6.
Full versions, source hashes and requirements are in environment-manifest.json.

An initial BF16 smoke path had batch/scalar value differences of approximately
.01756 for Mamba and .04058 for KDA; broad relative tolerance was not accepted for
the final freeze. The final common path is FP32 without autocast. GRU's cuDNN TF32
approximation is disabled locally: its earlier FP32 batch/scalar discrepancy was
.000351 with TF32 enabled. No official KDA code was patched.
This is numerical validation, not evidence of better navigation.
Optional causal-conv1d is not installed for Mamba: it uses its official library's
available path. These local smoke latencies are not an optimized deployment efficiency
comparison. A claimed speed advantage would require matched, verified optimized runtimes.

| Correctness test | Coverage / result |
|---|---|
| Legacy numerical parity | checkpoint, three controlled steps, all captured arrays bitwise equal |
| Shape and finite forward/backward | Mamba/GRU/KDA, [2,24,256], shared encoder/temporal/value head finite nonzero gradients |
| Causality | modifying frames 12-23 does not change outputs 0-11 within declared FP32 tolerance |
| Batch versus scalar candidates | 80 identical windows, atol=rtol=1e-4; measured errors saved, not rounded to zero |
| History and window isolation | scoring does not mutate real history/action; repeated windows give identical outputs |
| Save/load | strict state reload produces identical values for each backbone |
| Candidate execution | selected command is the scored smoothed command, without double smoothing |
| Filter/risk/fallback | mixed-safe and all-unsafe fixtures checked against inherited equations |
| Shared initialization | all non-temporal tensor hashes identical across three arms for each of 419/443/467/491 |
| Reward/action support | same 80 commands and identical immediate rewards across backbones |
| Official kernel/reference | chunk and recurrent versus official naive KDA recurrence; q/k/v/g/beta backward parity for chunk |
| Full official layer/reference | same projections/convolution/norm, only kernel replaced with official reference in diagnostic |
| Training entrypoint | python -m crowd_nav.train --help succeeds and exposes mamba/gru/kda; no training |

Full functional/gradient tests use untrained seed419 models; the other three seeds
check shared initialization only. They are NOT four trained models or paired outcomes.
CPU optimized KDA and FP16 are not tested. BF16 kernel/reference tests are separate
from the FP32 value path. This test scope cannot establish SR, CR, TR or usefulness.

## Registered resource measurements

Untrained seed419, RTX3060, FP32, warmed batch of 80 windows, T=24. Ten repetitions;
model forward only, excluding Python successor construction and the safety consumer.

| Backbone | Total params | Active value-path params | Temporal params | Batch80 ms | Peak torch allocation MiB |
|---|---:|---:|---:|---:|---:|
| Mamba | 2,223,337 | 2,166,785 | 2,048,000 | 21.99 | 83.33 |
| GRU | 1,754,857 | 1,698,305 | 1,579,520 | 4.92 | 58.42 |
| KDA | 1,506,809 | 1,450,257 | 1,331,472 | 6.13 | 48.12 |

Total params include inherited unused Q/forecast heads. Peak allocation includes
buffers retained after the backward smoke, not full deployment VRAM. Extra forward
allocation is separately registered in JSON. Mamba's optional fast convolution is
absent; these measurements must not be promoted to a paper efficiency claim.
Batch/scalar max value differences: Mamba 3.159e-6, GRU 2.980e-7, KDA 2.146e-6.
The known-prior-command legacy trace demonstrates score/execute command mismatches
of .285958, .085787 and .025736 m/s over three steps; VL-v2 scores its returned command.
All backbones' mixed-safe fixture has two safe candidates; all-unsafe fixture has zero.

## Files and reproducibility

| File | Change |
|---|---|
| crowd_nav/contracts.py | canonical field interpretations only |
| crowd_nav/policy/mamba_rl.py | common command scoring, read-only scoring function, KDA factory option, position contract, GRU numerical precision |
| crowd_nav/policy/kda_temporal.py | small adapter around the official layer |
| crowd_nav/policy/shared_initialization.py | backbone-independent shared initialization |
| crowd_nav/train.py | KDA CLI option, shared initialization and post-construction RNG reset; unchanged IL/MC targets |
| tools/verify_kda_vl.py | correctness-only fixtures, artifacts, reference tests and environment manifest |
| tools/run-dropin-verification.sh and Dockerfile.verify | repeatable local verification runtime |

Re-run from the CrowdNav root: bash tools/run-dropin-verification.sh.
Evidence is in artifacts/kda-vl-freeze: legacy-parity.json, backbone-correctness.json,
shared-init-hashes.json, environment-manifest.json, environment-requirements.txt,
three legacy traces and three candidate-contract NPZ files. No optimizer steps,
episode rollout benchmark, new training checkpoint or standalone per-test report.

## Stop boundary

Stop here. Engineering correctness is distinct from backbone performance and method
novelty. No evidence currently establishes KDA-VL superiority.
Before authorized paired training: identify the paper checkpoint/config explicitly,
freeze one shared raw ORCA dataset, use fresh three-arm VL-v2 initialization and a
separate output directory, and freeze the explicit evaluation protocol. Do not silently
reuse the old IL/MC weights or the default 50-episode eval setting as paper reproduction.

The next stage, only on instruction, is the planned paired full-budget training,
not an actor/forecast/commitment rescue. This freeze does not authorize it.

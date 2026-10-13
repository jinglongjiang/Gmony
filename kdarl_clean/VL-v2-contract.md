# VL-v2 Public Contract

## Scope and provenance

Parent: the restored original Mamba-VL CrowdNav project, not shixu.
Legacy tag: mamba-vl-legacy-20261006; commit 0c4ca45.
VL-v2 preserves the inherited environment, observation rule, action grid,
reward, history padding, scalar value head and MC-regression training route.
This contract is not a claim that every inherited implementation is bug-free.
No navigation training or paper benchmark was run for this freeze.

## Common data path

JointState -> packed 34-vector -> [8,13] tokens -> relational spatial encoder
-> [B,24,256] scene embeddings -> four temporal layers -> [B,24,256]
-> final temporal feature -> inherited Linear(256,1) -> scalar value.

Only the temporal operator differs between Mamba, GRU and KDA. Module widths,
depths and input/output interfaces match; parameter counts need not match.
Masks are None: short histories repeat their first real frame on the left.
No persistent neural state is shared across independent windows.

## Field contract and corrections

| Field | Packed index | VL-v2 interpretation | Legacy discrepancy |
|---|---:|---|---|
| radius | 4 | radius | none |
| gx | 5 | goal x | interpreted as v_pref in derived token features |
| gy | 6 | goal y | interpreted as gx |
| v_pref | 7 | preferred speed | interpreted as gy |
| theta | 8 | heading | none |
| human token x/y | human slots 0/1 | already robot-relative | spatial relation construction subtracted robot position again |

The first nine robot token entries remain the native packed fields, in the
order px,py,vx,vy,radius,gx,gy,v_pref,theta. Derived goal/speed entries now agree
with that order. simulate_next_frames reads preferred speed at index 7.
The spatial module shapes and parameters are unchanged; only the interpretation
of its existing human position fields is corrected for all backbones.
Executable evidence: legacy NPZ traces and backbone-correctness.json.

## Human selection: documented, not redesigned

The value-lookahead path takes the first five supplied humans before tokenization.
Tokenization then TTC-sorts those five, with a distance fallback for non-closing
humans. EnhancedSpatialEncoder distance-sorts those same five again.
It does not globally select the most dangerous five out of a 10/20-person crowd.
Synthetic 5/10/20-person traces include a close final human outside the first five
and confirm exclusion. No selection algorithm was changed.
Reward/clearance calculations still inspect all supplied humans, as inherited.

## Candidate scoring equals execution

There are 80 native grid indices: five exponential speeds and sixteen headings,
without a stop action. The grid itself is unchanged.
In test/val/eval, with a previous executed command and smoothing alpha=0.3:

    command[i] = alpha * previous_executed + (1-alpha) * grid[i]

That command is now used for robot successor, reward, clearance, value evaluation,
hard filtering and risk penalty, and is returned without a second smoothing pass.
Training still executes/scores unsmoothed grid commands, as in the original.
Existing all-unsafe fallback and the inherited index-0 tie bias are unchanged.
score_sarl_candidates is read-only. Selection appends the real current token,
not hypothetical successor tokens. The inherited epsilon exploration early-return
behavior is unchanged; this freeze does not claim to have redesigned history collection.

## Temporal implementations and precision

Mamba: official mamba_ssm.modules.mamba_simple.Mamba (Mamba-1 operator),
four inherited residual/post-LayerNorm blocks, state=64, conv=4, expansion=2.
GRU: inherited four-layer width-256 GRU and final LayerNorm, configured dropout.
KDA: official FLA KimiDeltaAttention, four layers, four heads of dimension 64,
expand_v=1, native short convolution=4, native q/k/v, delta update, channel decay,
native gated output normalization/projection. The inherited outer residual and
LayerNorm wrap the official layer. No MLA, MoE, LLM head or external gate.
KDA past_key_values=None and use_cache=False on every window.

The final public path is FP32, without autocast. GRU disables cuDNN TF32 locally
to avoid batch-size-dependent reduced-precision differences. KDA uses the official
CUDA kernels; there is no handcrafted compact or CPU substitute backend.
BF16 is tested separately against the official reference, not used in this freeze's
navigation value path. FP16 is not tested. Runtime versions are in the manifest.

## Initialization and future training

For each paired seed, initialize_shared resets all inherited non-temporal child
modules in an isolated CPU RNG scope. Shared tensor hashes match for 419/443/467/491.
Temporal operators retain their native initialization. The training entrypoint resets
sampler RNG after construction, so different backbone allocation does not shift it.
Existing dormant Q/forecast heads are retained from the parent and included in shared
hashes; they are not added mechanisms or losses in the scalar-value route.

Future comparison must initialize all three models from scratch under VL-v2.
Use one frozen ORCA demonstration pool with native raw fields; regenerate tokens.
Do not reuse legacy encoded tokens or compare legacy-trained Mamba with newly trained KDA.
Do not load an old IL checkpoint to bypass VL-v2 pretraining.
Preserve gamma=.99, scalar MC-return MSE, IL budget and 10k online MC budget.
Online policy data can differ; demonstrations, budget and evaluation cases must not.

The source train defaults use a 50-second limit and IL50; policy eval defaults have
50 episodes per environment whereas the paper describes 500. This freeze preserves
and hashes those source defaults rather than silently editing the evaluation protocol.
An explicit paper evaluation configuration and unique paper-checkpoint mapping are
still required before claiming paper reproduction. Published SR is not inherited as
a measured VL-v2 baseline. Formal training is not authorized by this freeze.

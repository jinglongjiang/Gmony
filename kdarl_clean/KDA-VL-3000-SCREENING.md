# KDA-VL minimum paired-budget screening

Status: COMPLETED. Both final-3000 and resumed final-10000 evaluations are complete.
This file preserves the chronological execution notes; statements marked as pending
below describe the earlier run-time state, not the current state. The authoritative
completed synthesis is KDA-VL-COMPREHENSIVE-REPORT.md.

Final-10000 (192 episodes per arm): Mamba 165/15/12, GRU 165/19/8,
KDA 169/15/8 (success/collision/timeout). One training seed only; resumed replay
was rebuilt, so these are not uninterrupted 10000-episode reproduction claims.

## Frozen scope

- Parent: the restored original Mamba-VL project, not shixu.
- Drop-in contract base: 19e6031. Replay metadata fix: 227210d.
  Numerically equivalent IL preprocessing: 717c238; config-wrapper fix: b216efd.
- One paired training seed: 419. Temporal backbones: Mamba-1, GRU, official FLA KDA.
- Native scene-level spatial encoder, T=24, width=256, depth=4, scalar value head,
  80-action value lookahead, reward, filter and execution contract are unchanged.
- Formal budget: identical ORCA pool, 50 fixed IL epochs, then 3000 online MC episodes.
- Checkpoints at 500/1000/2000/3000 (inherited 500-step interval also saves 1500/2500).
  Only episode 3000 is eligible for the primary comparison, never the best checkpoint.
- Test: cases 88000-88031 in each 5/10/20-person circle/square cell, 192 episodes per arm.
  Circle radius 4, square width 10, original 50-second limit and 0.25-second control clock.
  This is a screening protocol, not the paper's full six-scenario benchmark reproduction.
- No tuning from intermediate evaluation; no new predictor, memory interface or objective.

## Shared data and initialization

The existing raw ORCA pool contains 15000 successful 5-person demonstrations. Local and
server files have identical SHA256:

940ae49af4b641d23b7c8bdf47f548dddce5db128b7ea0beaabe1e897459a8e2

Raw state width is 34. Corrected VL-v2 token conversion is applied at training time.
The pool metadata specifies an 81-action teacher grid, whereas students use 80 actions.
Teacher action indices are not used in scalar MC-value IL. This difference is recorded,
not silently changed or presented as identical teacher/student action support.

The three inherited non-temporal module initialization hashes match on the actual 4090
runtime. Model parameters and complete source/config hashes are in protocol.json.

## Runtime

RTX4090, existing server environment, Python 3.10, PyTorch 2.9.1+cu128, Triton 3.5.1,
mamba-ssm 2.2.6.post3 and FLA 0.6.0 source commit
9f38d24980c46d46bd38614e743cdacd21906578. Official KDA forward/backward is finite.
No replacement Torch environment or large model weights were downloaded to the server.
Optional causal-conv1d is absent: measured Mamba runtime must not support a claim that
KDA structurally beats a fully optimized Mamba implementation.

Resource caveat: an independent four-arm Bayesian training job started on the same
GPU while KDA IL was running. Those processes are unrelated and are not modified or
terminated. The harness samples total device memory, so its peak includes other jobs;
it is not a per-model VRAM measurement. Training and inference wall times are observed
under resource sharing, not controlled architectural efficiency benchmarks.
The initial foreground SSH connection later disconnected; process inspection confirmed
that the existing remote harness and KDA training continued, without a training restart.

## Technical failures preserved

1. First launch stopped before any episode: inherited train.py requires a sibling
   env.config for the action-grid contract. Runner now supplies the unchanged file.
2. Next smoke showed online episodes were not stored: SARL act() returned None as the
   grid index; integer conversion rejected replay entries. The policy now records the
   actual selected grid index for both epsilon and greedy paths and exposes that metadata.
   This does not alter chosen commands, value targets, losses or action support.
   It is a common engineering correction, not a KDA method contribution.

The second failure says what this current code path did, not what the published
checkpoint historically learned. Historical RL coverage cannot be inferred from it.
Both failed launches and their protocol/log artifacts are retained separately.

Local and server regression checks pass: real grid indices, replay insertion, terminal
MC returns [0.99^2, 0.99, 1] for a three-frame success fixture. Successful smoke additionally
requires nonempty, finite optimizer state, not merely a completed episode counter.

## Smoke and formal comparison

Smoke completed: 100 episodes, 541.65 seconds, sampled total GPU memory peak 2003 MiB.
Final checkpoint SHA256:

9177621b909d935c4758c7258d2bdceef576bf152976c9538db1c24407c22a85

Policy tensors and AdamW optimizer state are finite; optimizer state is nonempty.
Value MSE at episodes 25/50: 0.0233/0.0135. This is technical stability evidence only,
not navigation performance evidence or a KDA-versus-baseline result.

The first formal IL launch was stopped during preprocessing, before its first optimizer
update. Original code converted all overlapping raw histories: 17,983,632 frame instances
for 749,318 original frames. Conversion now occurs once per frame before the identical
episode-local, left-repeated window construction. Samples, order, targets, window length,
dtype, optimizer and budgets are unchanged. No navigation result motivated this change.

Regression: synthetic lengths 1/3/24/31 with windows 1/3/24, plus 32 evenly spaced real
episodes (1591 frames, 3,971,136 token-window elements), all bitwise equal to the original
conversion-after-windowing path; max absolute error 0. The server regression also passes.
The paused preprocessing attempt and its log are retained; it is not a performance result.

The evaluation loader removes torch.compile's name prefix only, preserving every tensor
and using strict loading. Formal source/config hashes are re-frozen after preprocessing.

An additional launch failed before the first IL update because the new fixed-budget flag
was initially read as if its argument were RawConfigParser; the actual argument is
TrainConfig. This runner integration error is corrected by explicit wrapper propagation.
Regression now includes an actual two-epoch CPU GRU IL call, finite value loss and saved
final-epoch metadata. All three training-contract regressions pass; no result-based tuning.

The 100-episode smoke is random-init online stability only, with no IL or replay prefill.
Its weights, optimizer and RNG are never reused by any formal arm. Formal KDA restarts
with the same shared initialization and completes its own full common IL budget.

IL fixed-budget mode is opt-in. It disables inherited early stopping and best-epoch
rollback equally for all three arms. The default parent behavior remains unchanged.

Formal KDA IL completed all 50 epochs with 749318 samples and 2927 full batches per
epoch. Its saved metadata records final value MSE 0.0003225312102586031 and epoch 50.
The first epoch took 149.67 seconds. These are training-fit and runtime observations,
not held-out prediction errors or closed-loop navigation gains.
The online episode-500 checkpoint has 92 populated optimizer parameter states, all at
step 2000, confirming four actual optimization steps per episode. KDA subsequently
completed episode 3000, with optimizer step 12000 and finite weights/optimizer state.
Formal wall time was 16681.99 seconds, including preprocessing, IL, online MC and inherited
diagnostic evaluations. Its final checkpoint SHA256 is
55acb5c08a3c62b78ef9fe92477ffda42be16e07c23b2dc2af5218bbc5e2909f.
Mamba also completed epoch 50 and online episode 3000; its final optimizer step is 12000.
Final IL training MSE was 0.00019040274491999298. Formal wall time was 14608.37 seconds,
and final checkpoint SHA256 is
960f4f17abe4f34adddc1f6c0a0bef312e573697dc34f84bc9397faf83ec2abe.
GRU also completed its fixed IL budget and is running online MC. No primary closed-loop
comparison exists yet.

KDA online training MSE logged at episodes 500/1000/2000/3000:
0.0113/0.0087/0.0071/0.0061. These minibatch training losses cannot establish convergence,
held-out value accuracy or navigation superiority.

Pending: formal three-arm training, final-only closed-loop evaluation,
training curves, inference latency, parameter and memory comparison. No SR superiority,
KDA-specific benefit, convergence, or cross-seed stability claim is made at this stage.

## Artifact locations

Server: /root/kda-vl-3000-20261006/results

Local: /home/abc/workspace/nav_data/mamba/camrl/CrowdNav/artifacts/vl-3000-screening

Checkpoint/log artifacts are copied to local storage during execution. Frozen drop-in
verification results remain separate and are not overwritten by this experiment.

## Requested continuation to 10000

The user requested: finish all final-3000 evaluations, then resume the three arms in
parallel to cumulative episode 10000. No repeat IL, no model/reward/input changes.
The continuation is queued behind completion of the entire 3000 evaluation, not merely
the appearance of checkpoint files. Separate output preserves all 3000 artifacts.

The inherited RL checkpoint stores policy, target network, AdamW state, episode and
statistics; it does NOT store online replay, RNG state or environment case counters.
Consequently this is a matched resumed run, NOT an uninterrupted 10000-episode experiment.
All arms rebuild the same original ORCA replay prefill, use the original epsilon schedule
at cumulative episodes 3001-10000, and retain the 12000 completed optimizer steps. No IL
optimization is repeated. Expected final optimizer step is 40000 (7000 x 4 additional).
Resume failures raise rather than silently resetting to episode 1 or training new IL.

A new actual CPU GRU CLI regression resumed episode 1 to 2, skipped IL, rebuilt teacher
replay, restored AdamW step 1 and saved step 5 after four online updates. This is a small
engineering regression, not an additional navigation experiment or formal training arm.
All four training-contract tests passed locally and on the actual server runtime.
The regression initially failed because its fixture omitted sibling env.config and then
violated the inherited 50-second environment lock; the fixture was corrected, not the
formal simulator. Existing formal 3000 training was not restarted.

After all resumed arms finish, automatic final-10000 evaluation repeats the same 88000
test block and produces comparison.json. This repeated block is not fresh confirmation.
Status and errors are saved in status.json; failures are not hidden or converted into
successful completion. A disk guard stops only owned continuation jobs below 2 GiB free.

Server continuation: /root/kda-vl-10000-20261007/results

Local continuation mirror:
/home/abc/workspace/nav_data/mamba/camrl/CrowdNav/artifacts/vl-10000-continuation

The detached server controller is PID 1695792, source commit 1964f05. Its verified initial
state is WAITING_FOR_3000_EVALUATION. This means continuation is scheduled, not yet running;
KDA and Mamba are complete at 3000, while GRU is still finishing its 3000 budget.
No foreground SSH connection is required for either current training or the queued job.
Local mirror PID 3271927 copies completed artifacts every 120 seconds; the first sync
succeeded. It exits after the final completed/failed status is copied, or on a disk guard.
Morning status is in vl-10000-continuation/status.json. Final primary tables are generated
as vl-3000-screening/comparison.json and, when finished, vl-10000-continuation/comparison.json.
Neither outcome nor a completion time is promised before those artifacts exist.

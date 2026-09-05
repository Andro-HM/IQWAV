# IQWAV Agent Operating Rules

This file is the stable operating contract for AI coding harnesses working on IQWAV.

It is intended to be read by Kilo, Grok CLI, Codex, and other coding agents. It is not a replacement for `README.md`, `LOGS.md`, Git history, tests, or the source code.

## 1. Authority and ownership

- HM is the project owner and final technical authority.
- Sayan is the collaborator whose work may be reviewed selectively.
- Never confuse HM's repository state with Sayan's repository state.
- The live IQWAV repository is the source of truth.

When sources disagree, use this precedence:

1. current repository files and Git state
2. current test/experiment results
3. current `README.md` and `LOGS.md`
4. `IQWAV_HANDOFF.md`
5. old AI transcripts / exported sessions

Do not preserve a handoff claim that is contradicted by the live repository.

## 2. Required startup procedure

Before modifying IQWAV:

1. Read `README.md`.
2. Read the newest relevant entries in `LOGS.md`.
3. Read `IQWAV_HANDOFF.md` if it exists.
4. Inspect the current Git branch, HEAD, and working-tree state.
5. Inspect the existing implementation and tests relevant to the requested task.
6. Reconcile any stale handoff information against the live repository before proceeding.

Do not assume that a historical test count, branch, commit, or uncommitted state is still current.

## 3. Development style

IQWAV is built incrementally. Do not build several roadmap stages at once.

Normal sequence:

theory / scientific scope
→ bounded implementation
→ focused tests
→ full regression
→ manual / independent scientific validation where needed
→ documentation
→ commit only when HM explicitly asks

Rules:

- Keep each task bounded to one scientifically meaningful capability.
- Preserve established API semantics unless there is a demonstrated reason to change them.
- Do not perform unrelated refactors.
- Do not broaden claims beyond the exact signal model and data tested.
- Separate impairment injection, estimation, correction/synchronization, classification, and demodulation.
- Synthetic correctness is not real-world RF validation.
- A reliability score is not automatically a calibrated probability or confidence.
- If an estimator could validate itself circularly, use an independent truth-based validation path.
- Distinguish implementation bugs from estimator limitations and genuine identifiability limits.

## 4. Git safety

Unless HM explicitly instructs otherwise:

- do not commit
- do not push
- do not pull
- do not merge
- do not rebase
- do not cherry-pick
- do not switch/create/delete branches
- do not rewrite Git history

Reading Git state and diffs is allowed when required for the task.

If HM asks for a commit, include only the accepted files for that milestone and follow HM's requested sequence.

## 5. Sayan reconciliation policy

Sayan's work is candidate engineering input, not an alternate source of truth.

Never wholesale-copy or mass-merge Sayan's repository.

Use the established verdicts:

- KEEP — HM implementation remains production choice.
- TAKE — Sayan capability can be adopted essentially as-is after validation.
- ADAPT — useful idea, but integrate under HM architecture/semantics.
- PARK — preserve as a future candidate, do not integrate now.
- REJECT — do not use as the production replacement.

Frozen historical snapshots currently important:

- Snapshot A: `9e927de862f075bb3db16b995136d9155cc94de1`
- Snapshot B: `6bb80f6d6e522578a347e623d308263f055a32d9`

Do not inspect or integrate Sayan work newer than the allowed frozen snapshot unless HM explicitly authorizes a new reconciliation.

## 6. Testing and scientific validation

For production changes:

- run focused tests first
- run the full test suite after focused tests pass
- report exact pass/fail/skip counts
- preserve deterministic seeds/configuration when experiments depend on randomness
- use known synthetic truth for correctness checks
- use independent validation when reusing the production estimator would create circular evidence
- do not claim general RF robustness from synthetic tests alone

For AMC/model work later:

- prevent train/validation/test leakage by capture, generated stream, seed, or channel realization
- record seeds/configs
- use held-out evaluation
- report per-class and SNR/channel-stratified results where applicable
- models remain experimental until deliberately promoted

## 7. Documentation roles

`README.md`
- current project architecture/capability overview
- should describe what the system presently supports, not aspirational claims

`LOGS.md`
- chronological engineering evidence
- records meaningful accepted milestones, tests, decisions, experiments, and limitations

`IQWAV_HANDOFF.md`
- cross-harness transfer memory
- lets Kilo, Grok CLI, Codex, or another harness continue from the same project state
- is not the source of truth over Git/README/LOGS

## 8. Manual cross-harness handoff protocol

Do NOT update `IQWAV_HANDOFF.md` after every prompt or minor edit.

Only update it when HM explicitly asks for a handoff/finalization, normally immediately before switching coding harnesses.

When HM asks to finalize the handoff:

1. Read the existing `IQWAV_HANDOFF.md`.
2. Inspect current branch, HEAD, staging state, modified/untracked files, and recent commits.
3. Verify the latest relevant test state; do not invent a test count.
4. Summarize what changed since the state already recorded in the handoff.
5. Update the current checkpoint and next action.
6. Preserve durable older technical decisions and limitations unless they have genuinely been superseded.
7. Clearly distinguish committed, staged, unstaged, and untracked work.
8. Record the harness that performed the latest work when known.
9. Do not commit or push merely because a handoff was requested.
10. Stop after the handoff is finalized unless HM asks for more work.

A handoff request is an administrative transfer step, not permission to implement another milestone.

## 9. Scientific guardrails

Do not claim any of the following without new evidence:

- raw IQ sample rate can generally be uniquely inferred from samples alone
- absolute RF center frequency exists in baseband samples without acquisition metadata
- threshold-defined occupied-band width is a standards/regulatory OBW
- cumulative-power OBW excludes noise/interference unless explicitly compensated
- strongest FFT peak is automatically carrier, signal center, or CFO
- peak-above-noise is SNR
- current band SNR is Eb/N0, Es/N0, BER, EVM, or universal receiver quality
- baud rate is bit rate
- rectangular symbol-grid boundary offset is timing recovery
- current lag-1 CFO estimator is universal carrier acquisition
- static phase estimation resolves intrinsic BPSK/QPSK bit-label ambiguity
- FM OTA validation proves digital synchronization/AMC
- a feature near 57 kHz means RDS payload was decoded
- synthetic passing tests establish broad real-world RF robustness
- Sayan's controlled BPSK/QPSK classifier is general AMC

Blind FEC, interleaving, and framing remain later and potentially research-level problems.

## 10. Current handoff

Read `IQWAV_HANDOFF.md` for the current checkpoint, current uncommitted work, test baseline, exact API semantics, Sayan state, validation evidence, and next milestone.

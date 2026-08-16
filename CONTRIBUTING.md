# Contributing

Vyre is a Rust workspace of 34 crates. It builds a program out of registered
operations as IR, compiles the whole graph into one immutable artifact, emits
that artifact as a target payload, and runs it on a device. A change here is a
change to a compiler contract and is reviewed as one.

## Build and test

Build every crate and every target:

```bash
./cargo_full check --workspace --all-targets
```

Run the workspace suite:

```bash
./cargo_full test --workspace
```

`./cargo_full` is the wrapper at the workspace root. It declares the build
environment once and then execs cargo, so no command, script or document sets a
build-affecting variable or flag of its own. Pass it exactly what you would pass
cargo. Every command in this repository's documentation is written that way, and
a document that spells a bare `cargo` invocation fails the hygiene gate.

Per-crate test instructions live under `docs/testing/`, one page per crate.

## Gates

Every check in the workspace is a gate in one registry. There are no other
categories. Run the whole registry:

```bash
./cargo_full run -p xtask --bin xtask -- gates
```

Run one gate by name, or a related set with `--subset`:

```bash
./cargo_full run -p xtask --bin xtask -- docs-check
./cargo_full run -p xtask --bin xtask -- gates --subset docs
```

`./cargo_full run -p xtask --bin xtask -- --help` prints every gate and every
subset with what it judges. The subsets group gates by what they answer for:
`prepublish`, `composition`, `structure`, `docs`, `ir`, `manifest-rules`,
`source-rules`, `hot-path-rules`, `lint-rules`, `contract-rules`, `repo-rules`.

A gate reports findings. `xtask/gate-baselines.toml` pins the count each gate is
allowed: more findings than the pin fails, fewer is reported so the pin can be
lowered. The pin only moves down, and `gates --write-baseline` refuses to record
a count above the pin already written. A gate that owns a generated artifact
checks it by default and rewrites it only when you pass `--write`.

`xtask/ci-registry.toml` declares the wiring: which subsets hold each gate,
which workflows run it, every check CI runs that is not a gate, and every
workflow path the tree carries or once carried. Regenerate it with
`./cargo_full run -p xtask --bin xtask -- ci-registry --write` after adding a
gate, a subset or a workflow step. The `ci-registry` gate compares the file
against the registry, the subsets and the steps in both directions, and
`ci-steps` resolves every package, test, bench, example, binary and feature a
step names against the workspace manifests.

A `[[workflow]]` row states what happened to the path it names. `live` runs.
`paused` carries the reason and the condition that ends the pause. `superseded`
names the workflow that runs its checks and the gate that carries them.
`unprotected` names the verification class nothing performs. A row outlives its
file: deleting a workflow leaves the row naming a path the checkout does not
carry, and the gate fails until the row records where the checks went.

Run the smallest gate that owns the contract you changed, then the subset that
contains it.

## Backend work needs a device

Backend suites run against a real GPU. Before calling a backend failure
environmental, prove the device is visible and the capability contract holds:

```bash
nvidia-smi
./cargo_full test -p vyre-driver-wgpu --test capability_contract -- --nocapture
```

A GPU-required lane fails loudly when the probe is broken. Do not add a host
fallback, a skip guard, or a `no GPU` pass. `vyre-lints` rejects all three, and
a silent fallback is the failure class the workspace exists to prevent.

## What a change must satisfy

Placement follows two rules, stated in full in
[the placement rule](docs/lego-block-rule.md):

- Composed, not rewritten. A function that returns a `Program` built from IR
  that already exists belongs in `vyre-libs`, whoever calls it.
- Intrinsic means uncomposable. An operation belongs in `vyre-primitives` only
  when it needs its own emitter arm in every backend and its own arm in the
  reference interpreter.

A new operation carries a registry entry, reference behaviour, backend
behaviour where the backend supports it, tests, and catalog coverage.

Before a change is proposed:

- The contract it adds, strengthens or repairs is stated in one sentence.
- A test fails on the previous behaviour and passes on the new one.
- GPU-owned code was proved on a device.
- No stub, `todo!`, placeholder branch, or silent default return is introduced.
- No allocation, copy, blocking wait or unbounded growth is added to a dispatch
  path.
- A public API break carries its migration.
- The document that owns a changed claim changed with it, and the gate in
  `--subset docs` is green.
- One changelog fragment records the change.

## Changelog fragments

An observable change adds one file under `release/changes/unreleased/`, named by
its id, with exactly two keys:

```toml
category = "Fixed"
text = "One sentence naming what changed and what it changed from."
```

The categories are `Added`, `Changed`, `Deprecated`, `Removed`, `Fixed` and
`Security`. Any other value, and any third key, is a finding. `CHANGELOG.md` is
generated from the fragments and is never edited by hand.

## Commits

Split by contract, not by file. One commit changes one contract and carries
everything that contract needs: the source, its tests, the document that owns
the claim, and the changelog fragment. A reviewer reads the commit and sees the
whole change; a bisect lands on a tree that builds.

Keep a rename, a move and a reformat in their own commits, away from a
behavioural change. A diff that mixes them hides the behaviour inside the noise.

Write the message as what the change makes true, in the present tense, and name
the defect it closes rather than the files it touched.

## Where the manuals are

- [Architecture](docs/ARCHITECTURE.md): the layers and the production route.
- [Crate boundaries](docs/architecture/crates.md): what each crate owns and
  what it must not hold.
- [The placement rule](docs/lego-block-rule.md): which crate a new operation
  belongs in.
- [Add an operation](docs/extending/operation.md) and
  [add a backend](docs/extending/backend.md): the extension contracts, with a
  buildable example for each.
- [Conformance](docs/conformance/program.md): what a backend must prove.
- [Release](docs/release/process.md): the release train, the gates, the
  evidence.
- [THESIS.md](THESIS.md): the design argument behind the boundaries.

The book starts at [the summary](docs/SUMMARY.md).

## Security

Report a vulnerability through [SECURITY.md](SECURITY.md). Do not put exploit
details, credentials or private test targets in a public issue or pull request.

[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) applies to every interaction here.

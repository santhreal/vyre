# Script assertion ledger

`scripts/` holds 22 tracked files: 19 shell scripts and 3 Python scripts.
Each one is recorded below with its assertions, what makes it exit nonzero, every
caller found in the tree, whether the files it reads still exist, and the gate
that owns its assertions after the port. The rows carry 76 assertions and
19 findings.

A script leaves this document by being deleted: its rule belongs to a registered
gate, so the row is a record of a port that is finished, not of a file that still
runs. The ledger is empty when the registry owns every rule.

## Totals

- Files: 22. Assertions: 76. Findings: 19.
- Files whose subject is partly or wholly gone: 1.
- Files nothing invokes: 6.

### Subject gone

- `scripts/final-launch.sh`

### Nothing invokes it

- `scripts/apply-branch-protection.sh`
- `scripts/check_metal_macbook.sh`
- `scripts/check_signed_conformance_certificate.sh`
- `scripts/final-launch.sh`
- `scripts/install_wire_precommit_hook.sh`
- `scripts/wire_ci_local.sh`

## Rows

### `scripts/apply-branch-protection.sh`

Subject: present.

Invoked by: nothing; named in .github/CI_REQUIRED.md, .github/CODEOWNERS and six release evidence artifacts that record it as the source of the branch protection state.

Gate: xtask/src/gates/ci_contract.rs owns every assertion, and the script runs the `ci-required` gate before it applies anything. What is left is the gh mutation and its payload, which is an operator action against the GitHub API and not a rule. The six assertions about the workflow set that used to run only when an operator applied branch protection by hand are now a gate the sweep runs on every push, so this row has no findings left.

Assertions:

- gh is available and the repository is santhreal/vyre before anything is applied.
- The `ci-required` gate passes.

Exits nonzero on:

- gh missing
- wrong repository
- a failing `ci-required` gate

### `scripts/check_cuda_parity_perf_gate.sh`

Subject: present.

Invoked by: gpu-parity.yml.

Gate: xtask/src/gates/gpu_parity.rs.

Assertions:

- nvidia-smi succeeds, so the gate never reports a clean device without one.
- At least one tracked test target exists directly under vyre-driver-cuda/tests.
- At least one of those targets is named *gpu_parity*.
- The whole vyre-driver-cuda test suite passes on the live device.

Exits nonzero on:

- nvidia-smi failure
- no tracked test target
- no gpu_parity target
- any test failure

### `scripts/check_external_ir_extension_ci.sh`

Subject: present.

Invoked by: gates.yml.

Gate: xtask/src/gates/example_contract.rs.

Assertions:

- examples/external_ir_extension/Cargo.toml exists.
- The demo's tracked Rust source is at most 200 lines.
- The demo declares its own [workspace], so it cannot inherit workspace state.
- `cargo check --locked` on the demo manifest succeeds.

Exits nonzero on:

- missing manifest
- over the 200 line cap
- no [workspace] declaration
- check failure

### `scripts/check_feature_msrv.sh`

Subject: present.

Invoked by: gates.yml.

Gate: xtask/src/gates/feature_msrv.rs.

Assertions:

- Delegates to scripts/lib/check_feature_msrv.py after selecting the cargo runner, forwarding an optional --list.

Exits nonzero on:

- whatever check_feature_msrv.py exits nonzero on

### `scripts/check_metal_macbook.sh`

Subject: present.

Invoked by: nothing; the path appears as a string in vyre-bench/src/cli/bundle.rs.

Gate: xtask/src/gates/metal_remote.rs.

Assertions:

- A mode argument of driver, correctness, conformance, benchmark or all.
- VYRE_MACBOOK_SSH and VYRE_MACBOOK_VYRE_ROOT are set.
- vyre-driver-metal tests pass on the Apple GPU host.
- vyre-conform with the gpu feature passes with VYRE_BACKEND=metal.
- The smoke case runs on cpu-ref, wgpu and metal, each report is non-empty and passes validate-report.
- The metal report carries 16 named counters, and the resident-queue-closure report carries those plus three closure counters.
- wgpu-to-metal and cpu-ref-to-metal comparisons are produced, non-empty, and pass validate-comparison.
- Each comparison text carries baseline/candidate backend, profile backend, timing quality, compare exit code and the case id.
- The bundle manifest validates in both directions and carries schema, validator, suite, case id, both backends, comparison pairs, both fingerprints, an artifact count of 7 and a bundle hash.

Exits nonzero on:

- unknown mode
- missing SSH configuration
- any remote test failure
- an empty report or comparison
- a missing counter or manifest field
- a validate-report, validate-comparison or validate-benchmark-bundle failure

Findings:

- Nothing invokes it, so the entire Apple-GPU parity and benchmark-bundle contract runs only when an operator types it.
- Roughly 60 assertions are `grep -q` over JSON, so a counter renamed inside a nested object still matches, and a field present with a null value passes. The gate parses the JSON and asserts the fields.
- The artifact count of 7 is a literal in a grep pattern, so adding an eighth artifact fails with a message about a missing string rather than about the count.

### `scripts/check_signed_conformance_certificate.sh`

Subject: present.

Invoked by: nothing; named in conform/vyre-conform/tests/cert_artifact/release_script_contracts.rs.

Gate: xtask/src/gates/release_contract.rs.

Assertions:

- prove-release-shards.sh produces a non-empty merged conformance certificate.

Exits nonzero on:

- empty or missing merged certificate

Findings:

- It sets `export RUSTC_WRAPPER=""`, a build-affecting variable outside .cargo/config.toml. The gate does not set it; if the wrapper breaks a release build that belongs in the config file.

### `scripts/check_spirv_parity_perf_gate.sh`

Subject: present.

Invoked by: gates.yml.

Gate: xtask/src/gates/gpu_parity.rs.

Assertions:

- spirv-val is on PATH, because an unvalidated blob with a correct header is the defect the gate exists for.
- vyre-driver-spirv spirv_parity passes with the spirv-val feature enabled.

Exits nonzero on:

- spirv-val missing
- test failure

### `scripts/final-launch.sh`

Subject: partly gone: the release notes it passes to `gh release create` are docs/release/v<version>.md, and docs/ carries no Markdown after b1ed746d1c.

Invoked by: nothing; it is the manual launch entry point, and it is named by xtask/src/release/launch_state.rs, release/release-train.toml, release/vyre-release-evidence.toml, docs/optimization/OWNERSHIP.toml and conform/vyre-conform/tests/cert_artifact/release_script_contracts.rs.

Gate: xtask/src/gates/release_contract.rs owns every precondition; the publish, tag, push and gh release actions stay operator actions.

Assertions:

- Takes no argument other than --preflight.
- VYRE_RELEASE_APPROVED equals the token derived from release/release-train.toml, so a launch cannot happen without explicit approval.
- jq and gh are present and gh is authenticated.
- The origin remote matches the release repository named in the release train.
- HEAD is on main and not detached.
- The working tree is clean.
- Neither release tag already exists locally or on origin.
- The public repository's GitHub visibility is PUBLIC.
- The sharded all-backend conformance certificate is produced and copied to release evidence non-empty.
- vyre-release-gate passes before publish and again after.
- The launch completion evidence and launch state are written and committed.

Exits nonzero on:

- unknown argument
- missing approval token
- jq or gh missing or unauthenticated
- origin mismatch
- detached HEAD or non-main branch
- dirty tree
- tag already present locally or remotely
- public repository not PUBLIC
- empty conformance evidence
- release gate failure

Findings:

- `gh release create --notes-file docs/release/v<version>.md` names a path that no longer exists, so the launch chain fails at the release-creation step after it has already published to crates.io and pushed two tags. That is the worst possible place for a dead path and no gate covers it. The ported gate asserts the notes file for the configured version exists before any of it starts.
- It passes `-j1` to three cargo invocations, a build-affecting flag outside .cargo/config.toml.
- It calls `launch-state --output release/evidence/final/public-launch-state.json`; under the evidence lane's contract that becomes `launch-state --write` against the fixed path.

### `scripts/install_wire_precommit_hook.sh`

Subject: present.

Invoked by: nothing; named in CONTRIBUTING.md.

Gate: not a gate: it installs a local git hook. Its one assertion is reported and the hook target moves to `xtask sweep` once an operator decides how the hook is installed.

Assertions:

- The pre-push hook path is not an existing non-symlink file before the symlink is written.

Exits nonzero on:

- hook path exists and is not a symlink

Findings:

- Its one assertion protects an operator's existing hook. It is a local install action, not a tree property, so no gate can own it. It is the one file in this layer whose deletion needs an operator decision about how the hook is installed.

### `scripts/lib/cargo_runner.sh`

Subject: present.

Invoked by: 21 scripts.

Gate: not ported; the runner choice becomes std::process::Command on ./cargo_full inside the gates that need cargo.

Assertions:

- Selects VYRE_CARGO_RUNNER, then ./cargo_full, then cargo.

Exits nonzero on:

- never; it only assigns

Findings:

- Nothing remains open.

### `scripts/lib/check_feature_msrv.py`

Subject: present.

Invoked by: check_feature_msrv.sh from gates.yml.

Gate: xtask/src/gates/feature_msrv.rs.

Assertions:

- [workspace.package].rust-version is declared.
- rustup is runnable and the MSRV toolchain is installed.
- Every publishable member with features compiles on the MSRV under default features, no default features, and each feature alone.

Exits nonzero on:

- missing rust-version
- rustup missing or failing
- MSRV toolchain not installed
- empty members
- no publishable member declares a feature
- any matrix entry failing to compile

Findings:

- --list mode checks nothing and exits 0. It is a listing, not a gate mode, and the gate exposes it as a note rather than a pass.

### `scripts/lib/read_toml_values.py`

Subject: present.

Invoked by: toml_reader.sh, gates.yml.

Gate: replaced by typed TOML reads in xtask/src/gates/release_contract.rs.

Assertions:

- Every requested dotted key exists in the manifest.
- Every requested key resolves to a scalar, never a table or array.

Exits nonzero on:

- manifest unreadable
- key missing
- key not scalar

### `scripts/lib/release_train.sh`

Subject: present.

Invoked by: final-launch.sh, publish-release.sh.

Gate: xtask/src/gates/release_contract.rs.

Assertions:

- release/release-train.toml defines versions.vyre, tags.vyre_rc, tags.vyre and release_groups.vyre.repository, all scalar.

Exits nonzero on:

- any of the four keys missing or non-scalar

### `scripts/lib/repo_boundary.sh`

Subject: present.

Invoked by: final-launch.sh.

Gate: xtask/src/gates/release_contract.rs.

Assertions:

- release/repo-boundary.toml defines public_repository, private_repository, verify_public_repo_action and boundary_description, all scalar.

Exits nonzero on:

- any of the four keys missing or non-scalar

### `scripts/lib/sweep_targets.py`

Subject: present.

Invoked by: run_sweep_oracle_matrix.sh, run_volume_sweep_shard.sh, gates.yml.

Gate: xtask/src/gates/sweep_targets.rs.

Assertions:

- The root Cargo.toml declares workspace members.
- Every tracked <crate>/tests/sweep_*.rs sits in a declared workspace member.
- At least one tracked sweep source exists.
- Every [[test]] entry named sweep_* has a tracked source file.
- Every required-features entry names a feature the crate defines.
- The requested partition is non-empty.

Exits nonzero on:

- wrong argument count or unknown kind
- empty members
- sweep source outside members
- no tracked sweep source
- [[test]] entry with no source
- required-features naming an undefined feature
- empty partition

Findings:

- Every one of its six refusals exists because a runner that silently runs nothing reports success forever. They are the model for the whole port and all six become findings.

### `scripts/lib/toml_reader.sh`

Subject: present.

Invoked by: release_train.sh, repo_boundary.sh.

Gate: replaced by toml::from_str into toml::Table in the release gate.

Assertions:

- Caller passed MANIFEST, LABEL, EXPECTED_COUNT and exactly EXPECTED_COUNT keys.
- python3 is on PATH and read_toml_values.py exists.
- The reader produced exactly EXPECTED_COUNT values.

Exits nonzero on:

- argument count mismatch
- python3 missing
- reader missing
- value count mismatch

### `scripts/prove-release-shards.sh`

Subject: present.

Invoked by: final-launch.sh, check_signed_conformance_certificate.sh.

Gate: xtask/src/gates/release_contract.rs owns the certificate assertion; the sharded proof run stays an operator action.

Assertions:

- VYRE_RELEASE_SHARDS and VYRE_RELEASE_SHARD_WORKERS are positive integers.
- VYRE_RELEASE_PROFILE is debug or release.
- The vyre-conform binary exists and is executable after the build.
- Every shard worker succeeds.
- The merged certificate is written and its path printed.

Exits nonzero on:

- non-integer shard count or worker count
- invalid profile
- missing conform binary
- any shard worker failing

Findings:

- Nothing remains open.

### `scripts/publish-release.sh`

Subject: present.

Invoked by: final-launch.sh.

Gate: xtask/src/gates/release_contract.rs owns the readiness assertions; the publish loop stays an operator action.

Assertions:

- Takes no argument other than --preflight.
- VYRE_RELEASE_APPROVED equals the publish token derived from the release train.
- jq is present.
- Package readiness reports zero blockers.
- publish_order is non-empty.
- Every crate version becomes visible in the crates.io index before the next publish.

Exits nonzero on:

- unknown argument
- missing approval token
- jq missing
- any blocker
- empty publish order
- a publish failure
- index wait timeout

Findings:

- It passes `-j1` to cargo run.
- `package-readiness --output <path>` becomes `package-readiness --write` under the evidence lane's contract, reading the fixed path release/evidence/package/publish-readiness.json.

### `scripts/run_sweep_oracle_matrix.sh`

Subject: present.

Invoked by: gates.yml.

Gate: xtask/src/gates/sweep_targets.rs owns the roster assertions; the test run is a gate that shells into cargo.

Assertions:

- The matrix partition of the sweep roster is non-empty.
- Every tracked sweep_* oracle-matrix integration test passes, invoked per crate with the union of the required-features its own targets declare.

Exits nonzero on:

- empty roster
- any test failure

Findings:

- The reason it exists is that ci.yml runs the workspace suite with default features, so a test whose required-features name a non-default feature is silently skipped, and strict.yml builds --all-features without running anything. The roster derivation is what keeps a new sweep from being unrun, and it is preserved.

### `scripts/run_volume_sweep_shard.sh`

Subject: present.

Invoked by: gates.yml, run_sweep_oracle_matrix.sh.

Gate: xtask/src/gates/sweep_targets.rs.

Assertions:

- Shard index and shard count are non-negative integers, count is at least 1, and index is below count.
- The volume partition of the roster is non-empty.
- Shard count does not exceed the number of volume targets, so no shard runs nothing.
- The shard selects at least one target.
- Every selected volume wave passes.

Exits nonzero on:

- non-integer or out-of-range shard arguments
- empty roster
- shard count above target count
- a shard selecting nothing
- any test failure

Findings:

- A shard index outside the shard count used to select nothing and exit 0. The four assertions that close that are the valuable part and are preserved as findings.

### `scripts/wait-crates-index.sh`

Subject: present.

Invoked by: publish-release.sh.

Gate: not a gate: it polls an external registry during a publish. It stays an operator action inside the publish loop.

Assertions:

- Both crate and version arguments are given.
- VYRE_CRATES_INDEX_MAX_ATTEMPTS and VYRE_CRATES_INDEX_INTERVAL_SECONDS are positive integers.
- cargo_full is available.
- The crates.io index exposes the exact `crate = "version"` line within the attempt budget.

Exits nonzero on:

- missing arguments
- non-integer or non-positive tuning value
- cargo_full unavailable
- index timeout

Findings:

- Its assertions are all about its own arguments and an external service, so no tree property is lost when it goes. It is the one file here whose subject is not this repository.

### `scripts/wire_ci_local.sh`

Subject: present.

Invoked by: nothing; named in CONTRIBUTING.md and install_wire_precommit_hook.sh.

Gate: xtask/src/gates/wire_contract.rs.

Assertions:

- fmt, clippy with -D warnings, check, and eight named test targets across vyre-primitives and vyre-libs all pass.
- The wire contract suite produces identical test order across two runs, and both runs passed.

Exits nonzero on:

- any step failing
- a determinism diff between the two runs

Findings:

- It exports CARGO_INCREMENTAL=0, a build-affecting variable outside .cargo/config.toml, with a comment saying it mirrors CI. If local and CI builds must agree, that belongs in the config file.
- The determinism check previously ended in `|| true`, so two identically failing runs diffed empty and reported success. That repair is the shape to keep: determinism is only meaningful across two runs that passed.

## Injection matrix

Every ported gate has to go red on the same input the script it replaced failed
on. The gate lanes are one atomic cutover and nothing in it compiles alone, so
these injections run once against the merged tree. Each row is one edit, the gate
that must go red, and the number it must move to. Apply the edit, run the gate,
confirm the number, revert the edit, confirm the pin again. A gate that stays
green under its injection is not covering the assertion it inherited, whatever its
pin says.

The `findings` column is the count with the injection applied, given the pin in
`xtask/gate-baselines.toml` at the time of writing.

| Gate | Injection | Findings |
| --- | --- | --- |
| `backend-extension` | In `vyre-driver-wgpu/src/backend_dispatch.rs`, delete the `BackendPrecedence` inventory submission block. | 0 to 1 |
| `backend-extension` | In `vyre-driver/src/backend/registry/acquire.rs`, rename `registered_backends_by_precedence_slice` at its definition and its call. | 0 to 1 |
| `backend-extension` | Add `let id = "cuda";` to any file under `vyre-driver/src/backend/registry/`. | 0 to 1 |
| `readback-ring` | In `vyre-driver-wgpu/src/engine/record_and_readback/mod.rs`, rename `.arm_ticket(` to `.arm(` at its definition and its call sites. | 0 to 1 |
| `readback-ring` | In `vyre-driver-wgpu/src/lib.rs`, replace `ReadbackRingSet::new()` with `ReadbackRingSet::default()`. | 0 to 1 |
| `program-wire-fields` | Add `pub scratch_hint: u32,` to `Program` in `vyre-foundation/src/ir_inner/model/program/definition.rs`. | 0 to 1 |
| `program-wire-fields` | Delete every mention of `workgroup_size` from `vyre-foundation/src/serial/wire/encode/to_wire/mod.rs` and `vyre-foundation/src/serial/wire/decode/from_wire/mod.rs`. | 0 to 1 |
| `program-wire-fields` | Rename `pub struct Program` to `pub struct ProgramInner`. | gate errors, which is the intended outcome: the declaration is located, not named, so losing it is unmeasurable rather than clean |
| `frozen-contracts` | Add a method to `pub trait ExprVisitor` in `vyre-foundation/src/visit/expr_visitor/mod.rs`. | 1 to 2 |
| `frozen-contracts` | Delete `docs/frozen-traits/MutationClass.txt`. | 1 to 2 |
| `frozen-contracts` | Reindent the body of `pub enum AlgebraicLaw` by four spaces. | stays 1; indentation is not part of the contract |
| `file-size` | Append 200 blank lines to `vyre-foundation/src/optimizer/fact_cache/mod.rs` (measured 570, cap 599). | 75 to 76 |
| `file-size` | Append 60 lines to `vyre-libs/src/decode/inflate.rs` (measured 554, cap 582). | 75 to 76 |
| `file-size` | Add a row to the audit ceilings naming vyre-does-not-exist/src/lib.rs, a path the tree does not hold. | 75 to 76 |
| `gpu-loudness` | Add `#[cfg(not(feature = "gpu"))]` above a test in `vyre-driver-wgpu/tests/` with no loud abort within ten lines above or twenty below. | 2 to 3 |
| `gpu-loudness` | Add `if adapter.is_err() { return; }` to a test body. | 2 to 3 |
| `gpu-loudness` | Add `Backend::acquire_or_panic();` five lines below an existing finding site in `conform/vyre-conform/tests/cert_artifact/prove_failure_contracts.rs`. | 2 to 1, which is the allowance working rather than a failure |
| `unification` | Add a second `pub fn child_bodies` to any file under `vyre-foundation/src`. | 0 to 2, because a row over its ceiling reports every site |
| `unification` | Add `BufferAccess::infer(` to a file under `vyre-runtime/src/resident_work_queue`. | 0 to 1 |
| `unification` | Rename the directory `vyre-foundation/src/execution_plan` and update its `mod` declaration. | 0 to 1, reported as a path that does not exist rather than as a clean row |
| `evidence-paths` | In any artifact under `release/evidence`, change one cited path to a filename that does not exist but keeps a tree extension. | 18 to 19 |
| `evidence-paths` | Add `"manifest": "target/debug/build.rs"` to an artifact object, with `target/` gitignored. | 18 to 19, in the gitignored class rather than the missing class |
| `evidence-paths` | Change a cited path to `1.2.0`. | stays 18; a version string is not a citation |
| `docs-coupling` | Delete the `covers` key from the `docs/reference/wire-format.md` page row. | 0 to 1 |
| `docs-coupling` | Change one `covers` entry to a path the tree does not hold. | 0 to 1 |
| `docs-coupling` | In `docs/architecture/parsing.md`, change a cited source path inside a code span to one that does not exist. | 0 to 1 |
| `docs-coupling` | Edit a file `docs/reference/wire-format.md` covers without editing that page. | 0 to 2, one for the page and one for the missing changelog fragment |
| `docs-coupling` | Run it with `--base` naming a ref the checkout does not hold. | 0 to 1, reported as an unreachable base rather than as a gate that could not run |
| `example-capability` | In `examples/libs-template/Cargo.toml.liquid`, change `license` to `{{license}}`. | 0 to 1, naming the placeholder the gate cannot render |
| `example-capability` | In `examples/external_backend_extension/tests/backend_probe.rs`, change the expected dispatch output to `vec![9, 9, 9, 9]`. | 0 to 1, naming the failing test |
| `example-capability` | Delete the `[workspace]` table from `examples/external_backend_extension/Cargo.toml`. | 0 to 3, one for the isolation rule and one for each cargo invocation that then refuses to resolve |
| `example-capability` | Move `examples/external_backend_extension/Cargo.lock` aside so `--locked` cannot resolve. | 0 to 2, one per cargo invocation |
| `example-capability` | Track a file under a new `examples/<name>/` directory and give it no manifest. | 0 to 1 |
| `example-capability` | Track a 400-line Rust file and no lockfile under a new `examples/<name>/` directory carrying a manifest. | 0 to 2, one for the line cap and one for the missing lockfile |
| `invariant-paths` | In `vyre-spec/src/invariants.rs`, change a cited conformance test path to one that does not exist. | 0 to 1 |
| `doc-claims` | In `contracts/doc_claims_manifest.toml`, change one `phrase` to text its document does not contain. | 0 to 1 |
| `doc-claims` | Delete the `test` key from one claim. | 0 to 1, reported as an incomplete row rather than as a missing test |
| `hot-path-owned-dispatch` | In `vyre-driver/src/backend/compiled_pipeline.rs`, make `dispatch` the required method and give `dispatch_borrowed` a default that copies each row with `to_vec`. | 0 to 2, one finding for the requirement and one for the copy it forces |
| `hot-path-inventory` | In `vyre-libs/src/operation_catalog.rs`, serve `convergence_contract` by walking `inventory::iter` instead of probing the frozen index. | 0 to 1, quoting the statement that scans |
| `hot-path-nested-rows` | In `vyre-driver/src/backend/compiled_pipeline.rs`, delete the `dispatch_borrowed_into` declaration from the compiled-pipeline trait. | 0 to 1, naming the returning method that is then the only shape offered |
| `hot-path-nested-rows` | In `vyre-driver/src/backend/vyre_backend.rs`, replace the slot-preserving replacement in `dispatch_borrowed_into` with `*outputs = self.dispatch_borrowed(program, inputs, config)?;`. | 0 to 1, naming the slot the default replaces |
| `hot-path-unbounded-read` | Add `fs::read_to_string(` to a file under `vyre-driver/src` outside the reviewed cache modules. | 1 to 3 |
| `lint-unsafe-budget` | Add `#![allow(unsafe_code)]` to a crate root not in the reviewed budget. | 0 to 1 |
| `lint-unsafe-justification` | Add an `unsafe {` block with no SAFETY comment above it. | 2 to 3 |
| `lint-missing-docs-override` | Add `#![allow(missing_docs)]` to any crate root. | 0 to 1 |
| `proptest-coverage` | Delete two property-test files, taking the count from 182 to 180 against the floor of 181. | 0 to 1 |
| `repo-hygiene` | Add a second `BACKLOG.md` under any crate directory and track it. | 2 to 3 |
| `shader-source` | Add `out.push_str("@compute");` to a file under `vyre-driver-wgpu/src`. | 0 to 1 |
| `layering` | Add `vyre-lints.workspace = true` to `[dependencies]` in `vyre-spec/Cargo.toml`. | 0 to 29, one per member that reaches it, each naming the chain |
| `layering` | Add `wgpu.workspace = true` to `[dependencies]` in `vyre-spec/Cargo.toml`. | 0 to 48, every neutral member that reaches vyre-spec |
| `layering` | Delete the `[[crate]]` entry for `vyre-spec` from `docs/CRATE_OWNERSHIP.toml`. | gate errors: an unregistered member has an empty closure, so the roster is unreviewed rather than clean |
| `layering` | Change `layer` on the `vyre-spec` entry to a name NEUTRAL_LAYERS does not hold. | gate errors: a layer with no neutrality decision would be skipped |
| `layering` | Add `("invented", true)` to `NEUTRAL_LAYERS`. | gate errors: a decision no member uses is an allowance nothing needs |
| `layering` | Add `"not-a-crate"` to `BACKEND_APIS`. | gate errors: a boundary named after a crate the workspace never resolves cannot be crossed |
| `neutral-crates` | Add `naga.workspace = true` to `[dependencies]` in `vyre-primitives/Cargo.toml`. | 1 to 2, and the dotted form is what the shell rule could not see |
| `neutral-crates` | Add `vyre-ir = "0.1"` to `vyre-foundation/fuzz/Cargo.toml`, which the workspace excludes and cargo never loads with the graph. | 1 to 2 |
| `neutral-crates` | Add `"vyre-gone"` to `NEUTRAL_CRATES`. | gate errors: a neutrality rule over a manifest that does not exist reports success forever |

Three of these are negative controls rather than injections: the reindented
frozen enum, the version string cited as a path, and the loud abort added beside a
loudness finding. A gate that moves on those is over-reporting, which mutes it
just as surely as under-reporting.

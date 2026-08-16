# Script assertion ledger

`scripts/` holds 36 tracked files: 26 shell scripts and 10 Python scripts.
Each one is recorded below with its assertions, what makes it exit nonzero, every
caller found in the tree, whether the files it reads still exist, and the gate
that owns its assertions after the port. The rows carry 148 assertions and
35 findings.

A script leaves this document by being deleted: its rule belongs to a registered
gate, so the row is a record of a port that is finished, not of a file that still
runs. The ledger is empty when the registry owns every rule.

## Totals

- Files: 36. Assertions: 148. Findings: 35.
- Files whose subject is partly or wholly gone: 2.
- Files nothing invokes: 12.

### Subject gone

- `scripts/bench/cross_backend_comparison.sh`
- `scripts/final-launch.sh`

### Nothing invokes it

- `scripts/apply-branch-protection.sh`
- `scripts/bench/cross_backend_comparison.sh`
- `scripts/bench_smoke.sh`
- `scripts/check_bench_baselines.sh`
- `scripts/check_metal_macbook.sh`
- `scripts/check_signed_conformance_certificate.sh`
- `scripts/crate_ownership.py`
- `scripts/crate_readmes.py`
- `scripts/final-launch.sh`
- `scripts/install_wire_precommit_hook.sh`
- `scripts/testing_guides.py`
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

### `scripts/architecture_docs.py`

Subject: present: the one document it validates, docs/ARCHITECTURE.md, is tracked, as are docs/DOCS.toml, docs/generated/OP_SCHEMA.json, docs/optimization/OWNERSHIP.toml, docs/CRATE_OWNERSHIP.toml, release/release-train.toml and the backend evidence. CURRENT_DOCS carried five documents and an RFC when this row was written; four of the six were deleted and the list was trimmed to the survivor, so the assertions below that name a second document no longer apply.

Invoked by: architectural-invariants.yml, xtask/tests/tree_contracts/architecture_docs.rs; version-coupled by xtask-registry/src/docs/operation_schema.rs and cited by docs/optimization/OWNERSHIP.toml and xtask/src/release/conformance_workflows.rs.

Gate: xtask/src/gates/manifest_contract.rs takes the workspace, schema, backend evidence and lane assertions; xtask/src/gates/doc_contract.rs takes the five documents, their tokens, their dates and their forbidden patterns, and reports each missing document as one finding.

Assertions:

- workspace.members is a non-empty array of explicit paths and includes vyre-megakernel.
- release/release-train.toml declares versions.vyre.
- docs/generated/OP_SCHEMA.json declares schema_version 4 and is internally coherent: operation_count equals the operation row count and the tier counts sum to it.
- release/evidence/backends/backend-matrix.json has an empty blockers array and a preferred_backend_id with a matching probe row.
- docs/CRATE_OWNERSHIP.toml carries a vyre-megakernel crate row whose responsibility names ProgramGraph, and keeps no planned.vyre-megakernel entry.
- docs/optimization/OWNERSHIP.toml declares [lane.*] tables, each with a purpose, a layer, at least one write glob and at least one required command; every write and avoid pattern is repository-relative and matches something in the tree; every -p in a required command names a package a workspace manifest declares.
- docs/DOCS.toml classifies the four current architecture documents as current and the megakernel RFC as superseded.
- Each of the four architecture documents and the RFC carries a Last verified date and the current Vyre version.
- The RFC states Status: **Superseded**.
- None of the five documents retains a stale architecture pattern: 0.6.x, nine-op, WGPU as primary production path, Four CI laws, a codex identifier, or seven phrasings that describe vyre-megakernel as planned.
- Each of the five documents contains its required architecture tokens.

Exits nonzero on:

- any of the above, one at a time; validate raises on the first failure and never reports a second

Findings:

- architectural-invariants.yml fails on every tree today. read_text on docs/ARCHITECTURE.md raises before any live authority is read, so the six assertions that would still pass never run. Splitting them across two gates is what makes the surviving ones reachable again.
- validate raises on the first failure, so a tree with ten violations reports one. The gate collects findings instead, which is also what makes the pinned count meaningful.
- OPERATION_SCHEMA_VERSION = 4 is duplicated here and in xtask-registry/src/docs/operation_schema.rs. The gate reads the Rust constant instead of restating the number.

### `scripts/bench/cross_backend_comparison.sh`

Subject: gone: it wrapped a registered subcommand whose table now lives in the committed release evidence.

Invoked by: nothing; the path appeared in .gitignore only, and that stanza is gone too.

Gate: reported: it asserts nothing and its output directory is no longer published.

Assertions:

- Runs `xtask bench-crossback` for xor-1k and xor-1m and writes tables under docs/perf/.
- Both programs are gone with it. The gate derives the comparison from the committed release benchmark evidence and records one table under release/evidence/benchmarks/.

Exits nonzero on:

- either run failing

Findings:

- It is a wrapper around a registered subcommand and wrote generated Markdown into a gitignored directory, so a fresh checkout was red and one local run turned it green. Nothing invoked it.

### `scripts/bench_smoke.sh`

Subject: present.

Invoked by: nothing; named in CONTRIBUTING.md.

Gate: xtask/src/gates/bench_contract.rs already runs the smoke suite under a budget, so this wrapper carries no assertion of its own.

Assertions:

- Runs the vyre-bench smoke suite. Asserts nothing itself beyond the run succeeding.

Exits nonzero on:

- any bench failure

### `scripts/check_bench_baselines.sh`

Subject: present.

Invoked by: nothing; named in benches/RESULTS.md and the changelog only.

Gate: xtask/src/gates/bench_contract.rs.

Assertions:

- benches/RESULTS.md exists and carries machine:, gpu:, cpu:, rustc: and commit: fields.
- Every crate owning at least one benches/*.rs source has a `### <crate>` section in benches/RESULTS.md.

Exits nonzero on:

- missing RESULTS.md
- missing header field
- crate with a bench target and no section

Findings:

- Nothing invokes it. A published-baseline claim that no workflow checks is a claim.
- It walks the filesystem with `find` rather than tracked files, so an untracked benches/*.rs in a dev tree demands a section that CI never asks for.

### `scripts/check_bench_smoke_runtime.sh`

Subject: present.

Invoked by: bench-regression.yml.

Gate: xtask/src/gates/bench_contract.rs.

Assertions:

- contracts/perf_targets.toml declares a budget for crates.vyre-bench.targets.smoke_runtime.
- vyre-bench builds and the built binary is executable at the metadata target directory.
- `vyre-bench list --format json` succeeds.
- One smoke case (foundation.elementwise.add.1m, 30 measured samples) completes within the declared budget.

Exits nonzero on:

- missing budget
- missing or non-executable bench binary
- list failure
- wall clock over budget

Findings:

- The budget is parsed with awk over TOML text rather than a TOML reader, so a budget declared inline or with a comment on the line is read wrongly or not at all.

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

### `scripts/check_deep_bench_coverage.sh`

Subject: present.

Invoked by: gates.yml.

Gate: xtask/src/gates/bench_contract.rs.

Assertions:

- Delegates to scripts/lib/check_deep_bench_coverage.py after selecting the cargo runner.

Exits nonzero on:

- whatever check_deep_bench_coverage.py exits nonzero on

### `scripts/check_docs_references.py`

Subject: present: docs carries 41 tracked Markdown documents again, plus root Markdown, .github Markdown and the crate READMEs.

Invoked by: xtask/tests/docs_references.rs.

Gate: xtask/src/gates/doc_contract.rs.

Assertions:

- Every Markdown document in scope resolves each of its relative links, so no published document points at a path a reader cannot open.

Exits nonzero on:

- any unresolvable reference

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

### `scripts/check_public_api_snapshot.sh`

Subject: present (docs/public-api/*.txt survive; they are .txt, not the deleted mdbook).

Invoked by: public-api.yml.

Gate: xtask/src/gates/public_api.rs, with --refresh becoming ctx.write on a generating gate.

Assertions:

- public_api_snapshot_inventory.py resolves at least one publishable crate.
- `cargo public-api -sss -p <crate>` succeeds for every publishable crate with a src directory.
- Every publishable crate has a docs/public-api/<package>.txt snapshot.
- Every snapshot file names a currently publishable package, so a stale snapshot is a failure.
- The extracted surface is byte-identical to the committed snapshot, under LC_ALL=C sort.
- --refresh takes an optional crate name and rejects a flag as its argument.
- --refresh prints the per-crate diff before installing it.

Exits nonzero on:

- inventory failure or empty
- cargo public-api failure
- missing snapshot
- unowned snapshot
- surface drift
- unknown argument
- unknown crate name to --refresh

Findings:

- `set -uo pipefail` without `-e`.
- The refresh path writes into docs/public-api from whatever the tree holds at that instant. Under the contract the write half is ctx.write on the gate that owns the artifact, which keeps the diff print and the per-crate scoping.

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

### `scripts/crate_ownership.py`

Subject: present: docs/CRATE_OWNERSHIP.toml and both generated documents, docs/CRATE_GRAPH.md and docs/OWNERSHIP.md, are tracked.

Invoked by: nothing directly; consumed by xtask/src/gates/check_tier_deps.rs and xtask/src/gates/layering.rs, and named as the generator of docs/CRATE_GRAPH.md and docs/OWNERSHIP.md.

Gate: xtask/src/gates/ownership_registry.rs.

Assertions:

- docs/CRATE_OWNERSHIP.toml declares schema_version 2 and no `planned` table.
- Every [[crate]] row declares package, path, owner, layer and responsibility, and no removed allowed_dependencies field.
- Every [[crate.dependency]] declares package, purpose, features, conditions, kinds, optional, default_features, boundary and seam, with boundary public or private and kinds drawn from normal and build.
- No row duplicates a package, a path or a dependency package.
- The registry path set equals workspace.members and the package set equals the workspace packages.
- Each package's registry path matches its manifest location.
- Each crate's declared internal dependency set equals the set Cargo resolves, and every declared feature set, target condition set, kind set, optional flag and default-features flag matches Cargo exactly.
- Every dependency's declared seam equals the owner of the destination crate.
- docs/CRATE_GRAPH.md and docs/OWNERSHIP.md match the content generated from the registry.

Exits nonzero on:

- any registry schema or completeness failure
- path or package set mismatch
- dependency metadata drift
- a wrong seam
- a stale generated document

Findings:

- `--check` fails on every tree today, because it reads two generated Markdown documents that no longer exist. Its 30-odd registry assertions are the strongest manifest contract in the repository and none of them can pass while that read fails. The gate separates them: the registry assertions run, and the two missing generated documents are two findings with a pinned count.
- Nothing invokes it. The registry validation runs only through the Rust tree-contract test that shells into it.

### `scripts/crate_readmes.py`

Subject: present: it writes into crate READMEs, and the docs/testing/<crate>.md guides its generated block links are tracked again, 35 of them.

Invoked by: nothing directly; xtask/tests/tree_contracts/crate_readmes.rs shells into it, and all 35 crate READMEs name it as their generator.

Gate: xtask/src/gates/doc_contract.rs, which regenerates the crate-contract block under ctx.write.

Assertions:

- The ownership registry validates first, through crate_ownership.validate.
- docs/CRATE_GUIDES.toml declares schema_version 1 and defines profile and package tables.
- No package override names a crate the registry does not declare.
- No error profile names a layer no crate occupies.
- Every layer a crate occupies has an error profile.
- Every package override is a table.
- Every crate manifest declares a [package] table with a version, and a [features] table whose default is a string array.
- Every [[example]] row declares a name and a string-array required-features.
- Every crate's error_behavior text is non-empty.
- A release_status override is non-empty and uses only release-train version placeholders.
- release/release-train.toml declares [versions] including versions.vyre.
- No generated README contract contains a retired 0.4.x release claim.
- Every crate README exists and its generated crate-contract block matches the generated content.
- No README carries unbalanced or duplicate generated-contract markers, and none exceeds 2 MiB.

Exits nonzero on:

- a registry failure
- a guide-metadata failure
- an orphaned profile or unknown override
- a manifest without a version or a malformed features table
- a retired 0.4.x claim
- a missing or stale README contract block

Findings:

- Every generated block links docs/testing/<crate>.md for testing commands. All 31 of those guides were deleted at b1ed746d1c, so 31 crate READMEs point a reader at nothing. That is a live defect in tracked files, and the gate reports one finding per dangling guide link.
- It imports ContractError, load_registry, read_toml, validate and workspace_state from crate_ownership, so the registry contract and the README generator are one dependency chain. Both land in the same gate pair.

### `scripts/docs.sh`

Subject: present.

Invoked by: docs-ci.yml.

Gate: xtask/src/gates/rustdoc.rs.

Assertions:

- `cargo doc --no-deps --keep-going` succeeds, for the whole workspace or for the packages a diff touched.

Exits nonzero on:

- any rustdoc failure

Findings:

- `--changed-only` exits 0 when it finds no changed files and when it can resolve no affected package, so a run that documents nothing reports the same success as a run that documents everything. Under the contract the gate documents the workspace and reports what it built as a note.

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

### `scripts/lib/check_deep_bench_coverage.py`

Subject: present.

Invoked by: check_deep_bench_coverage.sh from gates.yml.

Gate: xtask/src/gates/bench_contract.rs.

Assertions:

- Each of five measured dimensions names a registered vyre-bench case id.
- vyre-driver-cuda/tests/module_cache_contracts.rs exists and defines repeated_dispatch_reuses_loaded_cuda_module.
- Every `--case <id>` reference in a tracked .yml/.yaml/.sh/.json/.toml/.md file names a registered case.
- The vyre-bench registry lists at least one case and every entry carries an id.

Exits nonzero on:

- bench list failure
- non-JSON registry
- empty registry
- missing representative case
- missing cache contract file or test name
- a case id passed to the runner that the registry does not contain

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

### `scripts/public_api_snapshot_inventory.py`

Subject: present.

Invoked by: check_public_api_snapshot.sh, semver-checks.yml.

Gate: xtask/src/gates/public_api.rs, which derives the same roster from the manifests it already walks.

Assertions:

- workspace.members is an explicit string array.
- Every member manifest is readable and declares a non-empty package.name.
- Every package.publish value is true, false, an empty array or an array of registry strings.
- Prints publishable members as directory:package, sorted by package name.

Exits nonzero on:

- wrong argument count
- unreadable manifest
- missing [package] or name
- unsupported publish value

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

### `scripts/testing_guides.py`

Subject: present: docs/CRATE_OWNERSHIP.toml, docs/testing/TESTING.toml, every crate manifest and all 35 docs/testing/*.md guides are tracked.

Invoked by: nothing directly; xtask/tests/tree_contracts/testing_guides.rs and docs_manifest_completeness.rs shell into it, and docs/DOCS.toml names it as the generator of every testing guide.

Gate: xtask/src/gates/doc_contract.rs, which regenerates the guides under ctx.write; xtask/src/gates/manifest_contract.rs owns the registry-to-members equality and the Cargo target enumeration.

Assertions:

- The ownership registry path set equals workspace.members.
- Every [[crate]] row declares a non-empty package, path, owner, layer and responsibility.
- docs/testing/TESTING.toml declares schema_version 1 and defines [defaults], [profile] and [package] tables.
- No package override names a crate the registry does not declare.
- No profile names a layer no crate occupies.
- Every layer a crate occupies has a profile.
- Every crate manifest's package.name equals the registry package.
- Every crate's merged metadata declares non-empty hardware, expected_skips and failure_behavior, string arrays test_classes and evidence_outputs, and a string-array commands.
- Every explicit Cargo target row is a table with a non-empty name, a string path and a string-array required-features.
- No two crates generate the same guide filename.
- docs/testing/ contains no guide that no workspace member owns.
- Every guide exists and matches the generated content, including the full Cargo target table for the crate.

Exits nonzero on:

- a registry or member mismatch
- a metadata schema failure
- an orphan profile or unknown override
- a package-name mismatch
- a guide filename collision
- a non-member guide in the directory
- a missing or stale guide

Findings:

- The extras assertion at line 397 globs docs/testing/*.md and reports guides no member owns. The directory is empty, so that assertion is now vacuous and stays vacuous until the guides come back.
- The stale assertion is the opposite: every one of the 31 expected guides is missing, so --check fails with 31 names. Both assertions survive in the gate, and the missing guides are the pinned findings.

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
| `program-wire-fields` | Delete every mention of `workgroup_size` from `vyre-foundation/src/serial/wire/encode/to_wire/mod.rs` and `decode/from_wire.rs`. | 0 to 1 |
| `program-wire-fields` | Rename `pub struct Program` to `pub struct ProgramInner`. | gate errors, which is the intended outcome: the declaration is located, not named, so losing it is unmeasurable rather than clean |
| `frozen-contracts` | Add a method to `pub trait ExprVisitor` in `vyre-foundation/src/visit/expr/mod.rs`. | 1 to 2 |
| `frozen-contracts` | Delete `docs/frozen-traits/MutationClass.txt`. | 1 to 2 |
| `frozen-contracts` | Reindent the body of `pub enum AlgebraicLaw` by four spaces. | stays 1; indentation is not part of the contract |
| `file-size` | Append 200 blank lines to `vyre-foundation/src/optimizer/fact_cache/mod.rs` (measured 570, cap 599). | 75 to 76 |
| `file-size` | Append 60 lines to `vyre-libs/src/decode/inflate.rs` (measured 554, cap 582). | 75 to 76 |
| `file-size` | Add a row to the audit ceilings naming `vyre-does-not-exist/src/lib.rs`. | 75 to 76 |
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
| `hot-path-nested-rows` | In `vyre-libs/src/parsing/c/preprocess/gpu_pipeline/dispatch.rs`, delete the `dispatch_borrowed_into` declaration from `ProgramOracle`. | 0 to 1, naming the returning method that is then the only shape offered |
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

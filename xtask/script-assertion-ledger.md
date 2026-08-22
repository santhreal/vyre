# Script assertion ledger

`scripts/` is recorded here in full. Each row names one script, what it
asserted, what made it exit nonzero, every caller found in the tree, whether the
files it reads still exist, and the gate that owns its assertions after the port.
A row is kept after its script leaves the tree, so the port it records stays
checkable against the gate that carries it.

A row whose script has left the tree records a port that is finished: it names
the registered gate that carries the rule, and the injection that proved that
gate red. A row whose script is still tracked records an operator action, which
is something a person does to the world rather than a rule about the tree. Every
rule is in the registry once no row names a tracked file.

The counts below are generated from the rows and from the tracked files by the
`script-ledger` gate, which also holds every row to those two facts.

## Totals

- Rows: 36. Assertions: 140. Findings: 28.
- Tracked files: 9: 8 shell and 1 Python.
- Rows whose script has left the tree: 27.
- Tracked files nothing invokes: 2.

### Left the tree

- scripts/architecture_docs.py
- scripts/bench/cross_backend_comparison.sh
- scripts/bench_smoke.sh
- scripts/check_bench_baselines.sh
- scripts/check_bench_smoke_runtime.sh
- scripts/check_cuda_parity_perf_gate.sh
- scripts/check_deep_bench_coverage.sh
- scripts/check_docs_references.py
- scripts/check_external_ir_extension_ci.sh
- scripts/check_feature_msrv.sh
- scripts/check_metal_macbook.sh
- scripts/check_public_api_snapshot.sh
- scripts/check_signed_conformance_certificate.sh
- scripts/check_spirv_parity_perf_gate.sh
- scripts/crate_ownership.py
- scripts/crate_readmes.py
- scripts/docs.sh
- scripts/install_wire_precommit_hook.sh
- scripts/lib/cargo_runner.sh
- scripts/lib/check_deep_bench_coverage.py
- scripts/lib/check_feature_msrv.py
- scripts/lib/sweep_targets.py
- scripts/public_api_snapshot_inventory.py
- scripts/run_sweep_oracle_matrix.sh
- scripts/run_volume_sweep_shard.sh
- scripts/testing_guides.py
- scripts/wire_ci_local.sh

### Nothing invokes it

- `scripts/apply-branch-protection.sh`
- `scripts/final-launch.sh`

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

### scripts/architecture_docs.py

Subject: gone: the script is not in the tree. docs/ARCHITECTURE.md, docs/DOCS.toml, docs/generated/OP_SCHEMA.json, docs/optimization/OWNERSHIP.toml, docs/CRATE_OWNERSHIP.toml and release/release-train.toml are tracked, and the gates below read them.

Invoked by: nothing; architectural-invariants.yml runs `architecture-contract` and `docs-references` in its place, and xtask/tests/tree_contracts/architecture_docs.rs holds the same documents.

Gate: `workspace-membership` takes the workspace, schema and backend evidence assertions; `doc-claims` takes the five documents, their tokens, their dates and their forbidden patterns, and reports each missing document as one finding.

Injection: Broke the `Last verified` token in docs/ARCHITECTURE.md; `doc-claims` named the document, proved red.

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
- OPERATION_SCHEMA_VERSION = 4 is duplicated here and in xtask-registry/src/docs/operation_schema/mod.rs. The gate reads the Rust constant instead of restating the number.

### scripts/bench/cross_backend_comparison.sh

Subject: gone: the script is not in the tree, and it wrapped a registered subcommand whose table now lives in the committed release evidence.

Invoked by: nothing; the path appeared in .gitignore only, and that stanza is gone too.

Gate: `bench-crossback` derives the cross-backend comparison from the committed release benchmark evidence and holds the recorded table to it.

Injection: Added a row to release/evidence/benchmarks/cross-backend-comparison.md; `bench-crossback` reported the recorded table against the evidence, proved red.

Assertions:

- Runs `xtask bench-crossback` for xor-1k and xor-1m and writes tables under docs/perf/.
- Both programs are gone with it. The gate derives the comparison from the committed release benchmark evidence and records one table under release/evidence/benchmarks/.

Exits nonzero on:

- either run failing

Findings:

- It is a wrapper around a registered subcommand and wrote generated Markdown into a gitignored directory, so a fresh checkout was red and one local run turned it green. Nothing invoked it.

### scripts/bench_smoke.sh

Subject: gone: the script is not in the tree.

Invoked by: nothing.

Gate: `bench-smoke-runtime` runs the smoke suite under the budget declared in contracts/perf_targets.toml, so this wrapper carried no assertion of its own.

Injection: Cut the declared smoke budget to one millisecond; `bench-smoke-runtime` reported the measured run against the budget, proved red.

Assertions:

- Runs the vyre-bench smoke suite. Asserts nothing itself beyond the run succeeding.

Exits nonzero on:

- any bench failure

### scripts/check_bench_baselines.sh

Subject: gone: the script is not in the tree.

Invoked by: nothing; named in benches/RESULTS.md and the changelog only.

Gate: `bench-baselines` holds every crate with a bench target to a published section in benches/RESULTS.md.

Injection: Renamed the `commit:` field of benches/RESULTS.md; `bench-baselines` reported the missing field, proved red.

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

### scripts/check_bench_smoke_runtime.sh

Subject: gone: the script is not in the tree.

Invoked by: nothing; bench-regression.yml runs `bench-smoke-runtime` in its place.

Gate: `bench-smoke-runtime`.

Injection: Cut the declared smoke budget to one millisecond; `bench-smoke-runtime` reported the measured run against the budget, proved red.

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
### scripts/check_cuda_parity_perf_gate.sh

Subject: gone: the script is not in the tree. The CUDA parity targets it counted are tracked under vyre-driver-cuda, and docs/optimization/OP_MATRIX.toml still carries the budgets it read.

Invoked by: nothing; gpu-parity.yml runs `cuda-parity` in its place.

Gate: `cuda-parity` derives the parity roster from the manifests and holds the CUDA driver to it. gpu-parity.yml runs it, and --device runs the measured half on a CUDA host.

Injection: Pointed the gate's CUDA crate at a member with no gpu_parity target, and then at a member with no test target at all; `cuda-parity` reported both, proved red.

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

### scripts/check_deep_bench_coverage.sh

Subject: gone: the script is not in the tree.

Invoked by: nothing; gates.yml runs `bench-coverage` in its place.

Gate: `bench-coverage`.

Injection: Renamed the module-cache test the coverage dimension names; `bench-coverage` reported the uncovered dimension, proved red.

Assertions:

- Delegates to scripts/lib/check_deep_bench_coverage.py after selecting the cargo runner.

Exits nonzero on:

- whatever check_deep_bench_coverage.py exits nonzero on

### scripts/check_docs_references.py

Subject: gone: the script is not in the tree. The documents it read are tracked: docs Markdown, root Markdown, .github Markdown and the crate READMEs.

Invoked by: nothing; xtask/tests/docs_references.rs and architectural-invariants.yml exercise the `docs-references` gate in its place.

Gate: `docs-references` resolves every path a tracked document names.

Injection: Named a document the tree does not carry from xtask/README.md; `docs-references` reported the unresolvable reference, proved red.

Assertions:

- Every Markdown document in scope resolves each of its relative links, so no published document points at a path a reader cannot open.

Exits nonzero on:

- any unresolvable reference
### scripts/check_external_ir_extension_ci.sh

Subject: gone: the script is not in the tree. The external IR extension example and the workflow job that builds it are tracked.

Invoked by: nothing; gates.yml runs `example-capability` in its place.

Gate: `example-capability` holds every example to its declared capability and to one line cap. The script's 200-line cap on this one example did not survive: the examples measure 158, 246 and 149 lines against a cap of 300 that applies to all of them, and a cap only one example is held to is a cap nobody can move.

Injection: Padded an example past the line cap; `example-capability` reported the example, proved red.

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

### scripts/check_feature_msrv.sh

Subject: gone: the script is not in the tree. The advertised rust-version and every feature selection it walked are in the manifests.

Invoked by: nothing; gates.yml runs `feature-msrv` in its place.

Gate: `feature-msrv` reads the advertised rust-version and holds every feature selection on the axis to it.

Injection: Emptied the advertised rust-version, and then set it to a moving channel; `feature-msrv` reported both, proved red.

Assertions:

- Delegates to scripts/lib/check_feature_msrv.py after selecting the cargo runner, forwarding an optional --list.

Exits nonzero on:

- whatever check_feature_msrv.py exits nonzero on

### scripts/check_metal_macbook.sh

Subject: gone: the script is not in the tree. The Metal driver, its telemetry suite and the benchmark cases the run measures are tracked.

Invoked by: nothing; the path was a string in vyre-bench/src/cli/bundle.rs, and `metal-parity` carries the assertions.

Gate: `metal-parity` derives the published counter roster from the driver rather than restating it, holds every published counter to a test that names it, and holds the measured cases to the benchmark catalog. --host and --remote-root run the driver suite, the conformance suite and one benchmark per case on an Apple GPU. The script's copied roster had already drifted: it listed 16 counters where the driver publishes 17.

Injection: Published a counter no test names; `metal-parity` reported the counter, proved red.

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

### scripts/check_public_api_snapshot.sh

Subject: gone: the script is not in the tree; docs/public-api/*.txt are tracked.

Invoked by: nothing; public-api.yml runs `public-api-snapshot` in its place.

Gate: `public-api-snapshot`, with --refresh becoming ctx.write on a generating gate.

Injection: Added an unexported symbol to docs/public-api/vyre-lints.txt; `public-api-snapshot` reported the snapshot against the crate, proved red.

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
### scripts/check_signed_conformance_certificate.sh

Subject: gone: the script is not in the tree. scripts/prove-release-shards.sh carries the four environment defaults it exported, at lines 5 to 8, and the merged certificate it checked is release evidence.

Invoked by: nothing; conform/vyre-conform/tests/cert_artifact records what the wrapper asserted.

Gate: None of its own. The wrapper set four defaults the script it called already defaults to, and asserted a non-empty merged certificate the merge step and the certificate suite both assert. `workspace-tests` runs that suite.

Injection: Removed the bounded worker pool from scripts/prove-release-shards.sh; the vyre-conform cert_artifact suite that `workspace-tests` runs reported the missing pool, proved red.

Assertions:

- prove-release-shards.sh produces a non-empty merged conformance certificate.

Exits nonzero on:

- empty or missing merged certificate

Findings:

- It sets `export RUSTC_WRAPPER=""`, a build-affecting variable outside .cargo/config.toml. The gate does not set it; if the wrapper breaks a release build that belongs in the config file.

### scripts/check_spirv_parity_perf_gate.sh

Subject: gone: the script is not in the tree. The validated SPIR-V target and its feature gate are in vyre-driver-spirv.

Invoked by: nothing; gates.yml runs `spirv-parity` in its place.

Gate: `spirv-parity` holds the validated target to its feature gate and runs the validation half behind --validate.

Injection: Removed the required-features entry from the validated target; `spirv-parity` reported the unregistered target, proved red.

Assertions:

- spirv-val is on PATH, because an unvalidated blob with a correct header is the defect the gate exists for.
- vyre-driver-spirv spirv_parity passes with the spirv-val feature enabled.

Exits nonzero on:

- spirv-val missing
- test failure

### scripts/crate_ownership.py

Subject: gone: the script is not in the tree. docs/CRATE_OWNERSHIP.toml and both generated documents, docs/CRATE_GRAPH.md and docs/OWNERSHIP.md, are tracked.

Invoked by: nothing; `crate-ownership` reads the registry and regenerates both documents under --write.

Gate: `crate-ownership` owns the registry shape and both generated documents; `check-tier-deps` and `layering` read the registry it holds.

Injection: Recorded a dependency on a crate the workspace does not carry; `crate-ownership` reported the row against the manifests, proved red.

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

### scripts/crate_readmes.py

Subject: gone: the script is not in the tree. The crate READMEs it wrote are tracked, and so are the testing guides its generated block links.

Invoked by: nothing; `crate-readmes` regenerates the block under --write.

Gate: `crate-readmes` regenerates the crate-contract block from the manifests and docs/CRATE_GUIDES.toml.

Injection: Edited a generated README block by hand; `crate-readmes` reported the file, proved red.

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

### scripts/docs.sh

Subject: gone: the script is not in the tree.

Invoked by: nothing; docs-ci.yml runs `workspace-docs` in its place.

Gate: `workspace-docs`.

Injection: Linked a symbol that does not exist from a doc comment in xtask/src/lib.rs; `workspace-docs` reported the broken link, proved red.

Assertions:

- `cargo doc --no-deps --keep-going` succeeds, for the whole workspace or for the packages a diff touched.

Exits nonzero on:

- any rustdoc failure

Findings:

- `--changed-only` exits 0 when it finds no changed files and when it can resolve no affected package, so a run that documents nothing reports the same success as a run that documents everything. Under the contract the gate documents the workspace and reports what it built as a note.
### `scripts/final-launch.sh`

Subject: present: the notes it passes to `gh release create` are release/evidence/docs/release-notes-body.md, which the `release-docs` gate generates.

Invoked by: nothing; it is the manual launch entry point, and it is named by xtask/src/release/launch_state.rs, release/release-train.toml, release/vyre-release-evidence.toml, docs/optimization/OWNERSHIP.toml and conform/vyre-conform/tests/cert_artifact/release_script_contracts.rs.

Gate: xtask/src/release/launch_state.rs and xtask/src/release/launch_contract.rs own the preconditions; the publish, tag, push and gh release actions stay operator actions.

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

Resolved:

- Release creation reads `release/evidence/docs/release-notes-body.md`, which `release-docs` generates and owns.
- Cargo invocations inherit build parallelism from `.cargo/config.toml`.
- `launch-state --write` regenerates the fixed public launch-state artifact before the launch-complete gate.

### scripts/install_wire_precommit_hook.sh

Subject: gone: the script is not in the tree. It symlinked scripts/wire_ci_local.sh, which is gone with it.

Invoked by: nothing.

Gate: None of its own: it installed a local git hook. The one assertion the hook made that no other gate makes is `wire-determinism`, which the sweep runs on every tree instead of on the trees of operators who remembered to install a hook.

Injection: Reserved a feature vyre-primitives does not declare for a wire suite; `wire-determinism` reported the target cargo would refuse, proved red.

Assertions:

- The pre-push hook path is not an existing non-symlink file before the symlink is written.

Exits nonzero on:

- hook path exists and is not a symlink

Findings:

- Its one assertion protects an operator's existing hook. It is a local install action, not a tree property, so no gate can own it. It is the one file in this layer whose deletion needs an operator decision about how the hook is installed.

### scripts/lib/cargo_runner.sh

Subject: gone: the script is not in the tree; runner resolution lives in xtask::cargo_runner.

Invoked by: nothing; operator scripts run ./cargo_full directly.

Gate: `compile` and `workspace-build` resolve the cargo runner through xtask/src/cargo_runner.rs.

Injection: Broken cargo runner resolution in xtask/src/cargo_runner.rs; `compile` proved red.

Assertions:

- Selects VYRE_CARGO_RUNNER, then ./cargo_full, then cargo.

Exits nonzero on:

- never; it only assigns

Findings:

- Nothing remains open.

### scripts/lib/check_deep_bench_coverage.py

Subject: gone: the script is not in the tree.

Invoked by: nothing.

Gate: `bench-coverage`.

Injection: Renamed the module-cache test the coverage dimension names; `bench-coverage` reported the uncovered dimension, proved red.

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
### scripts/lib/check_feature_msrv.py

Subject: gone: the script is not in the tree. The manifests it read are tracked.

Invoked by: nothing; `feature-msrv` walks the axis in process.

Gate: `feature-msrv`.

Injection: Emptied the advertised rust-version, and then set it to a moving channel; `feature-msrv` reported both, proved red.

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

Invoked by: toml_reader.sh.

Gate: None of its own: it reads TOML values for a helper sourced by an operator action. The gates read TOML through typed reads in xtask/src/toml_text.rs.

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

Gate: None of its own: it reads the release train for two operator actions. xtask/src/release/release_train.rs reads the same file for the release gates.

Assertions:

- release/release-train.toml defines versions.vyre, tags.vyre_rc, tags.vyre and release_groups.vyre.repository, all scalar.

Exits nonzero on:

- any of the four keys missing or non-scalar

### `scripts/lib/repo_boundary.sh`

Subject: present.

Invoked by: final-launch.sh.

Gate: None of its own: it resolves the repository boundary for an operator action. xtask/src/release/repo_boundary.rs resolves it for the release gates.

Assertions:

- release/repo-boundary.toml defines public_repository, private_repository, verify_public_repo_action and boundary_description, all scalar.

Exits nonzero on:

- any of the four keys missing or non-scalar

### scripts/lib/sweep_targets.py

Subject: gone: the script is not in the tree. The sweep sources and the manifest entries that reserve features for them are tracked.

Invoked by: nothing; `oracle-sweeps` derives the roster from the tree.

Gate: `oracle-sweeps` derives every tracked sweep target and the features its crate reserves, and runs a partition behind --sweep.

Injection: Reserved features for a sweep target with no tracked source, and then required a feature the crate does not define; `oracle-sweeps` reported both, proved red.

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

Gate: None of its own: it is sourced by two helpers of an operator action. The release gates parse TOML with toml::from_str into a toml::Table.

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

Invoked by: final-launch.sh.

Gate: none in the registry; conform/vyre-conform/tests/cert_artifact/release_script_contracts.rs holds the script to the merged certificate it must produce, and the sharded proof run stays an operator action.

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

### scripts/public_api_snapshot_inventory.py

Subject: gone: the script is not in the tree.

Invoked by: nothing; public-api.yml runs `public-api-paths` in its place.

Gate: `public-api-snapshot`, which derives the same roster from the manifests it already walks.

Injection: Added an unexported symbol to docs/public-api/vyre-lints.txt; `public-api-snapshot` reported the snapshot against the crate, proved red.

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

Gate: xtask/src/release/package_readiness/mod.rs owns the readiness assertions; the publish loop stays an operator action.

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

Resolved:

- Cargo invocations inherit build parallelism from `.cargo/config.toml`.
- `package-readiness --write` regenerates `release/evidence/package/publish-readiness.json` before the script reads it.

### scripts/run_sweep_oracle_matrix.sh

Subject: gone: the script is not in the tree. Every sweep source it selected is tracked.

Invoked by: nothing; gates.yml runs `oracle-sweeps` in its place.

Gate: `oracle-sweeps`.

Injection: Reserved features for a sweep target with no tracked source, and then required a feature the crate does not define; `oracle-sweeps` reported both, proved red.

Assertions:

- The matrix partition of the sweep roster is non-empty.
- Every tracked sweep_* oracle-matrix integration test passes, invoked per crate with the union of the required-features its own targets declare.

Exits nonzero on:

- empty roster
- any test failure

Findings:

- The reason it exists is that ci.yml runs the workspace suite with default features, so a test whose required-features name a non-default feature is silently skipped, and strict.yml builds --all-features without running anything. The roster derivation is what keeps a new sweep from being unrun, and it is preserved.

### scripts/run_volume_sweep_shard.sh

Subject: gone: the script is not in the tree. The volume waves it sharded are tracked sweep sources.

Invoked by: nothing; gates.yml runs `oracle-sweeps` in its place.

Gate: `oracle-sweeps`, which counts the volume waves separately and shards them behind --sweep.

Injection: Reserved features for a sweep target with no tracked source, and then required a feature the crate does not define; `oracle-sweeps` reported both, proved red.

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

### scripts/testing_guides.py

Subject: gone: the script is not in the tree. The per-crate guides it wrote are tracked under docs/testing, and so is docs/testing/TESTING.toml.

Invoked by: nothing; `testing-guides` regenerates every guide under --write.

Gate: `testing-guides` renders one guide per workspace member from the manifests and docs/testing/TESTING.toml, and reports a guide no member renders as well as a member with no guide.

Injection: Edited a generated guide by hand; `testing-guides` reported the file, proved red.

Assertions:

- Every workspace member has a guide under docs/testing.
- Each guide names only cargo targets the member's manifest declares or cargo discovers.
- Each guide states what the crate does when the hardware it wants is absent.

Exits nonzero on:

- a member with no guide
- a guide no member renders
- a guide that does not match what it is rendered from

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

### scripts/wire_ci_local.sh

Subject: gone: the script is not in the tree. The wire suites it ran are tracked under vyre-primitives/tests.

Invoked by: nothing; it was symlinked as a git hook by a script that is also gone.

Gate: `wire-determinism` carries the two-run ordering assertion, which was the only one no other gate made; `workspace-check`, `workspace-clippy` and `workspace-tests` carry the fmt, clippy, check and test steps.

Injection: Reserved a feature vyre-primitives does not declare for a wire suite; `wire-determinism` reported the target cargo would refuse, proved red.

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
| `file-size` | Add a row to the audit ceilings naming vyre-does-not-exist/src/lib.rs. | 75 to 76 |
| `gpu-loudness` | Add `#[cfg(not(feature = "gpu"))]` above a test in `vyre-driver-wgpu/tests/` with no loud abort within ten lines above or twenty below. | 2 to 3 |
| `gpu-loudness` | Add `if adapter.is_err() { return; }` to a test body. | 2 to 3 |
| `gpu-loudness` | Add `Backend::acquire_or_panic();` five lines below an existing finding site in `conform/vyre-conform/tests/cert_artifact/prove_failure_contracts.rs`. | 2 to 1, which is the allowance working rather than a failure |
| `unification` | Add a second `pub fn child_bodies` to any file under `vyre-foundation/src`. | 0 to 2, because a row over its ceiling reports every site |
| `unification` | Add `BufferAccess::infer(` to a file under `vyre-runtime/src/resident_work_queue`. | 0 to 1 |
| `unification` | Rename the directory `vyre-foundation/src/execution_plan` and update its `mod` declaration. | 0 to 1, reported as a path that does not exist rather than as a clean row |
| `evidence-paths` | In any artifact under `release/evidence`, change one cited path to a filename that does not exist but keeps a tree extension. | 18 to 19 |
| `evidence-paths` | Add `"manifest": "target/debug/build.rs"` to an artifact object, with the target directory gitignored. | 18 to 19, in the gitignored class rather than the missing class |
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
| `hot-path-inventory` | In `vyre-libs/src/plumbing/registration/operation_catalog.rs`, serve `convergence_contract` by walking `inventory::iter` instead of probing the frozen index. | 0 to 1, quoting the statement that scans |
| `hot-path-nested-rows` | In `vyre-driver/src/backend/vyre_backend.rs`, delete the `dispatch_borrowed_into` declaration. | 0 to 1, naming the returning method that is then the only shape offered |
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

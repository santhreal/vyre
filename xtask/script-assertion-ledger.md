# Script assertion ledger

`scripts/` holds 96 tracked files: 71 shell scripts, 21 Python scripts, 3 text
baselines and 1 lockfile. Each one is recorded below with its assertions, what
makes it exit nonzero, every caller found in the tree, whether the files it reads
still exist after the documentation deletion at commit b1ed746d1c, and the gate
that owns its assertions after the port. The rows carry 332 assertions and
118 findings.

## Totals

- Files: 96. Assertions: 332. Findings: 118.
- Files whose subject is partly or wholly gone: 18.
- Files nothing invokes: 34.
- Files that assert nothing: 6, being 4 data files and 2 run wrappers.

### Subject gone

- `scripts/architecture_docs.py`
- `scripts/bench/cross_backend_comparison.sh`
- `scripts/check_docs_index.sh`
- `scripts/check_docs_links.sh`
- `scripts/check_docs_references.py`
- `scripts/check_error_codes_cataloged.sh`
- `scripts/check_no_hot_path_inventory.sh`
- `scripts/check_no_string_wgsl.sh`
- `scripts/check_op_names.sh`
- `scripts/check_platform_consumer_docs.sh`
- `scripts/check_unification_baselines.sh`
- `scripts/cli_docs.py`
- `scripts/crate_ownership.py`
- `scripts/crate_readmes.py`
- `scripts/docs_manifest.py`
- `scripts/final-launch.sh`
- `scripts/testing_guides.py`
- `scripts/wgsl_to_rust/Cargo.lock`

### Nothing invokes it

- `scripts/apply-branch-protection.sh`
- `scripts/baselines/unfailing_tests.txt`
- `scripts/baselines/unwrap.txt`
- `scripts/bench/cross_backend_comparison.sh`
- `scripts/bench_smoke.sh`
- `scripts/check_bench_baselines.sh`
- `scripts/check_docs_index.sh`
- `scripts/check_docs_links.sh`
- `scripts/check_error_codes_cataloged.sh`
- `scripts/check_evidence_paths.sh`
- `scripts/check_expect_has_fix.sh`
- `scripts/check_gpu_test_loudness.sh`
- `scripts/check_invariant_paths_exist.sh`
- `scripts/check_max_file_size.sh`
- `scripts/check_metal_macbook.sh`
- `scripts/check_no_default_feature_megacrate.sh`
- `scripts/check_no_string_wgsl.sh`
- `scripts/check_op_names.sh`
- `scripts/check_parity_testing_not_leaked.sh`
- `scripts/check_performance_inventory_wave1.sh`
- `scripts/check_platform_consumer_docs.sh`
- `scripts/check_primitive_contract.sh`
- `scripts/check_signed_conformance_certificate.sh`
- `scripts/check_trait_freeze.sh`
- `scripts/check_unsafe_justifications.sh`
- `scripts/crate_ownership.py`
- `scripts/crate_readmes.py`
- `scripts/final-launch.sh`
- `scripts/install_wire_precommit_hook.sh`
- `scripts/release_docs.py`
- `scripts/testing_guides.py`
- `scripts/vyre_smoke.sh`
- `scripts/wgsl_to_rust/Cargo.lock`
- `scripts/wire_ci_local.sh`

### Asserts nothing

- `scripts/baselines/unfailing_tests.txt`
- `scripts/baselines/unwrap.txt`
- `scripts/bench/cross_backend_comparison.sh`
- `scripts/bench_smoke.sh`
- `scripts/unsafe_budget.txt`
- `scripts/wgsl_to_rust/Cargo.lock`

## Rows

### `scripts/apply-branch-protection.sh`

Subject: present.

Invoked by: nothing; named in .github/CI_REQUIRED.md and .github/CODEOWNERS.

Gate: xtask/src/gates/ci_contract.rs for every assertion; the gh mutation is not a gate and stays a manual operator action.

Assertions:

- .github/CI_REQUIRED.md exists and parses to at least one required status context.
- Every listed context is defined by a workflow, by job name or job id.
- Five required workflows exist and run on pull_request and push to main.
- None of those five uses a path filter, which would let a required check be skipped.
- ci.yml, conform.yml and gpu-parity.yml each carry a fan-in job with `if: always()`, a `.result` test and `exit 1`, so a skipped dependency fails closed.
- The repository is santhreal/vyre and gh is available before anything is applied.

Exits nonzero on:

- missing CI_REQUIRED.md
- gh missing
- wrong repository
- no contexts parsed
- a context no workflow defines
- a required workflow missing or wrongly triggered
- a path filter on a required workflow
- a fan-in job that is not fail-closed

Findings:

- This is the only place the CI_REQUIRED contract is checked, and it is checked only when an operator applies branch protection by hand. Six assertions about the workflow set therefore run on no schedule.
- .github/CI_REQUIRED.md has historically named reproducible-build.yml and mutation-testing.yml, which live under .github/workflows-paused/, so the context assertion is the one that catches that.

### `scripts/architecture_docs.py`

Subject: partly gone: the five documents it reads were deleted at b1ed746d1c; docs/DOCS.toml, docs/generated/OP_SCHEMA.json, docs/optimization/OWNERSHIP.toml, docs/CRATE_OWNERSHIP.toml, release/release-train.toml and the backend evidence all survive.

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

### `scripts/baselines/unfailing_tests.txt`

Subject: present.

Invoked by: nothing found.

Gate: reported: no script or workflow reads this file.

Assertions:

- Data. A pinned count of tests that cannot fail.

Exits nonzero on:

- not executable

Findings:

- Nothing reads it.

### `scripts/baselines/unwrap.txt`

Subject: present.

Invoked by: nothing found.

Gate: reported: no script or workflow reads this file.

Assertions:

- Data. A pinned unwrap count.

Exits nonzero on:

- not executable

Findings:

- Nothing reads it. A baseline nothing compares against pins nothing.

### `scripts/bench/cross_backend_comparison.sh`

Subject: gone: it writes into docs/perf/, and docs/ carries no Markdown after b1ed746d1c.

Invoked by: nothing; the path appears in .gitignore only.

Gate: reported: it asserts nothing and its output directory is no longer published.

Assertions:

- Runs `xtask bench-crossback` for xor-1k and xor-1m and writes tables under docs/perf/.

Exits nonzero on:

- either run failing

Findings:

- It is a wrapper around a registered subcommand and writes generated Markdown into a directory the repository no longer publishes. Nothing invokes it.

### `scripts/bench_smoke.sh`

Subject: present.

Invoked by: nothing; named in CONTRIBUTING.md.

Gate: xtask/src/gates/bench_contract.rs already runs the smoke suite under a budget, so this wrapper carries no assertion of its own.

Assertions:

- Runs the vyre-bench smoke suite. Asserts nothing itself beyond the run succeeding.

Exits nonzero on:

- any bench failure

### `scripts/check_architectural_invariants.sh`

Subject: present (cites the deleted docs/ARCHITECTURE.md in prose only).

Invoked by: gates.yml, architectural-invariants.yml.

Gate: xtask/src/gates/layering.rs, as the direct-edge half beside the transitive half.

Assertions:

- Each of six substrate-neutral crates has a manifest on disk.
- None of those six declares vyre-driver-wgpu, vyre-driver-cuda, vyre-driver-spirv, vyre-runtime, vyre-aot, wgpu or naga outside [dev-dependencies], excluding optional entries.
- No tracked Cargo.toml names the retired crates vyre-ir or vyre-wgpu.

Exits nonzero on:

- missing neutral crate manifest
- forbidden direct dependency edge
- stale legacy crate name

Findings:

- The legacy-name check runs `rg ... 2>/dev/null` and reads a nonzero exit as `no hits`. A ripgrep that is absent or that errors makes this assertion unreachable, which is the same class source_scan.sh was written to remove.
- It writes hits to /tmp/vyre_arch_legacy_hits.$$, a file outside the repository.
- The crate and dependency lists are hardcoded, so a new neutral crate is unchecked until someone edits the list. check_layering.py derives both from the registry.

### `scripts/check_audit_status_tags.sh`

Subject: present (audits/*.md are tracked).

Invoked by: gates.yml.

Gate: xtask/src/gates/audit_status.rs.

Assertions:

- Every numbered finding row in a status-managed audits/*.md file, above the `## Highest Leverage Execution Order` heading, begins with `open`, `in_progress` or `fixed`.
- At least one status-managed audit file exists.

Exits nonzero on:

- untagged finding row
- no status-managed audit file found

### `scripts/check_backend_extension_contract.sh`

Subject: present.

Invoked by: gates.yml.

Gate: xtask/src/gates/frozen_contract.rs.

Assertions:

- vyre-driver inventory_streams.rs collects BackendRegistration, BackendPrecedence and BackendCapability, freezes them in one LazyLock<Result<BackendRegistry, BackendError>>, and populates from inventory::iter::<BackendRegistration>.
- acquire.rs routes acquisition through registered_backends_by_precedence_slice and consults backend_dispatches.
- The core registry directory contains no concrete backend id literal (cuda, wgpu, spirv, metal, dxil).
- Each of five concrete driver crates has a manifest and src directory, depends on vyre-driver and inventory, implements VyreBackend, and submits BackendRegistration, BackendPrecedence, BackendCapability and supported_ops through inventory::submit!.

Exits nonzero on:

- any of the seven core requirements unmet
- concrete id literal in the core registry
- any of the eight per-crate requirements unmet for any of the five crates

Findings:

- The five driver crate names are hardcoded, so a sixth backend crate is unchecked until someone edits the list. The roster is derivable from the crates that depend on vyre-driver and submit BackendRegistration.

### `scripts/check_bench_baselines.sh`

Subject: present.

Invoked by: nothing; named in CHANGELOG.md and release/changes/unreleased.toml only.

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

### `scripts/check_ci_matrix.sh`

Subject: present.

Invoked by: gates.yml.

Gate: xtask/src/gates/ci_contract.rs.

Assertions:

- .github/workflows/ci.yml exists, contains a `matrix:` key, and mentions ubuntu-latest, macos-latest, windows-latest, stable and nightly.
- ci.yml contains no no-GPU escape hatch (`no-gpu`, `gpu-feature`, `vyre-driver-wgpu/no-gpu`).
- .github/workflows/gpu-parity.yml exists.

Exits nonzero on:

- missing ci.yml or gpu-parity.yml
- missing OS or toolchain string
- no matrix key
- escape hatch present

Findings:

- The OS and toolchain assertions are substring searches over the whole file, so a commented-out line or an unrelated mention satisfies them. `stable` also matches `unstable`. The gate cannot distinguish a declared matrix axis from prose, so it passes on a workflow whose matrix lost an axis. The repair is to parse the YAML and read the matrix axes.

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

### `scripts/check_direct_readback_ring_default.sh`

Subject: present.

Invoked by: gates.yml.

Gate: xtask/src/gates/frozen_contract.rs.

Assertions:

- record_and_readback.rs and its module directory contain readback_rings:, SubmittedReadback::Ring, .record_copy(, .arm_ticket( and .with_mapped_ticket(.
- vyre-driver-wgpu/src/lib.rs constructs ReadbackRingSet::new().

Exits nonzero on:

- any of the five patterns absent
- arena not owning a ReadbackRingSet

Findings:

- A third assertion was removed after being found unreachable: a multi-line regex against a line-oriented search matched nothing on any tree. Its invariant is which branch the code sits in, which no line scan can see, and it is now owned by two GPU tests in vyre-driver-wgpu/src/pipeline/tests/readback_ring_contracts.rs. The comment recording that is preserved in the gate as a note.

### `scripts/check_doc_claim_to_test.sh`

Subject: present (contracts/doc_claims_manifest.toml is tracked).

Invoked by: gates.yml.

Gate: xtask/src/gates/doc_contract.rs.

Assertions:

- contracts/doc_claims_manifest.toml exists and parses to at least one [[claim]].
- Every claim row declares id, doc, phrase and test.
- Every claim's doc file exists and contains the literal phrase.
- Every claim's test path exists.

Exits nonzero on:

- missing manifest
- zero claims parsed
- incomplete row
- missing doc
- phrase absent
- missing test path

Findings:

- The manifest is parsed with awk over TOML text, so a phrase containing a quote, a multi-line string, or a field written on a continuation line is read wrongly. The repair is toml::from_str.

### `scripts/check_docs_index.sh`

Subject: gone: docs_manifest.py --check validates docs/DOCS.toml pages, and all 132 declared pages were deleted at b1ed746d1c.

Invoked by: nothing; named in vyre-pass-engine/tests/platform_doc_consumer_boundary.rs only.

Gate: xtask/src/gates/doc_contract.rs, which reports each missing declared page as a finding.

Assertions:

- Delegates to scripts/docs_manifest.py --check.

Exits nonzero on:

- whatever docs_manifest.py --check exits nonzero on

Findings:

- Nothing invokes it, and its subject is gone, so it exits nonzero on every tree today.

### `scripts/check_docs_links.sh`

Subject: gone: the active set comes from docs_manifest.py --list-active over docs/DOCS.toml, and all 132 declared pages were deleted at b1ed746d1c.

Invoked by: nothing; named in vyre-pass-engine/tests/platform_doc_consumer_boundary.rs only.

Gate: xtask/src/gates/doc_contract.rs, which applies the same three classes to the Markdown that survives (root, .github, crate READMEs).

Assertions:

- Every relative Markdown link in the active documentation set resolves to a path inside the repository.
- No link escapes the repository root (OUTSIDE-REPO).
- No link names a path that does not exist (MISSING).
- No link names a gitignored path, which resolves for the author and fails for every other reader (GITIGNORED).
- git check-ignore exiting above 1 is a tool failure, not a clean result.

Exits nonzero on:

- any link in one of the three classes
- git check-ignore failure

Findings:

- Its scope collapsed to nothing when the pages were deleted, so the link contract has been unenforced since then. The three classes still apply to the surviving Markdown, and the gate scopes them there.

### `scripts/check_docs_references.py`

Subject: partly gone: docs/**/*.md are gone; root Markdown, .github Markdown and crate READMEs survive and the assertion applies to them unchanged.

Invoked by: xtask/tests/docs_references.rs.

Gate: xtask/src/gates/doc_contract.rs.

Assertions:

- Every Markdown document in scope resolves each of its relative links, so no published document points at a path a reader cannot open.

Exits nonzero on:

- any unresolvable reference

### `scripts/check_error_codes_cataloged.sh`

Subject: gone: docs/error-codes.md was deleted at b1ed746d1c.

Invoked by: nothing.

Gate: xtask/src/gates/doc_contract.rs, which reports the missing catalog as one finding and keeps the code extraction live.

Assertions:

- docs/error-codes.md exists.
- Every V### or E-/W-/B-/C- prefixed code appearing in a string literal under six crate src directories has a `code` row in the catalog.

Exits nonzero on:

- missing catalog
- uncataloged code

Findings:

- It exits 1 on every tree today, and nothing invokes it, so a documented-error-code contract has been unenforced since the deletion.
- The character class `[E|W|B|C]` includes a literal pipe, so `|-FOO` is accepted as a code prefix. Inside a bracket expression `|` is not alternation.
- `2>/dev/null` on the recursive grep turns a search failure into an empty code set, which reads as a fully cataloged tree.

### `scripts/check_every_source_file_is_reachable.sh`

Subject: present.

Invoked by: gates.yml.

Gate: xtask/src/gates/source_reachability.rs.

Assertions:

- Delegates to scripts/lib/check_every_source_file_is_reachable.py. Invokes no cargo, because the defect it detects is a file cargo never compiles.

Exits nonzero on:

- whatever check_every_source_file_is_reachable.py exits nonzero on

### `scripts/check_every_source_file_parses.sh`

Subject: present.

Invoked by: gates.yml.

Gate: xtask/src/gates/source_reachability.rs.

Assertions:

- rustfmt is installed before the scan runs, because a missing parser is not a clean tree.
- Delegates to scripts/lib/check_every_source_file_parses.py.

Exits nonzero on:

- rustfmt not installed
- whatever check_every_source_file_parses.py exits nonzero on

### `scripts/check_evidence_paths.sh`

Subject: present.

Invoked by: nothing; named in CHANGELOG.md, release/changes/unreleased.toml and vyre-pass-engine/tests/release_evidence_path_contract.rs.

Gate: xtask/src/gates/evidence_paths.rs.

Assertions:

- release/evidence exists and jq is available.
- The file-extension vocabulary is derived from the tree at run time and is non-empty.
- Every path-shaped string leaf, at any depth, in every JSON under release/evidence resolves on disk, resolved absolute, then against the workspace root, then against the artifact's own directory.
- No cited path that exists is gitignored, because evidence citing a local-only file is unverifiable by any other reader.
- git check-ignore exiting above 1 is a tool failure, per repository.

Exits nonzero on:

- missing evidence dir
- jq missing
- empty extension vocabulary
- any citation naming a nonexistent path
- any cited path that is gitignored
- check-ignore failure

Findings:

- Nothing invokes it. When it was written the tree carried 185 stale citations across 16 artifacts, and the only prior semantic check on an artifact was internal self-consistency, which a stale artifact passes trivially. That contract is now unenforced in CI.
- The derived extension vocabulary and the depth-independent leaf walk are the two shapes that make it work, and both are preserved: serde_json gives the same walk without jq.

### `scripts/check_expect_has_fix.sh`

Subject: present.

Invoked by: nothing; the path appears as a string in vyre-driver/src/registry/enforce.rs.

Gate: xtask/src/gates/lint_hygiene.rs.

Assertions:

- Every .expect("...") site in non-test Rust source has `Fix:` on the same line or within the next three lines.
- The count of sites without Fix: does not exceed the baseline.

Exits nonzero on:

- count above baseline

Findings:

- The baseline is `VYRE_EXPECT_BASELINE:-0`, so any caller can raise the ceiling from the environment and the ratchet passes. A pinned number belongs in xtask/gate-baselines.toml, which no caller can override.
- `set -uo pipefail` without `-e`, so a failure inside the loop body does not stop the script.
- `grep -rn ... 2>/dev/null` reads a failed search as zero sites, which is a clean tree.
- It scans the working tree from `.` rather than tracked files, so untracked scratch moves the count.

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

### `scripts/check_gpu_test_loudness.sh`

Subject: present.

Invoked by: nothing; the path appears as a string in vyre-driver-wgpu/tests/dispatch_adversarial.rs.

Gate: xtask/src/gates/gpu_loudness.rs.

Assertions:

- No Rust file contains a silent GPU skip: an is_err early return, a `skipped`/`no GPU`/`GPU unavailable` print, or a cfg/cfg_attr gate that ignores a test when no GPU feature is on, unless a loud abort (acquire_or_panic, a named panic!, or an assert carrying Fix:) sits within ten lines above or twenty below.

Exits nonzero on:

- any unpaired silent-skip site

Findings:

- Three of the ten patterns are unreachable. Inside single quotes `'#\\[cfg\\(not...'` passes a literal double backslash to grep -E, so the pattern requires a backslash character before `[cfg` in the Rust source, which never occurs. The cfg and cfg_attr classes named in the header comment are therefore not checked at all. The gate carries them as live patterns in Rust with single escaping.
- It scans the filesystem with `find` rather than tracked files.
- `grep -nE ... 2>/dev/null` inside an `if` reads a failed search as no hits.
- Nothing invokes it, so the AGENTS.md silent-fallback rule it cites has no enforcement in CI.

### `scripts/check_internal_deps_have_versions.sh`

Subject: present.

Invoked by: gates.yml.

Gate: xtask/src/gates/manifest_contract.rs.

Assertions:

- Delegates to scripts/lib/check_internal_deps_have_versions.py. Invokes no cargo.

Exits nonzero on:

- whatever check_internal_deps_have_versions.py exits nonzero on

### `scripts/check_invariant_paths_exist.sh`

Subject: present.

Invoked by: nothing; named in CHANGELOG.md only.

Gate: xtask/src/gates/doc_contract.rs.

Assertions:

- Every `conform/**.rs` path cited by vyre-spec/src/invariants.rs exists on disk, excluding the doc-comment example `conform/tests/<file>.rs`.

Exits nonzero on:

- any cited path missing

Findings:

- Nothing invokes it.

### `scripts/check_ir_wire_field_sync.sh`

Subject: present.

Invoked by: gates.yml.

Gate: xtask/src/gates/frozen_contract.rs.

Assertions:

- Some tracked file under vyre-foundation/src declares `pub struct Program`, located by search rather than by a hardcoded path.
- vyre-foundation/src/serial/wire/encode/to_wire.rs and decode/from_wire.rs both exist.
- Each of five serialized Program fields appears in the Program declaration.
- Each of those fields is mentioned in encode or decode.
- Every `pub` field of Program is either in the serialized list or in the transient list, so a new field cannot land unclassified.

Exits nonzero on:

- no Program declaration found
- missing encode or decode file
- serialized field absent from the struct
- serialized field absent from both encode and decode
- unclassified Program field

Findings:

- The per-field presence test is a substring search, so the field `entry` is satisfied by `entry_op_id` and by any prose containing the word. The gate matches the field declaration instead.
- It shells into perl to list struct fields. The Rust gate parses the field list directly.

### `scripts/check_layering.sh`

Subject: present. File deleted; every assertion is in `layering`.

Invoked by: nothing. The gates workflow now names the `manifest-rules` subset.

Gate: xtask/src/gates/layering.rs.

Assertions:

- Delegates to scripts/lib/check_layering.py after selecting the cargo runner.

Exits nonzero on:

- whatever check_layering.py exits nonzero on

### `scripts/check_max_file_size.sh`

Subject: present.

Invoked by: nothing; the path appears as a string in xtask-registry/src/gates/lego_audit.rs and lego_quick.rs.

Gate: xtask/src/gates/file_size.rs.

Assertions:

- Every .rs file under a src/ tree is within a line cap: 8000 for tests, benches, fuzz and the xtask crates, a per-file ceiling for a listed core file, 2500 for other core files, a per-file ceiling for a listed non-core file, 3000 otherwise.

Exits nonzero on:

- any file above its cap

Findings:

- 21 of the 26 AUDIT_EXCEPTIONS entries are unreachable. The branch order tests is_core_path before AUDIT_EXCEPTIONS, and those 21 paths are core paths, so their per-file ceilings are never read and CORE_AUDIT_EXCEPTIONS or the 2500 core cap applies instead. Only five entries are reachable: the two vyre-driver-cuda files, vyre-driver/src/pipeline.rs, and the two megakernel files that is_core_path deliberately excludes.
- 28 distinct exception entries name a file that does not exist: 15 of 26 in AUDIT_EXCEPTIONS and 25 of 70 in CORE_AUDIT_EXCEPTIONS. Each reserves a ceiling for nothing, which is the defect the unsafe-budget gate was rewritten to remove.
- It walks the filesystem rather than tracked files, so an untracked file under any src/ tree is judged and a caps table is compared against a tree git does not carry.

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
- It exports CARGO_BUILD_JOBS and optionally CARGO_TARGET_DIR into the remote shell. Build configuration belongs in the remote checkout's .cargo/config.toml.
- Roughly 60 assertions are `grep -q` over JSON, so a counter renamed inside a nested object still matches, and a field present with a null value passes. The gate parses the JSON and asserts the fields.
- The artifact count of 7 is a literal in a grep pattern, so adding an eighth artifact fails with a message about a missing string rather than about the count.

### `scripts/check_no_default_feature_megacrate.sh`

Subject: present.

Invoked by: nothing.

Gate: xtask/src/gates/manifest_contract.rs.

Assertions:

- The number of quoted feature slugs under [features].default in vyre-libs/Cargo.toml does not exceed 14.
- Under --strict, that count is 0.

Exits nonzero on:

- count above 14 in default mode
- count above 0 under --strict

Findings:

- --strict compares a live count of 14 against a ceiling of 0, so --strict can only fail. A mode that cannot pass is not a mode; the number is the ratchet and belongs in gate-baselines.toml.
- The count is taken with sed over manifest text rather than a TOML reader, so a default list written inline or across lines is miscounted.
- Nothing invokes it, so the megacrate default-feature ratchet has no enforcement.

### `scripts/check_no_hot_path_blocking_wait.sh`

Subject: present.

Invoked by: gates.yml.

Gate: xtask/src/gates/hot_path.rs.

Assertions:

- Occurrences of Maintain::Wait, pollster::block_on, thread::sleep, thread::yield_now, thread::park and park_timeout in vyre-driver-wgpu production sources equal 2, excluding tests, benches and wait_backoff.rs.
- Under --strict, every occurrence sits in one of four reviewed files.

Exits nonzero on:

- count above the ceiling
- count below the ceiling
- a strict-mode hit outside the allowed prefixes

Findings:

- The equality ratchet is deliberate and is preserved: slack above the measured count is room a regression hides in. Under the registry it becomes a pinned findings count that only ever moves down.

### `scripts/check_no_hot_path_inventory.sh`

Subject: present (docs/inventory-contract.md, cited in the header, is gone).

Invoked by: gates.yml.

Gate: xtask/src/gates/hot_path.rs.

Assertions:

- No `inventory::iter::<` call appears in eleven production trees, outside an allowlist of six init-only files and outside test modules.

Exits nonzero on:

- any hit outside the allowlist

Findings:

- This gate is the reason scripts/lib/source_scan.sh exists. It asked ripgrep for -P, this build has no PCRE2, every invocation errored into /dev/null, and the gate passed on every possible tree. The assertion is live again through the shared scanner and stays live in Rust.

### `scripts/check_no_hot_path_vec_vec.sh`

Subject: present.

Invoked by: gates.yml.

Gate: xtask/src/gates/hot_path.rs.

Assertions:

- Occurrences of Vec<Vec<u8>> in vyre-driver-wgpu production sources equal 14, excluding tests.
- Under --strict, every occurrence is inside a comment.

Exits nonzero on:

- count above the ceiling
- count below the ceiling
- a strict-mode hit in live code

Findings:

- The previous ceiling of 35 sat above an actual 24, so eleven new nested-Vec sites could land unseen. The equality ratchet replaced it and is preserved.

### `scripts/check_no_missing_docs_override.sh`

Subject: present.

Invoked by: gates.yml.

Gate: xtask/src/gates/lint_hygiene.rs.

Assertions:

- No crate-root `#![allow(... missing_docs ...)]` inner attribute appears in any src/lib.rs, so the workspace `missing_docs = deny` floor cannot be opted out of.

Exits nonzero on:

- any crate-root override

Findings:

- `find . -maxdepth 4 -name lib.rs` decides the roster, so a crate whose lib.rs sits deeper than four levels is never scanned, and an untracked lib.rs in a dev tree is. The Rust gate derives the roster from workspace.members.

### `scripts/check_no_owned_dispatch_hot_paths.sh`

Subject: present.

Invoked by: gates.yml.

Gate: xtask/src/gates/hot_path.rs.

Assertions:

- No `.dispatch(` call appears in vyre-libs/src, vyre-runtime/src or conform/vyre-conform/src outside test modules, so production and conformance paths use dispatch_borrowed.

Exits nonzero on:

- any hit

### `scripts/check_no_string_wgsl.sh`

Subject: present (the allowlist still names docs/ paths that are gone, which widens nothing because those files cannot appear).

Invoked by: nothing; named in CHANGELOG.md, vyre-libs/AUTHORING.md and vyre-primitives/README.md.

Gate: xtask/src/gates/shader_source.rs.

Assertions:

- No Rust file outside the allowlist contains a WGSL syntax token (@compute, @workgroup_size, @group(, @binding(, var<storage, var<uniform, var<workgroup, -> @location) while also containing push_str, format_args, format!, write!, writeln! or a raw string.
- No file under vyre-driver-wgpu/src has push_str calls together with WGSL tokens.
- The number of files containing naga::front::wgsl::parse_str under vyre-driver-wgpu/src and vyre-foundation/src is 0.

Exits nonzero on:

- violations above 0
- parse_str file count above 0

Findings:

- Both `progress` branches compare a count against 0 with `-lt`, so neither can ever fire. A branch that cannot execute is not a report, and the two messages it would print have never been printed.
- `grep -rl ... 2>/dev/null || true` on the outer file listing reads a failed search as no files, which reads as a clean tree.
- It scans the filesystem rather than tracked files, and it excludes only target and .git.

### `scripts/check_no_unbounded_cache.sh`

Subject: present.

Invoked by: gates.yml.

Gate: xtask/src/gates/hot_path.rs.

Assertions:

- Occurrences of HashMap::new() or VecDeque::new() in vyre-driver-wgpu/src and vyre-runtime/src equal 2, excluding tests, benches and fuzz.
- Under --strict, every occurrence is one of two reviewed files.

Exits nonzero on:

- count above the ceiling
- count below the ceiling
- a strict-mode hit outside the allowed files

### `scripts/check_no_unbounded_external_read.sh`

Subject: present.

Invoked by: gates.yml.

Gate: xtask/src/gates/hot_path.rs.

Assertions:

- Every read_to_end site in vyre-driver-wgpu/src outside tests is in the one reviewed cache module.
- The allow-prefix matches at least one live hit, so a stale exemption cannot sit unused.

Exits nonzero on:

- a hit outside the allow prefix
- an allow prefix matching nothing

Findings:

- The stale-exemption half is the shape every allowlist in this layer should have and most do not. The Rust gate applies it to every allowlist it carries.

### `scripts/check_no_under_reserve.py`

Subject: present.

Invoked by: check_no_under_reserve.sh.

Gate: xtask/src/gates/hot_path.rs.

Assertions:

- No try_reserve or try_reserve_exact call derives additional capacity from capacity(), measured over the call statement up to eight lines.
- A `//` line is documentation and is not a call site.

Exits nonzero on:

- never; it exits 0 and prints findings, and the shell wrapper decides

Findings:

- It walks the filesystem with rglob rather than tracked files, so untracked scratch is scanned.

### `scripts/check_no_under_reserve.sh`

Subject: present.

Invoked by: gates.yml.

Gate: xtask/src/gates/hot_path.rs.

Assertions:

- Delegates to check_no_under_reserve.py and fails when it prints anything.

Exits nonzero on:

- any under-reserve site

### `scripts/check_op_names.sh`

Subject: present (docs/op-naming.md, the cited rule source, is gone).

Invoked by: nothing; the path appears as a string in xtask/src/gates/check_cat_a.rs.

Gate: xtask/src/gates/naming.rs.

Assertions:

- vyre-libs/src exists.
- No public free function in a vyre-libs op source file is named with a compute_/do_/run_/make_/create_/new_ prefix.
- No public free function is named with an _op/_impl/_internal suffix.
- No public free function name contains an uppercase letter, which would need #[allow(non_snake_case)] to compile.

Exits nonzero on:

- missing vyre-libs/src
- any banned prefix, suffix or non-snake-case name

Findings:

- `for f in $op_files` splits an unquoted command substitution on whitespace, so a path containing a space is scanned as two nonexistent paths and the redirect fails the script.
- The skip list names five filenames by hand, so a new module-root or helper filename is scanned as an op file.

### `scripts/check_parity_testing_not_leaked.sh`

Subject: present.

Invoked by: nothing; the path appears as a string in vyre-libs/AUTHORING.md and xtask/src/gates/check_cat_a.rs.

Gate: xtask/src/gates/manifest_contract.rs.

Assertions:

- No Cargo.toml enables the vyre-driver-wgpu parity-testing feature outside a dev-dependencies section, except its declaration in vyre-driver-wgpu's own [features] block.

Exits nonzero on:

- any non-dev activation

Findings:

- It walks the filesystem with find rather than tracked manifests, so a Cargo.toml inside an untracked scratch checkout is judged.
- The section tracker is line-based, so `features = ["parity-testing"]` written inside an inline table on the crate line is not attributed to a section.

### `scripts/check_path_deps_resolve.sh`

Subject: present.

Invoked by: gates.yml.

Gate: xtask/src/gates/manifest_contract.rs.

Assertions:

- Delegates to scripts/lib/check_path_deps_resolve.py. Invokes no cargo, because the defect it detects is a workspace cargo cannot load.

Exits nonzero on:

- whatever check_path_deps_resolve.py exits nonzero on

### `scripts/check_performance_inventory_wave1.sh`

Subject: present.

Invoked by: nothing.

Gate: xtask/src/gates/bench_contract.rs, as two named test targets the gate runs.

Assertions:

- vyre-foundation optimizer_reference_parity_smoke passes.
- vyre-driver-wgpu dispatch_allocation_contract passes.

Exits nonzero on:

- either test failing

Findings:

- Nothing invokes it, so two P0 performance contracts named in the inventory are proven by no workflow.

### `scripts/check_platform_consumer_docs.sh`

Subject: partly gone: all 17 entries of PLATFORM_MARKDOWN_FILES were deleted at b1ed746d1c, and each is guarded by [[ -f ]], so that whole loop now scans nothing. Crate sources, crate READMEs and OP_MATRIX.toml survive.

Invoked by: nothing; named in CHANGELOG.md, vyre-lints/src/consumer_coupling.rs and vyre-pass-engine/tests/platform_doc_consumer_boundary.rs.

Gate: xtask/src/gates/doc_contract.rs.

Assertions:

- No Rust comment or doc comment in thirteen platform crates names a downstream consumer product (weir, surgec, gossan, keyhog).
- No crate-local README.md, ARCHITECTURE.md or CONFIG.md in those crates names one.
- No listed platform Markdown file or docs/optimization/OP_MATRIX.toml names one, except the release-coordination documents listed in vyre-lints/rules/release_coordination_docs.txt.

Exits nonzero on:

- any consumer name in a scanned comment or document

Findings:

- The 17-entry markdown list is now inert. Every entry is skipped silently by its own file guard, so the gate reports a consumer-neutral tree having read none of the documents its name is about. The Rust gate reports each missing listed document as a finding rather than skipping it.
- The exemption file vyre-lints/rules/release_coordination_docs.txt is shared with vyre-lints, which is the right shape and is preserved.

### `scripts/check_primitive_contract.sh`

Subject: present.

Invoked by: nothing; the path appears as a string in xtask/src/gates/hygiene_matrix.rs.

Gate: none needed: the registry already owns primitive-admission-gate, and this file is a shell adapter in front of it.

Assertions:

- Rejects any path argument, because source-file shape is not a primitive contract.
- Delegates to `xtask primitive-admission-gate`.

Exits nonzero on:

- any argument passed
- whatever primitive-admission-gate exits nonzero on

Findings:

- It is a compatibility entry point in front of a registered subcommand, which is exactly the shim shape the port removes. Its own assertion, that path arguments are refused, disappears with it because the registry gate takes no paths.

### `scripts/check_proptest_coverage.sh`

Subject: present.

Invoked by: gates.yml.

Gate: xtask/src/gates/proptest_coverage.rs.

Assertions:

- The number of tracked .rs files importing proptest is at least 181.

Exits nonzero on:

- count below the floor

Findings:

- This is a floor, not a ceiling, so the pinned number is inverted relative to every other ratchet here: growth is the goal and a fall is the failure. The gate reports one finding per missing file below the floor so the pin still counts findings, and the pin still only moves down.

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

### `scripts/check_repo_hygiene.sh`

Subject: present.

Invoked by: gates.yml.

Gate: xtask/src/gates/repo_hygiene.rs.

Assertions:

- Fifteen required repository files exist, and .github/ISSUE_TEMPLATE holds at least three .md files.
- CLAUDE.md and GEMINI.md each say `compatibility redirect`, name AGENTS.md, and are at most 8 lines.
- No .pytest_cache, .cursor or __pycache__ directory is present outside target and .git.
- No no-GPU escape hatch appears in .github or vyre-driver-wgpu/Cargo.toml.
- No file with a build or backup extension is present outside target, corpus and fixture trees.
- No node_modules, .venv, .next or dist directory is present.
- .github/workflows-paused/gpu-parity.yml does not exist, so GPU parity is not paused.
- No Rust file contains silent GPU skip language.

Exits nonzero on:

- any of the eight groups failing

Findings:

- `set -uo pipefail` without `-e`, so an unexpected error inside a check body continues.
- It prints a check mark per passing item, which is 20 lines of output on a clean tree. Under the contract a clean gate says nothing and the per-item confirmations become notes.

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

### `scripts/check_single_backlog.sh`

Subject: present.

Invoked by: gates.yml.

Gate: xtask/src/gates/repo_hygiene.rs.

Assertions:

- CHANGELOG.md exists.
- If BACKLOG.md exists it uses the four-column contract and does not carry the superseded seven-column table.
- No tracked .md file has a name containing PLAN, ROADMAP, BACKLOG, STATUS, HANDOFF, TASKS, BUILDOUT, PRD, BRIEF, TRAJECTORY, SEGMENTATION or GENERALIZATION.

Exits nonzero on:

- missing CHANGELOG.md
- wrong backlog table shape
- any committed parallel plan surface

Findings:

- An earlier revision asserted `-f BACKLOG.md` first, so the gate could never pass in CI and was never wired. The absence of a gitignored local file is not a violation, and the current shape is correct.

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

### `scripts/check_trait_freeze.sh`

Subject: present (docs/frozen-traits/*.txt survive; they are .txt).

Invoked by: nothing; named in vyre-foundation/tests/ci_script_frozen_contract_coupling.rs.

Gate: xtask/src/gates/frozen_contract.rs, with --refresh-snapshots becoming ctx.write.

Assertions:

- Each of seven frozen contracts has its declaring source file on disk.
- Each contract's declaration block is found in that file by keyword.
- Each has a docs/frozen-traits/<name>.txt snapshot.
- Each extracted block is byte-identical to its snapshot.

Exits nonzero on:

- missing source file
- keyword not found
- missing snapshot
- block drift

Findings:

- Nothing invokes it, so seven declared semver-major contracts have no drift enforcement in CI.
- The block is extracted with an awk brace counter that counts braces inside string literals and comments, so a contract body containing a brace in a doc comment shifts the snapshot. The Rust gate uses syn, which xtask already depends on.

### `scripts/check_unification_baselines.sh`

Subject: present (docs/MIGRATION.md, cited as the target list, is gone).

Invoked by: gates.yml.

Gate: xtask/src/gates/unification.rs.

Assertions:

- Every declared scan path of every ratchet row exists, so a row cannot pass by measuring nothing.
- vyre-foundation/src declares exactly one `fn child_bodies`, so nothing re-implements exhaustive child enumeration.
- No BufferAccess::infer/auto/derive_from helper exists in the lowering, wgpu or megakernel trees.
- No `fn cpu_reference` exists in vyre-foundation/src or vyre-reference/src.
- Exactly one fusion-planning entry point exists across three trees.
- No PipelineCacheStore implementation lives in vyre-driver-wgpu/src.

Exits nonzero on:

- a declared path missing
- any row count above its floor

Findings:

- `set -uo pipefail` without `-e`.
- Three of five rows previously scanned paths that had moved and scored 0, which is at or below every floor, so they passed by measuring nothing. The missing-path assertion that fixed it is the single most important shape in this whole layer and the Rust scanner enforces it for every gate: a scan path that does not exist is an error, not an empty result.
- The P-DELETE-1 row was pinned at 18 against an actual 22 and was measuring the wrong property. Its replacement counts one owner instead.

### `scripts/check_unsafe_budget.sh`

Subject: present.

Invoked by: gates.yml.

Gate: xtask/src/gates/lint_hygiene.rs.

Assertions:

- scripts/unsafe_budget.txt exists.
- The set of tracked .rs files carrying allow(unsafe_code) equals the reviewed list exactly, in both directions.

Exits nonzero on:

- missing budget file
- a file carrying the override that is not listed
- a listed file that no longer carries it

Findings:

- The two-directional set comparison is the right shape and is preserved. Three of the previous nine whitelist entries named a crate that no longer existed, which is why the rule is a set equality over the override marker rather than a path whitelist.

### `scripts/check_unsafe_justifications.sh`

Subject: present.

Invoked by: nothing; named in two crate lib.rs headers, a README and xtask/src/gates/hygiene_matrix.rs.

Gate: xtask/src/gates/lint_hygiene.rs.

Assertions:

- Every `unsafe {` block in production Rust source has a `// SAFETY: <text>` comment in the contiguous comment block up to eight lines above.
- No SAFETY comment is a cop-out (TODO, FIXME, unclear, investigate, unknown, tbd, ???).

Exits nonzero on:

- a block with no SAFETY comment
- a cop-out SAFETY comment

Findings:

- Nothing invokes it, so Law H has no CI enforcement.
- `grep -rn ... 2>/dev/null` reads a failed search as no unsafe blocks.
- It scans the filesystem from the repository root rather than tracked files, and it runs one `sed -n` per candidate line, so the cost is one process per line of backward scan.

### `scripts/check_workspace_filesystem.sh`

Subject: present.

Invoked by: gates.yml.

Gate: xtask/src/gates/manifest_contract.rs.

Assertions:

- Every Cargo.toml within four levels is either listed in [workspace.members] or declares its own [workspace].

Exits nonzero on:

- any orphan crate

Findings:

- `set -uo pipefail` without `-e`.
- `find -maxdepth 4` decides the roster, so a crate nested deeper is never checked, and the members list is parsed with awk over manifest text rather than a TOML reader.
- Two path patterns are excluded by name (target-codex/package, vyre-bench/competitors). The gate keeps `vyre-bench/competitors` as a declared exemption and reports it, so the orphan stays visible. `target-codex/package` is a build output directory, never tracked, so it could not match anything the gate reads; the gate reported it as an allowance naming no manifest and the allowance is deleted. `workspace-membership` therefore reports 1, the reviewed orphan, which is its honest number.

### `scripts/cli_docs.py`

Subject: partly gone: docs/CLI.toml, the crate READMEs and xtask/src/subcommands.rs survive; docs/CLI.md was deleted at b1ed746d1c.

Invoked by: docs-ci.yml, xtask/tests/tree_contracts/cli_docs.rs; named in docs/DOCS.toml, CHANGELOG.md, unreleased.toml and docs_manifest.py.

Gate: xtask/src/gates/doc_contract.rs owns the manifest assertions, the binary inventory, the help routes, the subcommand-registry equality and the generated README blocks, and honours ctx.write; the missing docs/CLI.md is one finding.

Assertions:

- docs/CLI.toml declares schema_version 1 and at least one [[binary]].
- Every [[binary]] declares package, name, readme, audience, hardware, environment, config, failure and exit_codes.
- No two rows name the same package and binary.
- Every audience is public or internal.
- Every declared readme exists.
- The declared binary set equals the bin targets cargo metadata reports.
- cargo build --workspace --bins succeeds.
- Every help route exits 0, prints non-empty valid UTF-8, and stays under 1 MiB.
- The xtask subcommand set discovered from its help equals the name rows of xtask/src/subcommands.rs SUBCOMMANDS.
- docs/CLI.md and the generated CLI block of every declared README match the generated content.

Exits nonzero on:

- any manifest failure
- an inventory mismatch
- a failed workspace bin build
- a help route that exits nonzero or prints nothing
- an xtask help/dispatch mismatch
- stale generated CLI documentation
- a README with only one generated marker

Findings:

- build_bins does environment.setdefault("CARGO_BUILD_JOBS", "1"), which is build configuration set outside .cargo/config.toml. The gate does not set it.
- docs-ci.yml fails on every tree today: docs/CLI.md does not exist, so the drift comparison reads a missing file.
- The subcommand-registry assertion parses xtask/src/subcommands.rs with a line-anchored regex on `name: "..."`. The gate reads the registry directly, so a reformatting of that file cannot silently empty the comparison.

### `scripts/crate_ownership.py`

Subject: partly gone: docs/CRATE_OWNERSHIP.toml survives, and both generated documents docs/CRATE_GRAPH.md and docs/OWNERSHIP.md were deleted at b1ed746d1c.

Invoked by: nothing directly; consumed by lib/check_layering.py, xtask/src/gates/check_tier_deps.rs and xtask/tests/tree_contracts/crate_ownership_registry.rs.

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

Subject: present: it writes into crate READMEs, which survive. It links docs/testing/<crate>.md, which is gone, so every generated block points at a file no reader can open.

Invoked by: nothing directly; xtask/tests/tree_contracts/crate_readmes.rs shells into it, and all 31 crate READMEs name it as their generator.

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

### `scripts/docs_manifest.py`

Subject: partly gone: docs/DOCS.toml survives and declares 132 page rows; every one of those documents, both generated navigation files and four of the twelve owner authorities were deleted at b1ed746d1c.

Invoked by: gates.yml, xtask/src/docs/docs_check.rs (registry gate docs-check), xtask/tests/tree_contracts/docs_manifest_completeness.rs, check_docs_index.sh, check_docs_links.sh, scripts/test_docs_manifest.py.

Gate: xtask/src/gates/doc_contract.rs, which validates the manifest, reports each declared-but-absent page and each missing owner authority as a finding, and regenerates the navigation under ctx.write.

Assertions:

- docs/DOCS.toml declares version 2.
- Every [[owner]] declares a unique id and an authority document that exists.
- Every [[page]] declares path, title, status in current/generated/superseded/archived, audience in user/extension/contributor/release, a known owner, a kind from the eleven-value set, a non-empty section, an authority that exists, and a generation of manual or generated.
- No two page rows declare the same path.
- Every page under archive/ or legacy/ is archived.
- Inactive pages set nav = false and active markdown pages set nav = true.
- status generated and generation generated imply each other.
- A generated page names exactly one generator, that generator exists, and the page is not its own authority.
- A manual page names no generator.
- The declared page set equals the Markdown actually present, in both directions.
- No user-facing or extension-facing manual page leaks an internal marker: a local:// link, BACKLOG.md, subagent or agent-swarm or worktree protocol wording, or a phase, slice or tranche number.
- docs/SUMMARY.md and docs/INDEX.md match the content generated from the manifest.

Exits nonzero on:

- any of the above; validate collects every failure and prints them all before exiting 1

Findings:

- It is the one script here that collects failures instead of raising on the first, which is the shape the gate contract requires. The port keeps that behaviour and the rest of the layer gains it.
- The registry gate docs-check already runs it and is pinned at output_lines = 0, so the pin is a line count of a script that today cannot produce zero lines. The findings pin replaces it.
- render_index runs cargo metadata --no-deps. The gate reads the workspace manifests it already walks and does not shell into cargo.

### `scripts/final-launch.sh`

Subject: partly gone: the release notes it passes to `gh release create` are docs/release/v<version>.md, and docs/ carries no Markdown after b1ed746d1c.

Invoked by: nothing; it is the manual launch entry point.

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

### `scripts/lib/cargo_runner.py`

Subject: present.

Invoked by: release_docs.py chain.

Gate: not ported; superseded by the Rust runner selection.

Assertions:

- Same runner order as the shell half, for Python tooling.

Exits nonzero on:

- never

### `scripts/lib/cargo_runner.sh`

Subject: present.

Invoked by: 21 scripts.

Gate: not ported; the runner choice becomes std::process::Command on ./cargo_full inside the gates that need cargo.

Assertions:

- Selects VYRE_CARGO_RUNNER, then ./cargo_full, then cargo.

Exits nonzero on:

- never; it only assigns

Findings:

- Exports CARGO_BUILD_JOBS=1. A build-affecting variable set outside .cargo/config.toml, so every gate that sources this file builds with a different configuration than a bare cargo invocation. Repair belongs in .cargo/config.toml.

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
- unregistered --case reference

### `scripts/lib/check_every_source_file_is_reachable.py`

Subject: present.

Invoked by: check_every_source_file_is_reachable.sh from gates.yml.

Gate: xtask/src/gates/source_reachability.rs.

Assertions:

- Every declared cargo target path (lib, build, bin, test, bench, example) names a tracked file.
- Every autodiscovered target root resolves.
- Every `mod name;` resolves to a tracked name.rs or name/mod.rs, honouring #[path] and inline module nesting.
- Every tracked .rs file is reachable from some cargo target root, or exempt.
- Every template-manifest exemption covers at least one tracked .rs file.
- Every trybuild pattern matches at least one tracked file.

Exits nonzero on:

- target path naming nothing
- unresolvable mod declaration
- orphaned .rs file
- exemption matching nothing
- trybuild pattern matching nothing
- no tracked .rs or Cargo.toml

### `scripts/lib/check_every_source_file_parses.py`

Subject: present.

Invoked by: check_every_source_file_parses.sh from gates.yml.

Gate: xtask/src/gates/source_reachability.rs.

Assertions:

- Every tracked .rs file outside a scaffolding template root parses under edition 2021, measured by rustfmt --check.
- rustfmt exiting other than 0 or 1 without an error line is a tool failure, not a clean tree.
- A template exemption whose files all parse is slack and fails.

Exits nonzero on:

- unparseable tracked .rs file
- rustfmt tool failure
- exemption covering no file
- template exemption no longer needed
- every file exempt

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

### `scripts/lib/check_internal_deps_have_versions.py`

Subject: present.

Invoked by: check_internal_deps_have_versions.sh from gates.yml.

Gate: xtask/src/gates/manifest_contract.rs.

Assertions:

- Every `workspace = true` internal dependency names an existing [workspace.dependencies] entry.
- Every internal dependency on a published member carries a version outside dev-dependencies.
- No internal dependency on a `publish = false` member carries a version.
- [workspace.dependencies] entries naming members obey both halves, since every inheritor gets them.
- The workspace has at least one publishable and one unpublished member, so neither half scans nothing.

Exits nonzero on:

- dangling workspace inheritance
- path-only dependency on a published member
- versioned dependency on an unpublishable member
- root Cargo.toml untracked
- member manifest untracked
- empty members
- one half of the roster empty

### `scripts/lib/check_layering.py`

Subject: present. File deleted; every assertion is in `layering`.

Invoked by: nothing. It ran from check_layering.sh, which is also deleted.

Gate: xtask/src/gates/layering.rs.

Assertions:

- Every workspace member has a docs/CRATE_OWNERSHIP.toml [[crate]] entry.
- Every layer a member declares has a neutrality decision in NEUTRAL_LAYERS.
- Every NEUTRAL_LAYERS decision is used by at least one member.
- Every BACKEND_APIS name is present in [workspace.dependencies].
- Every resolved internal cargo edge stays inside the crate's declared dependency closure.
- No crate in a substrate-neutral layer reaches ash, cudarc, metal, naga or wgpu.
- `cargo tree` resolves; a failure is fatal rather than an empty graph.

Exits nonzero on:

- unregistered member
- layer with no neutrality decision
- unused neutrality decision
- BACKEND_APIS name absent from workspace deps
- layer violation
- cargo tree failure
- empty workspace.members
- empty [[crate]] registry

The gate reads the graph from the manifests and `Cargo.lock` rather than from
`cargo tree`, so the cargo-failure assertion has no subject left: there is no
cargo invocation to fail. The manifest edge set is the member's own default
features, which is what `cargo tree --edges=normal` prints, and the lockfile
supplies third-party edges, so a neutral crate that reaches a backend API only
through another third-party crate is now caught where the tree-based form saw it
only if cargo printed it.

### `scripts/lib/check_path_deps_resolve.py`

Subject: present.

Invoked by: check_path_deps_resolve.sh from gates.yml.

Gate: xtask/src/gates/manifest_contract.rs.

Assertions:

- Every `path` in every dependency table of every tracked manifest resolves to a tracked Cargo.toml inside the repository.
- Every `workspace = true` dependency and inherited package field names an existing root table entry.
- Every workspace.members entry resolves, and every glob pattern matches at least one Cargo.toml.
- Every [patch] path entry resolves.
- root workspace.dependencies is non-empty, so the inheritance scan is not vacuous.

Exits nonzero on:

- unresolvable or escaping path
- untracked target manifest
- dangling workspace inheritance
- members glob matching nothing
- no tracked Cargo.toml
- empty root workspace.dependencies

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

### `scripts/lib/source_scan.sh`

Subject: present.

Invoked by: 10 shell gates plus vyre-foundation/tests/ci_script_frozen_contract_coupling.rs.

Gate: replaced by the Rust scanner in xtask/src/gates/scan.rs, which is fallible by type rather than by exit status.

Assertions:

- Every scan path passed to a rule exists on disk (returns 2 otherwise).
- `git ls-files` succeeds; a listing failure is fatal rather than an empty result.
- A grep exit status other than 0, 1 or 123 is a failed search, not a clean tree.
- A tracked file absent from the working tree is reported, not silently dropped.
- `vyre_file_has` exits 2 when its file is missing or the search fails, because `set -e` does not fire inside an `if` condition.

Exits nonzero on:

- scan path missing
- git ls-files failure
- grep status not in {0,1,123}
- vyre_file_has on a missing file

Findings:

- The reason this file exists is a live finding class: nine gates read a failed ripgrep as a clean tree, and check_no_hot_path_inventory.sh passed on every possible tree because this ripgrep build has no PCRE2.

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

- It sets CARGO_BUILD_JOBS on three command lines and reads CARGO_TARGET_DIR to locate the binary. Both are build configuration that belongs in .cargo/config.toml, and the target directory is available from `cargo metadata` without the variable.

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

### `scripts/release_docs.py`

Subject: present.

Invoked by: nothing directly; named in CHANGELOG.md, docs/DOCS.toml, unreleased.toml, docs_manifest.py and xtask/tests/release_docs.rs.

Gate: kept as the one script exception. xtask/src/gates/changelog.rs runs `python3 scripts/release_docs.py --check` and honours ctx.write by running --write.

Assertions:

- release/release-train.toml has non-empty [versions] and [release_groups].
- Every release group declares an owner/repository, a known version key and at least one package.
- No package belongs to two release groups.
- The train declares exactly three approval-gated external actions with non-empty unique ids.
- release/changes/unreleased.toml declares schema_version 1, at least one fragment, unique non-empty ids, supported categories, and unique non-empty text.
- CHANGELOG.md contains every required_release_note_token from the train.
- scripts/final-launch.sh contains ten guarded launch steps, in order.
- CHANGELOG.md matches the content generated from the train and the fragments.

Exits nonzero on:

- unreadable or invalid train
- invalid fragments
- missing release token
- a missing or out-of-order launch step
- a stale changelog

Findings:

- Its launch-order assertion names the literal `-- launch-state --output`, which the evidence lane is removing. The token becomes `-- launch-state --write` in the same commit that changes final-launch.sh.
- Porting it properly means moving the changelog renderer into Rust: textwrap.wrap at width 79 with break_long_words and break_on_hyphens off, the [Unreleased] section splice, and the train-identity lines. That is a byte-exact text generator, and a reimplementation that differs by one wrap point rewrites CHANGELOG.md on the first --write. The port needs a fixture corpus of fragment sets with the expected rendering committed beside it, generated by this script before it is deleted, and a gate proving the Rust renderer reproduces every fixture byte for byte. Until that corpus exists the gate calls the script.

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

### `scripts/test_docs_manifest.py`

Subject: present: it drives docs_manifest.py against temporary fixtures and reads no deleted document.

Invoked by: gates.yml.

Gate: xtask/src/gates/doc_contract.rs carries the same five negative fixtures as Rust unit tests in the gate module, which is where a test of a gate belongs.

Assertions:

- docs_manifest.validate accepts a manifest whose authority and generation records are coherent.
- It rejects a duplicate page path, an unclassified page present on disk, an inactive page with nav = true, an unknown owner, a missing authority source, a missing generator and an unarchived page under archive/.
- It flags a BACKLOG.md reference and a phase identifier on an extension-facing page.
- It accepts both on a contributor-facing page.
- It rejects a generated page that names no generator.

Exits nonzero on:

- any assertion failure in the five unittest cases

Findings:

- This is the only test in scripts/, and it is the only negative-fixture coverage the documentation manifest has. Losing it would mean the port's collect-every-failure behaviour is asserted by nothing, so the five cases move into the gate module's tests rather than into the gate.

### `scripts/testing_guides.py`

Subject: partly gone: docs/CRATE_OWNERSHIP.toml, docs/testing/TESTING.toml and every crate manifest survive; all of docs/testing/*.md was deleted at b1ed746d1c.

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

### `scripts/unsafe_budget.txt`

Subject: present.

Invoked by: check_unsafe_budget.sh.

Gate: kept as data at scripts/unsafe_budget.txt is not an option once scripts/ is gone; it moves to xtask/unsafe-budget.txt beside gate-baselines.toml.

Assertions:

- Data. The reviewed list of files permitted to carry allow(unsafe_code).

Exits nonzero on:

- not executable

### `scripts/vyre_smoke.sh`

Subject: present.

Invoked by: nothing.

Gate: xtask/src/gates/smoke.rs.

Assertions:

- rustc, cargo and the workspace MSRV are reported.
- vyre-driver-wgpu compiles in release, used as the adapter-probe signal.
- examples/three_substrate_parity/manifest.toml exists.
- dispatch_allocation_contract, pipeline_cache_contract and dispatch_hot_path pass.

Exits nonzero on:

- compile failure
- missing parity manifest
- any of the three contract suites failing

Findings:

- Nothing invokes it, so three named wgpu contract suites have no scheduled run outside the workspace default suite.
- Step 2 calls itself an adapter probe and is a compile check, so it cannot observe an adapter at all. The comment admits it. The gate keeps the compile assertion under an honest name and reports the missing probe as a finding.
- `set -uo pipefail` without `-e`.

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

### `scripts/wgsl_to_rust/Cargo.lock`

Subject: gone: scripts/wgsl_to_rust holds only this lockfile.

Invoked by: nothing.

Gate: reported: a lockfile with no manifest and no source beside it.

Assertions:

- Data. A lockfile for a tool directory that carries no Cargo.toml and no source.

Exits nonzero on:

- not executable

Findings:

- scripts/wgsl_to_rust/ contains one file, Cargo.lock. The crate it locked does not exist in the tree. Nothing reads it and nothing can.

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
| `backend-extension` | In `vyre-driver-wgpu/src/backend_impl.rs`, delete the `BackendPrecedence` inventory submission block. | 0 to 1 |
| `backend-extension` | In `vyre-driver/src/backend/registry/acquire.rs`, rename `registered_backends_by_precedence_slice` at its definition and its call. | 0 to 1 |
| `backend-extension` | Add `let id = "cuda";` to any file under `vyre-driver/src/backend/registry/`. | 0 to 1 |
| `readback-ring` | In `vyre-driver-wgpu/src/engine/record_and_readback.rs`, rename `.arm_ticket(` to `.arm(` at its definition and its call sites. | 0 to 1 |
| `readback-ring` | In `vyre-driver-wgpu/src/lib.rs`, replace `ReadbackRingSet::new()` with `ReadbackRingSet::default()`. | 0 to 1 |
| `program-wire-fields` | Add `pub scratch_hint: u32,` to `Program` in `vyre-foundation/src/ir_inner/model/program/definition.rs`. | 0 to 1 |
| `program-wire-fields` | Delete every mention of `workgroup_size` from `vyre-foundation/src/serial/wire/encode/to_wire.rs` and `decode/from_wire.rs`. | 0 to 1 |
| `program-wire-fields` | Rename `pub struct Program` to `pub struct ProgramInner`. | gate errors, which is the intended outcome: the declaration is located, not named, so losing it is unmeasurable rather than clean |
| `frozen-contracts` | Add a method to `pub trait ExprVisitor` in `vyre-foundation/src/visit/expr.rs`. | 1 to 2 |
| `frozen-contracts` | Delete `docs/frozen-traits/MutationClass.txt`. | 1 to 2 |
| `frozen-contracts` | Reindent the body of `pub enum AlgebraicLaw` by four spaces. | stays 1; indentation is not part of the contract |
| `file-size` | Append 200 blank lines to `vyre-foundation/src/optimizer/fact_cache.rs` (measured 570, cap 599). | 75 to 76 |
| `file-size` | Append 60 lines to `vyre-libs/src/decode/inflate.rs` (measured 554, cap 582). | 75 to 76 |
| `file-size` | Add a row to the audit ceilings naming `vyre-does-not-exist/src/lib.rs`. | 75 to 76 |
| `gpu-loudness` | Add `#[cfg(not(feature = "gpu"))]` above a test in `vyre-driver-wgpu/tests/` with no loud abort within ten lines above or twenty below. | 2 to 3 |
| `gpu-loudness` | Add `if adapter.is_err() { return; }` to a test body. | 2 to 3 |
| `gpu-loudness` | Add `Backend::acquire_or_panic();` five lines below an existing finding site in `conform/vyre-conform/tests/cert_artifact/prove_failure_contracts.rs`. | 2 to 1, which is the allowance working rather than a failure |
| `unification` | Add a second `pub fn child_bodies` to any file under `vyre-foundation/src`. | 0 to 2, because a row over its ceiling reports every site |
| `unification` | Add `BufferAccess::infer(` to a file under `vyre-runtime/src/megakernel`. | 0 to 1 |
| `unification` | Rename the directory `vyre-foundation/src/execution_plan` and update its `mod` declaration. | 0 to 1, reported as a path that does not exist rather than as a clean row |
| `evidence-paths` | In any artifact under `release/evidence`, change one cited path to a filename that does not exist but keeps a tree extension. | 18 to 19 |
| `evidence-paths` | Add `"manifest": "target/debug/build.rs"` to an artifact object, with `target/` gitignored. | 18 to 19, in the gitignored class rather than the missing class |
| `evidence-paths` | Change a cited path to `1.2.0`. | stays 18; a version string is not a citation |
| `invariant-paths` | In `vyre-spec/src/invariants.rs`, change a cited conformance test path to one that does not exist. | 0 to 1 |
| `doc-claims` | In `contracts/doc_claims_manifest.toml`, change one `phrase` to text its document does not contain. | 0 to 1 |
| `doc-claims` | Delete the `test` key from one claim. | 0 to 1, reported as an incomplete row rather than as a missing test |
| `hot-path-owned-dispatch` | Add a `.dispatch(` call taking an owned row to a file under `vyre-runtime/src`. | 114 to 116, one finding for the occurrence and one for its being unreviewed |
| `hot-path-inventory` | Delete a reviewed allowlist entry that currently matches an occurrence. | 12 to 13 |
| `hot-path-unbounded-read` | Add `fs::read_to_string(` to a file under `vyre-driver/src` outside the reviewed cache modules. | 1 to 3 |
| `lint-unsafe-budget` | Add `#![allow(unsafe_code)]` to a crate root not in the reviewed budget. | 0 to 1 |
| `lint-unsafe-justification` | Add an `unsafe {` block with no SAFETY comment above it. | 2 to 3 |
| `lint-missing-docs-override` | Add `#![allow(missing_docs)]` to any crate root. | 0 to 1 |
| `proptest-coverage` | Delete two property-test files, taking the count from 182 to 180 against the floor of 181. | 0 to 1 |
| `audit-status` | Remove one status tag from a row in a status-managed audit document. | 0 to 1 |
| `repo-hygiene` | Add a second `BACKLOG.md` under any crate directory and track it. | 2 to 3 |
| `shader-source` | Add `out.push_str("@compute");` to a file under `vyre-driver-wgpu/src`. | 0 to 1 |
| `layering` | Add `vyre-lints.workspace = true` to `[dependencies]` in `vyre-spec/Cargo.toml`. | 0 to 29, one per member that reaches it, each naming the chain |
| `layering` | Add `wgpu.workspace = true` to `[dependencies]` in `vyre-spec/Cargo.toml`. | 0 to 48, every neutral member that reaches vyre-spec |
| `layering` | Delete the `[[crate]]` entry for `vyre-spec` from `docs/CRATE_OWNERSHIP.toml`. | gate errors: an unregistered member has an empty closure, so the roster is unreviewed rather than clean |
| `layering` | Change `layer` on the `vyre-spec` entry to a name NEUTRAL_LAYERS does not hold. | gate errors: a layer with no neutrality decision would be skipped |
| `layering` | Add `("invented", true)` to `NEUTRAL_LAYERS`. | gate errors: a decision no member uses is an allowance nothing needs |
| `layering` | Add `"not-a-crate"` to `BACKEND_APIS`. | gate errors: a boundary named after a crate the workspace never resolves cannot be crossed |

Three of these are negative controls rather than injections: the reindented
frozen enum, the version string cited as a path, and the loud abort added beside a
loudness finding. A gate that moves on those is over-reporting, which mutes it
just as surely as under-reporting.

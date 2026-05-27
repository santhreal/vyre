# CRITIQUE: Surgec UX Flow Audit

Date: 2026-04-22
Reviewed: 2026-04-23
Scope: end-to-end user experience after the recent `surgec` landings
Method: source audit, doc audit, fixture/test audit, workflow audit, and local command-path checks on a GPU-equipped workstation

This critique focuses on what a security engineer would actually try to do from the outside. The engine surface moved quickly; the user-facing flow did not keep up. Findings are organized by the eight requested flows and written as user-visible failures.

## 1. First-time install

1. CRITICAL | [.github/workflows/publish.yml](/media/mukund-thiru/SanthData/Santh/.github/workflows/publish.yml:46) | The release path only covers crate publishing and never builds or attaches prebuilt binaries, so the first-time install story is "compile the whole Rust stack yourself" on every platform. A new user looking for a curl-able binary or GitHub release asset finds nothing.
Suggested fix: add cross-platform release packaging for Linux/macOS/Windows and document binary installation before asking users to compile from source.

2. CRITICAL | [libs/tools/surgec/README.md](/media/mukund-thiru/SanthData/Santh/libs/tools/surgec/README.md:1) | The primary README has no install section at all. A first-time user cannot tell whether `cargo install surgec` is supported, whether crates.io is current, which Rust version is required, or whether GPU drivers/system packages are mandatory.
Suggested fix: add an explicit install matrix with crates.io status, minimum Rust toolchain, supported OSes, GPU prerequisites, and a source-build fallback.

3. HIGH | [libs/tools/surgec/Cargo.toml](/media/mukund-thiru/SanthData/Santh/libs/tools/surgec/Cargo.toml:16) | The default feature set enables `gpu`, but the install docs never explain what happens on hosts with broken WGPU/CUDA/Vulkan stacks. The user symptom is a source build that appears normal until runtime, where scan startup fails with backend acquisition errors that were never part of the installation checklist.
Suggested fix: document the GPU backend requirements explicitly and provide a supported CPU-only installation mode with clear tradeoff language.

4. HIGH | [libs/tools/surgec/Cargo.toml](/media/mukund-thiru/SanthData/Santh/libs/tools/surgec/Cargo.toml:59) | The package exposes two binaries and no documented `cargo run --bin surgec` path for contributors working from a checkout. The result is immediate confusion for anyone validating the tool from source, especially because `cargo run -p surgec` is ambiguous in practice.
Suggested fix: document the exact source-development run commands and make the contributor path explicit in the README.

5. HIGH | [libs/tools/surgec/Cargo.toml](/media/mukund-thiru/SanthData/Santh/libs/tools/surgec/Cargo.toml:36) | The binary depends on a deep graph of workspace path crates, so the install experience from a clean clone is "build the world" rather than "build the CLI". Users pay workspace-wide compile cost and cache-lock contention on first contact, which reads as tool flakiness rather than engine sophistication.
Suggested fix: slim the install graph for the CLI, publish the supporting crates cleanly, and document the expected first-build footprint and duration.

## 2. First scan

6. CRITICAL | [libs/tools/surgec/src/main.rs](/media/mukund-thiru/SanthData/Santh/libs/tools/surgec/src/main.rs:239) | The obvious first command in the task brief, `surgec scan <dir>`, is not supported. The CLI errors with "missing arguments" because it requires `surgec scan <rules_path> <target_path>`, so the shipped standard library is not discoverable from the primary scan command.
Suggested fix: make `surgec scan <dir>` the default path by auto-discovering bundled stdlib rules and reserving explicit rules paths for advanced usage.

7. CRITICAL | [libs/tools/surgec/README.md](/media/mukund-thiru/SanthData/Santh/libs/tools/surgec/README.md:53) | The README examples assume the user already knows where `rules/` lives and how to point the scanner at it. On a clean checkout of a target repo, the first scan fails conceptually because the docs teach "scan your corpus with this repo's source tree beside you", not "install and scan a project".
Suggested fix: add a zero-config "scan a directory with the built-in rule pack" example first, then relegate custom-rule paths to a separate advanced section.

8. HIGH | [libs/tools/surgec/src/main.rs](/media/mukund-thiru/SanthData/Santh/libs/tools/surgec/src/main.rs:311) | Every local scan recompiles rules, acquires the backend, scans, and only then renders output. There is no visible warm-cache path or rule-pack caching at the user level, so first-run latency looks like the normal product experience even when the work is mostly setup cost.
Suggested fix: separate compile and scan phases in the UX, cache bundled rule packs automatically, and print cold-vs-warm timing so users know what the tool is doing.

9. HIGH | [libs/tools/surgec/tests/launch_rules.rs](/media/mukund-thiru/SanthData/Santh/libs/tools/surgec/tests/launch_rules.rs:1) | The launch gate still targets an older set of rules and does not exercise the recently landed `static_iv_reuse`, `weak_kdf_iterations`, or `prompt_injection` rules. A first-time user scanning a Python repo cannot trust that the flagship new detections actually fire end-to-end.
Suggested fix: add positive and negative launch tests plus fixtures for all newly shipped rules before claiming them as landed.

10. HIGH | [libs/tools/surgec/tests/fixtures/rules](/media/mukund-thiru/SanthData/Santh/libs/tools/surgec/tests/fixtures/rules:1) | There are no shipped fixtures for `static_iv_reuse`, `weak_kdf_iterations`, or `prompt_injection`, so there is no canonical corpus a new user can run to verify that the install and first-scan flow works. The missing experience is "I installed it, I scanned the sample, and I saw the promised detections."
Suggested fix: add minimal reproducible fixtures for each new rule and reference them from the README as the first verification step after install.

## 3. Authoring a new rule

11. HIGH | [libs/tools/surgec/AUTHORING.md](/media/mukund-thiru/SanthData/Santh/libs/tools/surgec/AUTHORING.md:47) | AUTHORING tells users to add fixtures under `tests/corpus/...`, but the repo actually stores rule fixtures under `tests/fixtures/rules/...`. A new rule author following the doc literally will put files in the wrong place and then conclude the test harness is broken.
Suggested fix: update AUTHORING to match the actual fixture layout and include a copy-pasteable directory skeleton for a new rule.

12. HIGH | [libs/tools/surgec/rules/README.md](/media/mukund-thiru/SanthData/Santh/libs/tools/surgec/rules/README.md:14) | The rules guide points to `CONFIGURATION.md`, but that document is not present. The user symptom is a broken documentation trail exactly at the point where they need to learn how rule loading, labels, and runtime configuration interact.
Suggested fix: either add the missing configuration guide or remove the reference and inline the necessary setup details into `rules/README.md`.

13. HIGH | [libs/tools/surgec/rules/README.md](/media/mukund-thiru/SanthData/Santh/libs/tools/surgec/rules/README.md:118) | The rules guide references `docs/SEVERITY_TAXONOMY.md`, but that file is missing. That leaves new authors without a source of truth for severity selection, which directly degrades CI policy, SARIF ranking, and triage consistency.
Suggested fix: add the taxonomy doc and link every severity example back to it, or remove the dead link and embed the taxonomy where authors need it.

14. CRITICAL | [libs/tools/surgec/rules/README.md](/media/mukund-thiru/SanthData/Santh/libs/tools/surgec/rules/README.md:101) | The authoring guide documents a `[[test_inputs]]` workflow, but there is no corresponding implementation surface in the scanned source or tests. A new author can spend time writing metadata that the product never consumes, which is worse than missing docs because it teaches a false workflow.
Suggested fix: either implement `[[test_inputs]]` end-to-end or delete the section immediately and replace it with the real test harness contract.

15. HIGH | [libs/tools/surgec/AUTHORING.md](/media/mukund-thiru/SanthData/Santh/libs/tools/surgec/AUTHORING.md:63) | The review checklist stops at `cargo check`, `cargo test --no-run`, and `cargo doc`. It never walks a user through "author a rule, compile it, scan a target, confirm the finding, inspect the output", so the documented loop is build-centric rather than detection-centric.
Suggested fix: replace the checklist with a literal 10-minute happy path that ends in a real `surgec scan` result, not just workspace compilation.

## 4. CI integration

16. CRITICAL | [libs/tools/surgec/src/main.rs](/media/mukund-thiru/SanthData/Santh/libs/tools/surgec/src/main.rs:25) | The binary exits nonzero only on execution errors, not on findings. A GitHub Actions user wiring in `surgec scan` today gets a successful job even when detections are present, which breaks the most basic CI expectation.
Suggested fix: define and document explicit exit-code semantics for "clean", "findings present", and "tool failure", with flags to tune policy.

17. HIGH | [libs/tools/surgec/README.md](/media/mukund-thiru/SanthData/Santh/libs/tools/surgec/README.md:1) | There is no GitHub Actions example, no sample job, and no documented invocation for CI. Users wanting to gate pull requests are forced to reverse-engineer the command line and guess where SARIF should be uploaded.
Suggested fix: add a copy-pasteable GitHub Actions recipe covering install, cache strategy, scan invocation, exit-code policy, and SARIF upload.

18. CRITICAL | [libs/tools/surgec/src/output/sarif.rs](/media/mukund-thiru/SanthData/Santh/libs/tools/surgec/src/output/sarif.rs:171) | SARIF regions are emitted with byte offsets and byte lengths only. Most CI consumers and code-scanning UIs expect line/column locations for useful annotations, so the product emits technically structured output that is still poor as a developer review experience.
Suggested fix: map findings back to source line/column spans and include snippets and logical locations in SARIF results.

19. CRITICAL | [libs/tools/surgec/src/scan/diff_scan.rs](/media/mukund-thiru/SanthData/Santh/libs/tools/surgec/src/scan/diff_scan.rs:77) | The GitHub review-comment conversion uses `result_slots.first()` as the review `line`. That is not a source line; it is an internal slot identifier. Any CI or review integration built on this path will comment on the wrong lines or fail unpredictably.
Suggested fix: compute review comment locations from actual source spans and refuse to emit comments when a trustworthy line mapping is unavailable.

20. HIGH | [libs/tools/surgec/src/main.rs](/media/mukund-thiru/SanthData/Santh/libs/tools/surgec/src/main.rs:563) | CI users can only choose `json` or `sarif`, with no documented summary mode and no stable machine-friendly "policy result" envelope. The symptom is brittle action scripting: every team has to invent its own parser to answer "did this job find anything actionable?"
Suggested fix: add a stable CI summary format plus a machine-readable policy object that separates scanner health from finding counts and severities.

## 5. Editor support

21. CRITICAL | [libs/tools/surgec/src/main.rs](/media/mukund-thiru/SanthData/Santh/libs/tools/surgec/src/main.rs:32) | There is no LSP or editor-facing subcommand at all. A user looking for inline diagnostics, background scan, or IDE integration discovers that the product stops at a batch CLI even though the rule language and provenance system are already sophisticated enough to justify richer tooling.
Suggested fix: define an editor integration surface, starting with an LSP or daemon mode that reuses compiled bundles and emits location-rich diagnostics incrementally.

22. HIGH | [libs/tools/surgec/src/main.rs](/media/mukund-thiru/SanthData/Santh/libs/tools/surgec/src/main.rs:151) | Argument parsing is fully manual. That means no shell completion generation, no typed flag metadata, no structured per-command help, and no editor-friendly introspection of valid flags. The CLI feels handmade where users expect discoverability.
Suggested fix: move to a declarative CLI layer that can generate help, completion, and schema metadata automatically.

23. HIGH | [libs/tools/surgec/src/main.rs](/media/mukund-thiru/SanthData/Santh/libs/tools/surgec/src/main.rs:218) | The scanner tells users "Fix: see surgec scan --help" for unknown flags, but there is no real subcommand help path behind that message. In editors and terminals alike, the corrective guidance points to a dead end.
Suggested fix: implement subcommand-specific help immediately and ensure every recovery message points to a working command.

24. HIGH | [libs/tools/surgec/src/output/report.rs](/media/mukund-thiru/SanthData/Santh/libs/tools/surgec/src/output/report.rs:38) | The JSON result surface lacks line/column, source snippets, matched predicate names, and matched label families. That makes editor integrations much harder because the raw data is not rich enough to support "jump to cause" or "hover to see why this matched".
Suggested fix: extend the JSON schema to include human-usable source spans and rule-evaluation context rather than only internal engine identifiers.

25. MEDIUM | [libs/tools/surgec/rules/README.md](/media/mukund-thiru/SanthData/Santh/libs/tools/surgec/rules/README.md:1) | The rule docs provide no editor support story for rule authors: no syntax highlighter, no schema, no predicate auto-complete, and no "lint this rule file" command. Authoring remains a blind text-editing exercise.
Suggested fix: publish a rule schema, a `surgec validate`/`surgec lint` command, and basic editor metadata for predicate completion and inline rule errors.

## 6. Debugging a false positive

26. CRITICAL | [libs/tools/surgec/src/output/report.rs](/media/mukund-thiru/SanthData/Santh/libs/tools/surgec/src/output/report.rs:38) | When a rule fires wrongly, the JSON report does not tell the user which predicate matched, which label family contributed, or which clause actually triggered the result. The user sees "rule fired" without enough evidence to tighten the rule safely.
Suggested fix: include a predicate-by-predicate match explanation with clause IDs, contributing labels, and the final decision path in the default output.

27. HIGH | [libs/tools/surgec/src/output/explainer.rs](/media/mukund-thiru/SanthData/Santh/libs/tools/surgec/src/output/explainer.rs:7) | `--explain` is wired to exploit-chain code-flow annotation, not rule-evaluation explanation. A user debugging a false positive reasonably expects "why did this rule fire on this file?", but the current explain path answers a different question entirely.
Suggested fix: split exploit-chain enrichment from rule-explanation and make the latter the default meaning of `--explain` for scan results.

28. HIGH | [libs/tools/surgec/src/scan/provenance.rs](/media/mukund-thiru/SanthData/Santh/libs/tools/surgec/src/scan/provenance.rs:15) | Provenance is reduced to hashes and an adapter fingerprint. That is useful for integrity, but not for debugging. A false-positive reporter cannot tell which compiled rule pack, which source rule file, or which engine build produced the match without additional archaeology.
Suggested fix: augment provenance with rule file paths, bundle identity, scanner version, and a human-readable evaluation fingerprint.

29. HIGH | [libs/tools/surgec/src/output/sarif.rs](/media/mukund-thiru/SanthData/Santh/libs/tools/surgec/src/output/sarif.rs:238) | SARIF messages collapse findings to `rule_name (severity: ...)`, and rule descriptors omit `help`. In code-scanning UIs, that leaves maintainers with almost no remediation context or explanation of what the rule actually saw.
Suggested fix: populate SARIF help text, rule descriptions, and remediation guidance directly from the rule metadata and evaluation trace.

30. MEDIUM | [libs/tools/surgec/src/scan/suppressions.rs](/media/mukund-thiru/SanthData/Santh/libs/tools/surgec/src/scan/suppressions.rs:1) | Suppressions exist internally, but the user-facing scan flow does not provide a clear "why was this suppressed or kept?" inspection path. Debugging a false positive becomes guesswork around hidden policy rather than an explicit reviewable decision.
Suggested fix: expose suppression decisions in output and add a command to explain active suppressions for a finding.

## 7. Debugging a false negative

31. CRITICAL | [libs/tools/surgec/src/main.rs](/media/mukund-thiru/SanthData/Santh/libs/tools/surgec/src/main.rs:32) | There is no `surgec explain rule <name> --against <file>` or equivalent diagnostic path. When a user says "this should have fired", the CLI offers no direct workflow to trace rule applicability against a specific file.
Suggested fix: add a targeted explain command that runs one rule against one artifact and prints the evaluation tree, failed predicates, and data bindings.

32. CRITICAL | [libs/tools/surgec/docs/vs-mythos.md](/media/mukund-thiru/SanthData/Santh/libs/tools/surgec/docs/vs-mythos.md:58) | The docs claim "Predicate-level trace" for "Why did this NOT fire", but the shipped CLI surface does not provide that capability. This is a user-facing promise gap, not just missing polish.
Suggested fix: either ship the predicate-trace UX immediately or remove the claim until the command exists.

33. HIGH | [libs/tools/surgec/tests/vyre_output.rs](/media/mukund-thiru/SanthData/Santh/libs/tools/surgec/tests/vyre_output.rs:13) | The output-focused tests skip when expected corpus files are absent and do not validate the negative-debugging experience. That means the exact flow users need for false negatives remains unexercised by the test suite.
Suggested fix: add explicit end-to-end tests for missed detections, including expected explain output and rule-evaluation traces.

34. HIGH | [libs/tools/surgec/src/output/report.rs](/media/mukund-thiru/SanthData/Santh/libs/tools/surgec/src/output/report.rs:45) | The report includes `clause_index` but not the reason a clause failed or the predicates that short-circuited. For false negatives, that is almost the worst possible debug surface: enough internals to imply traceability, not enough to actually explain the miss.
Suggested fix: record both positive and negative clause-evaluation traces and render them on demand for a given rule/file pair.

35. HIGH | [libs/tools/surgec/src/main.rs](/media/mukund-thiru/SanthData/Santh/libs/tools/surgec/src/main.rs:311) | Local scan only returns findings. It does not surface rule-application counts, skipped files, parser failures, unsupported-language reasons, or filtered-out candidates. A user cannot tell whether a miss came from no match, no parse, no labels, or no rule coverage.
Suggested fix: add a diagnostic mode that reports scan coverage, skipped artifacts, parser failures, and per-rule application statistics alongside findings.

## 8. Performance reality-check

36. HIGH | [libs/tools/surgec/SCOPE.md](/media/mukund-thiru/SanthData/Santh/libs/tools/surgec/SCOPE.md:115) | The project narrative claims surgec can run "1000× faster than any" competitor, but the normal CLI does not print timing, throughput, cold/warm separation, or benchmark provenance. Users see a wall-clock runtime with no way to reconcile it to the headline claim.
Suggested fix: print honest end-of-scan performance summaries and tie benchmark claims to reproducible methodology and build metadata.

37. HIGH | [libs/tools/surgec/benches/vs_competition.rs](/media/mukund-thiru/SanthData/Santh/libs/tools/surgec/benches/vs_competition.rs:64) | The competition benchmark tells users to inspect `docs/BENCHMARK.md`, but that document is missing. The result is a broken paper trail exactly where skeptical users go to understand how the advertised speedup was measured.
Suggested fix: add the missing benchmark guide with corpus details, cold/warm methodology, hardware notes, and commands to reproduce every published claim.

38. HIGH | [libs/tools/surgec/src/main.rs](/media/mukund-thiru/SanthData/Santh/libs/tools/surgec/src/main.rs:311) | Normal scan output contains findings but no elapsed time, throughput, bytes scanned, files scanned, or compile-vs-execution split. On a 1GB corpus, users cannot tell whether the product is fast, slow, stalled, or dominated by compilation overhead.
Suggested fix: emit structured performance counters in every run and reserve silent output for an explicit quiet mode.

39. HIGH | [libs/performance/matching/vyre/benches/RESULTS.md](/media/mukund-thiru/SanthData/Santh/libs/performance/matching/vyre/benches/RESULTS.md:102) | The backend benchmark record itself admits that GPU execution-path timing is still missing and the gap remains open, yet the user-facing tool surface does not disclose that limitation anywhere. That makes performance claims look stronger than the currently instrumented evidence supports.
Suggested fix: align CLI claims and docs with the actual benchmark coverage until GPU execution timing is fully surfaced and validated.

40. MEDIUM | [libs/tools/surgec/src/scan/watcher.rs](/media/mukund-thiru/SanthData/Santh/libs/tools/surgec/src/scan/watcher.rs:110) | Watch mode emits raw JSON snapshots but no timing, no diff summary, and no indication of whether the latest result was a warm incremental pass or a full rescanning event. For users judging real-world performance from repeated scans, the UX hides the very information they need.
Suggested fix: add watch-mode telemetry showing rescan cause, files touched, elapsed time, and warm-cache status on every iteration.

## Bottom line

The dominant UX failure is not engine weakness; it is product surface mismatch. The scanner core exposes serious capability, but the first-run path, authoring path, CI path, and debugging path still require source-code archaeology. The recent landings improved internals faster than the CLI, docs, and diagnostics caught up.

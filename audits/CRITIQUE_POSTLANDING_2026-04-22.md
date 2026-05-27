# CRITIQUE_POSTLANDING_2026-04-22

Read-only post-landing regression hunt over the recent `vyre` / `surgec` landing wave.

Scope read for this pass:
- `libs/performance/matching/vyre/CLAUDE.md`
- `libs/performance/matching/vyre/audits/RELEASE_GATE.md`
- `libs/tools/surgec/SCOPE.md`
- prior critiques under `libs/performance/matching/vyre/.internals/audits/CRITIQUE_CODEX*_2026-04-22*.md`

Only fresh regressions, seam failures, or newly introduced fidelity gaps are listed below. Where a prior critique closed one class of bug but the follow-up landing introduced a new one, that is called out explicitly.

1. CRITICAL | `libs/tools/surgec/src/scan/collector.rs:294-305` | The live scan path still dispatches every `ClauseDispatchPlan` against every file/layer without consulting any applicability gate, so the recent applicability work is not load-bearing in the hot path. Fix: build `FileMetadata` per collected file/layer and prune plans before `build_clause_inputs` / `dispatch_rules`.

2. HIGH | `libs/tools/surgec/src/main.rs:313-354`, `libs/tools/surgec/src/scan/collector.rs:200-215` | The default CLI path still uses `scan_gpu_with_context`, not `scan_gpu_with_chains`, so the new exploit-graph reconstruction never reaches normal `surgec scan` output. Fix: make exploit-chain reconstruction part of the primary scan pipeline and thread structured chains into both JSON and SARIF.

3. HIGH | `libs/tools/surgec/src/main.rs:524-599`, `libs/tools/surgec/src/output/explainer.rs:20-59` | The CLI SARIF path bypasses the real exploit-graph code and instead synthesizes `codeFlows` from `result_slots`, so the “explainer” is not evidence-backed. Fix: route SARIF generation through the real chain builder, or omit code flows until they are derived from actual graph reachability.

4. HIGH | `libs/tools/surgec/src/output/explainer.rs:3-6`, `libs/tools/surgec/src/output/explainer.rs:36-55` | The explainer treats `result_slots` as ordered taint hops (`source` → `touched` → `sink`), but those slots are just firing thread ids, not a data-flow path. Fix: delete the synthetic path construction and emit only graph-backed steps produced by exploit-chain reconstruction.

5. HIGH | `libs/tools/surgec/src/main.rs:552-561`, `libs/tools/surgec/src/scan/dispatch.rs:54-58` | The recent SARIF region fix introduced a new correctness bug: `result_slots` are `gid`s that require the scan pipeline’s offsets buffer for translation, but CLI SARIF now writes them directly as `byte_offset`. This is distinct from the earlier “region missing” bug in `CRITIQUE_CODEX_SECURITY_PERF_2026-04-22.md #40`. Fix: preserve the offsets buffer in the result envelope and translate `gid -> byte offset` before emitting SARIF.

6. HIGH | `libs/tools/surgec/src/scan/exploit_graph.rs:273-282`, `libs/tools/surgec/src/scan/dispatch.rs:54-58` | Exploit-graph ordering uses `result_slots.min()` as a file offset, so chain ordering is currently keyed by thread ids rather than byte positions. Fix: feed reconstructed byte offsets into `NodeRecord::order_key` instead of raw `gid`s.

7. HIGH | `libs/tools/surgec/src/scan/distributed.rs:26-35` | `WireFinding` drops severity, confidence, provenance, primitive metadata, and decode-layer ancestry, so distributed scans cannot preserve the post-landing result contract. Fix: ship the full `FileFinding` envelope or a wire-compatible equivalent instead of a truncated transport struct.

8. HIGH | `libs/tools/surgec/src/main.rs:258-279`, `libs/tools/surgec/src/scan/distributed.rs:26-35` | The distributed mode returns a raw serialized `Vec<WireFinding>` instead of the normal `JsonScanOutput`, so the same `scan` command now emits two incompatible JSON schemas depending on flags. Fix: normalize distributed output through the same renderer used by local scans.

9. HIGH | `libs/tools/surgec/src/main.rs:281-310` | `surgec scan --watch` watches the target corpus only; rule changes are never observed or recompiled, so the recent watch-mode landing does not provide rule hot-reload on the main entrypoint. Fix: watch both rule roots and target roots, and recompile rules on rule mutations.

10. HIGH | `libs/tools/surgec/src/main.rs:281-310`, `libs/tools/surgec/src/scan/watcher.rs:1-8` | The CLI `--watch` path completely bypasses the new `scan::watcher` implementation and uses `stream_watch` instead, so the shipped watch semantics differ from the dedicated watch module that was just landed. Fix: consolidate on one watch implementation and delete the shadow path.

11. HIGH | `libs/tools/surgec/src/scan/stream_watch.rs:102-106`, `libs/tools/surgec/src/main.rs:283-287` | File deletions are reported as “changes”, and the callback then constructs a `Collector` for the deleted path; the next rescan fails on `scan target does not exist` and kills watch mode. Fix: distinguish deletions from modifications and either emit tombstones or rescan the containing corpus instead of the deleted file path.

12. MEDIUM | `libs/tools/surgec/src/scan/stream_watch.rs:10-11`, `libs/tools/surgec/src/scan/stream_watch.rs:41-46` | The active watch path is a 500 ms mtime poller rather than the `notify`-backed hot-reload described in the new watcher module, so it can miss coalesced edits and has materially worse latency than the documented path. Fix: remove the polling watcher from the public scan path and use the notify-based implementation consistently.

13. HIGH | `libs/tools/surgec/src/scan/watcher.rs:3-8`, `libs/tools/surgec/src/scan/watcher.rs:95-99`, `libs/tools/surgec/src/scan/watcher.rs:116-139` | The new watcher claims it recompiles affected rule files, but every change triggers `compile_and_cache(rules_dir, cache_root)` across the entire tree. Fix: track changed `.srg` paths and rebuild only the invalidated documents before merging.

14. HIGH | `libs/tools/surgec/src/scan/watcher.rs:142-156` | The watcher cache key is only the raw `.srg` file hash, so compiler changes, grammar-generator changes, label-set changes, and dependency changes all reuse stale `.srgb` blobs. Fix: salt the cache key with compiler version, grammar-gen wire version, and dependency fingerprint metadata.

15. HIGH | `libs/tools/surgec/src/scan/watcher.rs:194-199` | The dedicated watcher scans via `collector.scan_gpu(&backend)`, which flattens away file and decode-layer context that the rest of the landed output stack now depends on. Fix: use `scan_gpu_with_context` and keep the file-aware envelope all the way to stdout.

16. MEDIUM | `libs/tools/surgec/src/scan/watcher.rs:219-223` | `emit_findings` prints bare `dispatch::Finding` JSON lines, so path context, decode ancestry, and provenance disappear even though those fields exist in the normal scan output. Fix: emit the same structured JSON schema as `JsonScanOutput`.

17. HIGH | `libs/tools/surgec/src/scan/diff_replay.rs:49-78` | `diff_replay` dedupes by `rule_name -> whole-file hash`, so duplicate rule names across files or artifacts alias each other and produce false “not new” / “new” decisions. Fix: key by stable per-rule identity, not a plain name string.

18. HIGH | `libs/tools/surgec/src/scan/diff_replay.rs:3-7`, `libs/tools/surgec/src/scan/diff_replay.rs:66-74` | The module claims dedupe is done via the landed rule-provenance chain, but it actually hashes the entire `.srg` file and assigns that hash to every rule in the file. Fix: use the actual `Provenance` chain (rule source hash + program hash + adapter fingerprint) per rule, not one file hash.

19. HIGH | `libs/tools/surgec/src/scan/diff_replay.rs:81-84` | `diff_replay` runs `Collector::scan_gpu`, so even if the dedupe logic were correct it still strips file/decode provenance from the findings it returns. Fix: diff replay must run on file-aware findings and preserve the same evidence payload as normal scans.

20. HIGH | `libs/tools/surgec/src/scan/provenance.rs:28-46` | The landed `rule_source_hash` is not a hash of rule source; it ignores clause predicates, file selectors, severity tiers, report blocks, and artifact scope, so materially different authored rules can collide. Fix: hash the canonicalized SURGE AST (or canonical source) for the specific rule instead of a small metadata subset.

21. MEDIUM | `libs/tools/surgec/src/scan/provenance.rs:49` | `adapter_fingerprint` is only `backend.id():backend.version()`, which is not enough to reproduce adapter-specific behavior across different physical GPUs or driver stacks. Fix: include concrete adapter/vendor/device information from the backend capability surface.

22. HIGH | `libs/tools/surgec/src/output/sarif.rs:105-123`, `libs/tools/surgec/src/main.rs:537-567` | The new provenance field never reaches SARIF at all, even though JSON output exposes it. Fix: attach provenance hashes and adapter identity in SARIF `properties` so offline replay and diffing survive format conversion.

23. HIGH | `libs/tools/surgec/src/output/sarif.rs:215-299`, `libs/tools/surgec/src/main.rs:524-599` | There are now two SARIF builders with diverging semantics: `output::sarif::{to_sarif,to_sarif_with_chains}` and the hand-rolled `build_sarif` in `main.rs`. The CLI uses the latter, so the newly landed chain helpers are effectively dead code. Fix: keep one SARIF construction path and make every caller use it.

24. HIGH | `libs/tools/surgec/src/compile/fuse.rs:42-48` | The fusion module documentation explicitly states that many security ops still emit empty bodies and that fusion “treats this cleanly,” which is a post-landing stub admission in shipped code. Fix: either implement those ops fully or fail compilation when a fused rule body is inert; do not normalize empty-body execution as a supported state.

25. HIGH | `libs/tools/surgec/src/compile/fuse.rs:190-201`, `libs/tools/surgec/src/compile/fuse.rs:219-234`, `libs/tools/surgec/src/compile/compile.rs:178-186` | The applicability salvage path introduced after `CRITIQUE_CODEX_2026-04-22.md #74-75` keys selectors by plain `rule_name`, while compiled rules still keep only unqualified names. Duplicate rule names across top-level and artifact scopes will therefore alias to the wrong applicability mask. Fix: give every compiled rule a stable qualified identity and key fusion metadata on that identity.

26. HIGH | `libs/tools/surgec/src/compile/specialize.rs:16-21`, `libs/performance/matching/vyre/audits/RELEASE_GATE.md:548-666` | `specialize_program` only wires decode/fixpoint/all-of/zone/count specializations; none of the landed I.4 neural prefilter or I.9 subgroup DFA claims are represented here, so those innovations are not compile-time load-bearing. Fix: add explicit specialization hooks for every shipped innovation or stop advertising them as landed.

27. HIGH | `libs/tools/surgec/src/compile/specialize.rs:153-196` | `specialize_decode_chain` is hard-coded to exact buffer names (`decoded`, `transitions`, `accept`, `matches`), so semantically equivalent scanner programs silently miss decode fusion if a lowerer names buffers differently. Fix: detect decode/scan shapes structurally from the program graph instead of by stringly-typed buffer names.

28. CRITICAL | `libs/tools/surgec/src/scan/collector.rs:161-195`, `libs/tools/surgec/src/scan/collector.rs:254-271`, `libs/performance/matching/vyre/audits/RELEASE_GATE.md:507-546` | The real scan path still reads whole files into host memory and runs decode layers on CPU before dispatch, so the headline I.1 fused decode pipeline and I.3 zero-copy NVMe→GPU ingest are not load-bearing. Fix: move ingestion and decode chaining into the dispatch pipeline and keep decoded buffers GPU-resident.

29. HIGH | `libs/tools/surgec/src/scan/collector.rs:217-251` | The helper path used by collector tests still bakes in eager whole-file materialization, which cements the old ingest contract and makes it harder to evolve toward streaming GPU ingestion. Fix: refactor file collection to expose a streamed/shared buffer abstraction instead of `Vec<CollectedFile>` of owned bytes.

30. HIGH | `libs/tools/surgec/rules/stdlib/go_frontend.srg:1-31`, `libs/tools/surgec/src/compile/structural.rs:29-55` | The new Go frontend rules use `go:go_call`, `go:channel_send`, `go:channel_receive`, and `go:deferred_call`, but the structural compile path still only lowers C/C++ `call_to(@family)` rules. Fix: either implement Go structural lowering end-to-end or remove these rules from the shipped stdlib until they are executable.

31. HIGH | `libs/tools/surgec/rules/stdlib/go_frontend.srg:4`, `libs/tools/surgec/rules/stdlib/go_frontend.srg:12`, `libs/tools/surgec/rules/stdlib/go_frontend.srg:20`, `libs/tools/surgec/rules/stdlib/go_frontend.srg:28`, `libs/tools/surgec/rules/labels/*` | The Go rules reference `@worker_family`, `@channel_family`, and `@cleanup_family`, but no corresponding label TOML files exist under `rules/labels/`. Fix: add the missing family definitions or reject the rules during launch-rule assembly.

32. HIGH | `libs/tools/surgec/tests/launch_rules.rs:1-6`, `libs/tools/surgec/rules/stdlib/go_frontend.srg:1-31` | The launch-rule gate still frames itself as “the 10 missing v3 rules” and never references the newly added Go frontend rules, so the shipped launch corpus is no longer aligned with the claimed 20-rule minimum. Fix: update the launch gate to enumerate the real launch set and include the Go rules or remove them from the promised minimum.

33. HIGH | `libs/performance/matching/vyre/vyre-libs/tests/go_frontend_corpus.rs:52-60`, `libs/performance/matching/vyre/CLAUDE.md:49` | The Go frontend corpus test runs every parser composition through `vyre_reference::run`, not an actual GPU backend, so it does not validate the GPU-resident frontend the gate now claims is shipped. Fix: add a real GPU execution gate for the Go frontend and keep the reference runner only as an oracle.

34. HIGH | `libs/tools/surgec/tests/launch_rules.rs:64-88` | The launch-rule test suite compiles/lower-checks the `.srg` files, then dispatches hand-built primitive programs instead of the compiled rule programs or the actual `Collector` scan path. Fix: run launch fixtures through `compile_paths` plus the real scan pipeline so the rule surface, lowering, and dispatch path are tested together.

35. HIGH | `libs/tools/surgec/src/scan/confidence.rs:41-52` | The new confidence heuristic boosts findings whose rule name or primitive contains `sanitiz`, `clean`, or `escape`, which in practice rewards the presence of sanitizers rather than treating them as evidence that exploitability is lower. Fix: invert this heuristic or replace it with graph-backed evidence from actual sanitizer bypass paths.

36. MEDIUM | `libs/tools/surgec/src/scan/confidence.rs:30-38` | The confidence code treats `result_slots.len()` as exploit-path length, but `result_slots` are firing thread ids, not path steps, so the “path length” penalty has no semantic meaning. Fix: compute confidence from actual chain depth / proof structure, not slot count.

37. MEDIUM | `libs/tools/surgec/src/main.rs:316-320`, `libs/tools/surgec/src/scan/dispatch.rs:183-200` | Confidence is now computed once in `dispatch_rules` and then recomputed again in `main`, creating two sites that can drift if the heuristic changes. Fix: compute confidence once in the canonical result builder and treat it as immutable output data.

38. HIGH | `libs/tools/surgec/src/main.rs:322-328` | `--diff base..head` still performs a full scan and only filters by changed path afterwards, so the “incremental” path pays the whole scan cost and only hides unchanged results at the end. Fix: use the changed-file set to constrain collection before dispatch starts.

39. HIGH | `libs/tools/surgec/src/main.rs:267-270`, `libs/tools/surgec/src/scan/distributed.rs:12-17` | Distributed scans collapse bundle mode and compiled-rule mode into a single `rules_path: String`; when `--bundle` is used the coordinator sends an empty rules path and the transport contract loses whether the worker should load a bundle or a rules tree. Fix: make the wire task explicitly distinguish rule roots from bundles and include bundle metadata on the wire.

40. HIGH | `libs/tools/surgec/src/scan/confidence.rs:67-75` | The confidence test helper was not updated for the landed `provenance` field, so the new core result type already has compile-breaking test scaffolding. Fix: update every `Finding` fixture to initialize the full post-landing struct and add a builder helper to avoid future drift.

41. HIGH | `libs/tools/surgec/src/output/explainer.rs:88-100` | The explainer test fixture also omits the landed `provenance` field, which means the synthetic explainer path is already out of sync with the real result type. Fix: centralize `Finding` test construction and require full-field initialization.

42. HIGH | `libs/tools/surgec/src/output/sarif.rs:433-455` | The SARIF chain tests still build `dispatch::Finding` without the new provenance field, so the stale alternate SARIF path is not even maintained against the new result model. Fix: update the fixtures and delete the duplicate SARIF path if it is no longer authoritative.

43. HIGH | `libs/tools/surgec/tests/exploit_graph_e2e.rs:9-20` | The new exploit-graph end-to-end test helper constructs `DispatchFinding` without `confidence` and `provenance`, which is a direct regression in the just-landed graph test harness. Fix: move exploit-graph fixtures onto a shared `FindingBuilder` that tracks schema changes centrally.

44. HIGH | `libs/tools/surgec/src/scan/auto_suppress.rs:83-95` | The auto-suppression test helper omits the landed `confidence` field, so one of the new “post-processing” features is already compiling against an outdated finding shape. Fix: consolidate test fixture construction for `FileFinding` / `Finding`.

45. HIGH | `libs/tools/surgec/src/output/tfidf.rs:71-83` | The TF-IDF test helper also omits `confidence`, another sign that the recent result-model change was not propagated through the new auxiliary features. Fix: use a single shared test helper for finding construction across all output modules.

46. HIGH | `libs/tools/surgec/src/scan/diff_scan.rs:159-169` | The diff-scan test fixture omits `confidence`, so the “scan-path hygiene” landing left even adjacent review tooling stale against the core finding model. Fix: update the test fixture and add a compile-fail guard that rejects partial `Finding` constructors in the crate.

47. HIGH | `libs/tools/surgec/src/scan/distributed.rs:194-195` | The distributed test uses `Duration::from_millis(50)` without importing `Duration`, so the newly landed distributed path contains a basic compile regression in its own test module. Fix: import `std::time::Duration` in the test module and add `cargo check --tests -p surgec` to the local landing checklist.

48. MEDIUM | `libs/tools/surgec/src/scan/collector.rs:331-341` | The `FileFinding` doc example is stale against the landed `Finding` schema because it omits `provenance`, so even the public documentation no longer type-checks conceptually. Fix: update all examples to construct the full post-landing result type.

49. HIGH | `libs/tools/surgec/README.md:27-38` | The README still documents `CompiledDocument::evaluate_file(...)`, which no longer matches the shipped scan interface, so the public API docs reverted to a pre-landing contract. Fix: rewrite the usage section around `compile_paths` + `Collector` / CLI commands that actually exist.

50. MEDIUM | `libs/tools/surgec/README.md:43-50` | The pipeline overview still points readers at `parser/lexer.rs`, `parser/grammar.rs`, and `compiler/`, which are stale paths after the recent restructure. Fix: update the architecture section to current crate/module locations.

51. HIGH | `libs/tools/surgec/AUTHORING.md:3-18`, `libs/tools/surgec/src/main.rs:281-354`, `libs/tools/surgec/src/scan/watcher.rs:194-199`, `libs/tools/surgec/src/scan/diff_replay.rs:81-84` | AUTHORING still says “surgec never runs a Program,” but the current binary, watch path, and diff-replay path all dispatch GPU programs directly. Fix: rewrite AUTHORING around the actual split of responsibilities instead of the old dissolve-era invariant.

52. MEDIUM | `libs/performance/matching/vyre/Cargo.toml:100`, `libs/tools/surgec/grammar-gen/Cargo.toml:18` | The session-specific clap pin mismatch is still present (`=4.5.21` vs `=4.6.0`), which keeps the workspace on split CLI dependency versions right after a landing wave that already tripped dependency sloppiness. Fix: converge on one clap pin across the workspace and add a dependency-drift check.

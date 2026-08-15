# Changelog

All notable changes to vyre are documented here. Follows Keep a Changelog.

## [Unreleased]

### Added

- Grouped affine INT4 linear now provides a typed batched program builder that
  dequantizes each immutable weight tile once and reuses it across independent
  resident batch rows. Release evidence measures normalized per-inference
  latency.
- The vyre-safetensors adapter reads bounded safetensors headers and sharded
  indexes without reading tensor payloads, confines shard paths to the
  checkpoint root, rejects duplicate or unmapped tensors, validates
  caller-supplied dtype and shape requirements, and streams complete shards
  against an exact trusted BLAKE3 set before returning an immutable checkpoint
  identity.
- The runtime now owns immutable resources, reusable artifact instances, and
  mutable leased state through one budgeted residency boundary. Cold and warm
  admission, rollback, cancellation, generation-checked reset, completion,
  eviction, and manager destruction release resources without exposing stale
  state.
- ProgramGraph now composes reusable Programs through canonical typed value
  identities, explicit consumer and output ports, symbolic or concrete shapes,
  access and lifetime contracts, and validated state transitions. Its bounded
  VGR0 wire format embeds existing VIR0 Programs and rejects implicit casts,
  rank drift, alias conflicts, dangling state, malformed framing, and hostile
  counts before mutation.
- ProgramGraph now validates complete compositions and derives canonical
  topological schedules, inclusive value liveness, and deterministic
  interval-colored allocation plans. Invocation-local values reuse only
  nonoverlapping slots, while immutable weights, sequence-state generations,
  and caller-visible outputs retain dedicated storage.
- ProgramGraph now derives one versioned, domain-separated BLAKE3 identity from
  canonical topology and Programs, typed port contracts, artifact schema,
  validated model configuration, exact symbolic bindings, and verified
  immutable-weight identities. Mutable sequence-state contents are excluded, so
  cache growth reuses compiled artifacts while any executable or provenance
  change invalidates the key.
- The neural library now provides gated RMSNorm with float32 accumulation,
  source-dtype rounding, learned scaling, float32 SiLU gating, and exact
  last-dimension row isolation. The reference interpreter now executes
  canonical F16 and BF16 loads, stores, and F32 conversions with
  round-to-nearest-even semantics while preserving raw element-byte APIs.
- The neural library now provides last-dimension L2 normalization for grouped
  query and key heads. It accumulates sum-of-squares in F32, applies epsilon
  inside the canonical inverse-square-root contract, isolates rows exactly, and
  converts output once to F32, F16, or BF16.
- The neural library now executes floating channel-major depthwise causal
  convolution with exact left padding, masks, bias, SiLU, F16/BF16 conversion,
  and output truncation. A short-chunk route emits an explicit next-state
  generation whose outputs and tail match full prefill across arbitrary token
  partitions and reset.
- The neural library now executes recurrent gated delta attention with F32 Q/K
  normalization, grouped heads, scaled queries, exponential decay, sigmoid
  beta, F32 matrix-state continuation, source-dtype output, and explicit
  validated prior and next state generations.
- The neural library now composes F32 query and key normalization,
  cache-position partial rotary embedding, explicit query-to-KV head grouping,
  and dynamically bounded causal attention in one typed ProgramGraph. Prompt
  and cached-decode routes exclude future cache rows and support configurable
  head ratios, head widths, and rotary dimensions.
- The neural library now executes a reusable dense gated-MLP ProgramGraph with
  learned RMSNorm, checkpoint-native output-major gate and up projections, F32
  SwiGLU math, output-major down projection, and residual addition. F16, BF16,
  and F32 storage use F32 normalization, projection accumulation, activation,
  and residual arithmetic with source-dtype boundaries.
- The neural library now executes chunk-size-64 gated delta prefill with F32
  cumulative log-decay, a strict lower-triangular solve, initial-state
  correction, chunk output reconstruction, and explicit final matrix state.
  F16, BF16, and F32 inputs retain F32 internal math. Guarded rows in the final
  structural tile cannot read padding, change state, or appear in truncated
  output.
- The reference and SPIR-V backends carry a tracked hostile-input closure
  target, `vyre-driver-reference/tests/hostile_input_closure_contract.rs` and
  `vyre-driver-spirv/tests/hostile_input_closure_contract.rs`. Both backends
  previously held their adversarial case in a file name the repository ignores,
  so the assertion ran only where someone had generated it and no branch could
  review it. The obligations come from `vyre_driver::hostile_input_closure`,
  and the ignore rule that asked for a crate-owned name is now satisfied for
  these two crates.
- `vyre_reference::reference_eval_lane_rotated` executes a program with the
  workgroup and invocation step order rotated left by a caller-chosen amount.
  Reversal is a symmetric permutation, so an implementation that confuses lane
  identity with step position can be made reversal-symmetric and stay wrong; a
  rotation separates the two and catches a subgroup collective that resolves
  its peers by physical step position.
- `vyre-foundation` has a criterion benchmark over the whole optimizer pass
  pipeline: the eight release corpus families, wide kernels at 16 and 64
  buffers, and a 4x8 loop nest. The pipeline had per-pass timing inside the
  scheduler report and no measurement of the pipeline as a whole, so a rewrite
  of the walk every pass runs through had nothing to be measured against.
- Every registry closure gate in the workspace is a tracked test. The gate
  asserts that each `pub fn ... -> Program` builder a crate publishes is
  reachable from its `inventory` registry or named by one of its tests, that
  the source enumeration finds at least a declared floor, and that a builder
  listed as an exception has not since gained coverage, which is what keeps the
  exception list only-shrinkable. Nineteen crates ran that gate from a file no
  checkout had: a blanket ignore rule for
  `<crate>/tests/adversarial_registry_closure.rs` kept each copy out of the
  repository, so the gate existed on one machine, ran in no
  continuous-integration job, and appeared in no review. Seven of those crates
  publish builders and now carry `<crate>/tests/registry_closure.rs` with that
  crate's floor and exception list; the other twelve enumerate no builders at
  all, so their gate could only ever pass, and a target that cannot fail is
  worse than none because it reads as coverage. Two of the invisible gates were
  failing. `vyre-driver` left the three hostile-input probe programs every
  backend's obligations are asserted against unregistered, untested and
  unlisted; those probes are now pinned by a contract target, because a probe
  whose buffer count or workgroup shape drifts leaves the backend assertions
  green while an over-supplied buffer case becomes the correct call and a zero
  workgroup dimension launches. `vyre-primitives` left three builders uncovered
  and unlisted; each is a second call convention over a builder that is swept,
  and each now says so on its own line. The same ignore rule also hid the lexer
  oracle's hostile-input half, which asserts that bytes outside the accepted
  subset fail loudly instead of tokenizing to a wrong answer; it is tracked
  under a name that says what it drives.
- `structure-gate` derives the set of crates that submit `inventory`
  registrations from the tree and rejects any `use <that crate> as _;` in
  workspace sources, naming `vyre-registry-link` as the way to read the
  registry instead. The submitting set is read at scan time rather than listed,
  so a new submitting crate is judged the moment it submits, and a companion
  contract fails when the scan finds no submitters, which is the state that
  would accept every discarding import in the tree.
- `xtask feature-isolation` judges whether every declared feature compiles on
  its own. A crate that builds under its default features and under
  `--all-features` can still be uncompilable under one feature alone, because
  `--all-features` is a union that supplies whatever a feature forgot to
  require, and only the consumer who enables that feature sees the break. The
  gate derives the axis from the tracked manifests at run time and holds all
  140 (member, feature) pairs, 35 members with one `--no-default-features`
  probe each plus 105 declared features, to the outcome recorded in
  `xtask/feature-isolation.toml`. A pair with no row, a row naming a pair no
  manifest declares, and a row whose outcome disagrees with the sweep are each
  a failure, so a new member or feature is red until a decision is recorded for
  it. The declaration half runs inside the gate sweep; the compile half is
  `--sweep` and CI owns it.
- A registration behind a feature the registry walker does not enable is
  invisible rather than absent, so nothing was red when one fell out. Two rules
  in `xtask-registry` close that class, both derived from the tree at run time
  rather than from a list: every file that submits a registration is reachable
  through the walker's declared feature selection, resolved transitively
  through the source crate's own feature table, and every feature that gates a
  registration is one the source crate's widest aggregate turns on. A new
  domain feature carrying registrations turns both red until the aggregate
  covers it, and a consumer naming the aggregate then needs no change.
- `graph_primitive_binding_contracts` reads the operation registry at run time
  and asserts that every registered graph primitive declares bindings that are
  the contiguous range from zero, and that a `changed` convergence flag sits
  read-write directly above the read-write frontier it reports on. A count of
  zero is deliberately not a signal: it is the documented sentinel for a
  runtime-sized storage buffer.
- A gate takes the published `min_` key set from the serialized
  `WallClockMinima` at run time and fails unless each key is either a sample
  floor or a member of the aggregate percentile field list. A minimum added to
  the record that no reader checks now fails the suite by name instead of
  reaching the artifacts unexamined.
- `vyre_foundation::transform::visit::try_for_each_node` walks every node in a
  body and every nested body, stopping at the first `Break`, and
  `for_each_node` now delegates to it. A short-circuiting scan outside the
  crate previously had to implement the abstract-by-default `NodeVisitor` and
  write a no-op body for every variant it did not care about, which is the cost
  that made one scan hand-roll its own descent with a catch-all arm instead.
  `node_buffer_refs`, `expr_buffer_ref` and their result types are public for
  the same reason: a lowering crate answering "what does this statement do to a
  buffer" now reads the exhaustive owner rather than restating it.
- Two suites close the classes above from the tree at run time. The Naga
  scanner's gate plants an atomic and a store in every body slot of every
  body-carrying `Node` variant and asserts the walk reaches both, and
  separately asserts that every buffer `node_buffer_refs` calls a write is
  reported as one; the variant set comes from `NODE_VARIANT_NAMES`, so a new
  variant is red before either question is asked.
  `vyre_backend_forwarding_closure` parses the trait declaration and the
  forwarding macros and fails when a method belongs to neither, when the
  grid-sync wrapper reaches a `Program`-carrying entry point without deciding
  the split, and when the wrapper hand-writes a forward the owner already
  emits.
- A gate resolves every name a `run:` step passes to cargo or to a shell
  against this tree: each `-p` package is a workspace member, each `--test` and
  `--bin` target is declared or auto-discovered by the package the same command
  line addresses, and each `scripts/` path is published. Steps are extracted by
  indentation and each is its own scope, so a target is never resolved against
  a neighbouring step's package. Nothing is listed in the test: a workflow
  added tomorrow is judged tomorrow, and renaming a target without updating its
  workflow is red locally instead of in CI.
- `loop_legality_collector_closure` plants a scalar read, a buffer load, a
  store and a binding in every operand and body slot of every declared `Node`
  variant and asserts each read set reports it. The variant set comes from
  `NODE_VARIANT_NAMES` through the shared IR fixtures, so a new variant is red
  before any of the three questions is asked.
- The scalar sweep is closed against the frozen public surface: every
  `NodeStorage` literal variant in the `vyre-foundation` snapshot must have a
  matrix row, a row naming a variant the surface does not carry fails, each row
  must sweep the full sampled depth or carry the reason it does not, each row
  must keep the operation count it declares today, and every operation a row
  leaves undeclared must still be refused by name. The dual sweep is closed the
  same way against the `vyre-reference` snapshot: every marker implementing
  `ReferenceEvaluator` is either exercised against its u32 contract or recorded
  with the payload shape it takes instead, and a marker that is both, or that
  the frozen surface no longer carries, fails. Adding a scalar literal, adding
  an evaluator, or widening the interpreter's operation table turns the suite
  red until a row records the decision.
- `vyre-primitives` gates three duplication classes it has reintroduced after
  previous cleanups, each deriving its member set from source at run time so a
  new member is red on its first run rather than absent from a hand-maintained
  roster. No dispatch-grid function may compute its own ceiling division, and
  every domain declaring one must be able to reach the owner, which is checked
  against the feature closure parsed from the manifest. No binding record may
  have its full name set spelled twice anywhere in the workspace, and a new
  record must publish canonical names. Every op routing a convergence loop
  through `routed_persistent_fixpoint` must register with the one routing
  contract, and none may observe the routing obligations privately. Each gate
  asserts its own member count first, because a structural signature that stops
  matching the tree otherwise turns into a test that passes by finding nothing.
- `scan_prefilter_width_closure` reads the `PrefilterWidth` variants out of the
  width table's own source at run time and fails when a width has no recorded
  dispatch ABI, when a recorded row names a width the source no longer
  declares, when a shipped program's bindings disagree with the row's counter,
  mask and sink layout, when a declared mask is never read by the emitted body,
  and when the widths stop forming a mask prefix chain. An empty or unparsable
  member set panics rather than passing, because an empty set makes every
  closure assertion vacuous.
- A gate over `vyre_primitives::graph` rejects the argument-transposition
  class. It walks the module directory and parses every declared signature at
  run time, so a CSR entry point added later is covered without editing a list:
  closure entry points must receive the graph as a bundle and must not declare
  three or more consecutive parameters of one type, and the wider slice-taking
  family must give each role a single name across the tree.
- `vyre-libs` gates the row contract the C typedef-annotation family agrees on,
  deriving its member set at run time from the `pub use` re-exports of
  `vyre_libs::parsing::c::parse::vast::typedef_ann` so a new builder is red on
  its first run until someone records what it writes. No builder may declare a
  zero-length buffer for an empty node table, every builder must size its row
  tables by the one shared stride at several row counts, and every pass must
  carry every VAST field it does not declare that it writes, checked by running
  each program on a table whose carried fields all differ per row.
- Two duplication classes in `vyre-libs` are gated, each deriving its member
  set from source at run time so a new member is red on its first run rather
  than missing from a hand-maintained roster. Every quantized dispatch entry
  point published as `_via` must carry a row asserting how it rejects a
  malformed backend readback, checked against the re-export list parsed from
  `vyre_libs::solvers::quantized_dispatch`. Every
  `ResidentCsrQueueMaterializer` variant must have a case pinning the step
  sequence it launches, checked against the variants parsed from
  `vyre_libs::graph::dispatch::csr_frontier_queue_scratch`. Each gate assertion
  names the fix rather than the failed comparison.
- `vyre_libs` runs every C-AST parity case on the reference interpreter through
  `vyre_driver_reference::CpuRefBackend`, dispatching the same programs the
  driver arms dispatch. A parity failure used to be observable only on a
  machine with a working adapter, so a case no arm named looked exactly like a
  case that passed. Splitting the arms separates a kernel that disagrees with
  its oracle, which fails here, from a device that disagrees with the kernel,
  which fails only there.
- `c_frontend::parity_matrix::assert_case_table_covers_fixture_file` reads a
  fixture family's own source at run time, collects the fixture builders it
  declares, and fails when the family's `CASES` table does not name one of
  them. Adding a construct and forgetting to enumerate it used to leave it
  proven on no backend and indistinguishable from a construct that passed,
  which is how GNU `__restrict` normalization stayed unproven on every device.
  A companion contract reads the fixture directory itself, so a new family is
  red until it either has a case table or is recorded as proven another way
  with the reason.
- `vyre_test_support::case_table::ArmCoverage` reads a shared case table's
  declared group names at run time, records which groups a crate's arm actually
  asserted, and fails naming every declared group that crate has no branch for.
  A corpus shared through an include has a hole a per-crate copy also had: the
  table declares a group, one crate grows an arm, the other does not, and the
  crate without the arm still passes because nothing in it mentions the group.
  Group-count and case-count floors make a collapsed table fail rather than
  report a clean sweep of an empty set. Both the dense-matvec and exploded-IFDS
  tables are enrolled, in four arms across two crates.

### Changed

- The standalone `vyre-harness` package is gone. Semantic operation identity,
  tier classification, and registration now live in `vyre-foundation`; library
  fixture views live in `vyre-libs`; conformance execution and parity policy
  live in `vyre-conform`; self-substrate behavior tests live with their owner.
- Scan products now return the foundation-owned `ByteRange { tag, start, end
  }`. The deprecated `Match` and `LiteralMatch` surfaces and the duplicate
  primitive range type are gone.
- The reference interpreter now consumes foundation-owned IR, diagnostics, and
  operation metadata directly instead of depending on the public `vyre` facade.
- PTX f32 canonicalization now uses native flush-to-zero multiplication plus
  NaN selection, preserving signed zero and canonical NaN semantics with fewer
  instructions and registers.
- Release benchmark commands now run 300 warmup samples before measurement so
  accelerator clock preconditioning is explicit and reproducible.
- Foundation now exposes IR-specific `IrError` and `IrResult` contracts instead
  of a cross-domain error sink. Reference interpretation, backend execution,
  WGPU device selection, and runtime framing return owner-local typed failures.
- Documentation pages now declare audience, owner, authority, kind, and
  generated/manual ownership. Crate dependency records declare purpose,
  features, target conditions, visibility, and destination seam, and optimizer
  pass reference pages are generated from the live pass registry.
- Floating-point parity now uses one foundation-owned comparison contract for
  semantic operation witnesses, including each operation's declared tolerance.
  The library operation catalog distinguishes the complete semantic inventory
  from its deterministic executable-fixture projection.
- C typedef row phases remain canonical callable operations. The operation
  matrix marks them as inlined callees whose execution coverage belongs to
  fixture-backed parent operations.
- Five checks that no workflow invoked now run in
  `.github/workflows/gates.yml`: the transitive layer closure, benchmark
  registry coverage, internal dependency versions, the feature-isolated MSRV
  sweep, and the oracle-matrix sweeps with their volume waves. Each step names
  the class it closes and why no other workflow sees it.
- `scripts/check_layering.sh` discarded `cargo tree` stderr, so a cargo that
  could not resolve the workspace printed a green result and exited 0. It now
  derives every workspace member, requires a `docs/CRATE_OWNERSHIP.toml`
  decision for each, and holds every transitive internal edge to the declared
  closure.
- `scripts/check_feature_msrv.sh` printed a hardcoded 19-entry matrix and
  exited 0 without compiling anything. It now derives one entry per publishable
  member per declared feature from tracked manifests and compiles each alone on
  the toolchain `[workspace.package].rust-version` names, which must be
  installed.
- The internal dependency version rule, the benchmark coverage rule, and both
  sweep runners derive their crate, case, and test rosters from tracked
  manifests and sources instead of hardcoded lists, no longer write outside the
  repository, and fail on a cargo failure instead of reporting a clean tree.
  The volume sweep runner had a three-crate list that left one tracked volume
  wave in no shard, and a shard index outside the shard count selected nothing
  and exited 0.
- The target-compiler shell has one owner, `vyre-driver/src/target_dialect.rs`.
  Every backend had written the same shell: hold the payload format, validate
  the profile, walk the selected modules, infer the dispatch grid from the
  logical element count, copy the canonical bindings through, and name the
  dialect in two error strings. Twenty-nine of the SPIR-V backend's seventy-six
  target-compiler lines were duplicated against cuda, metal and wgpu. A backend
  now declares a `TargetDialect` with its payload identity, its device limits
  and its emitter, and the shell derives the grid size, the workgroup size and
  the resource bindings so no dialect can disagree about them. Operator-facing
  failure text is unchanged.
- The hostile-input closure obligations every backend owes are stated once, in
  `vyre-driver/src/hostile_input_closure.rs`. Each backend crate had written
  the probe programs and the assertion text again against its own backend type,
  so the wording drifted per crate while the contract did not. A backend's
  adversarial target now supplies the backend and names itself. The reference
  backend keeps the hostile-bytes and trailing-input obligations it had; the
  SPIR-V backend keeps zero-workgroup and trailing-input, and reports rather
  than fails when no Vulkan device is present.
- The generated bitset and reduce sweeps are two parameterized matrices instead
  of twenty-five near-identical per-operation files.
  `sweep_bitset_oracle_matrix` carries one case list and one assertion body per
  call shape and fails when a registered `vyre-primitives::bitset` operation is
  neither swept nor exempted to a named suite; `sweep_reduce_oracle_matrix`
  does the same for the reducers. Every operation now runs the union of the
  populations the separate files used, and the in-place, exclusive-scan, and
  `cpu_ref_into` paths are swept for the first time. The CRC and graph-motif
  oracle matrices no longer restate the implementation they check: they compare
  against published CRC-32/ISO-HDLC and FNV-1a check vectors, a table-free
  bit-at-a-time walk, the concatenation law, and an edge-mask dictionary.
  Program-shape queries and the CSR frontier-step driver each have one owner
  rather than a copy per suite.
- Release tooling now has one owner per shared shape. Subcommand dispatch and
  the agreement check between the assignment table and each delegate crate live
  in `xtask::subcommands`; the operation table cells shared by the catalog and
  the operation inventory live in `xtask-registry::docs::schema_cells`; the
  optimizer catalog row published by both the integration matrix and the corpus
  pass manifest lives in `xtask-registry::release::optimizer_pass_rows`;
  bounded file reads route through `xtask::output_arg::read_text_bounded`. The
  xtask integration tests link as three binaries instead of ten and the
  registry tests as one instead of two.
- The graph single-source rules derive what they judge from the tree. They
  carried a table of eleven wrapper file names, each with identifier and prose
  fragments its source had to contain and a line ceiling, and every rename in
  `vyre-primitives` or test reorganisation in `vyre-libs` reported a refactor
  as a regression while a real fork that kept the old words would have passed.
  The wrapper set is now the directories under `vyre-libs/src/graph/dispatch/`
  whose name is also a module in `vyre-primitives/src/graph/`, the reference
  functions a wrapper must name are the ones its primitive publishes, and a new
  wrapper is judged without editing the rules. A floor on the derived set keeps
  a broken pairing from making every rule vacuous.
- The WGPU target compiler is declared as a
  `vyre_driver::target_dialect::TargetDialect` value instead of a hand-written
  implementation. The shell it duplicated (the compiler struct, the profile
  constructor, the factory, and a local word-comparison helper) is gone; what
  remains is the dialect record and the WGSL module emitter, so the device
  limits and payload format a backend reports are stated once in the same place
  every other backend states them.
- The WGPU C-frontend token tests share the fixture builder, the CPU pipeline
  runner, and the row readers with the CPU-side tests instead of carrying a
  second copy of each. The support file became a directory module, because a
  top-level file under `tests/` is its own test binary and therefore cannot
  reach the shared support module, which is what forced the copy. The shared
  fixture builder now models the lexer for whitespace and comment lexemes: they
  contribute source text and no token row, so a preprocessor fixture states
  real source layout instead of inventing rows for trivia.
- An expected expression-shape row is stated as one call. It was an eight-field
  tuple written across ten lines, so a precedence ladder read as three hundred
  lines of columns and two suites expecting the same row for the same operator
  matched as duplicated text. `binary_row` and `conditional_row` join
  `shape_none_row` in the shared support module and fix only what the grammar
  determines: a binary row's third link is the sentinel, and a ternary's
  operator spelling, precedence band, and associativity are constants.
  Precedence and associativity stay explicit arguments for binary operators,
  because deriving them from the parser's own table would make the assertion
  agree with the implementation by construction.
- The encoded-order IR rewrite walk in `vyre-pass-engine` has one owner,
  `optimizer::rewrite_walk`. Three copies of the same recursive rebuild
  existed, in the walk module, in const folding, and in resident arena-delta
  decoding, so a fix to the descent order reached one pass and not the others.
  Each pass now supplies only its per-expression decision: `fold_decision` for
  const folding and `arena_delta_decision` for the combined resident deltas,
  which applies const fold, then pattern-match action, then canonicalize swap.
  The fused level-wave kernel skeleton and the arena row buffer declarations
  are likewise owned once, by `optimizer::arena_kernel`, and the region entry
  unwrap by `optimizer::rewrite_program_entry`. Program output is unchanged;
  `vyre-pass-engine/tests/encoded_rewrite_walk_contract.rs` pins the walk and
  both decision functions against the rebuilt tree they must produce.
- The irregular Aho-Corasick bench case cluster has one owner per concern.
  `vyre-bench/src/cases/scan_ac_irregular/mod.rs` held module wiring, the
  shared constants, and one of the two case implementations, and the literal
  scan and the count preflight each carried their own copy of the workgroup
  validation, grid derivation, resident reset-then-scan dispatch, transfer
  accounting, and BenchRun assembly. The case now lives in `literals.rs`, the
  sampling path in `sample.rs`, the 4 MiB fixture in `haystack.rs`, and the
  match wire format in `match_triples.rs`; `support.rs`, which was named for a
  shape rather than a concern, is gone. Reported metrics, error strings, and
  readback ranges are unchanged.
- Three concepts in `vyre-lower` had two copies each. The constant behind an
  index operand is written in one of two encodings, a `Literal` op whose first
  operand is a pool index or an operand id that is itself a pool index, and
  bank-conflict and coalescing classification each resolved both encodings with
  its own copy, so a third encoding would have had to be added twice;
  `analyses::constant_u32_operand` now owns it, beside the producer map it
  reads. The one-buffer descriptor fixture was written out as a struct literal
  in `descriptor/` and again in `analyses/op_histogram/`, so a new
  `KernelDescriptor` field had to be chased through both; it is stated once,
  through the public builder. The load-count precondition fixture was
  byte-identical in texture promotion and AoS-to-SoA layout, doc comment
  included, and now lives with the traversal that defines the op shape it
  builds.
- The Aho-Corasick emit paths in `vyre-libs/src/scan/` read the flat
  output-record span through one owner. Six builders each wrote their own loop
  over `out_begin..out_end` binding `pattern_id` from `output_records`, and
  four of them also wrote their own per-pattern presence write,
  `presence[rs_base + (pattern_id >> 5)] |= 1 << (pattern_id & 31)`, spelled
  out node by node: the bounded presence and per-region presence builders, the
  fused presence-and-positions builder, the anchored region admission walk, and
  the fused region evidence program. A change to the record layout or to the
  bitmap indexing had to land in every copy, and a copy that was missed would
  produce a program that still built, still dispatched, and reported a
  different set of patterns. `bounded_ranges::output_record_loop_node`,
  `pattern_bitset_or_node` and its `presence_bit_write_node` form,
  `match_span_start_nodes`, and `bounded_walk_matched_nodes` now own the record
  loop, the bitset write, the floored match-start computation, and the
  accept-gated bounded walk. The emitted IR is unchanged: the presence
  reference, per-region ground-truth, fused presence-and-positions,
  transition-walk ownership, region-chain and conformance-matrix suites pass
  without repinning.
- `Program::canonicalized` reports what canonicalization changed instead of
  rebuilding the tree to discover it changed nothing. An already-canonical
  program keeps its entry body, its buffer table, and its memoized fingerprint
  rather than being replaced by an equal copy, and a program that only needs
  its buffer table sorted keeps its entry body. Canonicalization stays a fixed
  point in one application and a semantic no-op, both now asserted over the
  shipped release corpus and over shapes the corpus does not generate: a
  `Block` that owns a binding is a scope, so it survives, while a `Block` that
  owns none is spliced into its parent at every body position.
- The GPU queue bench families in `vyre-bench` have one owner per concern. The
  skewed-CSR and IFDS cases each carried their own copy of the
  queue-materialize dispatch sequence, the queue-driven traversal plan, the
  queue-closure payload and its CPU oracle, the single-dispatch frontier step,
  and the four-column Rust lexer dispatch, so a fix to any of them reached one
  family and not the other. Those concerns now live in
  `vyre-bench/src/cases/queue_materialize.rs`,
  `vyre-bench/src/cases/queue_traverse_plan.rs`,
  `vyre-bench/src/cases/queue_closure.rs`,
  `vyre-bench/src/cases/queue_closure_oracle.rs`,
  `vyre-bench/src/cases/frontier_step.rs`, and
  `vyre-bench/src/cases/rust_frontend/lex_columns.rs`, and nine cases are
  described by a `WorkloadDescription` plus `CaseOps` over a shared payload
  instead of a hand-written `BenchCase` impl. Case ids, metric names, error
  strings, and readback ranges are unchanged. The `vyre-bench` duplication pin
  drops from 3397 to 2491 lines.
- One CSR frontier step is stated once. `graph::csr_frontier_step` already
  owned the Program builder for both edge directions, but the host reference
  was written twice: `csr_forward_traverse` scattered into the destination bit
  and `csr_backward_traverse` gathered from it, each with its own row scan, its
  own edge-kind filter, its own bitset-word arithmetic, and its own copy of the
  CSR input validator, which the reverse walk reached across a sibling module
  to call. The walk is now one function whose argument is the direction, so the
  read endpoint and the write endpoint of an allowed edge are the only thing
  that varies, and the two published entry points every step op republishes
  under its own name (`cpu_ref` and `cpu_ref_into`, plus the predicate-level
  pairs for a masked step) come from `define_csr_frontier_step_cpu_ref` instead
  of six hand-copied parameter lists. `csr_forward_traverse_with_op_id` and
  `csr_backward_traverse_with_op_id` are gone: they differed only in the
  direction constant they passed, and `predicate::edge` and
  `predicate::size_argument_of` now build through the predicate program
  builders that already existed for that purpose. A missed copy is why a bounds
  guard can hold on one direction and not the other: the forward walk
  range-checked its destination against `node_count` while the reverse walk
  range-checked only against the frontier length, and a reference that is wrong
  in one direction blesses a wrong program in that direction. Emitted IR is
  unchanged, and the 2048-case hostile sweep in `csr_frontier_step` still
  compares both directions against two scalar walks written independently of
  the primitive.
- One queued-row CSR expansion ABI, one owner. Five entry points build the same
  queue-driven traversal and differ only in how lanes are assigned to a queued
  row, yet each restated the whole surface: `csr_queue_strided` declared
  `CsrQueueStridedForwardParams`, a field-for-field second name for
  `csr_frontier_queue::CsrQueueForwardTraverseParams`, then destructured it,
  repeated the zero-node and zero-capacity refusal with its own diagnostic, and
  reassembled the step spec; the delta pair repeated its thirteen-argument
  positional list in both files. A second name for one buffer contract is how a
  field added for one lane strategy silently misses the others, and the strided
  path also carried a hand-written CPU reference that only forwarded to the
  queue reference it is checked against. The queued-row ABI now owns its
  refusal rule and its step spec, one macro states the positional argument list
  per family, the strided op publishes the queue reference under its own names
  instead of copying it, and the offset-count overflow contract is one
  assertion both forward entry points run against the invalid-output program
  shape.
- A CSR closure states its call shapes once. Iterating a one-step CSR traversal
  to a fixpoint is one algorithm with several call shapes: allocate the two
  frontier buffers or borrow the caller-owned pair, observe each attempted step
  or not, report a malformed graph or panic on it. `csr_bidirectional`
  published six of those shapes and `csr_forward_or_changed` three, each
  retyping the same seven to ten argument names above a body that only
  forwarded them, and the `vyre-libs` reference facades for both ops retyped
  the pair again to add a dispatch-call count. Plumbing repeated per op is how
  one shape keeps a `#[must_use]`, a clippy allowance, or a fix the siblings
  never receive, and a facade with a hand-written body is one edit away from
  being a second implementation of a fixpoint it is only supposed to count.
  `graph::csr_closure_entry_points` now owns the argument list and generates
  the allocating, borrowing and panicking shapes above the hooked driver that
  carries the semantics; each op supplies its own documentation and its own
  diagnostic. The infallible shape is exported, so the composition facades in
  `vyre-libs` pass a counting hook to the primitive driver instead of restating
  it, and the traversal pipeline reference, which counted nothing at all, is
  now the primitive shape published under the pipeline name.
- The wgpu dispatch guards that every path repeats have one owner each.
  `vyre-driver-wgpu/src/dispatch_timeout.rs` answers the three questions a
  `DispatchConfig::timeout` raises: the absolute instant the budget expires at,
  whether a requested budget is even serviceable by this backend's queue and
  readback window, and whether the budget is already spent at a given phase.
  The spent-budget comparison existed in seven copies across the asynchronous,
  batched, resident and pending-dispatch paths, each with its own diagnostic
  wording, so a correction to the comparison reached only the copies it was
  applied to and a path holding a stale copy would keep waiting on a dispatch
  whose budget had run out instead of reporting the overrun. Wall-clock
  telemetry is likewise one call: `numeric::WGPU_NUMERIC.elapsed_nanos_u64`
  replaces three local nanosecond conversions that each re-decided what happens
  when an elapsed duration does not fit `u64`. Reported nanosecond values are
  unchanged.
- Four binding and readback decisions in `vyre-driver-wgpu` are stated once.
  `pipeline::binding::declared_byte_size` owns the
  element-count-times-element-size product with its overflow refusal,
  previously written out in handle validation, in persistent padded sizing, and
  on the input path of record-and-readback;
  `pipeline::binding::accepts_params_handle` owns the params-slot predicate
  that pre-recorded and persistent dispatch each spelled out.
  `TimestampRecorder::pass_writes` and `TimestampRecorder::finish_and_resolve`
  own the timestamp write pair and the query resolve, which three and two
  callers respectively had inlined. Borrowed dispatch into caller buffers now
  runs the same `readback_persistent_outputs` pass as the owned path rather
  than a second copy of the output loop, so a change to output trimming cannot
  reach one entry point and miss the other, which would have returned
  differently trimmed bytes for the same program depending on whether the
  caller supplied the output buffers. Emitted shader text, binding order, and
  readback ranges are unchanged.
- The in-memory and on-disk pipeline caches ask the same question about a
  rule-graph change. `pipeline::cache_impact::RuleImpactQuery::impact_mask`
  owns the reachability walk and its error mapping; both invalidation paths
  used to carry the eight-argument list and their own mapping around it, so the
  two could disagree about which entries a change reaches and one cache would
  serve a pipeline the other had already discarded.
  `disk_cache_invalidation.rs` no longer decides what invalidation means, only
  where entries live and how they are removed, so it is named
  `disk_cache_entries.rs`. `device.rs` in the wgpu device runtime repeated its
  parent and named nothing; it owns device acquisition, the process-wide cached
  device, feature negotiation against the adapter, and the host waits
  acquisition needs, so it is `vyre-driver-wgpu/src/runtime/device/acquire.rs`.
- The borrow-preserving structural `Node` rewrite has one owner,
  `vyre_foundation::transform::rewrite_walk`. Eight sites carried their own
  `match node { .. }` that rebuilt every variant: induction-variable
  substitution, fusion alpha-renaming, cache-key canonicalization, const-buffer
  folding, and four walks in the pass engine (encoded-order rewrite,
  cross-scope CSE occurrence collection and substitution, and same-scope let
  dedupe). They differed only in what they decided at each position, never in
  which positions exist, so a variant added to the IR had to be answered eight
  times and a fix to the descent order reached one pass. `rewrite_node` offers
  every rewritable position to a `NodeRewrite` policy in the order the program
  is written, identifiers and operands first and then child bodies, which is
  the order the pass engine's expression arena numbers expressions in, so a
  pass consuming a per-expression GPU verdict stays aligned with the encoder.
  The match is exhaustive with no catch-all: a new `Node` variant fails to
  compile until the author says which of its positions a rewrite owes a visit.
  Every hook answers `None` for unchanged, so an unchanged subtree is returned
  as the same allocation rather than an equal copy, and
  `Program::canonicalized` keeps that property on an already-canonical body.
- The two CSE passes in `vyre-pass-engine` that walk IR while counting
  expression-arena positions share one cursor, `optimizer::arena_cursor`. Each
  had its own copy of advancing an index per node, remembering a position to
  rewind to, and skipping a nested body's worth of ids, and the nesting each
  skipped was written out variant by variant, so a new statement-carrying
  variant would have misaligned an arena verdict against the node it was
  computed for without any error. `ArenaCursor` takes its nesting from
  `transform::visit::child_bodies` and stays in the pass engine, because the
  numbering is the encoder's and not the IR's. The hoisting decision and the
  same-scope let-dedupe decision stay separate: they are different decisions
  over the same walk.
- The four `*_via_encoded` passes prove their decode against one test
  dispatcher, `optimizer::arena_kernel::FixedOutputDispatcher`, which takes the
  pass name, the input count it must be bound with, and the grid it must be
  asked for. Each pass had its own copy, and they had already drifted on the
  assertion that matters: three assert a single workgroup, which is their real
  contract, while const-fold dispatches `ceil(expr_count / WORKGROUP_X)`
  workgroups and its copy asserted a literal one-workgroup grid, correct only
  for the one-expression fixture every test used. The grid is now a stated
  expectation rather than a copied literal, so a pass whose dispatch shape
  changes has one place to say so.
- The three non-native grid-sync dispatch routes in `vyre-driver` ask one owner
  whether a split produced segments. `reject_empty_grid_sync_split` in
  `grid_sync` states the invariant once; the resident timed route, the resident
  fixpoint route, and the host split route each carried the same eight-line
  guard with the same error text, so a route added or reworded without it would
  dispatch nothing, leave the caller's output buffers untouched, and return
  success. The split tests likewise share their fixtures: `grid_sync_chain`
  builds the returning-region-per-segment program eleven call sites wrote out
  by hand, `barrier_chain` covers the two that vary the barrier ordering or the
  workgroup size, `cross_segment_store_program` states the cross-segment
  accumulator regression once instead of four times, and `apply_out_stores` is
  the single reading of a segment body the stand-in device backends interpret.
  Two copies of that walk existed, so a nested node form one of them forgot
  would hide a dropped store behind a passing test. Dispatch behavior, segment
  counts and error strings are unchanged.
- The unfused resident-sequence fallback in `vyre-driver` has one owner,
  `backend::resident_sequence`. Two decisions were written five times between
  the `VyreBackend` defaults and the grid-sync split decorator: the launch
  configuration a sequence step gets, which is that step's grid override and
  nothing else from the caller's config, and the conversion of a readback range
  list into the one `download_resident_ranges_into` call that ends the
  sequence. `dispatch_resident_steps`, `resident_step_config` and
  `read_resident_ranges_into` now hold them, so the plain sequence default, the
  timed default, the repeated default and both decorator overrides read the
  same sequence. A copy that drifted would have launched a step with the wrong
  grid or read back a different set of ranges than the sequence it belongs to,
  on the backends that do not fuse the sequence and therefore have no other
  check on it. Dispatch order, launch counts and readback ranges are unchanged.
- The six planners in `vyre-driver` that reserve their scratch before they
  decide anything declare their storage-reservation failure adapter with one
  line. `reservation_policy::storage_reserve_failure_adapter!` owns the
  conversion, which carries the field being reserved, the entry count
  requested, and what the allocator said; result compaction, device diagnostic
  aggregation, benchmark pass selection, the megakernel barrier planner, the
  megakernel frontier memory planner, and multi-query execution each wrote that
  function out identically. A seventh fact added to the shared reservation
  layer had to be threaded through six copies, and a planner missed in that
  pass would have reported a reservation failure with less context than the
  layer had already produced. Each planner's rendered message stays its own,
  because it names the planner and the sharding that fixes it. Error variants,
  message text and reservation behavior are unchanged.
- The megakernel wave-policy corpora in `vyre-driver` are read from
  `megakernel_fixtures` by every test that drives them. The barrier planner
  suite rebuilt the two-wave cycle and the four-wave chain inline next to the
  fixtures it already imported, the frontier suite declared its own two-wave
  growing pair, and both generated sweeps wrote their own copy of the
  width-by-depth layered DAG edge generator. That module exists because a
  corpus edited on one side turns a backend parity gate into two suites that
  agree about nothing and still pass, so an inline copy of a corpus defeats the
  gate it was written for. `CYCLE_DEPENDENCIES`, `LONG_CHAIN_DEPENDENCIES`, the
  new `GROWING_PAIR_WAVES`, which is the first two `DIAMOND_WAVES` by
  construction rather than by retyped numbers, and `layered_dag_dependencies`
  now own those shapes. Every planned barrier count, group width and peak byte
  figure the suites assert is unchanged.
- The megakernel empty-template cache and the resident launch recommendation
  cache now share one bounded least-recently-used map, so eviction order and
  tick saturation are decided in one place instead of two.
- Megakernel builder and scheduler body inspection now walks nested node bodies
  through one shared preorder helper built on the foundation-owned child-body
  enumeration, instead of two hand-written recursive matches that each had to
  list the nesting variants.
- Persistent, finite and JIT megakernel lane bodies now share one assembly path
  and one published-slot claim body, so the node reservation bound, the
  slot-base binding order and the IO polling block are decided in a single
  place. Emitted IR and reservation error text are unchanged.
- The duplication baseline pins for vyre-driver and vyre-runtime now record the
  measured tree: 1463 to 1112 and 520 to 412 duplicated lines. The vyre-lower
  total line count is corrected to the measured value.
- The duplication baseline records the measured tree for three more crates:
  vyre-driver 1112 to 1082 and vyre-driver-wgpu 5710 to 5709 after their lanes
  merged, and vyre-foundation 5443 to 5339. Duplication is cross-file, so a
  crate's count moves when a sibling lands or drops a copy of its text, and
  each pin is measured against the merged tree rather than carried over from
  the branch that lowered it.
- A test-only IR extension payload is declared through
  `vyre_test_support::test_expr_extension!` or `test_node_extension!` instead
  of six hand-written trait methods. Five of the six are the same in every test
  that needs an inert opaque leaf: report success from validation, hand back
  `self` for downcasting, and answer the identity questions from a literal.
  Only the kind string, the debug identity, the result type, the CSE answer and
  the fingerprint byte differ, so those are the arguments. A test that needs
  reachable structure, a wire body, or a validation failure still writes the
  impl out, because those are the cases where the boilerplate is the contract.
- The WGSL emitter runs one carrier-scope protocol. A value produced inside a
  structured child body and read after it has to round-trip through a
  function-scope local, or naga's writer emits `let _eN = ...;` inside the
  closed child block and the reader after it fails validation with `no
  definition in scope for identifier _eN`. That protocol, snapshot the carrier
  state, find the op's position by identity, collect the ids that escape, seed
  each carrier local from the parent's pre-op value, emit the child, reload
  every carrier in the parent block, restore the parent state, was written
  three times: once in `emit_structured_block`, once in `emit_structured_if`,
  once in `emit_structured_for_loop`. The store-then-reload publish inside
  `bind_result` was written twice more, once per local pool, and
  `value_handle_for_id` spelled load-and-coerce four times.
  `emitter::carrier_scope` now owns it: `with_carrier_scope` for the block and
  if forms, `register_carrier_targets` plus `store_carrier_seeds` plus
  `publish_carriers` for the loop, which splits the seed across the index
  setup, and `publish_through_local` for both pools of `bind_result`. The two
  pools are one `LocalPool` value decided once per bind, so the publish, the
  trace name and the traced local type cannot disagree. A copy missed by a fix
  to any of these would not fail to build: it would emit a shader that passes
  vyre's own descriptor checks and is then rejected by wgpu at pipeline
  creation for one structured form only, or worse pass validation and read a
  stale carrier, which is how an accumulator seeded from a reused descriptor id
  summed from the wrong constant. The seed store keeps its two genuinely
  different meanings as an explicit `CarrierSeed`: a loop seeds verbatim
  because the index type already decided the carrier's type, while a block or
  if arm coerces to the local's type first. `collect_op_referenced_ids` reads a
  table of operand roles per structured op instead of four hand-written arms,
  so a new structured op is a row rather than a fifth arm to forget. Emitted
  text is unchanged: the pinned WGSL corpus, the determinism check and the
  carrier-scope regression suite all pass without a repin.
- The WGSL emitter maps between a canonical naga type handle and a scalar kind
  through one function per direction. `BodyBuilder::scalar_kind_of_type`,
  previously named `binding_types_lookup` for a lookup it does not perform, is
  the only reader of the seven canonical handles, and `coerce_value_to_type`
  had its own byte-identical copy of that if-chain deciding what to coerce
  toward; `canonical_type_for_scalar_kind` is the only writer, and the
  carrier-local allocator and the binary-operand unifier each had their own
  copy of it. `yields_bool` names the six comparisons and two logical
  connectives once, where `scalar_kind_of_expression` and
  `is_bool_expression_inner` each listed them. A naga release adding a
  comparison operator, or vyre admitting another scalar width, previously had
  to be tracked into two places per direction, and the failure of a missed copy
  is not a build error: a comparison whose result reads back as its operand
  kind is stored into a numeric local and rejected at pipeline creation as
  `InvalidStoreTypes`, or coerced with a `select` that silently reinterprets
  the value. `unify_binary_operand_types` also stopped rebuilding
  `Expression::Binary` at each of eight exits; it computes the operand pair and
  constructs the expression once, and its two operator sets are named for the
  rule they encode, that naga requires matching operand kinds and that the
  arithmetic and shift operators additionally reject a bool. The three
  `Expression::As` arms that differed only in target kind are one arm binding
  the kind. Emitted WGSL is unchanged and the pinned corpus passes without a
  repin.
- The `vyre-emit-naga` test surface states two things once. Four recursive
  probes over an emitted `naga::Block`, whether it contains a barrier, a loop
  or an atomic at any depth and how many `If` statements it holds, existed in
  full in both the crate's inline emitter tests and
  `<crate>/tests/adversarial_emit_program_matrix.rs`, eight copies of the same
  descent across nested blocks, both if arms, and a loop's body and continuing
  block. They are now one counting walk plus four predicates in
  `naga_block_probe.rs`, which the integration test includes with `#[path]`
  because the probes are test scaffolding and do not belong on the crate's
  public surface. The smallest descriptor that emits a global and a statement,
  one binding and two literals feeding a `StoreGlobal`, was written out twice,
  once in the pattern-audit tests and once in the entry-emission tests;
  `tests::single_store_desc` owns it. Both duplicates were the kind that fails
  quietly: structured lowering moves the depth at which statements are wrapped
  in `Statement::Block`, so a probe updated in one file and not the other
  reports an emitted barrier as absent and the assertion that was meant to
  catch a regression passes instead.
- The `vyre-emit-ptx` test surface states each kernel shape once. Five atomic
  RMW descriptors, four subgroup reductions, two subgroup shuffles, five
  predicated-store preludes and thirteen cast kernels each carried their own
  inline copy of the same op list, operand indices and literal table, so a slot
  index, a literal position or a dispatch width could drift in one copy while
  the test kept its name and kept passing against a program it no longer
  describes. `atomic_kernel` in `<crate>/src/tests/mod.rs` is the only writer
  of the one-slot atomic body and is now shared with the unsupported-op refusal
  in `subgroup.rs` that had a fifth copy; `subgroup_reduce_kernel`,
  `subgroup_shuffle_kernel`, `predication_kernel`, `cast_kernel` and
  `chained_cast_kernel` own the remaining shapes, and the pre-existing
  `f32_cast_kernel` and `u32_cast_kernel` delegate to `cast_kernel` instead of
  rebuilding it. `assert_xor_butterfly_steps` states once that a 32-lane XOR
  butterfly takes log2(32) exchange steps, where three reduction tests each
  spelled out the same two count assertions and their messages. Emitted PTX is
  unchanged: every fixture reproduces its callers' descriptor field for field.
- Three facts the `vyre-emit-ptx` tests depend on are now stated once each. The
  canonical MMA op kind, six coupled fields naming an `m16n8k16` row-by-column
  f16 multiply accumulating in f32, appeared in four places: twice in
  `index_facts.rs`, once in the schedulability probe in
  `<crate>/src/emitter/schedule.rs`, once in the MMA emission contract;
  `tests::f16_mma_kind` is now the only writer, so a shape or element-type
  change cannot leave one probe testing a combination the emitter no longer
  sees. The nine-op chain a four-way vector load fuses from, a base index and a
  stride feeding four loads at result ids 2, 4, 6 and 8 with the index add
  between them, appeared three times with the load kind as the only difference;
  `tests::four_load_chain` takes that kind and callers append their own tail,
  which also states that the `LoadConstant` shape `const_buffer_promote` leaves
  behind is the same chain rather than a similar-looking one. The two
  whole-program emit probes, grid-sync-in-a-loop refusal and nested-return
  branching, each carried their own copy of the region wrapper their programs
  are measured inside and of the lower-then-emit pipeline that reports which
  stage refused; `<crate>/tests/emit_probe/probe.rs` owns both, so a probe
  cannot wrap its body differently from the probe it is being compared against.
  Emitted PTX is unchanged.
- The cross-emitter parity suite states the audit contract once. Three tests
  each listed all four audit layers, substrate-neutral plus one per target, and
  asserted that each carried the kernel id it was handed; two of them differed
  only in whether they audited a descriptor before or after verification, so
  the copies were near-identical bodies whose divergence would have been
  invisible. `assert_audits_carry_kernel_id` is the single list, and it also
  pins the PTX report's target, which only one copy checked. The smallest store
  descriptor was built twice in the same file, once as the corpus entry and
  once inline for the id test with a different binding name; `store_one_kernel`
  takes the id and the slot and owns the shape.
- Asking whether the emitted WGSL entry point contains a given operation is one
  probe. Three tests each opened the entry point's expression arena, iterated
  it and hand-matched a `naga::Expression::Unary` or `Expression::Binary`
  variant with the operator pinned in the pattern and the rest wildcarded,
  which is a shape that silently stops matching when naga adds a field to
  either variant: the assertion then reports the operation as absent rather
  than failing to compile. `entry_has_unary` and `entry_has_binary` sit beside
  the block probes in `<crate>/src/tests/naga_probe.rs`, renamed from
  `naga_block_probe.rs` now that it probes statements and expressions rather
  than blocks alone.
- The duplication pins for the four emitter crates now sit at what the tree
  measures: vyre-emit-metal 29 to 22, vyre-emit-naga 404 to 221, vyre-emit-ptx
  295 to 105, vyre-emit-spirv 49 to 43. A pin that passes with room under it
  hides the next copy, so each is lowered to the measurement rather than left
  as headroom. vyre-megakernel stays at 60: both of its duplicated files
  partner only with crates outside the emitter boundary, so nothing there could
  be collapsed from this side.
- Compiling a small graph into a canonical artifact is one fixture. Four crates
  each carried their own copy of it, and `vyre-megakernel` and `vyre-runtime`
  carried theirs byte-identically because their assertions compare artifact
  digests, which only agree while the two copies agree. The copies also each
  restated which buffer declaration a value contract implies, so a fixture
  whose declared element count disagreed with its contract shape was accepted
  by whichever crate wrote it and rejected by the next one to copy it.
  `tests/support/artifact_fixtures.rs` owns the mapping, the one-node graph and
  the compile request, and rejects a uniform or workgroup contract instead of
  silently making it global storage. Shared through `#[path]` the same way
  `tests/support/preferred_dispatch_backend_contract.rs` is. The facade's own
  `artifact_workflow` test keeps its inline construction: what it proves is
  that every type needed to build and compile a graph is reachable through
  `vyre`, so naming those types is the contract rather than setup.
- The C-frontend test corpus spells its token streams once. A fixture that
  listed one `FixtureToken::new("lexeme", TOK_KIND)` per line now reads
  `c_tokens("lexeme lexeme ...")` against
  `tests/support/c_frontend/spelling.rs`, which derives each raw kind the way
  the lexer does and promotes keywords through `reference_c_keyword_types`, so
  a stream that used to occupy fifteen lines in two crates occupies one line in
  one place. The build-annotate-classify triple every CPU reference case opened
  with is `c_frontend::token_fixture::annotate_and_classify`, and the PG row
  assertion that spelled out `&fix.tok_starts` and `&fix.tok_lens` is
  `assert_pg_preserves_fixture_row`. Every assertion the corpus made is still
  made and still executed; 136 fixture streams across 34 files became one-line
  spellings.
- Three shapes repeated across the `vyre-libs` test corpus have one owner each.
  `vyre-libs/tests/support/optimizer.rs` owns `assert_optimizer_is_idempotent`
  and the debug first-difference reporter that names which node moved when the
  optimizer fails to reach a fixed point; three idempotence contracts carried
  their own copy of both. `vyre-libs/tests/support/ir_fingerprint.rs` owns the
  pinned-fingerprint family guard that proves a collapsed clone family still
  emits what every former copy emitted. Seven files that reimplemented
  little-endian `pack`/`unpack` read `vyre_primitives::wire::pack_u32_slice`
  and `decode_u32_le_bytes_all`, the crate that owns that layout.
- The CUDA resident dispatch contract family states its setup once. Five files
  asserted against a four-lane u32 resident buffer seeded [1, 2, 3, 4], an
  elementwise add or multiply over one wrapped workgroup, and a readback
  expectation that depends on whether the borrowed host-buffer fallback is
  opted in, and each file carried its own copy of all three: the lane width,
  the seed, the program shapes, the sequence-step and read-range literals, the
  borrowed-fallback predicate, and the native compact-readback telemetry block.
  `vyre-driver-cuda/tests/resident_dispatch_contracts/resident_lane_fixture.rs`
  now owns them. A copy that had already drifted is what the split cost:
  `repeated_sequence_contracts.rs` carried a second test with the same name as
  the parent's release-path contract, asserting the same borrowed-fallback
  counter over a multiply program but never checking the output lanes, so a run
  where the multiply lowering wrote wrong values reported green. The surviving
  contract now sweeps both the add and the multiply program and asserts the
  lanes for each.
- The generated CUDA reference matrices share one sweep runner. Six files each
  carried the same runner loop: dispatch a case on the direct and the compiled
  CUDA path, diff both against the reference interpreter, accumulate the lanes
  each comparison actually checked, and assert the total against cases times
  lanes times two. Fifteen copies of that loop, five copies of the guarded lane
  store, two copies of the six comparison-word builders
  `vyre-driver-cuda/tests/common/mod.rs` already exported, and per-dtype
  program builders that differed only in which buffer types they declared.
  `vyre-driver-cuda/tests/common/mod.rs` now owns `generated_lane_program`,
  `guarded_generated_store`, `guarded_generated_store_at`,
  `GeneratedMatrixCase`, `assert_u32_matrix_sweep` and
  `assert_f32_matrix_sweep`, and the binding order every matrix depends on is
  fixed in one place instead of restated per file. Case tables, adversarial
  corpora and ULP bounds stay local, because they differ on purpose. Three
  sweeps got stronger rather than shorter: the scalar bool tables, the cast
  table split by output dtype, and each of them now asserts its own lane
  coverage, where a single combined total let one group's over-count hide
  another's shortfall.
- The generated CUDA resident reference matrices share one sweep runner and one
  program builder. Five contract files each carried their own runner loop over
  the resident dispatch path, and `program_builders.rs` carried thirteen
  near-identical guarded-store program builders that differed only in the
  declared buffer types. The parent now owns `ResidentMatrixCase`,
  `assert_resident_u32_sweep` and `assert_resident_f32_sweep`, and the builders
  delegate to `resident_lane_program`. The in-place atomic reduction builder is
  deliberately left standalone: it binds its accumulator read-write, performs
  no guarded store, and is checked against a different reference, so folding it
  into an output sweep would assert something else. The lane-coverage totals
  are now asserted per table rather than combined, which stops one table's
  shortfall from being hidden by another's surplus.
- The CUDA C-preprocessing parity tests share their two dispatchers. The
  payload, tokenize, macro-expansion and filter files each declared a private
  reference dispatcher that evaluates the program on the host and a private
  CUDA dispatcher that runs it on the device once owned and once borrowed, four
  copies of each. `vyre-driver-cuda/tests/common/c_preprocess_oracles.rs` now
  owns `ReferenceOracle` and `CudaOracle`; the parity assertion each file makes
  stays in that file, because the two arms being compared are the point of the
  test.
- The release conformance facts derived from `docs/optimization/OP_MATRIX.toml`
  are derived once.
  `xtask::release::conformance_op_matrix::evaluate_op_matrix_coverage` owns the
  covered and missing required-op counts, the supported release-backend row
  count, and the six blockers that follow from them; the registered-op matrix
  in `xtask-registry` and the per-backend conformance artifacts in `xtask` each
  carried their own copy of that arithmetic and of five of the six blocker
  messages, differing only in which set of observed op ids they judge the
  matrix against. The two artifacts report the same field names, so a
  correction applied to one copy left the other reporting a different coverage
  number for the same matrix and a release could be signed on whichever
  artifact was read first. `release_backend_rows.rs` held half of that decision
  for both callers and is folded into the matrix module, so the count is no
  longer reachable without the judgement it feeds.
- Every evidence-producing xtask subcommand parses `--output`, reports the
  written artifact and chooses its exit code through one owner in
  `xtask::output_arg`. Each command used to carry its own copy of the option
  loop, the usage-error exit, the parent-directory creation and the
  wrote-then-exit epilogue, so exit 2 for a usage error and exit 1 for a
  failing gate were a dozen separate decisions that CI reads as one contract.
  `parse_output_and_flag_arg` covers the commands that take a second valueless
  flag, `parsed_or_exit` maps a usage error to exit 2,
  `report_evidence_artifact` prints the path and exits 1 on a non-empty blocker
  list, and `write_json` now creates the parent directory itself rather than
  each caller doing it first. Help text, blocker text and exit codes are
  unchanged.
- `xtask::delegate::run_delegated_main` owns what a delegated binary's `main`
  does: answer `--help` from the callee's own dispatch table, reject a missing
  subcommand, then resolve the name. `xtask-registry` and `xtask-evidence` each
  wrote that out, including the exit code for an unimplemented subcommand, so
  the two entry points could disagree about a contract that `xtask` and CI read
  as one. Each `main` is now a single call naming the package and why `xtask`
  routes there, and the `dispatch` wrapper each crate exported for its own
  `main` alone is gone. Help text, error text and exit codes are unchanged.
- The release gate reads a benchmark report's blocker array and failed-case
  summary in one place. `check_benchmark_report_summary` owns the
  summary-versus-case-evidence mismatch and the failed-case message; the three
  gates that open a benchmark report each carried that arithmetic and its four
  messages, differing only in whether the report is named by evidence suffix or
  by path, so a correction reached one report and not the next. The four
  `require_case_metric_*` requirements now share one walk over the cases array,
  which also makes explicit that only the present-metric requirement treats a
  missing cases array as its own failure and that the positive-metric
  requirement reads the recorded percentile rather than accepting a bare
  scalar. An unreachable second copy of the markdown-evidence check is deleted:
  it had no callers and had drifted to demand an evidence-sources list from
  every markdown file, while the live check demands it only under
  `evidence/docs/`.
- The hygiene scan's pattern names are declared where the scan emits them.
  `xtask::gates::hygiene_matrix` exports the hidden-fallback, resource-bound
  and cargo-wrapper pattern lists, and the release-evidence check that requires
  each family to have been covered now references them instead of restating
  nineteen names. Adding a pattern to the scan previously left the gate
  accepting evidence that never looked for it.
- The `Expr::Call` ABI in `vyre-reference` has one owner,
  `vyre-reference/src/execution/call.rs`. Callee resolution and its
  per-call-site cache, the arity check, the declared byte width of an argument,
  the 64 MiB input budget, the CPU-reference lookup and the output decode were
  each written twice, once for the statement evaluator in `expr.rs` and once
  for the hashmap interpreter in `node_step.rs`, and the two copies had already
  drifted in how they sized the output buffer and spelled the output type
  match. Both arms now bind that one ABI and supply only their own argument
  evaluator, and `ResolvedCall` replaces the two identical resolved-callee
  structs. A copy missed while changing the ABI would have made one reference
  execution path encode a call differently from the other while both still
  reported success, so the oracle would disagree with itself depending on which
  entry point a test used. The width table gained a registry-derived gate:
  `signature_param_spellings_all_have_a_declared_width` walks every registered
  callable signature at run time and fails when a parameter spelling has no
  declared width, instead of letting it fall through to the one-byte default
  and truncate an oracle input.
- The async byte copy in `vyre-reference` has one owner,
  `vyre-reference/src/execution/async_transfer.rs`, and the copy itself goes
  through two new window methods on the reference buffer. Each of the two
  reference node executors carried its own transfer enum, its own
  offset-and-size-to-byte-count conversion, its own short-tail-zero-padding
  read and its own past-the-end-dropping write, and the two writes disagreed:
  the statement executor recovered a poisoned byte lock with `into_inner()`
  while the hashmap executor panicked. A poisoned lock means a writer panicked
  mid-store, so the recovering arm could copy half-mutated bytes into an output
  buffer and hand the conform gate a golden value no correct backend can
  reproduce. Both arms now read and write through `read_window` and
  `write_window` in `oob.rs`, which own that policy, and the fail-closed test
  covers both helpers instead of only the hashmap path.
- The proptest `DataType` generator in `vyre-spec` has one owner,
  `vyre-spec/tests/spec_variants/mod.rs`. The layout sweep and the
  `OpSignature` byte-accounting sweep each carried their own generator and the
  two had already diverged: the signature copy omitted opaque, device-mesh,
  block-sparse and quantized types, so nothing checked that a signature
  carrying one of them survives serialization. Both sweeps now draw from the
  shared generator, which widens the signature sweep to the full type space.
- The scalar-leaf and quantized-storage `DataType` tables in `vyre-spec` have
  one owner, `vyre-spec/tests/spec_variants/mod.rs`. The leaf list was retyped
  in the proptest generator, the seeded signature matrix, the seeded
  wire-payload matrix and the edge matrix, and the nine quantized storage types
  were retyped in five places including two seeded generators that only
  produced five of them. Every sweep now indexes the shared tables, and
  `data_type_surface.rs` checks `DataType::is_quantized_storage` against the
  shared table in both directions over every scalar leaf instead of four
  hand-picked negatives, so widening one without the other turns red.
- The `CollectiveOp` proptest generator in `vyre-spec` has one owner,
  `vyre-spec/tests/spec_variants/mod.rs`, replacing the copy in the op-wire
  sweep and the copy in the collective sweep. The frozen collective wire-tag
  table now appears only in `vyre-spec/tests/collective_op_contracts.rs`, which
  owns the round trip, the unassigned-tag rejection and the world comm-group
  constant; the subset copy in the extension-category test is gone.
- The `vyre-spec` expression-variant catalog test no longer retypes the 22
  variant names it is checking. It now pins a floor on the catalog size,
  requires the names to be unique, and requires each to be a well-formed
  UpperCamelCase identifier, so a new variant does not have to be spelled twice
  to pass.
- The `IntrinsicTable` missing-backend contract in `vyre-spec` has one owner,
  `vyre-spec/tests/spec_contract_errors.rs`, which holds the fuller family. The
  two duplicated cases and the `required_backends()` helper they shared are
  gone, and the remaining file is renamed to `data_type_width_contracts.rs`
  after the backend half left it.
- The flat-evaluator test harness in `vyre-reference` has one owner,
  `vyre-reference/tests/flat_expr_eval/mod.rs`. The adversarial proptest, the
  adversarial gap suite and the subnormal flushing contract each carried their
  own wrapper program, zero invocation, expression-evaluation entry, literal
  `BinOp`/`UnOp` builders and float-bit extractor, and the copies had already
  drifted in panic text. The IEEE-754 canonicalization the sweeps compare
  against stays an independent restatement rather than a call into the
  interpreter, so the oracle still does not check itself.
- The duplication baseline pins for vyre-reference and vyre-spec now record the
  measured tree: 971 to 743 and 446 to 244 duplicated lines. Both total line
  counts are corrected to the measured values.
- Twelve duplication pins now sit at what the merged tree measures rather than
  at what each lane measured in isolation: vyre 40 to 22, vyre-aot 54 to 31,
  vyre-debug 71 to 54, vyre-driver-cuda 3252 to 3238, vyre-driver-wgpu 4229 to
  4017, vyre-foundation 5125 to 5117, vyre-libs 11759 to 11539, vyre-megakernel
  60 to 8, vyre-primitives 7145 to 7038, vyre-runtime 412 to 345,
  vyre-pass-engine 299 to 288. Duplication is measured across files, so
  collapsing a copy in one crate moves every crate that shared it:
  vyre-megakernel lost 52 lines it never edited, and vyre-primitives lost 107.
  A pin left at the isolated measurement would have carried that slack as
  headroom for the next copy. `total_lines` is measured too, not carried
  forward, because a stale total makes the ratio it exists to report
  meaningless.
- The five `vyre-lints` scans share one source walk in
  `vyre-lints/src/scan.rs`, and the CLI reads a lint registry instead of four
  near-identical entry points. `ViolationKind` gains `ALL`, `position` and
  `as_str` so the CLI and the JSON writer read the enum rather than restating
  it, and a kind that reaches no lint now fails the suite. The CLI reports its
  declared scan scope under `--print-default-roots`, so the production-root
  test asserts the property those roots must satisfy instead of restating the
  list. `raw_ir_in_libs` kept a second workspace-relative path function that
  disagreed with `vyre-lints/src/paths.rs` on the vyre-libs prefix; it is gone.
- The three `vyre-debug` descriptor and WGSL suites share one program fixture
  in `vyre-debug/tests/support/mod.rs`. The descriptor a dump renders, the
  descriptor a diff compares and the WGSL a backend emits are only comparable
  while their fixtures agree, and the fixture was previously copied byte for
  byte into each target. `vyre-macros` pass metadata is a table in
  `vyre-macros/src/pass.rs` rather than three parallel token functions, and the
  macro test suites take their pass bodies from
  `vyre-macros/tests/support/mod.rs`.
- Duplication pins for three crates now record the measured tree: vyre-macros
  89 to 41, vyre-lints 66 to 0, vyre-debug 54 to 9. Their total line counts are
  corrected to the measured values, which were already stale before these
  changes. A no-op test whose body was a comment claiming to pin a test count
  is deleted, and the two `raw_ir_in_libs` case files, named for a helper and
  for one of their own tests, are now named for what they hold.
- The `vyre` facade wire suites share one program fixture in
  `vyre/tests/support/mod.rs`. The round-trip suite proves a valid program
  survives an encode and a decode byte for byte and the malformed-wire suite
  proves hostile bytes never decode into one, so two copies of that program let
  the two halves of the claim drift. In `vyre-grammar-gen` the LR table the
  wire round trip exercises is the sample grammar `vyre-grammar-gen/src/lr.rs`
  already owned, not a second copy of it. Pins lowered to the measured tree:
  vyre 22 to 0, vyre-grammar-gen 137 to 119.
- The resident work-queue buffer set the megakernel benchmark cases dispatch
  over has one owner, `vyre-bench/src/cases/resident_queue.rs`. Each of the
  three cases carried its own copy of the four-buffer control/ring/debug/io
  encoding, its own post-dispatch transfer accounting and its own resident-pool
  metric point. Each case keeps what it measures: its own ring builder, its own
  CPU reference, and its own grid override, because the latency case drains a
  single workgroup while the condition case drives the full grid and that
  difference is the measurement.
- The seven `foundation.*` micro benchmark cases are rows on one owner,
  `vyre-bench/src/cases/micro.rs`, which states the program, fixture, reference
  and work accounting each case differs in and nothing else. `gpu_case.rs` is
  absorbed into it, and its metric sweep now covers both work-accounting arms
  across every lane rather than one.
- The vyre-bench duplication pin records the measured tree: 2485 to 2179
  duplicated lines, with total_lines corrected from 37430 to the measured
  37504.
- Every structural optimizer pass in `vyre-foundation` now holds its rewrite
  rule and nothing else. Eighteen analysis bodies restated the same shape, an
  O(1) node-kind bitset filter followed by a scan of the entry tree for a
  candidate, and sixteen transforms restated a recursive descent plus the same
  changed-flag bookkeeping; both are stated once in
  `vyre-foundation/src/optimizer/passes/driver.rs`. Three of those descents
  applied themselves to every child twice, once through the child map and again
  through the body map, which is `2^depth` node visits for a nest of that
  depth. The descent also preserves borrows: the owned entry walk the passes
  used cannot report `no change`, so each of them rebuilt the whole entry tree
  on every run and dropped the cached facts behind it, including on the runs
  that changed nothing. A pass whose rule does not fire now hands back the
  caller's program unchanged, which is the common case under the optimizer
  fixpoint. Legality is stated once per pass instead of twice, so an analysis
  can no longer schedule a pass for a node its own rewrite then declines.
- The `vyre-foundation` duplication pin sits at what the tree measures, 5125
  down to 3333 duplicated lines, with the total line count corrected from a
  stale 95984 to a measured 94304. It was lowered four times as each owner
  landed, because a pin with room under it hides the next copy.
- Which operand position of a structured kernel op names a child body now has
  one owner on both sides in `vyre-lower`. `child_body_operands` already owned
  the reading side, but every test fixture that built a branch spelled the
  operand vector out by hand, so a moved position could go unnoticed on one
  side; `if_then`, `if_then_else` and `for_loop` in
  `vyre-lower/src/descriptor_builder.rs` own the writing side and a new test
  drives each constructor's output through the reader. The shared-memory and
  constant-buffer promotion analyses both read their load-count precondition
  through `vyre-lower/src/analyses/load_counts.rs` instead of each carrying its
  own recursive walk, and a test pins the two to the same answer on a nested
  body. The three copies of the lowered-loop locator in the lowering tests
  collapse onto `vyre-lower/src/lower/loop_site.rs`, and `fixture_builders.rs`,
  whose module was never imported and whose constructors were a second copy of
  the descriptor builder's, is deleted.
- The duplication baseline pin for vyre-lower records the measured tree: 279 to
  205 duplicated lines, with `total_lines` measured at 13187. The header block
  listing rows pinned below their measurement is gone with its only entry.
- `KernelOpKind` is enumerated in exactly one place.
  `vyre-lower/src/op_facts.rs` owns an exhaustive match with no wildcard arm
  answering both per-kind questions the workspace asks: which operand names the
  first child body, and whether the op must be kept when its results are
  unused. `child_body_operands` and the dead-op purity predicate both read it,
  replacing two separate lists of the same variant universe, and
  `vyre-emit-ptx/src/patterns/predicated_execution/mod.rs` reads it instead of
  restating sixteen variant names. What stays in the PTX pass is the judgment
  that only a plain global or shared store is maskable by a `@%p` instruction
  predicate. Adding a variant now fails to compile until both facts are stated
  for it, rather than defaulting to no child body and removable.
- `vyre-libs` feature `full` is every dialect the crate ships, meaning every
  domain that submits an operation registration: it gains `logical`, `visual`,
  `security`, `parsing` and `matching-nfa`. It stops short of the compiler's
  own self-use composition domains, which submit nothing. Before this, `full`
  was a subset and two consumers each carried the same ten-name list to reach
  the rest.
- The registered-target-compiler contract has one owner,
  `tests/support/target_compiler_contract.rs`, which every backend reaches with
  its own backend id, payload format identity and version, entry point and
  output access class. Four backends had restated all four statements, and the
  copies had drifted: two asserted the payload format version and two did not,
  three asserted the completion digest and one did not, each looked its
  registration up through only one of the two registry routes, and the value
  the fixture program stores was a separate literal from the value the
  assertion read back. The shared contract asserts both registry routes resolve
  to one registration, one emitted module per compiler-selected fusion group,
  entry-point agreement between the bundle and the payload entry, and the
  payload seal against the compiled artifact; whatever is target-native about a
  module is inspected by the caller's own closure inside its own crate.
  `vyre-driver-spirv/tests/shared_target_contract_discrimination.rs` proves the
  contract rejects a falsified payload format identity, format version, entry
  point, unlinked backend id and precedence rank, so a helper that accepted
  every argument cannot pass.
- The resident asynchronous overlap contract has one owner,
  `tests/support/resident_async_overlap_contract.rs`. Two backends had carried
  it verbatim apart from acquisition, message text and one assertion.
  Device-measured time is now a parameter rather than a shared assumption, so a
  backend without a device timer on that path is still held to exact per-slot
  outputs and to enqueue and wait time being separated.
- `vyre-driver-spirv` builds its `DeviceProfile` from
  `DeviceProfile::conservative` plus the four limits it probes, instead of
  restating all thirty-two fields. The driver already owned the
  unprobed-capability defaults and nothing used them; a new capability field
  had to be answered by hand in `vyre-driver-spirv/src/lib.rs` and the answer
  was always the conservative one.
- The duplication pins for the four driver crates in this change record what
  this branch measures in isolation: vyre-driver-cuda 3238 to 3187,
  vyre-driver-wgpu 4017 to 3918, vyre-driver-spirv 194 to 118,
  vyre-driver-reference unchanged at 23 with its stale total corrected. Every
  `total_lines` is measured, not carried forward. Duplication is cross-file, so
  collapsing one side of a copy moves the other: vyre-driver-metal fell 251 to
  175 and vyre-driver 1082 to 1069 without either being edited, and both sit
  below pins this change does not touch.
- `bellman_shortest_path` and `sinkhorn_iterate` take a named binding record
  plus an extents record instead of nine and thirteen positional arguments. The
  IR-parity test for Sinkhorn had been passing its ten same-typed buffer names
  in binding order rather than parameter order, so it emitted a program that
  named its kernel matrix `u_curr` and its convergence flag `kv`; a record
  forces each name to be spelled at the call. `bellman_tn_order_program` and
  `sinkhorn_full_clustering_program` forward the same records.
- Harness selection and convergence-flag width for the persistent fixpoint are
  one decision with one owner, `routed_persistent_fixpoint`, which returns both
  halves together. The grid form indexes `changed[iteration]`, so taking the
  harness without its flag width was an out-of-bounds atomic write on the
  second iteration.
- The buffer layout every CSR graph primitive appends to the read-only
  ProgramGraph bundle has one owner. `graph::program_graph` gains
  `word_buffer`, `frontier_buffer`, and `push_frontier_changed_buffers`;
  thirteen call sites across the forward, backward, frontier-step, degree-sum,
  motif, reachability, tensor-flow and persistent-BFS builders read them
  instead of restating the declarations. Two of those sites disagreed on the
  frontier word count for a zero-node graph, one declaring a zero-word storage
  binding that no backend can allocate, and the excluding forward step bound
  its extra input at the index its own module documents as the output frontier.
  `bitset_words` had three spellings; the two forwarding shims are gone and
  every caller reads `bitset::bitset_words`.
- The frozen `vyre_spec` operator variant space is enumerated in one place,
  `tests/support/spec_variant_tables.rs`, instead of once per suite. Five
  suites carried their own list of every `BinOp`, `UnOp`, `AtomicOp` and buffer
  `DataType`, and the lists had drifted in both directions: the random-IR
  corpus was missing `WrappingAdd`, `WrappingSub` and `MulHigh`, and the wire
  round-trip sweep was missing six `UnOp` and six `BinOp` variants, so a
  variant no list named was a variant no suite exercised. The tables cannot be
  derived from the types, which are `#[non_exhaustive]` with no variant
  iterator, so the completeness gate reads the variant set out of the rustdoc
  public-API snapshot at run time and fails naming any variant a table omits.
  The xorshift sweep generator the suites had copied verbatim now lives in
  `tests/support/sweep_rng.rs` and rejects a zero seed, which is a fixed point
  of xorshift and would emit a constant corpus.
- Every witness input expansion routes through `WitnessInputPlan`. Three copies
  of the planner existed beside the owner: the per-op ULP audit and the
  cross-backend parity matrix each carried a full reimplementation differing
  only in the prefix on each error string, and a rename layer aliased the
  owner's type and forwarded its functions under the copies' names. The owner
  gained the two operations only the copies had, `buffer_indices` and
  `plan_witness_inputs_owned_into`, and each copy's unique test moved to it.
  Three expansions of one fixture is three chances for a gate to compare
  against an input stream the dispatcher never saw.
- `vyre-conform` depends on `vyre-libs` with `features = ["full"]` rather than
  restating ten of the aggregate's members by hand. A hand-kept list of
  aggregate members drifts silently against the aggregate, which is how the
  same shape on the registry walker's primitives edge lost four dialects and
  made three operations invisible.
- Four duplication pins record what the tree measures: `conform` 765 to 367
  duplicated lines, `vyre-spec` 244 to 84, `vyre-foundation` 4342 to 4127, and
  `vyre-libs` 11446 to 11425, each with `total_lines` measured. A pin with room
  under it hides the next copy.
- `vyre_lower::op_facts` was both a module and a function inside it, so a doc
  link to either was ambiguous and `crate::op_facts` resolved to whichever
  rustdoc preferred. The function is `facts_for`, reached as
  `vyre_lower::facts_for` or `vyre_lower::op_facts::facts_for`. Five call sites
  and the crate root re-export name the new spelling.
- The host dataflow closures are read from their owner. `vyre-libs` carried a
  module whose six public functions each did nothing but call the identically
  named function in `vyre_foundation::pass_substrate::dataflow_fixpoint`, and
  it had grown a copy of the owner's whole test module: five assertions
  verbatim, plus two that could not fail because they compared the forwarder
  against the function it forwards to. The forwarders are gone and the module
  re-exports the owner, so the documented paths still resolve. What remains is
  the one thing this crate adds, the call counter, in a file named for it.
- The release evidence readers in `xtask-evidence` read each shared concept
  through one owner. The eight wall-clock minima are a single `WallClockMinima`
  record flattened into the three artifacts that publish them, the schema-2
  JSON read, the workload matrix, the `u64` field read and the array length
  read are one function each, the optimization family list is
  `vyre_foundation::optimizer::corpus::RELEASE_OPTIMIZATION_FAMILIES`, and the
  benchmark case fixtures are built rather than restated. A copy that a reader
  forgets to update is a gate that passes on evidence nobody checked.
- Three duplication pins record what the tree measures: `xtask-evidence` 475 to
  91 duplicated lines, `xtask-registry` 58 to 49, and `vyre-foundation` 4127 to
  4028, each with `total_lines` measured. A pin with room under it hides the
  next copy.
- Three duplication pins record what the tree measures after the owners above:
  `vyre-driver` 1047 to 994 duplicated lines, `vyre-emit-naga` 221 to 155, and
  `vyre-reference` 743 to 676, each with `total_lines` measured. A pin with
  room under it hides the next copy.
- `vyre_driver_cuda::optimizer` is `vyre_driver_cuda::resident_dispatcher`. The
  module holds no optimization pass: it holds `CudaProgramDispatcher`, the
  resident buffer pool, the static-upload cache and their eviction policy.
  Optimization semantics live in `vyre-foundation`, so an `optimizer` module
  inside a concrete driver crate named the caller rather than the contents, and
  four private helpers were named after the file instead of their work. The
  `CudaProgramDispatcher` re-export at the crate root is unchanged.
- Program capability scanning and backend capability enforcement are proved in
  one place, `vyre-foundation/tests/capability_contracts.rs`. An inline test
  module beside `vyre_foundation::program_caps` exercised the same public
  surface and the two had drifted in opposite directions. The suite derives its
  expectations from the public API snapshot at run time: every boolean
  `RequiredCapabilities` field must have a scan case that a program needing the
  feature sets and a scalar program does not, and every `supports_*` parameter
  of `check_backend_capabilities` must have a case where withholding it rejects
  a program that needs it, so a capability added later fails the suite until
  somebody records what proves it.
- The IR-shape proptest corpus generates all twenty `Node` variants, up from
  eight. Async transfers and waits, traps and resumes, the four collectives,
  indirect dispatch, regions and opaque extensions were never produced, so
  every property that walks the corpus was only ever compared on stores, lets,
  assigns, ifs, loops, blocks, barriers and returns. A gate enumerates the
  variant names the IR registry declares at run time and fails until each is
  either generated or recorded with a reason, so a variant added later cannot
  silently escape the corpus. Case counts on the two cross-walk properties are
  raised to match the wider shape space.
- One table drives the scalar storage-graph sweep and one table drives the dual
  scalar evaluator sweep, replacing four per-width harnesses that restated the
  corpus construction, the strided draw, the case loop and the diagnostic
  assertions once per width. Each row declares its literal, its corpus, its
  draw, its case counts and its expectations, and the operation set it
  exercises is derived from those expectations rather than listed a second
  time. The collapse raised coverage the copies had let drift: `f32` sweeps
  4096 binary and 8192 unary cases rather than 2048 and 4096, `i32` samples
  8192 unary cases rather than one pass over its corpus, and `i32` gained the
  saturating subtraction and multiplication cases the interpreter defines.
- The structural graph, causal and logic kernels are reached at
  `vyre_primitives::graph`, not through a second set of names in `vyre-libs`.
  `graph::dispatch::structural_kernel_pipeline` held sixty-six wrappers across
  a `dispatch` module and a `references` module, one per primitive builder and
  one per CPU oracle, each restating the primitive's parameter list verbatim
  and calling it with those parameters in that order. Nothing outside the
  module's own tests called one, so the layer bought a second signature to keep
  in step with the first and no behaviour, and a parameter added on one side
  was a compile error at sixty-six call sites or a silent divergence. What
  survives is the module's test, which pins the primitive contracts the graph
  dispatch layer relies on. Two ceilings on the Newton-Schulz IR shape now come
  from `vyre_foundation::transform::visit::walk_exprs` instead of a hand-rolled
  counter over `#[non_exhaustive]` enums whose catch-all arm read an unlisted
  variant as a leaf, so a tree that grew through a new variant counted as
  small.
- Two duplication pins record what the tree measures: `vyre-libs` 11425 to
  10453 duplicated lines and `vyre-primitives` 6981 to 6384, each with
  `total_lines` measured. Collapsing one side of a copy lowers both crates'
  counts, and a pin with room under it hides the next copy.
- `vyre_libs::solvers::numerical_kernel_pipeline` re-exports the Sinkhorn entry
  points of `vyre_primitives::math::sinkhorn_iterate` instead of wrapping them.
  Seven wrappers forwarded every argument and added nothing, so their whole
  body was a restatement of the primitive's parameter list, up to twelve
  positional arguments, free to drift from the list it forwards to and
  unprovable from either side. A composition names the primitive it composes.
  The public names are unchanged.
- The bounded-ranges candidate gate is a width value rather than a shape per
  width. `PrefilterWidth` records how many mask buffers a gate binds, which
  region generator the assembled program carries, which dispatch shape an error
  names and which fallible entrypoint a fail-closed panic points at;
  `PrefilterGate` holds the mask names as a prefix of `[end byte, bigram,
  trigram bloom]` and is only constructible through a per-width constructor, so
  an unbound slot cannot reach a buffer declaration. One assembler emits every
  match-emitting shape at every width, and the ungated scan is width zero of
  the same table instead of a fourth builder. Each width previously restated
  the six-buffer input ABI, the replay call and the build quintet, so the
  binding of the match sink was computed in three places from three copies of
  the mask count: a width added or a mask resized in one copy left the sink
  bound over a mask, which writes match triples into a read-only table and
  loses recall silently rather than failing.
- The `vyre-libs` duplication pin records what the tree measures after the
  width table lands: 10649 to 10598 duplicated lines, with `total_lines`
  measured. What remains between the three match-emitting entry points is their
  frozen positional signatures and the shared value they each construct from
  them, which no owner can absorb without changing the public ABI.
- CSR closure entry points take the graph as one named bundle.
  `vyre_primitives::graph::csr_closure_inputs` owns `CsrGraphView { node_count,
  edge_offsets, edge_targets, edge_kind_mask }` and `CsrClosureInputs { graph,
  allow_mask, max_iters }`, and every closure entry point in
  `vyre_primitives::graph::csr_bidirectional`,
  `vyre_primitives::graph::csr_forward_or_changed`,
  `vyre_primitives::graph::csr_backward_or_changed` and
  `vyre_primitives::graph::persistent_bfs` now receives them instead of seven
  or nine positional slots. Neither struct has a constructor: a struct literal
  is the only way to build one, so a transposed buffer is a compile error
  rather than a wrong closure. The dispatcher-backed consumers in
  `vyre_libs::graph::dispatch::csr_bidirectional`,
  `vyre_libs::graph::dispatch::csr_forward_or_changed` and
  `vyre_libs::graph::dispatch::persistent_bfs` take the same bundle, so the
  flat shape no longer exists at any layer.
- The C typedef-annotation builders in
  `vyre_libs::parsing::c::parse::vast::typedef_ann` share one owner for the row
  plumbing every one of them needs:
  `vyre_libs::parsing::c::parse::vast::typedef_ann::row_io` declares the node
  table, the packed and expanded haystacks and the declaration-context table,
  sizes each by the one row stride, and emits the store loop that copies a VAST
  row forward with named field overrides. Five passes each carried their own
  copy of that loop and their own buffer declarations, so each was free to size
  a table by a stride of its own or to drop a field it only carries forward.
  Neighbour-row kind reads move to
  `vyre_libs::parsing::c::parse::vast::build::vast_row_fields`, which now also
  publishes the forward-neighbour read the passes were writing by hand. Eight
  forwarding wrappers in
  `vyre_libs::parsing::c::parse::vast::build::typedef_visibility::chain` are
  gone and their callers name `decl_context_row_access::decl_context_base`
  directly.
- `vyre_libs::graph::dispatch::traversal_dispatch_pipeline` names the
  `vyre_primitives::graph` program builders it composes instead of
  re-publishing them. Twenty-one wrappers forwarded every argument and returned
  the result unchanged, so each body was a restatement of the primitive's
  parameter list, free to drift from the list it forwards to and unprovable
  from either side. Nothing outside the module's own tests called any of them.
  This follows the same removal already applied to the sibling
  `structural_kernel_pipeline`.
- The `vyre-libs` duplication pin records what the tree measures rather than
  leaving room under it, since a pin with slack hides the next copy.
- `c_frontend::token_fixture::classify_without_annotation` names the weaker of
  the two classification chains, which the C-AST contracts restated as a
  builder call plus a classifier call seventeen times. The classifier reads the
  typedef flags an annotation pass writes, and the two chains disagree on the
  kind of a GNU attribute's payload identifier, so a contract about declarator
  and specifier kinds may skip annotation and one about attribute payloads may
  not. Both are now spelled as which chain they mean.
- The three-dispatch scope-aware typedef annotation sequence, identifier
  prehash then brace-scope precompute then annotate, has one owner in
  `c_frontend::parity_matrix::arm_annotated_vast`. It was written twice in the
  driver's parity support and once more in a family root, and that third copy
  dispatched the global-fast annotator instead: a kernel documented as having
  no scope model, compared against the exact oracle it cannot reproduce under
  shadowing. Removed with it, having no callers left, a wrapper whose body
  forwarded a fixture's source bytes to the sequence.
- The dense byte-tile Four-Russians matvec corpus has one owner,
  `tests/support/dense_matvec_cases.rs`, with one arm per crate:
  `vyre_primitives::bitset::four_russians` pins its byte-LUT builder,
  word-count helper, CPU reference and dispatch Program, and
  `vyre_libs::encoding::bitset_transform_pipeline` pins its own sizing, LUT
  builder, parity oracle and composed Program. The corpus generator, the
  frontier masking and the naive boolean-semiring oracle were written twice,
  and the two copies swept different bounds: 0..=18 byte tiles by 1..=5
  destination words on the primitive side, 0..=24 by 1..=4 on the substrate
  side, so 384 cases in the union were exercised by neither. Both arms now run
  the union, plus a saturated-frontier group neither had, which is the only way
  the all-ones LUT row of every tile is reached at once.
- The exploded-supergraph (IFDS) CPU-reference corpus has one owner,
  `vyre_test_support::exploded_ifds_cases`, which declares the cases and owns
  what a correct CSR for them is. `vyre_primitives::graph::exploded` pins its
  allocating, fallible and workspace-reusing builders against it, and
  `vyre_libs::graph::dispatch::exploded` pins its host reference, its
  node-count helper and its dispatched path. The mixed intra/inter/GEN/KILL
  case stream was written twice and the copies ran different counts, 1024
  against 512, so the dispatched path was never asked about the upper half of a
  corpus its own file defined. The rule semantics that were hand-checked per
  suite, KILL suppression, GEN injection and inter-edge fact propagation, are
  now declared as dense edge expectations both arms are held to.
- The fixed CSR graphs the closure contracts are written against have one
  owner, `vyre_primitives::graph::csr_closure_inputs::graphs`, and an
  unrestricted edge filter has one spelling,
  `vyre_primitives::graph::csr_closure_inputs::CsrClosureInputs::allow_all`.
  The four-node chain and the four-node diamond were rebuilt from three array
  literals at each call site, four crate-local `linear_graph` helpers returned
  the chain as an owned triple so a caller had to keep `off`, `tgt` and `msk`
  alive to borrow a view from them, and more than thirty call sites restated
  the whole seven-field closure group only to set the allow mask to every kind.
  `CsrGraphShape` owns the arrays with `'static` lifetime and borrows itself as
  the view, so a contract now names the shape it means instead of agreeing with
  its siblings by coincidence.
- The dirty-output contract for a dense byte-tile matvec Program is asserted
  once, by `tests/support/dense_matvec_cases.rs`, which drives the reference
  interpreter with the all-ones output buffer and takes the LUT builder and
  Program builder as arguments. The two arms ran byte-identical interpreter
  setups and differed only in which builders they named, which is why they were
  the largest cross-crate duplicate pair in the repository. The primitive arm
  passes
  `vyre_primitives::bitset::four_russians::four_russians_dense_matvec_byte_lut`
  and the substrate arm passes
  `vyre_libs::encoding::bitset_transform_pipeline::four_russians_dense_matvec_program`,
  and the failure message names which arm failed.

### Removed

- Self-substrate no longer publishes source-text validators for deleted
  C-frontend test files or parser release artifacts. Diagnostic and
  preprocessing conformance now belongs to the live frontend and conformance
  paths.
- Neural operations and opaque-payload helpers now use their category-owned
  module paths. Flat compatibility re-exports and the `matching::ops` shim are
  gone; unclassified backend failures use `BackendError::Other`.
- Backend registration is now consumed from `vyre-driver`, the `ReferenceKind`
  alias is gone in favor of `vyre-spec::CpuFn`, and `gpu_int_literal_scan()` no
  longer accepts an ignored source-length parameter.
- The WGPU host-ingress and raw persistent-kernel compiler routes are gone.
  Persistent product execution uses authenticated artifact sessions; concrete
  pipeline compilation remains available only as a hidden oracle helper for
  driver cache tests.
- The macro crate now exports only the production-used AST registry and
  semantic pass registration generators. Test-only operation registration,
  algebraic-law derive, no-op builder marker, and generated decoder stubs are
  gone.
- The `check_warning_budget` gate script is gone: its baseline lived at a
  gitignored path with no file, so it exited before measuring anything, it
  counted cargo summary lines as warnings, and a trailing `|| true` reported
  zero warnings for a failed build. `strict.yml` already builds the workspace
  with all targets and all features under deny-warnings, so the enforced
  ceiling is zero. The `check_tier_b_rule_contracts` gate script is gone: it
  required a repository-root `rules/` tree that does not exist here, and its
  second rule scanned a directory that does not exist behind a suppressed
  error.
- Three `vyre-primitives` graph shapes are gone, all behind the `graph`
  feature. `CsrQueueStridedForwardParams` is deleted: the row-strided queue
  expansion takes the same inputs as the scalar one, so
  `csr_queue_strided_forward_traverse_with` now accepts
  `CsrQueueForwardTraverseParams`, whose fields carry the same names.
  `csr_forward_traverse_with_op_id` and `csr_backward_traverse_with_op_id` are
  deleted; the direction-parameterized frontier-step reference covers what they
  were for and nothing called them. Every other CSR closure and queue entry
  point keeps its exact name, arity and argument order even where it is now
  generated from a macro or published as a re-export, so a caller that used one
  of those needs no edit. The default-feature public-API snapshots for
  `vyre-primitives` and `vyre-libs` are unaffected, since `graph` is not a
  default feature.

### Fixed

- Driver decorators now preserve the concrete backend device profile, including
  device-timestamp capability and timing quality.
- Enforced benchmark contract failures now retain correctness, timing metrics,
  device identity, and measured speedup in the failed case report instead of
  collapsing into an unprobed error shell.
- Resident throughput batches preserve complete device-timestamp totals and
  normalize them per logical item. String bitmap scatter uses subgroup ballots
  to materialize 16 independent output rows in one resident dispatch, with
  exact CPU-oracle parity.
- `xtask heuristic-audit` now resolves both standalone Vyre checkouts and the
  enclosing Santh workspace without duplicating the Vyre path.
- Public API checks now discover every committed crate snapshot, parse exact
  package names, use dependency-noise-free output, and reject ordinary snapshot
  updates that remove or change an existing item.
- Empty QK-gain tensor shapes now declare a zero-byte output range instead of
  an unknown-size backend allocation, while overflowing positive shapes fail
  closed with an actionable trap program instead of wrapping their element
  count.
- Grouped-query attention now composes the canonical max, normalization-sum,
  and weighted-write primitives with explicit KV-head bases. Overflowing row or
  element counts fail with a sharding error before buffer declarations are
  built.
- WGPU resident dispatch now splits `GridSync` programs at launch boundaries
  before compilation, preventing oversized resident fixed-point grids from
  deadlocking inside a software global barrier.
- The WGPU stream-sharding error is now nameable as
  `engine::multi_gpu::StreamShardError` without changing existing signatures.
- Workspace documentation now resolves NFA conversion and megakernel table
  links.
- The reduction benchmark now measures atomic-scalar and workgroup-tree sums on
  the same GPU at 32 and 1,048,576 elements. It verifies both routes exactly,
  selects the measured winner per size, and records contention and barrier
  counters. NVIDIA idle clocks no longer invalidate a cold, low-utilization
  microbenchmark as thermal instability.
- V055 now accepts a post-barrier loop exit only when its full return path is
  workgroup-uniform. It derives same-address loads from an acquiring barrier
  and rejects intervening writes, divergent indices, atomics, and
  lane-dependent guards. The DCE fixpoint loop therefore removes one redundant
  barrier per iteration without weakening the unsafe-exit rejection.
- Public API snapshots now cover every workspace package whose Cargo manifest
  permits publication, including CUDA and every emitter/runtime library. The
  manifest-derived gate rejects both missing snapshots and stale snapshots for
  packages that no longer publish.
- Removed the unreachable `vyre-bench` dataflow baseline module whose
  undeclared feature could never be enabled and whose engine dependency does
  not exist in this workspace. Benchmark feature guards now have a manifest
  agreement gate, so a hidden undeclared case cannot recur.
- Regex DFA replay now gives open-ended repetitions an explicit finite policy
  instead of treating their minimum as a maximum. Whole-buffer variable-length
  matches derive exact starts from candidate origins, and region evidence
  returns one longest extent per pattern and origin.
- Weighted paged-corpus scans now expose per-device timing and byte balance.
  The physical two-adapter benchmark verifies exact single-device parity and
  records paired end-to-end speedup, topology, staging overhead, and raw
  samples.
- Package readiness now validates unpublished, version-matched release
  dependencies through local registry patches after Cargo normalizes path
  dependencies for packaging. Cross-repository `weirflow` archive evidence now
  records its real files, examples, Rust sources, and file-list digest.
- Workspace crate ownership now comes from one manifest-checked registry. The
  tier gate rejects missing crates, undeclared production edges, and stale
  generated graph or ownership guides, while planned compiler boundaries stay
  visibly separate from current workspace members.
- Testing guides are now generated for all 36 workspace members from Cargo
  features and targets plus maintained hardware, evidence, skip, and failure
  metadata. The documentation gate rejects missing, orphaned, or stale guides.
- Every workspace crate README now carries a manifest-backed contract for
  purpose, boundaries, a runnable example, features, errors, testing, release
  status, and ownership. Retired 0.4.x package claims and README drift fail the
  documentation gate.
- Operation documentation now has one generated JSON authority covering every
  linked library, primitive, intrinsic, and runtime dialect operation.
  Schema-derived inventories and subsystem catalogs expose exact tiers,
  categories, program or dialect signatures, Cargo feature routes, oracles,
  backend support evidence, algebraic laws, composition chains, and counts.
- The root README now derives every workspace crate's publication and support
  status from manifests and maintained metadata. Operation tier counts come
  from the canonical operation schema, backend claims come from executable
  backend evidence, and the architecture identifies Metal as Apple-active
  instead of planned.
- Architecture guides now use the generated 36-crate dependency graph, joined
  operation registries, CUDA-first backend evidence, typed cross-program
  composition, and explicit runtime/compiler/driver megakernel boundaries. The
  earlier device-bytecode-interpreter RFC is retained as superseded rationale.
- Documentation coverage now reports measured gates instead of universal
  completeness. Public guides distinguish generic consumers from named
  integrations, and the documentation gate rejects missing or gitignored
  repository inputs hidden in code spans and shell examples.
- The documentation matrix now covers every indexed public document and
  workspace crate README. Each row records audience, owner, authority, source
  artifacts, verification date, executable examples, version coherence, support
  status, and claim-evidence blockers.
- Every current public guide is now revalidated for Vyre 0.7.2. Historical
  architecture, migration, release, operation, and testing documents are
  explicitly archived or superseded, generated views identify their source, and
  crate-local paths remain reproducible in a clean checkout.
- Release operations now use one runbook and one generated checklist derived
  from release-train versions, repositories, package groups, tags, approval
  actions, and validated changelog fragments. The guarded launcher pushes
  candidate tags before publication, final tags afterward, and records
  completion only after external actions succeed.
- Command-line documentation now inventories and executes all 12 workspace
  binaries and 84 subcommands, publishes exact help, exit-code, environment,
  configuration, hardware, and failure contracts in crate READMEs, and gates
  drift in documentation CI. The vyre-wgpu demo is documented and exercised on
  the real GPU lane, while helper --help routes are side-effect free.
- The `types` feature of `vyre-primitives` now depends on `vyre-foundation`,
  which its shape-predicate evaluator has always aliased. Enabling only that
  feature against the published crate failed to compile; in-workspace builds
  hid it because another member always enabled a feature that pulled the
  foundation in.
- The documentation reference rule read an absolute path in a code span as
  workspace-relative, so a document mentioning `/dev/null` was reported as
  claiming a missing `dev/null` in this repository. An absolute path outside
  the checkout is no longer treated as a claim this repository can satisfy; one
  inside the checkout is still checked. The feature matrix for `vyre-libs` also
  named nine scan modules by a path that resolved nowhere, and one of them, the
  regex compiler, has been a directory rather than a file since it was split.
- Every gate now resolves the checkout it reports on from the working directory
  at run time. Two checkouts of this repository that share one cargo target
  directory compute the same unit hash for a member, so cargo hands one
  checkout a binary compiled by another; a `VYRE_CHECKOUT_ROOT` declared in
  `.cargo/config.toml` and read with `env!` was supposed to make the value a
  fingerprint input and did not, because cargo does not export a `relative =
  true` config variable to the process it runs. The shared `xtask` binary baked
  a worktree's path and `dup-scan` therefore read that worktree's pins,
  reporting this tree's `xtask` at 473 duplicated lines against a pin of 465
  while this tree's own file said 411. The variable is gone, the walk has one
  owner in `structure-gate`, and the contract asserts the property rather than
  the mechanism: the resolved root must contain the directory the run was
  invoked in.
- The duplication gate measures the repository rather than the working
  directory. It walked the tree with `walkdir` and counted every `.rs` file
  present, including files an ignore rule excludes, so a pin recorded on a
  workstation described that workstation while CI measured a smaller tree,
  which is the direction that lets a pin pass by accident. This tree carried
  twenty-two ignored `.rs` files under one rule alone. The file list now comes
  from `git ls-files --cached --others --exclude-standard`, which is what a
  commit would carry: tracked files plus new files no rule excludes, so a copy
  still counts before it is committed. Running the gate outside a git checkout
  now fails with that as the remedy instead of measuring whatever is on disk.
- The `vyre-libs` operation-registry target declares the features its
  assertions need. It pins the numeric tolerance of thirteen registered
  compositions, eleven of which are only compiled under an `nn` or `math`
  feature, and it had no manifest entry, so `cargo test -p vyre-libs` built it
  with default features and it failed on a missing registration rather than on
  a wrong tolerance. It now requires `math`, `math-linalg`, and the four `nn`
  domains that own those ops.
- The registered witness programs for
  `vyre-primitives::math::quantized::i4x8_matvec_f32_scaled` and
  `i4x8_batched_matvec_f32_scaled` bound their output buffer to a slot named
  `vector_scale` and sized it for one f32, so every lane above the first wrote
  out of bounds on the declared fixture. The registry safety rules read that as
  an out-of-bounds access on valid input, an out-of-bounds access under grid
  over-fire, and a cross-lane write-write race, three failures for one wrong
  buffer. Both now name the output `out` and declare only the three input
  buffers, matching the other packed-INT4 registrations.
- The operand namespace table of a lowered kernel op has one owner again,
  `vyre_lower::operand_class`. Structural verification and data-dependency
  queries answered from two tables that disagreed on a structured loop operand
  past the body index: one called it metadata, the other an SSA reference. The
  merged table keeps the reference, because a use that is not counted makes a
  live value look dead to elimination and hoisting.
  `vyre_lower::operand_semantics` and `vyre_lower::verify::classify_operand`
  are gone; `vyre_debug::source_walker` is now
  `vyre_debug::source_assignments`, and the value-range report types are
  reachable from `vyre_lower::analyses`.
- Every rule that reads the live operation registry now links the crates that
  submit into it. `inventory` registrations live in the object file of the
  declaring crate, and a linker pulls an archive member out of an rlib only
  when a symbol inside it is referenced, so naming a crate with a discarding
  import left the registrations out of any binary that called nothing in it.
  The production binary calls the catalogs while generating documents and saw
  all 354 registrations; the test binaries called nothing and saw zero, so
  three registry rules passed while judging an empty registry and a fourth
  reported it. Reads go through `xtask_registry::live_registry`, which calls
  each source crate's catalog and asserts what it contributed, and two new
  rules keep the account current: one requires every crate publishing an
  `operation_catalog` module to be linked there, and one requires the registry
  to hold exactly what the counted sources contributed. The tier fail-closed
  rule also mutated a tier to the value it already held, so it accepted the
  unmutated schema; the mutation is now derived from the tiers the schema uses.
  The duplicate-analysis rule no longer asserts that a signature-only
  registration exists, which the registry refuses by design, and states the set
  equality it relies on instead.
- The registered witness programs for
  `vyre-primitives::hardware::subgroup_ballot` and
  `vyre-primitives::hardware::subgroup_shuffle` passed an unguarded buffer load
  as the collective's operand, and the reference interpreter resolved a lane's
  subgroup peers by the position the schedule happened to step them in. Every
  lane of a subgroup contributes to a collective, so an operand written as
  `load(buffer, idx)` is a read at every lane index in the subgroup rather than
  only at the indices a store guard admits: the ballot performed 112
  out-of-bounds loads and the shuffle 224 on their own declared-valid fixtures,
  on the natural grid and one workgroup past it. Both now compute the operand
  into a control-flow-guarded per-lane local and take the collective in uniform
  control flow, where the participating-lane set is well defined instead of
  dependent on the active mask of a divergent branch. The interpreter now
  captures its lane snapshots indexed by lane, so a ballot returns its own
  subgroup's mask and a shuffle sources the lane it was asked for under any
  step order; before, a reversed schedule gave lane 0 the mask of the
  neighbouring subgroup. Category C registrations are enumerated from the
  registry at run time and run through every safety rule, including reversed
  and rotated lane orders, so a new intrinsic cannot arrive with a witness
  program no gate executes.
- The `vyre_reference::workgroup::Invocation::bind` and `bind_loop_var`
  documentation examples name `vyre_reference::ReferenceError` instead of
  `crate::ReferenceError`. A doctest compiles as its own crate, so the old path
  resolved to nothing and both examples failed to build.
- `scripts/check_bench_baselines.sh` requires a `benches/RESULTS.md` section
  for every crate that owns a bench source file, not for every directory named
  `benches`. The directory search demanded a measured section for
  `vyre-grammar-gen`, whose `benches/` directory holds documentation and no
  target, so the gate could not pass without an invented number.
  `benches/RESULTS.md` now carries the criterion medians of `cargo bench -p
  vyre-primitives --bench wire_throughput` and `cargo bench -p vyre-bench
  --bench release` with the machine, GPU, CPU, toolchain, and commit they were
  measured on.
- `cargo xtask check-tier-deps` judges every production dependency in the
  workspace, derived from the layer each crate already declares in
  `docs/CRATE_OWNERSHIP.toml`. It carried its own hardcoded table of crate
  names mapped to tier numbers, and it only inspected dependency entries
  written with an inline `path`, of which the workspace has three. Every
  `dep.workspace = true` edge, which is how the other several hundred are
  written, went unjudged, and the name table had gone stale in a way that made
  the two edges it could see report as violations: `xtask-registry` and
  `xtask-evidence` were unlisted and fell to a default library tier while
  `xtask` was pinned above them, so two crates in the same tooling layer were
  reported as an inversion. The gate now states one thing, the order of the
  layers, and reads the layer of each crate from the registry, so a rename
  cannot make it stale and a crate added without a declared layer, a layer
  declared without a recorded position, or a recorded position no crate claims
  each fail the gate. `structure-gate` moves to a new `standalone-tooling`
  layer below `foundation`, which is what it already is: it depends on no crate
  in the workspace so it keeps answering while the workspace does not compile.
- The clone-family IR pins for `nn::softmax` and `nn::layer_norm` track the
  current shared child-region names. Renaming the reduce-family owners from
  `vyre-libs::substrate::*` to `vyre-libs::builder::*` moved the fingerprint of
  every program embedding them, because a region generator identity is part of
  the wire encoding, and the pinned digests were left on the old names. The
  drift went unnoticed because the target needs `nn-attention`, which the
  package-scoped test command does not enable. Rewriting only those identity
  strings in the built programs reproduces the previous digests exactly, so no
  node, buffer, expression or workgroup value moved. A companion rule now pins
  the region identities each entry point carries, which names a rename directly
  instead of reporting an opaque digest change, and asserts the reduce-family
  owner is still reached by more than one caller.
- `vyre-libs` runs the registry/coverage closure gate.
  `vyre_test_support::assert_registry_closure` is the workspace's single
  enumerator of `pub fn -> Program` builders, and no crate called it, so the
  contract it exists to enforce was unenforced everywhere: a builder that is
  neither submitted through `inventory` nor named by a test still compiles,
  still appears in the generated catalog documents, and still diverges from its
  reference arm with nothing red. `vyre-libs/tests/registry_closure.rs` is the
  caller, with an empty waiver and a floor under the enumerated builder count
  so a broken source scan fails instead of reporting a clean sweep of an empty
  set. The enumerator also stopped excluding itself by one hardcoded file name,
  which was the name of a gitignored file rather than the one its own
  documentation gives; it now excludes any test file that calls it, so a waiver
  entry cannot cover itself.
- Every scalar rule leaf is runnable again.
  `vyre_libs::rule::condition_op::condition_program` declared its verdict slot
  as a backend-allocated output with no static element count, which fails IR
  validation with V130, so all eleven leaves (six file-size predicates, two
  pattern-count predicates, the two literals, and the pattern-existence check)
  built successfully and were refused before execution. The builder was neither
  registered nor named by a test, which is why nothing reported it. It now
  declares the one element it writes, and a frame contract pins the binding
  order, the slot each accessor reads, the region identity, the verdict store,
  and the element count, executing every leaf through the reference
  interpreter. The leaf table is checked against the operation ids declared
  under `vyre-libs/src/rule` on each run, so a twelfth predicate turns the
  suite red until it is pinned.
- The conformance certificate regression pin passes, and a future drift in it
  is attributable. Seven pinned values had been stale long enough that the gate
  was red on every checkout: the region-chain bundle carries the test operation
  id in an `Expr::call`, so respelling that id from
  `vyre-conform.test.identity_u32` to `vyre_conform_test::identity_u32` when
  the operation and target registries were unified put one more byte in its
  wire body, 324 to 325, and moved its digest; and all five signatures had
  moved because the signable body covers `reference_output_blake3`, the digest
  of what the reference returns, which a wire-identical program can change
  underneath. Each new signature was reproduced independently from the bundle
  digest, the corpus digest, and the single output word each program stores
  before being written down. The test now asserts those words through the same
  planning and interpreter entry point the issuer uses, so a signature that
  moves while the words hold is a framing change and one that moves with them
  is a semantic one, where before either read as an opaque hash mismatch. The
  `vyre-conform` pin generator `_compute_pins.rs`, a second copy of all five
  bundle builders and the signing body whose only output was the constants to
  paste, is deleted: a generator that can drift from the test it feeds prints
  pins for programs nobody is checking, and the failing assertion already names
  the value to write.
- Adding an IR `Node` variant can no longer be handled by a catch-all arm
  nobody chose. The AST registry macro emits `NODE_VARIANT_NAMES` and
  `node_variant_name` from the declaration site,
  `vyre_foundation::transform::visit::node_shape` records for every variant
  whether it nests statements, carries operand expressions, or holds an opaque
  payload, and `child_bodies` is the one exhaustive owner of child enumeration.
  Two traversals that re-derived that list were wrong: the reference
  interpreter's barrier scan claimed an exhaustive match but let `Node::Region`
  fall into its default, so a barrier inside a region body read as absent; and
  `walk_exprs` skipped the `offset` and `size` operands of asynchronous copies
  and the `address` operand of a trap, hiding those buffer references from
  every analysis built on it. Loop unrolling's local-declaration check now also
  treats a region body as scope-transparent. Tail duplication's read check now
  answers yes for a statement form it does not recognise instead of no, so an
  unfamiliar tail costs a missed duplication rather than code sunk past a live
  read. The duplicate visitor implementations in `vyre_foundation::visit` now
  delegate to that owner rather than restating the variant list, and the
  descendant scan uses an explicit worklist so a deep tree cannot overflow the
  native stack.
- The pass scheduler judged a rewrite against facts taken from the wrong
  program. A pass that reported no change while rewriting the program had its
  effect, linearity, shape, and cost certificates recorded against the program
  it returned rather than the program it was handed, so the next pass was
  measured against a certificate describing its own input and a cost regression
  was accepted as flat. Facts are now derived once per program, keyed on the
  program fingerprint, and reused across every gate and every fixpoint
  iteration, which also removes one deep clone of the program per running pass.
- Contract tests resolve the checkout they report on from the working directory
  at run time. `vyre_test_support::monorepo::vyre_workspace_root` delegates to
  `structure_gate::workspace_root`, the one owner of that question, instead of
  its own compiled-in manifest directory, which cargo emits as a relative path
  and which names whichever checkout built the binary when a target directory
  is shared. Seven `vyre-foundation` contract suites covering workflow script
  references, frozen trait contracts, bench corpus duplication, example
  orphans, validator error codes, wire fuzz infrastructure, workspace naming,
  and the consumer boundary went through that compiled-in path and now ask the
  helper. `ConsumerBoundaryScan::for_crate` takes the crate directory as a
  path.
- Every reader of the operation registry and the backend registry reads it
  through `vyre-registry-link`, which references a real symbol in each
  submitting crate and asserts that each linked source reached the registry. An
  `inventory` registration lives in the object file of the declaring crate, and
  the linker keeps that object only when a symbol inside it is referenced, so
  `use vyre_libs as _;` and `std::hint::black_box(METAL_BACKEND_ID)` linked
  nothing: the conformance test binaries registered two backends where the
  build declared four, and their registry rules judged that partial set without
  failing. The five driver crates now expose `registered_backend_id`, a
  function call that anchors the object file and reports whether the target
  compiled the registration at all, and the discarding imports and
  `force_link_backend_inventory` helpers they replaced are gone.
- The layer ordering places the registry link owner where its dependencies
  allow it. It has to name every registration source, including the concrete
  drivers, and it is read by the conformance and tooling crates, so it sits
  above the facade and below conformance rather than in tooling, which the
  conformance crates cannot depend on. The crate-guide generator now also
  rejects an error profile for a layer no crate occupies, alongside the check
  for a layer with no profile. Only the second direction was checked, so a
  profile survived a crate absorption while describing a layer that no longer
  existed, and the two layers introduced since had no profile at all, which
  left the generator failing for every crate rather than for the ones that were
  wrong.
- The public-API snapshot for the CUDA driver names the crate its adapter
  signatures actually come from. Ten of its public items take or return
  `FrontierTypedPlan`, `DeviceResidentTokenFactGraph`, `MegakernelScaleSample`,
  and `MegakernelScheduleError`, which moved to `vyre-libs` when the pass
  engine narrowed to pass execution; the snapshot still spelled them under the
  pass engine, so the stability gate reported drift on a surface nobody had
  changed and would have gone on reporting it in front of every real change to
  that crate.
- `WgpuBackend::compile_pipeline` is the single compile entry point, taking the
  program, the dispatch config, and an optional pre-authenticated target. Two
  near-identical wrappers existed, one for a caller-supplied WGSL and
  descriptor pair and one that lowered them itself, and both carried a
  `dispatch_arena` parameter that no line of either body read: six call sites
  threaded an arena through to nothing, which reads as a live dependency of
  compilation on arena state that does not exist. The authenticated pair is now
  the `pipeline::AuthenticatedTarget` record rather than two loose arguments,
  and the dead parameter is gone from the three signatures and every caller.
  Public signatures and emitted pipelines are unchanged.
- The `vyre-driver-wgpu` pipeline tests build their store-a-constant program
  through one `stores_u32` fixture instead of six copies of the same six-node
  builder, so adding an IR field to that shape no longer has to be chased
  through three test files. The output-slot test was a verbatim copy of
  `vyre-driver`'s own policy test, all 4096 cases of it, which proved the
  shared policy twice and proved nothing about this backend: it now asserts
  what is actually wgpu's to get right, that a reservation failure is reported
  against the `WGPU pipeline` label and directs the caller to split the
  dispatch batch before readback, plus the grow, hold, shrink and empty
  boundaries of the resize itself.
- Two rewrites that enumerated `Node` themselves reached fewer operands than
  the analysis that fed them. Cross-scope CSE recorded occurrences inside the
  offset and size of an asynchronous copy and the address of a trap, but its
  substitution pass fell through to a clone for those variants, so it hoisted a
  `let` binding whose only reader was never rewritten to use it: a program with
  a repeated asynchronous-copy offset gained a dead binding and kept the
  duplicate expression. Const-buffer folding never descended into
  `Node::Region` at all, cloning the body instead, and since a wrapped
  `Program` always carries a root region the pass folded nothing on one; its
  own tests passed only because each flattened the region first. It also cloned
  asynchronous-copy offsets and sizes and trap addresses unfolded. Both now
  drive the shared rewrite walk, so the positions an analysis inspects and the
  positions a rewrite visits are the same list.
- The wire-roundtrip and ProgramStats property suites draw statements and
  programs from the one random-IR owner, not just expressions. `ir_arbitrary`
  already owned identifiers, data types, literals and expressions after an
  earlier collapse, but each suite still carried its own copy of the five
  statement leaves, the three body-carrying statements, and the nine-buffer
  program wrapper the generated body is placed in. The buffer table is what
  makes a generated body valid, since the statement generator stores into
  `out`, `rw` and `bytes_out` and the expression generator loads from every
  declared name, so a suite holding its own copy had to be kept in step with
  both generators by hand and the copies had already drifted once. The stats
  suite generates seven statement forms the wire suite does not, so the shared
  control flow enters its choice weighted at three, which leaves each of `If`,
  `Loop` and `Block` the same one-in-eleven share it had when all eleven arms
  were written out.
- The grid-sync split fixtures read a segment body through
  `transform::visit::child_bodies` instead of a hand-written match with a
  catch-all arm. The walk applies each literal store a test backend stands in
  for, so a nesting form it does not descend into makes a store invisible and a
  split that dropped a write looks correct. The catch-all meant a statement
  variant that gains a body would have been skipped silently; the nesting is
  now stated once, in the crate that owns the IR.
- The program-header validation rules are stated once, in
  `validate_program_level`. The differential property test that compares the
  single-walk validator against the multi-walk one it replaced carried a
  verbatim copy of them: the region-wrapper rule, the workgroup-dimension rule,
  the duplicate-name, duplicate-binding and empty-workgroup-buffer rules, the
  output-buffer contract, the output markers, and the buffer lookup a walk
  needs. Those are not what the property compares, so a diagnostic corrected in
  production and not in the copy would have failed the property for a reason
  that has nothing to do with the walk, and the two copies had already drifted
  on the V105 suggestion text. The second arm still walks the node tree
  independently through `nodes::validate_nodes`, which is the only thing the
  property is for; mutating the production walk to skip the last statement of a
  body turns it red.
- The live CUDA INT4 parity contracts diff against the CPU oracles
  `vyre-primitives` publishes behind `cpu-parity`, not against a private
  reimplementation of them.
  `vyre-driver-cuda/tests/int4_quantized_gpu_parity.rs` carried its own
  packed-nibble pack and extract, its own dot, matvec, batched matvec, batched
  matmul and top-1 references, and its own little-endian word packing and
  reading, all asserted bit-exactly. A correction to a shipped oracle therefore
  left the CUDA arm pinning bit-equality against a definition nobody ships, and
  the GPU could have matched a stale reference while diverging from the product
  one. The lane patterns, shape tables, deterministic generators and binding
  order stay local to the CUDA arm, each with one owner, and the shape tables
  that genuinely differ between the fixed-pattern and generated sweeps now say
  so by name instead of reading as drift.
- A failed CUDA enqueue is cleaned up by one function. Four sites, two in the
  resident async dispatch path, one in the resident batch path and one in the
  host dispatch path, each carried the same ordering decision written out by
  hand: synchronize the stream so the device is no longer reading the buffers,
  record the sync point in telemetry, release the launch resource lease, drop
  the resident in-flight guard, then forget the allocations that the device may
  still own. That ordering is the whole correctness argument, and four copies
  of it can drift one at a time, with the failure landing as a use-after-free
  on the abandon path where nothing routine exercises it.
  `vyre-driver-cuda/src/backend/enqueue_cleanup.rs` now owns
  `FailedEnqueueGuards` and `abandon_failed_enqueue`, which perform that
  sequence once and return the same error text the four sites produced.
- The CUDA e-graph device-image upload contracts build their snapshots from
  named fixtures, and `assert_span_matches_foundation` has one definition.
  `upload_layout_contracts.rs` redefined that helper identically to the
  parent's, so the parent's copy was shadowed and a correction to it would have
  silently missed one file. Seven inline snapshot literals across four files
  were three distinct shapes, now `shared_eclass_add_snapshot`,
  `duplicate_add_snapshot` and `distinct_add_snapshot`, each documented with
  the property it exists to exercise.
- Enabling one `vyre-libs` composition feature no longer resolves the whole
  primitive substrate. `vyre-libs` used to enable 19 `vyre-primitives` domains
  unconditionally, so a consumer that wanted substring scanning also compiled
  the neural-network, parsing, topology, geometry and optimization domains, and
  no feature declared what it actually used. The unconditional set is now the
  three domains the ungated module trees need: `graph` for
  `vyre-libs/src/graph/ast_walk.rs`, `text` for
  `vyre-primitives/src/text/char_class.rs`, and `inventory-registry` so
  primitive registrations reach the catalog. Every other domain rides the
  feature that gates its consumer: `math` on the five `math-*` features and on
  `visual`, `nn` on the four `nn-*` features, `matching` on
  `matching-substring`/`matching-dfa`/`decode`, `nfa` on `matching-nfa`,
  `decode` on `decode`, `visual` on `visual`, `hash` on `hash` and `intern`,
  `bitset` on `logical`, `parsing` on `c-parser` and `python-parser`,
  `predicate` and `label` on `security`, `predicate` also on `c-parser`. `geom`
  and `opt` are dropped; no `vyre-libs` module uses them. `vyre-pass-engine`
  now takes `vyre-libs` with `default-features = false`, since it uses only
  `dispatch_buffers` and `graph`, so an optimizer-only consumer resolves seven
  primitive features instead of thirteen.
- The feature-isolation gate records the eight pairs it was missing:
  `vyre-registry-link` with no default features and with each of its `cuda`,
  `metal`, `operations`, `reference`, `spirv` and `wgpu` features, and
  `vyre-test-support` with `ir-fixtures`. The gate derives its axis from the
  manifests at run time, so both crates turned it red the moment they landed
  after the recorded set was written. Every one of the eight compiles. The
  `SOURCES` builder in `vyre-registry-link/src/backend.rs` also stops binding a
  `mut` vector that only the feature-enabled builds mutate; the linked sources
  are now one `vec!` whose elements carry the `cfg`, so the no-default-features
  probe compiles without a warning.
- The feature-isolation axis judges two kinds of selection it never covered.
  The plain default build of each member, which is what `cargo check -p
  <member>` resolves and which neither the `--no-default-features` probe nor
  any single-feature probe is, and every selection a workspace edge asks of a
  sibling, spelled in the `feature` column as a comma-joined list that opens
  with `(default)` when the edge keeps defaults. Both holes were load-bearing.
  `vyre-aot` and `vyre-pass-engine` do not compile under their own default
  build, which is why the public-API snapshot gate could not extract either
  surface, and `vyre-libs` asks `vyre-primitives` for `graph`,
  `inventory-registry` and `text` together, a combination no single-feature
  probe reaches. Cargo unifies features across a build, so a whole-workspace
  check hides a break inside such a selection behind whichever unrelated member
  happens to enable the missing piece. `--sweep --write` now merges what a run
  observed over the rows already recorded, and `--only-unrecorded` narrows the
  compiling to the selections that have no row yet, so recording one decision
  no longer costs a sweep of the whole axis, which is how the recorded set went
  stale.
- No test, bench, or fixture in the workspace resolves a repository path from a
  directory fixed at compile time. Every checkout of this repository shares one
  cargo target directory, and cargo hashes a workspace member by its path
  relative to the workspace root, so two checkouts compute the same unit hash
  and hand each other compiled binaries. Forty-eight source files read
  documents, workflows, scripts, manifests, evidence and pins through a
  compiled-in directory, and own-crate fixture and golden-file locators
  resolved that way too, so a byte-stability test compared its output against
  the other checkout's golden file and asserted byte stability against that
  tree. `vyre_test_support::monorepo::vyre_workspace_root` now delegates to
  `structure_gate::workspace_root`, the one owner of that answer, and
  `ConsumerBoundaryScan::for_crate` and `assert_registry_closure` take the
  crate directory as a path rather than a compiled-in string.
  `structure-gate/tests/checkout_provenance.rs` scans every `.rs` file under
  every workspace member's `src/`, `tests/` and `benches/`, plus the root
  `tests/` directory, with the member list read from the root manifest and
  every directory walked at run time, so a new crate or a new file is covered
  without an edit to the gate.
- Every path cited inside `release/evidence` resolves on disk, and the gate
  that says so now reads the whole document. Twenty-seven citations across five
  artifacts named files that three source moves had left behind:
  `module_cache.rs` and `eqsat.rs` became directories, and the megakernel
  schedule, device scratch, VAST walk builders, subgroup shuffle and
  conformance matrix sources moved to the crates that own them. Twenty-five of
  the twenty-seven were regenerated from the current tree by `hygiene-matrix`,
  which is the only correct fix for a scan artifact; the other two came from a
  hardcoded marker table in `xtask-evidence/src/release/backend_matrix.rs` and
  a stale row in `docs/optimization/THRESHOLD_POLICY.toml`, both of which now
  name the file that owns the tokens they prove. Citation discovery in
  `scripts/check_evidence_paths.sh` walked one shape only, a top-level key
  holding an array of objects, so 81 of 634 citations were never read; it now
  takes every `path` string at any depth and reports the route to it. The one
  dead citation that hid there, an unexpanded shell template naming a README in
  another repository, belonged to a pair of orphaned artifacts,
  `readme-contracts.json` and `readme-proof.md`, certifying a crate this
  workspace does not contain and that no longer exists under that name where it
  did; nothing here registered, generated or read them.
- The release publish order is derived from the manifests instead of listed in
  source. It was a hardcoded table of twenty-six steps, and moving library code
  into `vyre-libs` gave that crate five consumers while the table still held it
  at index twenty-one, so the recorded evidence certified an order that
  publishes `vyre-pass-engine`, `vyre-driver`, `vyre-runtime`,
  `vyre-driver-cuda` and `vyre-driver-wgpu` against a `vyre-libs` version
  crates.io does not have yet, with `blockers: []` throughout. The order is now
  a topological sort over the crates the metadata matrix calls publishable,
  keyed on the same dependency edges the order check enforces, with
  name-ordered ties so one tree yields one order. It fails closed rather than
  guessing: a publishable crate whose manifest is not on disk is a blocker, and
  a cycle among publishable crates is reported per member instead of resolved
  into some order.
- The release-evidence path gate reads every citation in a document rather than
  one key. It matched the key `path`, which named 629 of the 3418 filesystem
  paths those artifacts cite; the rest sit under `manifest`, `artifact`,
  `evidence_link`, `source_artifact`, `workflow` and bare array members, and
  nine were dead, all naming one crate's manifest after that crate had been
  folded into another. A citation is now any string with no whitespace whose
  last component carries an extension the tree itself uses, so the vocabulary
  extends with the first file of a new kind and never needs an edit here, and
  version strings, operation ids and schema ids stay out. Reported locations
  carry the key that holds the citation. The hand-written duplication
  consolidation scan records the exact command behind each of its findings and
  was re-measured: `CpuOp` moved to `cpu_op.rs`, and the `Target` it named is
  no longer in the scanned set.
- The timed CPU reference in `vyre-bench` has one owner,
  `vyre-bench/src/cases/reference_sample.rs`, and it saturates. Eleven cases
  hand-rolled the same timer and seven of them cast `Duration::as_nanos()`
  straight to `u64`, so a reference slower than roughly 18 seconds was reported
  as a small number instead of a large one, inverting the speedup it was
  compared against.
- Region fusion reads which buffers each side touches, and both of the walks
  that answered it enumerated buffer positions per node variant and ended in a
  catch-all. Neither had an arm for the four collective statements, which name
  their operands as buffers and carry no operand expression at all, so an
  all-reduce, all-gather, reduce-scatter or broadcast was reported as touching
  nothing; an atomic read-modify-write was reported as a pure read. The
  positions are now owned exhaustively in
  `vyre-foundation/src/transform/visit.rs`, where adding a variant fails to
  compile, one walk collects both directions, and a node carrying an opaque
  payload reports the sets as a lower bound rather than as empty, which
  declines the fusion instead of guessing at it. The test that should have
  caught this compared two inline copies of the walk against each other and
  never called the production one.
- The tool that generates every operation document from the live registry now
  links every crate feature that gates a registration. Its dependency edge on
  `vyre-primitives` named fourteen domain features by hand, and `geom`, `opt`,
  `decode` and `visual` had fallen out of that list long after the documents
  were generated with them on, so nine files of registrations were invisible:
  the walker reported a smaller registry, and every generated document agreed
  with it. Nothing was red except three backend matrix rows reporting `no live
  registration` for `clifford2_geometric_product`, `tfn_scalar_mix` and
  `homotopy_euler_predictor`, which also took down the inventory contract in
  `vyre-foundation`. Both edges now name the source crate's own aggregate
  feature instead of a list kept by hand, and the regenerated documents are
  byte-identical, which is the proof that the documents were right and the edge
  was wrong.
- Grid-fence counting for wave-structure assertions descends through the
  workspace's single exhaustive owner of node nesting. Seven copies each
  re-derived the walk with a trailing catch-all arm, which classifies an
  unrecognised nesting variant as containing no fences and would let an
  under-fenced program pass its own structure test. The grid-fence builder, the
  launch-width reader, and the declared-flag-width reader also had three copies
  each.
- `Log2`, `Exp2`, `Tan`, `Acos`, `Asin` and `Atan` had two different f32 parity
  windows depending on which gate ran a program. The per-op ULP audit had
  forked the transcendental classifier behind `f32_ulp_tolerance` and its
  operator set was a strict superset, so those six got the 128 ULP backend
  window in the audit and the 4 ULP elementary window everywhere else.
  `vyre-foundation/src/fp_parity.rs` now classifies the union, which is the
  correct direction because each lowers to an approximate native instruction,
  and the fork is gone. `Reciprocal` stays in the elementary window: cuda and
  wgpu both lower it to a division. A gate enumerates every frozen `UnOp` and
  fails on one that neither classification table names, so a new variant cannot
  inherit the elementary window by omission.
- `vyre-primitives` feature `graph` enables `fixpoint`.
  `graph::persistent_bfs::program` reads
  `fixpoint::persistent_fixpoint::grid_sync_barrier`, the single owner of the
  grid-wide fence node, so the crate did not compile with `graph` alone.
  Nothing was red because every aggregate that exercises graph also turns
  `fixpoint` on.
- Six intra-doc links were unresolved or pointed at a private item, and each
  was reported only by the public-API gate's rustdoc pass rather than by a
  build. A module carrying both an outer doc comment on its `mod` declaration
  and an inner `//!` header has the two merged and resolved in the PARENT
  scope, which is why `vyre-driver`'s `TargetDialect` link failed from inside
  the module that defines it; that one is now fully qualified. The five links
  naming a private item are plain code spans, because a link to an item a
  reader cannot reach is not a link.
- The frontier leaderboard reads a metric percentile through the same reader as
  every other artifact inspector. Its private copy accepted only an integer
  p50, so an artifact recording a float percentile was reported as missing the
  metric entirely.
- The pass-family benchmark manifest records a family as covered only when the
  case proving it passed. It used to push each specification's declared
  families whether or not the artifact behind them could be read, and the
  specification list is the same list as `required_pass_families`, so
  `uncovered_pass_families` was structurally always empty and the release gate
  demanding it be empty could not fail on the one thing it exists to catch: a
  missing or blocked artifact. A test asserts that state, so the unconditional
  form turns the suite red.
- The pre-emission scan that decides which buffers Naga emits as `atomic<...>`
  and which keep `BufferAccess::ReadWrite` takes descent, operand positions and
  per-node buffer direction from the exhaustive owners in
  `vyre_foundation::transform::visit`, and returns both sets together. The
  write half was a hand-rolled recursive descent ending in `_ => {}`, so the
  four collective variants were reported as writing nothing: a buffer written
  only by an `AllReduce`, `AllGather`, `ReduceScatter` or `Broadcast` was
  auto-downgraded to `ReadOnly` and emitted as `var<storage, read>`. The atomic
  half restated `node_operands` as fifteen `NodeVisitor` method bodies.
  `node_buffer_refs` disagreed with the old scan about
  `Node::IndirectDispatch`, which names its count buffer as a read because the
  host writes it and the shader only reads it; the scan now agrees, and no
  emitted shader changes because the Naga emitter rejects `IndirectDispatch`
  before producing WGSL.
- A decorator that wraps a `VyreBackend` forwards the contract through
  `backend::forward`, split into the non-dispatch surface and the
  `Program`-carrying surface. `GridSyncSplitBackend` restated all 57 methods by
  hand and left seven on their trait defaults, so through the grid-sync
  registry wrapper a device-buffer-capable backend reported
  `UnsupportedFeature` for allocation, upload, download, free and device-buffer
  dispatch, a backend with distributed collectives reported none, and
  `cooperative_grid_sync_fits` answered `false` for every device. Device-buffer
  dispatch of a program that needs the host-side grid-sync split now fails
  closed naming the borrowed path, because the split carries each segment's
  state through host byte buffers and the device-buffer path exposes no
  readback between segments.
- Benchmark case declaration and span timing each have one owner.
  `api::metric::elapsed_ns` narrows every measured nanosecond count, replacing
  three spellings across 35 sites in 28 files: a bare `as u64` cast, a
  `min(u64::MAX)` clamp, and a `try_from().unwrap_or()`. The bare cast wrapped,
  so a span past about 18.4 seconds reported as a short one and the slowest
  sample read as the fastest. `cases::honest_case` owns the suite list,
  metadata and memory floor every honest case declares; `search.binary.u32.1m`
  had omitted the smoke suite from its own copy of that list and was never
  smoke-tested. `cases::reference_sample::run_against_reference` accounts both
  halves of a reference comparison, closing two records that published a
  baseline carrying only a wall time and one that read the baseline's
  written-byte total off the device output.
  `cases::release_workloads::resident_batch` owns resident batch dispatch and
  its metric points, replacing two hardcoded reset-byte constants with the
  uploaded payload length and routing the metadata condition workload through
  the shared run assembly it had bypassed, which is where it had been silently
  omitting its throughput metrics. `api::case::prepared_as` and
  `api::case::prepared_as_mut` own borrowing a prepared payload as its own type
  and own the wording when it is the wrong type. Sixteen cases hand-rolled that
  downcast; the read-only and mutable flavours meant nine copies survived a
  first collapse, and the message had drifted into three unrelated sentences
  for one condition. An `Option`-returning downcast is a different operation
  and stays where it is. `case_declaration_contracts` derives its coverage from
  the case registry and a walk of the crate source, so a new case or a
  reintroduced narrowing fails by name rather than going uncovered.
- Every per-node validation rule lives in
  `vyre-foundation/src/validate/node_rules.rs`, which both the production
  single-pass validator and the legacy multi-walk arm call. The two walks each
  carried their own copy of the rule bodies, so a correction to one was not a
  correction to the other: the empty async stream tag reported as `V117` from
  an `AsyncLoad` or `AsyncStore` and as `V128` from an `AsyncWait`, giving one
  condition two stable rule identities, and the `V121` store-type message was
  missing a closing backtick on one side. The rule is now `V128` for all three
  async node kinds, `V117` is retired from the registry and the code catalog,
  and the walks differ only in how they traverse the node tree. An `AsyncLoad`,
  `AsyncStore` or `Trap` also validates its own operand expressions, which went
  unchecked in both walks, so a load from an undeclared buffer inside a
  transfer size or a trap address was accepted while the same load in a store
  index was rejected.
- `docs/CRATE_OWNERSHIP.toml` declares `xtask-evidence`'s production edge on
  `vyre-foundation`, the `foundation-ir` seam it reads the release optimization
  family list across. The manifest gained the dependency without the
  hand-maintained ownership contract gaining the row, which fails both the
  ownership and the crate-README gates.
- A gate that needs its own crate directory resolves it at run time through
  `structure_gate::member_directory`, reached as
  `vyre_test_support::monorepo::vyre_crate_directory(env!("CARGO_PKG_NAME"))`.
  Nine files, eight of them gates added this cycle, built a repository path
  from a compiled-in manifest directory. Every checkout here shares one cargo
  target directory and cargo hashes a member by its path relative to the
  workspace root, so two checkouts compute the same unit hash and hand each
  other compiled binaries: the reused binary then reads the OTHER tree's pin,
  fixture or golden file and asserts about that tree while claiming to describe
  this one. The resolver walks the root manifest's member roster rather than
  joining the package name onto the root, because a member's directory is not
  always its package name.
- The packed-bitset addressing skeleton, the one-dimensional dispatch grid and
  the double-buffered workgroup scan sweep each have one owner in
  `vyre-primitives`: `graph::frontier_bits`, `graph::lane_grid` and
  `reduce::workgroup_tree::hillis_steele_inclusive_sum_nodes`. Each was
  hand-written in a dozen or more graph primitives, and the copies disagreed at
  the zero case. Two dispatch grids returned zero groups for an empty input,
  which the CUDA launcher rejects outright because it requires every grid axis
  above zero, and the empty input reaches those grids because their input
  validators accept it. Three copies of a ceiling division underflowed at zero
  and were safe only because every caller happened to floor its argument first,
  and a fourth returned zero groups. A frontier word count multiplied three
  unsigned factors without saturating, so three large factors wrap to a small
  allocation. A kernel body restated per primitive drifts where no per-op
  oracle looks, because an oracle compares evaluated output and never sees the
  grid or the allocation size.
- Two CI jobs named cargo test targets that do not exist and would have failed
  with `no test target named` on the first run that reached them. The
  architecture gate ran `--test architecture_docs --test
  canonical_first_workgroup_guard` after both became modules of the
  `tree_contracts` target, and the conformance CPU job ran `--test
  generated_graph_oracle_matrix`, a `vyre-primitives` target deleted when its
  content moved and never repointed, so the nine `sweep_graph_*_oracle_matrix`
  sweeps it stood for went unrun there. The existing CI inspector could not see
  either, because it asserts that a command STRING appears in a workflow file,
  which is as true of a command naming a deleted target as of one naming a live
  target.
- The six byte counts that decide a megakernel wave's device-memory plan travel
  as one value, `vyre_driver::megakernel_execution::MegakernelByteLayout`,
  instead of a positional list of six `u64` arguments restated at every hop
  from the caller that measures them to `plan_megakernel_memory_budget`. The
  list was written out nine times across `vyre_driver::megakernel_execution`,
  `vyre_driver::megakernel_frontier`,
  `vyre_driver_cuda::megakernel_plan_cache`,
  `vyre_driver_cuda::megakernel_scheduler` and
  `vyre_driver_cuda::megakernel_barrier_planner`, and any two of the six could
  be exchanged at a call site without a compiler complaint because they share a
  type: swapping the scratch and output counts, or the per-node and per-edge
  counts, produced a plan that was arithmetically valid and wrong. Named fields
  make each of those a type error. `MegakernelExecutionRequest` carries the
  same value rather than a second copy of the same six fields, and the CUDA
  plan cache's unbounded sparse probe now says so in its own argument instead
  of leaving a bare `u64::MAX` in the eighth position.
- The workspace member roster has one reader,
  `structure_gate::workspace_members`. It was parsed out of the root manifest
  three times, once inside `structure-gate` and once in each of two gates, and
  every copy carried its own filter over the member list, so a member added
  under a path one copy accepted and another skipped would be gated by one and
  invisible to the other. The CI reference gate resolves a package's directory
  through `structure_gate::member_directory` for the same reason instead of
  walking the roster itself.
- The loop restructuring passes ask three questions before they reorder
  statements, and each now has one owner.
  `vyre_foundation::optimizer::passes::loops::var_reads`, `touched_buffers` and
  `bound_names` are public, and
  `vyre_foundation::transform::visit::node_bound_name` answers which statement
  binds a name with no catch-all arm. The walks they replace named their own
  variants and ended in `_ => {}`, so a `Var` read in `Node::Trap.address` or
  in an async copy's `offset` reported ABSENT: `loop_fusion` fused two loops
  across a scalar one of them assigns, which silently changes the values the
  program computes, and `legality::bindings_flow_across` weakened the capture
  guard for both fusion and fission. The rematerialization pass asked the same
  question through a `_ => false` arm and could inline a stale definition
  across a rebinding.
- The fusion-alias hazard rule `V116` has one owner, the whole-program pass in
  `vyre-foundation/src/validate/fusion_safety.rs`, which both the production
  single-pass validator and the legacy differential arm call. Production
  carried a second copy inside its frame stack that recorded only two of an
  async transfer's four operands, so an atomic sitting in an `AsyncLoad` or
  `AsyncStore` `offset` or `size` expression was not recorded as an access and
  did not raise a hazard against an unsynchronized read of the same buffer: the
  program validated clean and was fused. The hazard is a relation between two
  nodes rather than a property of a frame, and one rule with two
  implementations gave one program two answers depending on which arm read it.
  `V113`, the malformed-alias-frame code the deleted copy raised, is retired
  from the registry and the code catalog.
- Both validator walks agree that `Node::Region` scopes its body. The legacy
  multi-walk arm recorded no scope log for a region body, so a `let` inside a
  region was never undone and stayed live past the region, past an enclosing
  `If` branch, and on through the rest of the program, which let it shadow or
  collide with a later binding of the same name without a diagnostic. The pass
  that flattens a small region into its parent sequence re-wraps it in a
  `Node::Block` on a name collision, and that is sound only while a region
  body's bindings die at the region boundary.
- The reference interpreter evaluates `BinOp::WrappingAdd` and
  `BinOp::WrappingSub` on `u32` and `i32` operands. Both are aliases of `Add`
  and `Sub`, which the interpreter already evaluates with wrapping arithmetic
  and which the constant folder and every emitter lower identically, but only
  the `u64` branch listed them; a graph that named either on a 32-bit width was
  refused with `unsupported u32 binary operation` instead of evaluated. Four
  generated per-width sweeps covered the same operation table and none of them
  swept the pair, so the gap was invisible to all four.
- The `program_wire` fuzz corpus is checked in, and every seed in it is
  replayed against the fuzz target's own three invariants on an ordinary test
  run. The corpus directory was covered by the `cargo fuzz init` default ignore
  rule while a gate asserted that its regression seeds were checked in, so that
  gate passed on the machine that had once run the fuzzer and failed on every
  fresh checkout. It also proved less than it read: the entry-count floor it
  enforced was met by fifty-nine near-copies of one seed, and one entry was
  byte-identical to another, which spends fuzzer budget re-deciding an input it
  has already decided. The replacement decodes every entry, requires each
  rejection to carry a `Fix:` hint, requires each decode to survive a `to_wire`
  round-trip under `structural_eq`, and requires each named valid-program seed
  to still decode, so a wire-format change that strands an old seed is red
  without libFuzzer and without a nightly run.
- `vyre-libs::security::aliases_dataflow` and `vyre-libs::security::taint_kill`
  name themselves on the region they hand out. The first returned a fused
  program straight from `fuse_programs`, so its entry region was
  `vyre.program.root` and the operation had no composition identity at all; the
  second reparented its children to its own id but never wrapped them, so its
  entry region was `vyre-primitives::bitset::and_not` and the operation was
  indistinguishable from the primitive it composes. Its macro-generated
  siblings all call `tag_program`, which also carries the self-exclusive region
  marking the hand-written copy dropped. `library_operation_provenance` reads
  the linked operation registry at run time and fails naming any library
  operation whose entry region is not a region generated by its own id, so a
  builder that forgets the tag cannot ship uncovered.
- The lane-count-to-dispatch-grid ceiling division has one owner,
  `vyre_primitives::dispatch_grid::lane_grid`, ungated at the crate root and
  re-exported from `vyre_primitives::graph` so every existing call path
  resolves unchanged. It previously lived inside the `graph` domain, where a
  domain that does not enable `graph` cannot see it: `decode` does not, and
  `vyre_primitives::decode::rle_segment_lengths` therefore split its lane count
  into whole blocks plus a tail block and floored the sum, its own fourth
  spelling of the same arithmetic. An owner a caller cannot reach is not an
  owner, and the caller that cannot reach it writes the copy. Routed onto the
  owner with it: `vyre_primitives::graph::union_find`,
  `vyre_primitives::graph::persistent_bfs::layout`,
  `vyre_primitives::math::scallop_join`,
  `vyre_primitives::math::scallop_join_wide`,
  `vyre_primitives::math::bigint_add_carry` and
  `vyre_primitives::math::scallop_persistent`, which carried a private ceiling
  helper of its own. The persistent-BFS copy spelled it `((count - 1) / width)
  + 1`, which underflows at zero and was safe only because its one caller
  guarded zero separately.
- `vyre_primitives::math::bellman_shortest_path::BellmanBuffers` publishes
  `CANONICAL` and `TERSE` binding-name sets, matching the
  `SinkhornBuffers::CANONICAL` it already had. Four sites spelled the same six
  names, one of them in another crate, while the record documented itself as
  the only place they were spelled. Every field is a `&str`, so a full
  re-spelling is a positional list wearing field names: a transposed
  `src`/`dst` or `dist`/`next_dist` compiles, looks deliberate from either
  copy, and silently relabels which buffer the relaxation reads. The two sets
  stay disjoint so an assertion that reads a name back out of a program cannot
  pass by matching the other set.
- The persistent-fixpoint routing contract is asserted in one place,
  `vyre_primitives::fixpoint::routing_contract`, over a description of what an
  op builds on either side of one workgroup width. Each routed convergence op
  previously carried its own copy of the same four obligations, and the copies
  had drifted in what they accepted: both pinned the grid fence count to a
  literal 8 without stating that the count is two fences per wave, so neither
  would have caught a build whose wave count changed. The contract now checks
  all four at four iteration budgets, and both ops register every way their
  dispatch outgrows one workgroup rather than only the widest state buffer:
  `vyre_primitives::math::bellman_shortest_path` registers node-widened and
  edge-widened spans, and `vyre_primitives::math::sinkhorn_iterate` registers
  scaling-widened and kernel-widened spans. The span is the widest declared
  buffer once a program carries atomics, so four nodes with 257 edges, and a
  17-by-17 kernel over two 17-element scaling vectors, both cross the threshold
  while the ping-pong state still fits one workgroup.
- `release-evidence` runs its thirteen child subcommands through
  `xtask::delegate::dispatcher`, the one owner of which binary runs a
  subcommand. It resolved that child from the running executable, which was
  correct only while it lived in `xtask`; after it moved to `xtask-evidence` it
  re-entered itself, met that crate's six-entry table, and exited 1 with `is
  not implemented in xtask-evidence` for twelve of the thirteen. Only
  `backend-matrix` worked, because `xtask-evidence` happens to own it. Each
  failure was still labelled `xtask <name>`, so one wrong process read as
  twelve gates failing and the release evidence surface reported sixteen
  blockers where seven exist. The same resolution in `xtask::gates::sweep` and
  `xtask::release::release_gate` was three copies of the same eight lines of
  failure handling and is now one call.
- `vyre_primitives::graph::csr_backward_or_changed` named the per-edge kind
  array `masks` and the scalar edge-kind filter `edge_kind_mask`, inverting the
  roles every sibling CSR module gives those two names. A call repointed from a
  sibling module fed the scalar where the per-edge array belonged. Both
  parameters now carry the tree-wide role names.
- `vyre_primitives::decode::ziftsieve` publishes `ZiftsieveBuffers` and
  `ZiftsieveExtents`, so the five buffer names and three extents an indexed
  literal-copy program binds are named at every construction site instead of
  being passed as an eight-long positional list through three entry points and
  a composition wrapper. Every one of the five names is a `&str`: transposing
  `seq_literal_len` with `seq_literal_offset` compiled, read as deliberate from
  either side, and copied each literal run to the length it should have had
  rather than to its output offset. `ZiftsieveBuffers::CANONICAL` publishes the
  binding set the registered fixture already used, matching
  `SinkhornBuffers::CANONICAL`. `vyre_libs::decode::ziftsieve::ziftsieve_gpu`
  takes the same records and its tests now assert what the composition adds
  rather than restating the primitive's decode semantics: three of its four
  tests were copies of the primitive's own cases, while nothing asserted the
  family-scoped buffer rewrite that is the module's reason to exist.
  `NOTE_ZIFTSIEVE_GPU_DESIGN` is gone; a public constant holding a source path
  is not an API.
- `vyre_libs::parsing::c::parse::vast::build::vast_row_fields` clamps the index
  before loading a prior row kind. The subtraction was unclamped, and because a
  select in this IR evaluates both arms, row 0 wrapped its index to near
  `u32::MAX` and the untaken arm addressed a row far past the end of the table.
  Three of the five hand-written copies of the read already clamped. The
  global-typedef fast pass also read its forward neighbour with no out-of-range
  fallback while every sibling and the CPU oracle substituted the sentinel;
  both reads now come from the shared owner.
- Every INT4 entry point in `vyre_libs::solvers::quantized_dispatch` rejects a
  backend output buffer count other than one through
  `shapes::expect_one_output`, which already owned that diagnostic while four
  of the six inlined their own copy of it. The copies were each half a
  contract: only `i4x8_batched_matmul_top1_f32_scaled_via` rejected two output
  buffers, and only it rejected a buffer shorter than the decoded shape, while
  the other four rejected only a longer one. `unpack_i4x8_via` checked neither.
  `decode_output_exact` compares byte lengths for inequality, so both
  directions always held for all six and no suite asserted more than one of
  them. Three parity assertions also compared results through `zip`, which
  passes on a truncated readback; they now compare the whole buffer.
- The sparse-queue step sequence of
  `vyre_libs::graph::dispatch::adaptive_traverse` is asserted once, over a
  table naming which queue materializer each graph width and frontier selects.
  Five near-identical cases previously each rebuilt the dispatcher, graph,
  scratch and packed frontier and then spelled their own step expectation, and
  the copies disagreed about what to check: two cases exercised the same
  materializer with the same expectation, only the wider one asserted that no
  word-partial buffers are allocated, three of five asserted the upload set,
  and two of five asserted the plan cache. No case sat on the block count at
  which frontier word prefixes stop inlining block offsets, so shifting that
  threshold by one left the suite green.
- The C-AST parity families evaluate one case list on both backends. Each
  family has a CPU classification arm reading the oracles and a backend parity
  arm dispatching the same kernels, and each arm used to enumerate its cases by
  writing one test function per fixture it named, so the two lists were
  independent and had drifted.
  `c_frontend::fixtures::declarator_matrix_constructs::fixture_gnu_restrict_qualifier`
  was named by the CPU arm and by no backend arm, leaving GNU `__restrict`
  normalization proven on the oracle and unproven on every device;
  `c_frontend::fixtures::semantic_gap_constructs::fixture_inner_typedef_shadows_outer`
  reached no backend arm either, so the one construct in that family whose
  purpose is scope-dependent typedef visibility was never dispatched; six
  declarator-matrix cases and `fixture_anonymous_struct_union` reached a
  backend classifier but never a backend property-graph lowerer. Both arms now
  iterate the family's `CASES` table beside its fixtures, and
  `c_frontend::parity_matrix` owns the four stages, every program they
  dispatch, and the comparisons, so a construct cannot be proven on one side
  and unproven on the other.
- `c_frontend::parity_matrix::assert_pg_mirrors_every_vast_row` checks the
  property graph against the typed VAST at every row a fixture produces, and
  asserts the row count equals the token count so the span columns cannot
  compare unrelated rows. The lowerer emits one graph row per VAST row, so that
  is its whole contract; each family previously pinned a hand-written index
  list for a few of its cases, which is a member set that goes stale in
  silence, and the rows those lists left out were the ones nothing covered.
- The asm-alias, mixed-initializer and incomplete-initializer classification
  contracts live with the rest of their family in `vyre-libs`. Nothing in them
  dispatches anything, so holding them in a driver crate's tests made the CPU
  classification of three C constructs depend on that driver compiling and left
  a package-scoped `vyre-libs` test run unable to check them. Deleted with the
  move: a second fixture builder for `typedef int (*fn_t)(int); fn_t f;`, whose
  four assertions were already made against the byte-identical
  `c_frontend::fixtures::declarator_matrix_constructs::fixture_nested_typedef_complex_declarator`.
- IR variant descent has one owner per reference mode.
  `vyre_foundation::transform::visit::child_bodies_mut` is new and owns the
  body slots of a node held by unique reference, alongside `child_bodies` for a
  shared read and `vyre_foundation::transform::rewrite_walk::rewrite_node` for
  a borrow-preserving rebuild. `vyre_foundation::visit::node_map::map_body`
  took its slot list from a hand-written match ending in a catch-all that
  returned the node unchanged, so a body-bearing variant the list had not been
  told about made every pass composed on it a silent no-op inside that variant,
  including `rematerialize_cheap_let` and the pass engine's constant
  propagation. The scalar namespace also has one owner,
  `vyre_foundation::transform::visit::node_scalars`, reporting the bound name,
  what the statement does to it, and the operand expressions in one record;
  `node_operands` and `node_bound_name` are derived from it, and
  `vyre_foundation::visit::bound_names` no longer classifies an unrecognised
  variant as binding nothing. `vyre_foundation::optimizer::cost` folds the
  divergence dimension over the owning descent instead of a search that stops
  at the first match, and `vyre_foundation::transform::autodiff::grad` reads
  its forward types and adjoint targets from the owner and reports an
  unsupported node by its registry variant name rather than by a `Debug`
  rendering truncated at sixty characters.

## [0.7.1] - 2026-08-01

### Fixed

- Release benchmark source fingerprints now exclude operator-internal files such
  as `AGENTS.md`, `CLAUDE.md`, and `SKILL.md`. A public checkout therefore
  reproduces the same runtime source identity as the private release workspace.

## [0.7.0]  -  2026-07-30

One release. The work that had been staged as 0.6.6 is folded in here: it could not
ship as a patch, because making its release gate pass required canonicalizing
eigenvector sign in the shared Jacobi body, and that changes the observable output of
a published op.

The only source edit an upgrade requires is the dataflow-import rename. See the
migration table under "Removed".

### Fixed: fusing a narrow synchronizing arm produced an intermittently wrong kernel (`vyre-foundation`)

`fuse_programs` set the fused workgroup size to the axis-wise maximum over the
arms and fused anyway. For an arm whose invocations are independent that is only
a launch-size change. For an arm that synchronizes its workgroup or keeps state
in workgroup memory it changes the meaning of the arm: such an arm guards its
body for its own width, so under a wider workgroup the invocations with no work
skip the guarded body and never reach the barrier the working invocations wait
on. A workgroup barrier that is not reached by every invocation in the workgroup
is undefined.

Piping `sinkhorn_scale::consumer_b` (workgroup 256) into `scan_prefix_sum` at
n=4 (workgroup 4, two workgroup buffers, five barriers) produced a kernel that
returned the wrong final lane on 49 dispatches out of 500 of the same input:
the prefix sums came back as `[4, 7, 17, 8]` instead of `[4, 7, 17, 19]`. Being
intermittent, it read as flakiness rather than as unsound fusion.

`fuse_programs` now refuses such a batch with the new
`FusionError::WorkgroupGeometry`, naming the arm, both geometries, and what in
the arm makes the widening unsafe. Arms whose invocations are independent are
still widened, and arms that already agree on their workgroup still fuse even
when both synchronize. If you hit the refusal, dispatch that arm separately.

A fused program also keeps `non_composable_with_self` as the OR over its arms.
It used to be reset to `false`, so a second round of fusion could place two
copies of a scratch-carrying body in one kernel. The same loss is fixed in the
decode-scan fusion pass, the streaming decode adapter, and the two scan
programs that tag themselves with a region: all of them rebuilt through
`Program::wrapped`, which constructs a NEW program and so starts the metadata
fresh. Use `with_rewritten_buffers`, `with_rewritten_entry`,
`with_rewritten_wrapped_entry` or `map_entry` to change part of a program.

`vyre-foundation/tests/fusion_workgroup_geometry.rs` and
`vyre-foundation/tests/fusion_composability_metadata.rs` pin both.

### Fixed: the raw-byte C syntax parser under-counted tokens (`vyre-frontend-c`)

Any source above two 1024-token blocks reported a token count that was too low,
with no error. 4096, 8192 and 66560 semicolons all reported 2048 tokens; 2049
reported 1025.

Sparse token compaction runs a block-total stage, one 1024-lane workgroup per
block, that writes each block's token count to `block_totals[block]`. That
stage's only sized buffer is `block_totals`, one word per block, and its input
arrives as a resident device blob whose length the dispatch grid inference cannot
read. Inference therefore chose `ceil(num_blocks / 1024)`, which is one workgroup
for every source under a million tokens. Block 0 computed its total and every
later block kept the zero it was allocated with, so the scanned prefix that ranks
tokens in the compact stage collapsed to `block_totals[0]`.

Both block-total dispatches now state their grid instead of leaving it to be
inferred. `vyre-frontend-c/tests/raw_syntax_multi_block_token_counts.rs` pins the
exact count at each block boundary.

### Fixed: three rewrite passes reused result ids across bodies (`vyre-lower`)

Result ids are unique across a whole `KernelDescriptor`, not per body: the PTX
emitter keeps one flat result-id to register map for the entire kernel, so an id
two bodies both define resolves to whichever producer the emitter walked last.
Three passes broke that.

- `branch_collapse` inlined a collapsed `StructuredIfThen` body into its parent
  but left the child body populated. Child indices are positional, so the slot
  cannot be removed without reindexing its siblings; it is now emptied instead.
- `egraph_saturation` and `shared_mem_promote` recursed over the body tree and
  rebuilt their result allocator at every level, so a nested body seeded its
  high-water mark from its own subtree. Both now thread one allocator from the
  descriptor root, the same shape the 0.7.0 `loop_unroll` fix uses.

The debug-only post-pass verify surfaced this on the int4 CUDA parity suites. It
stayed hidden because three descriptor builders, including the soundness fuzzer,
assigned ids per body and so failed `verify` before any rewrite ran.

### Fixed: GPU dead-code elimination supplied one input buffer too few (`vyre-self-substrate`)

The persistent-BFS analysis program `gpu_dce` dispatches gained a `converged`
output in this release. A ReadWrite buffer binds as InputOutput, so it consumes
an input slot as well as an output slot, and the direct path kept filling eight
slots for a program declaring nine. Every dispatch failed with "expected 9 input
buffer(s) from Program declarations but received 8". The resident path already
passed all nine. A new suite runs the pass against a recording dispatcher, so a
future slot-count drift fails without a GPU.

### Removed: the `strict-fp` feature (`vyre-harness`, `vyre-test-harness`)

`strict-fp` claimed to forbid multiply-add contraction and demand bit-identical
f32 results. It forbade nothing: no emitter read it, and its only effect was to
force `f32_ulp_tolerance` to 0 for backend-vs-reference comparisons. Since
contraction is a documented backend right, and both cuda and wgpu fold `a*b+c`
into one FMA, that made `cargo test --workspace --all-features` unable to pass:
`newton_schulz_poly5_f32` drifted 4 ULP, `newton_schulz_5step` 2 and `ema_apply`
1, with the two backends agreeing bit-for-bit with each other and differing only
from the CPU reference.

If you enabled `strict-fp`, drop it from your feature list. The elementary and
transcendental ULP budgets are unchanged and still apply. Bounding contraction
is an emitter job; a tolerance constant cannot do it.

### Security: two advisories cleared in the dependency graph

- `crossbeam-epoch` moves 0.9.18 to 0.9.20, clearing RUSTSEC-2026-0204: the `fmt::Pointer`
  impl for `Atomic` and `Shared` dereferenced the underlying pointer, so formatting a null
  pointer was an invalid dereference.
- `anyhow` moves 1.0.102 to 1.0.104, clearing RUSTSEC-2026-0190: adding context with
  `Error::context` and then calling `Error::downcast_mut` on the result violated borrow
  rules and was undefined behaviour.

`cargo deny check` is now green on advisories, bans, licenses, and sources.

### Added: a composite op can take a whole buffer (`vyre-foundation`)

An op could only receive scalars, so a phase that indexes a table could not be
split into its own composition. The only way to name a buffer at a call site was
`Expr::Var`, which the validator reads as a scope-bound variable, so every such
program was rejected with "reference to undeclared variable". The
composition-discipline gate therefore told over-budget ops to split into
compositions while the pipeline refused to compile the result.

- `Expr::BufferRef { buffer }` names a buffer. It is not a value: it has no type,
  and the validator rejects it (V051) anywhere except a call argument. Build one
  with `Expr::buffer_ref("table")`.
- An op signature declares such a parameter as `buffer<u32>`. The validator checks
  that the argument is a buffer reference (V053), that the buffer is declared
  (V052), and that its element type matches (V054).
- Inlining a call with a buffer argument RETARGETS the callee's loads, atomics, and
  `BufLen` at the caller's buffer, keeping the callee's index expressions. A scalar
  argument still substitutes its value, so `BufLen` of a scalar parameter stays 1
  and `BufLen` of a buffer parameter is now the caller buffer's real length.
- Wire format rev 5 adds expression tag 22 for it. The decoder still reads rev 4,
  since rev 5 only appends a tag. See `docs/wire-format.md`.
  `framing::wire_format_version_is_supported` is now the single owner of the accepted
  range: three decode paths had each spelled the comparison for themselves, and one was
  missed when the range widened.
- `V047` and `V051` through `V054` are cataloged in `docs/error-codes.md`. Call-signature
  validation moved to its own `validate::call_rules` module.

### Fixed: the reference interpreter computed nothing for a composite op (`vyre-reference`)

`Expr::Call` was always dispatched to the op's registered CPU function. A composite
op is defined by its IR body and registers no CPU function, so it landed on the
non-executable sentinel in `LoweringTable::empty()`, which clears the output buffer
and returns. The interpreter reported success and produced zeros.

- The interpreter now inlines every composite body before execution, through the
  single `program_for_interpreter` funnel, so only intrinsics reach the CPU dispatch.
  `vyre_foundation::ir::inline_composite_calls` is the new entry point;
  `UnresolvedCalls` selects whether an unresolvable call is an error or is left in
  place.
- Reaching the sentinel is now a hard error naming the op, instead of a silent
  empty result.
- Inlining returns a call-free program untouched instead of rebuilding its node tree,
  so running this on every reference execution costs nothing when there is no call.

### Changed: typedef annotation is three ops instead of one monolith (`vyre-libs`)

`vyre-libs::parsing::c11_annotate_typedef_names` carried every phase inline: 613
statement nodes against a 200 budget, control-flow depth 20 against 6, and 37 loops
against 8. The composition-discipline gate has no exemption list, so the op was red,
and it could not be split because a callee could not take a buffer.

- The three per-row phases are now registered ops of their own, each answering one
  question about one row: `c11_typedef_scope_open_for_row`,
  `c11_typedef_visible_name_for_row{,_packed_haystack}`, and
  `c11_typedef_decl_kind_for_row{,_packed_haystack}`. They take the node table and
  haystack as buffer references and the row index as a scalar.
- The calls inline before lowering, so the emitted kernel is unchanged and the C
  parser's oracle parity is unaffected.
- `emit_typedef_visibility_scan` and `emit_current_declaration_annotation`, the two
  wrappers the annotator no longer uses, are removed.
- `vyre_libs::dialect_init::ensure_ops_resolvable` installs the driver registry as the
  process op lookup. A builder that emits a call now calls it, so the program it returns
  still inlines and validates for a caller who never touches `vyre-driver` directly.

### Added: device convergence flags for persistent BFS (`vyre-primitives`, `vyre-self-substrate`)

A persistent-BFS closure that exhausts its `max_iters` budget while still growing
produces an under-approximated frontier. Until now the device path returned that
partial frontier with no way to tell it apart from a real fixpoint, so a caller
silently reasoned over a truncated reachability set.

- Every persistent-BFS program now writes a `converged` output: one u32 word for
  the single-query programs, a per-query u32 array for the batch programs. It is
  `1` when a step added nothing before the budget was exhausted, and `0` when the
  loop ran all `max_iters` steps while still growing, or when `max_iters == 0`.
  `BINDING_CONVERGED` names the binding.
- `validate_persistent_bfs_converged_flag` rejects any other value, and
  `cpu_ref::PersistentBfsConvergence` plus `try_cpu_ref_converged` give the CPU
  reference the same signal so device and reference results are comparable
  flag-for-flag.
- `vyre-libs` borrow checking now uses it: `enforce_borrow_closures_converged`
  fails the dispatch with a `Fix:` message when any forward loan-issue or
  backward loan-use closure did not converge, because borrow-checking a truncated
  loan reachability set silently drops conflicts.
- The optimizer's dispatched DCE uses it too. `build_dce_bfs_program` declares the
  `converged` word its module doc already promised was part of the layout, sets it
  on the early-exit fixpoint branch, and leaves it zero when the loop burns its
  whole budget while still growing. `gpu_dce` now reads it and fails closed:
  liveness is a reachability closure, so DCE over a truncated one deletes live
  code. The failure is a miscompile, not a missed optimization, which is why this
  path refuses rather than degrades.

### Added: per-iteration frontier density telemetry (`vyre-primitives`)

- `persistent_bfs_with_density`, `persistent_bfs_batch_with_density`, and
  `try_persistent_bfs_batch_with_density` build programs that declare one extra
  u32 output, `density_active` (`BINDING_DENSITY_ACTIVE`,
  `DENSITY_ACTIVE_BUFFER`), holding the frontier popcount after each traversal
  step. The batch layout is `q * max_iters + i`. A host reconstructs every
  per-iteration density aggregate from this array plus the seed popcount instead
  of a per-step device round-trip.
- The density write is a recompute-and-store, not an accumulating atomic, so it
  lands the same value when the grid-sync split re-executes a segment to a
  fixpoint. An atomic would double-count there.
- The base `persistent_bfs` and `try_persistent_bfs_batch` programs are
  byte-for-byte unchanged, so callers that do not want telemetry pay nothing.
- `try_cpu_ref_density` is the CPU reference counterpart. New device-parity
  suites cover both the converged flag and the density array.

### Added: closure-driven grid-sync splitting (`vyre-driver`)

- `dispatch_with_grid_sync_split_via_into` and its allocating wrapper
  `dispatch_with_grid_sync_split_via` take an opaque single-launch dispatch
  closure instead of a `&dyn VyreBackend`. A host-loop fixpoint solver can move
  its convergence loop onto the device without holding a backend handle, plugging
  in the CPU reference, CUDA, or wgpu as a closure.
- The split, input-refresh, and adaptive-convergence logic moved into a shared
  `dispatch_grid_sync_split_generic`, so the backend entry and the closure entry
  run the same code and converge to identical output. Neither path has its own
  copy of the loop.

### Added: `DispatchConfig::dispatch_grid` (`vyre-driver`, `vyre-reference`)

- The CPU reference interpreter inferred its coverage from buffer shapes,
  distributing the dispatch only across workgroup axes larger than one. A program
  fanning a `[256, 1, 1]` workgroup across `grid.y`, which is how batched
  persistent BFS runs one query per block, collapsed to `grid.y == 1` and
  computed only the first query with no diagnostic.
- `dispatch_grid: Option<[u32; 3]>` states the real per-axis workgroup grid. When
  set it overrides shape inference entirely, so the interpreter covers every
  workgroup the GPU would. `None` keeps the previous inference. It takes
  precedence over `dispatch_elements`, which is a 1-D floor.

### Changed: selective fused positioned evidence (`vyre-libs`)

- Add `GpuLiteralSet::prepare_resident_fused_scan_positioned_from`, which keeps
  per-region presence complete for every literal while emitting match triples
  only for an appended positioned-evidence segment. Dense admission-only rows
  no longer consume atomic counter capacity or readback bandwidth.
- The bounded-range suffix3 prefilter gains the same shape:
  `classic_ac_bounded_ranges_suffix3_presence_and_positions_by_region_program_filtered_ext`
  and `try_build_ac_bounded_ranges_suffix3_presence_and_positions_by_region_program_filtered`
  keep presence complete for every pattern while filtering only the atomic triple
  append to IDs at or above a supplied boundary.

### Fixed: paged corpus windows read exactly their planned byte length (`vyre-libs`)

- `fill_window_from_paths` appended each file in the window whole. A file that
  grew between planning and reading overran its window, so the haystack no longer
  matched the offsets the plan had computed and reported matches at wrong
  positions. It now reads at most the remaining window budget per file and errors
  with the path and both byte counts if a file overruns or the window comes up
  short.

### Removed: product-specific names in the dataflow-import API (`vyre-lower`, `vyre-libs`)

vyre's dataflow-import surface named a specific downstream consumer. The API is
generic: it imports alias and reaching-definition facts from any external
dataflow engine. The names now say that, which also stops the published API from
coupling vyre to one sibling product.

| Before | After |
| --- | --- |
| `analyses::weir_alias` | `analyses::alias_import` |
| `analyses::weir_reaching_def` | `analyses::reaching_def_import` |
| `dead_store_with_weir_alias_facts` | `dead_store_with_external_alias_facts` |
| `licm_with_weir_alias_facts` | `licm_with_external_alias_facts` |
| `load_forwarding_with_weir_alias_facts` | `load_forwarding_with_external_alias_facts` |
| `loop_fission_with_weir_alias_facts` | `loop_fission_with_external_alias_facts` |
| `loop_fusion_with_weir_alias_facts` | `loop_fusion_with_external_alias_facts` |
| `security::weir_ifds` | `security::external_ifds` |
| `route_security_taint_through_weir_ifds` | `route_security_taint_through_external_ifds` |
| `security_witness_path_from_weir` | `security_witness_path_from_external_path` |
| `WeirIfdsSecurity{Buffers,Dispatch,RouteError}` | `ExternalIfdsSecurity{Buffers,Dispatch,RouteError}` |
| `WEIR_IFDS_SECURITY_BACKEND_ID` | `EXTERNAL_IFDS_SECURITY_BACKEND_ID` |
| `cfg(feature = "weir_ifds_external_engine")` guard | `cfg(feature = "external_ifds_engine")` guard |

- The shared fact schema's producer id changes from `weir` to
  `external-dataflow`. A serialized fact header carries the producer id, so a
  consumer matching on the old string must update. The schema version itself is
  unchanged.
- `rules/security_predicates.toml` renames the `weir_mapping` key to
  `external_mapping` across all ten predicates. This is a Tier-B data file, so a
  user who copied it to extend the catalog updates the key name.
- The rename is mechanical. There are no behavior or signature changes beyond the
  names in this table.


### Fixed: unrolled loops reused result ids and the CUDA backend miscompiled the address (`vyre-lower`)

A store indexed by the invocation id came out of the CUDA path as a store at a
constant offset, so every lane wrote the same element and element 0 was never
written. `sinkhorn_iterate` and both of its catalog wrappers returned zeros on
CUDA while the reference and wgpu agreed.

- `loop_unroll` reseeded its free-id counter from the subtree it was currently
  visiting, at every level of the recursion. Ids inside a child body are small, so
  unrolling a short loop nested under a long one minted ids that were fresh within
  that subtree and already owned by a sibling or an ancestor. It now threads one
  descriptor-wide counter through the whole recursion.
- The PTX emitter keys its literal map on the raw result id across the whole
  kernel, so a store index produced by `GlobalInvocationId` in one body resolved to
  a sibling body's `Literal(1)` and the address folded to a constant. The
  descriptor was well formed by every check that existed; only the ids were
  ambiguous.
- `verify` gained invariant 6, `ResultIdReusedAcrossBodies`. It previously
  collected produced ids fresh per body, so it caught only duplicates sitting side
  by side in one op list and cross-body reuse verified clean.

### Fixed: a split op could not be lowered (`vyre-foundation`, `vyre-lower`)

The canonical pre-emit pipeline inlines through `inline_calls`, whose default
resolver returned `None` for every op id, so any program containing an `Expr::Call`
failed with `InlineUnknownOp` before a backend saw it. Only `vyre-aot` passed a real
resolver. The composition-discipline gate meanwhile instructs an over-budget op to
split into sub-ops connected via `Expr::Call`, so the prescribed remedy produced code
the pipeline refused to compile. The default resolver now asks the installed dialect
lookup, the same dependency-inversion boundary the reference interpreter uses. An op
resolves when it is registered and carries a composition body; intrinsics and
unregistered ids still do not.

This unblocks half of the composition-discipline split. A callee still takes only
scalar arguments, so a phase that indexes a table cannot yet be factored out.

### Changed: eigenvectors come back with a canonical sign (`vyre-primitives`)

An eigenvector is only defined up to sign, so the Jacobi rotation accumulation was free
to return `v` or `-v` and both were correct. That made every consumer of
`jacobi_eigen_body` unpinnable: a backend that rounded one rotation differently landed
on the opposite sign, and anything dividing by the vector flipped with it.

- `jacobi_eigen_body` now canonicalizes each eigenvector column so its first component
  larger than `EIGENVECTOR_SIGN_EPSILON` in magnitude is positive. `symmetric_eigen_jacobi`
  and `tensor_train_decompose` inherit it.
- If you consumed raw eigenvector columns and applied your own sign convention, remove it.
  If you compared columns against stored values, half of them may now differ by a sign.
  Eigenvalues, the eigen-decomposition itself, and anything invariant to sign (a
  reconstruction, a projection) are unchanged.

### Fixed: registered ops that no backend actually executed (`vyre-primitives`)

Two ops were registered with no fixtures, so they counted as covered while nothing ever
checked a value.

- `tensor_train_decompose` shipped without an oracle on the grounds that a truncated SVD
  is basis-dependent. Two of the three ambiguities are removed rather than tolerated: sign
  by the canonicalization above, and the degenerate eigen-subspace by moving the fixture
  from a wide 2x4 unfolding to a tall 4x2 one. A wide unfolding makes the Gram matrix
  rank-deficient, and a degenerate null space has no single correct eigenbasis. The oracle
  is derived analytically from the closed-form decomposition.
- `multi_block_prefix_scan_inclusive_sum` had `test_inputs: None`. It now runs a
  64-element inclusive scan against the closed-form triangular-number expectation.

### Fixed: the cross-backend parity matrix never resolved calls (`vyre-conform-runner`)

The parity harness did not install the process-wide dialect lookup, so validation rejected
any op carrying an `Expr::Call` with V016 before it reached a backend. This was not
specific to the coverage bundle that surfaced it; it applied to every op with a call. The
harness now installs the registry first, and the expr-variant bundle calls a registered
callee instead of a placeholder id.

### Fixed: `docs/INDEX.md` listed documents that are not published

The index gate enumerated `docs/` from the filesystem, so every gitignored operator
document failed it, and the index in turn pointed at 22 documents that `.gitignore`
excludes and a published crate therefore does not contain. The gate now enumerates tracked
files, and the private rows are gone.

### Fixed: two guards disagreed about release runbooks

`vyre-lints` and `scripts/check_platform_consumer_docs.sh` both enforce the
downstream-consumer naming boundary and each carried its own exemption list, so
`docs/RELEASE.md` was exempt in one and scanned by the other. The list now lives in
`vyre-lints/rules/release_coordination_docs.txt` and both read it.

### Fixed: the workspace now builds from a clean clone (`vyre-conform-enforce`)

- 31 governance test suites embedded `docs/optimization/ALL_AXES_ACCELERATION_PLAN.md`
  with `include_str!`. That file is private operator state that `.gitignore` excludes
  from the public repository, so `cargo test` failed to compile on every fresh clone
  with `couldn't read ...: No such file or directory`. Nothing caught it because the
  file is always present in a maintainer's checkout.
- The removed assertions only checked that the private document contained literal row-ID
  strings such as `VX-1081`. Every requirement they were meant to prove is still asserted
  directly against the committed `docs/optimization/*.toml` artifacts, which carry the
  same row ranges, so coverage is unchanged.
- `tests/clean_checkout_build_governance.rs` now fails if any Rust source embeds a path
  that git does not track, if the private acceleration plan is embedded again, or if a
  tracked file matches a `.gitignore` rule. The scanner skips `include_str!` occurrences
  inside string literals and comments, so the lint that greps for the macro name and the
  raw-string test fixtures that contain sample source are not reported as violations.

### Fixed: release gate resolved three identifiers incorrectly (`xtask`)

- The publish train named the dataflow product's package `weir`. The publishable package
  is `weirflow`; `weir` is only its library target name, and the bare `weir` name on
  crates.io belongs to an unrelated crate. `package-readiness` reported a blocker that no
  version bump could clear. `release_train::weir_package_name` is now the single owner and
  the three sites that hardcoded the name read from it.
- Every gate resolved the security compiler consumer at `libs/surge/surgec`, but it lives
  at `surge/surgec`. Eight sites carried the wrong prefix, so gates reported the tree as
  absent: `distributed-parser-coherence` alone raised 51 blockers claiming `src/lib.rs`
  does not exist for a crate with 229 test files, 5 benches, and 2 fuzz targets on disk.
  `release_train::compiler_consumer_relative_path` is now the single owner.
- `vyre-grammar-gen` had fallen out of the publish train after 0.6.2 and went stale on
  crates.io while every sibling advanced to 0.6.5, which is why in-workspace consumers had
  to pin it path-only. It is back in the train and publishes first, having no internal
  dependencies.
- `package-readiness` now reports zero blockers.

### Fixed: release runbooks contained unrunnable instructions (`docs`)

- A rename sweep had replaced the product name with a two-word phrase inside identifiers,
  producing `git tag vyre-0.4.1-dataflow consumer-0.0.1`, the xtask subcommand
  `vyre-dataflow consumer-release-gate`, the path `release/vyre-dataflow consumer-evidence.toml`,
  and the sentence `The The dataflow consumer repository`. Tags, subcommand names, and
  paths are literal strings an operator types, so they are restored across `RELEASE.md`,
  `RELEASE_CHECKLIST.md`, `RELEASE_ENGINEERING.md`, `PUBLISH_GATE.md`, and the v0.4.1 and
  v0.4.2 release notes.
- The consumer-coupling lint gained a narrow exemption for release runbooks, which name the
  products in the combined release train on purpose. Architecture docs, guides, and all
  Rust source stay under the guard. Three tests pin the exemption, prove it does not leak
  to neighbouring documents, and prove it never covers Rust source.
- The same sweep had neutralized the negative fixture in the coupling lint's own test, so
  the fixture no longer contained the string the lint must flag and the test failed while
  the lint was correct. Restored, with a comment recording why the fixture keeps the name.

### Fixed: stale dependency pins (`vyre-spec`, `vyre-primitives`, `vyre-intrinsics`, `vyre-driver-wgpu`)

- Eight internal dev-dependencies pinned `version = "0.6.1"` alongside their path, three
  releases behind the workspace. They are now path-only, matching the documented pattern:
  cargo strips path-only dev-dependencies at publish, so they cannot demand a stale
  crates.io version or block the publish train again.
- `examples/libs-template` pinned `vyre`, `vyre-foundation`, `vyre-spec`, and
  `vyre-reference` at `0.4.2` while pinning `vyre-libs` at `0.6.5`. The template is what a
  consumer copies, so it resolved a two-minor-old API. All pins now track the release.

### Changed: third-party dependency pins refreshed (`Cargo.toml`)

- Every third-party dependency is exact-pinned with `=`, so `cargo update` cannot move
  them and freshness is a deliberate edit. Seventeen pins advance to the current patch or
  minor release: `serde` 1.0.229, `thiserror` 2.0.19, `rand` 0.10.2, `tokio` 1.53.1,
  `bytemuck` 1.25.2, `proc-macro2` 1.0.107, `toml` 1.1.3, `faer` 0.24.4, `memchr` 2.8.3,
  `regex-syntax` 0.8.11, `rustc-hash` 2.1.3, `clap` 4.6.4, `regex` 1.13.1,
  `regex-automata` 0.4.16, `quote` 1.0.47, `crossbeam-channel` 0.5.16, `openssl` 0.10.81.
- `wgpu`/`naga`, `syn`, and `wide` stay on their current majors. Each of those bumps
  changes APIs vyre calls directly, so they are code changes rather than pin edits and do
  not ride a release-engineering release.

### Changed: repository identity moved to `santhreal` (`docs`, `.github`, crate metadata)

- `repository` and `homepage` metadata, `CODEOWNERS`, issue-template links, `CITATION.cff`,
  and the governance evidence now name `santhreal/vyre`. The workspace `homepage` points at
  `https://santh.dev`.
- README carries crates.io, docs.rs, and license badges.

### Added: adversarial coverage for loop peeling and induction rebinding (`vyre-lower`)

- Second-pass edge cases for loop peeling, induction-variable rebinding helpers, and
  shared-memory uniformity, including the control-flow shapes where a rebind must not fire.

### Fixed: private operator state is no longer stageable (`.gitignore`)

- Planning, status, audit, and agent-handoff documents were being staged out of
  subdirectories that the root-only ignore patterns did not cover, including a 625KB
  backlog and a 909KB operator plan. The patterns now apply at every depth.

### Changed: composition provenance and deduplication audits use canonical ownership

Cat-A wrappers now use the foundation-owned `tag_program` operation. The helper
preserves the primitive program metadata, keeps primitive generator ids as
children, and records the Cat-A operation as their parent. INT4 quantization
wrappers and predicate builders use this single path.

The generated `vyre-libs::catalog::*::consumer_a` and `consumer_b` registrations
have been removed. Primitive coverage now counts real composition callers only.
The operation matrix contains 371 tracked rows: 206 library rows, 149 primitive
rows, 9 intrinsic rows, 5 runtime rows, and 2 foundation IR rows.

The similarity audits now classify operations by canonical implementation
family. Source similarity parses Rust functions and methods with `syn`,
normalizes local bindings, and retains semantic identifiers such as called
operations, types, and constants.

### Fixed: the PTX cache key recomputed the program digest on every dispatch (`vyre-driver-cuda`)

`ptx_for_program_cached_with_key` derived its cache key from
`lower_subgroup_reductions(program.clone(), caps)`. The normalized program
digest that feeds that key is memoized on the program VALUE, and that value was
created and dropped inside a single dispatch, so the memo's only writer was a
temporary and the memo could never be read. A memo whose only writer is a
temporary is a memo that cannot ever be read.

Neither piece was wrong on its own, which is why reading either one could not
find this. `Program::clone` forwards all six memos correctly, and the digest
itself is a sound pure function of the program. The defect lived in the LIFETIME
of the value the key was derived from: a caller's program stayed permanently
cold because nothing ever computed its digest, so every dispatch cloned a cold
program, computed the digest on the clone, and dropped it.

The key is now derived from the caller's own program when the subgroup lowering
pass is a no-op, which is the ordinary case. The pass is already fully
copy-on-write internally: of its three returns, two hand the input straight back
and only the third rebuilds the entry. Pointer equality on the shared `Arc`
fields is therefore an O(1) witness that nothing was rewritten, and in that case
the two programs are the same value differing only in which memos are warm, so
the key receives byte-identical input. A program the pass does rewrite is still
keyed on its lowered form, because keying it on the unlowered digest would file
lowered PTX under the unlowered program's identity and serve a later dispatch a
kernel containing subgroup reductions it never requested.

Measured on an RTX 5090 with the `exatok` encode profile, 45 warm encodes per
corpus shape over 19 distinct program shapes. The digest walk cost 79.0 ns per
IR node (R-squared 0.907) and was 91.8 percent of the host PTX phase, making it
the largest single host term on the encode path. Digest computations per encode
fell from 6 to 0 on the `cjk` and `code` shapes, 4 to 0 on `prose`, and 3 to 0
on `short_pretokens`. The residual per-node rate in that phase fell from 83.1 to
4.0 ns per node, a factor of 21. Programs reach 12,410 IR nodes and 3.9 MB of
PTX for one dispatch, so the cost of getting the memo lifetime wrong grew with
the workload.

Host allocations per dispatch fell about 16-fold on the `short_pretokens`
fixture, from roughly 1,600 calls to roughly 100, deterministic across five
consecutive runs. That is a counted, load-independent instrument separate from
the phase probe and it corroborates the same change. Token ids are unaffected:
the cache key receives identical bytes, and `exatok` parity, determinism, device
parity and specials exactness gates all pass unchanged.

### Fixed: the parallel DCE fixpoint exited a synchronizing loop unordered (`vyre-self-substrate`)

The device DCE program's iteration body ended with an unconditional barrier and
then an early exit: once a step added no bit, lane 0 recorded convergence and the
invocation returned. That exit sat AFTER the body's last synchronizing node, so
one invocation could take the back edge and write while a sibling had not yet
reached the exit, and the sibling then left the kernel while the rest kept
iterating, freezing the data it owned partway through. Nothing hangs, because a
barrier does not count invocations that already returned, so the cost is
ANSWERS rather than liveness and a single workgroup is enough to hit it.

The shape was always there. It was not a hazard until now because a `Return`
nested inside a loop used to be emitted as nothing by `vyre-emit-ptx`, and the
program carried an explicit correctness argument resting on that: on device the
loop ran its full iteration budget and a `converged` gate, not the `Return`, was
what made the early exit real. Lowering a nested `Return` to a real branch turned
that documented no-op into a live exit and made the argument false at the moment
it landed, which is what surfaced the program to the V055 back-edge validator.

The body now ends with an unconditional barrier AFTER the exit branch. That is
safe here for a specific reason worth keeping: the exit condition reads a value
the preceding barrier settles, so it is workgroup-uniform, every lane sees the
same value, and the trailing barrier is reached by all lanes or by none, never by
a subset. It stays at body level, since a barrier inside the convergence gate
would desynchronize a workgroup whose lanes are allowed to read that flag stale.
Cost is one extra barrier per non-converged iteration against a body that already
costs two (INFERRED from the emitted node sequence, not timed).

The stale emitter claim in that program's comment is corrected in place, because
the file records that this reasoning had already misled three separate attempts,
and a wrong mechanical note in the one comment written to prevent a fourth is how
the fourth happens. V055 was not weakened. It still refuses any exit after a
loop's last barrier, including provably uniform ones like this; teaching it to
prove uniformity is real analysis and is deferred, with this program as the
motivating example.

Two suites now hold this shut, both host-only and GPU-free. In
`vyre-self-substrate`, `dce_program_back_edge_contract` asserts the built program
VALIDATES, which is the property that survives any future change to how a
`Return` lowers, and four of its tests mutate the real program to require the
refusal back: trailing barrier removed, barrier moved inside the convergence gate
(the plausible wrong fix), exit moved past the barrier with the barrier count
unchanged, and an unconditional exit after the barrier. That last one recorded a
correction: the rule refuses a provably UNIFORM exit too, so its reach is any
exit textually after the last barrier, not only a lane-dependent one (OBSERVED,
from the test failing against the opposite expectation).

In `vyre-primitives`, `loop_back_edge_audit` asks the question directly instead
of waiting for a downstream symptom, since all four instances of this shape found
so far were found because something else went red. It builds every shipped
program whose file contains both a loop and a barrier and validates it on the
host: thirteen programs at five iteration budgets each, all clean, so there is no
fifth instance among them (a measured absence over that set, not a proof about
the crate). Exactly four of the thirteen put a barrier inside a loop body and are
governed by the rule at all. Two of those four end in an unconditional barrier
and are exit-proof, meaning an exit added later stays ordered; the two density
variants are merely exit-free, legal because they hold no exit. That gap is
recorded rather than closed: an exit added there is refused loudly at validation,
and closing it would cost a real barrier per iteration in a program with no
defect to justify it.

### Fixed: cold-start launch width stranded a third of every SM (`vyre-driver`)

Blocks per compute unit is an integral division: a unit hosts whole workgroups
only. On an RTX 5090 the per-SM budget is 1536 threads, so a 1024-wide group
hosts exactly ONE block and 512 of every SM's 1536 thread slots are unreachable
for the life of the launch. The cold-start estimator had no occupancy term. It
scored candidates on tail waste and per-group overhead alone, which made 1024
the outright winner for any element count that is a multiple of 1024, precisely
where its idle-lane penalty vanishes. Every unmeasured tunable 1-D dispatch on
this class of device therefore launched at two thirds occupancy by arithmetic,
before any kernel ran.

The cooperative consequence is larger than the occupancy one. A grid-sync
program fits a single cooperative launch only while its grid stays inside the
device's resident-thread ceiling, and that ceiling is a function of the width
the tuner RESOLVES, not the width the program declares: 170 SMs x 1024 resident
threads is 174,080 lanes at width 1024, against 170 x 1536 = 261,120 at any
width dividing 1536 evenly. The bad width cut the cooperative ceiling by a third
and pushed programs that should have run as one launch onto the host split
route, turning one dispatch into many on a workload whose measured cost is
already dominated by host-side launch preparation.

Cold start now prefers the candidate maximizing resident threads per compute
unit, breaking ties toward the wider group. On this device that selects 512: 3
blocks per SM, 1536 resident threads, zero stranded slots, and the full 261,120
lane ceiling. `VyreGridSyncAot` confirmed the selection against the hardware
with `cuOccupancyMaxActiveBlocksPerMultiprocessor` on a real emitted kernel
rather than by arithmetic.

This is a residency rule and not a rule against 1024. Where the per-SM budget
divides evenly by 1024, such as a 2048-thread SM, 1024 strands nothing, ties on
residency, and the latency estimate still selects it.

Three protections are pinned by test. Callers that pin geometry are unaffected:
`workgroup_override` and `grid_override` keep their existing precedence, which
is why `exatok`, which sets both, never saw this. Measured feedback still
outranks the preference, so a real timing can select a width cold start would
never propose. And a backend that reports no per-SM budget stays byte-identical
to previous behavior: `LaunchGeometryLimits::max_threads_per_sm` of `0` means
unreported, the residency methods answer `None` rather than a guessed `0`, and
the candidate filter is inert, so wgpu selects exactly what it selected before.

The residency division now has one definition in the workspace,
`vyre_driver::validation::blocks_per_compute_unit`, which CUDA's cooperative
preflight in `vyre-driver-cuda/src/occupancy.rs` also routes through. Two copies
of this arithmetic had already drifted apart once. The shared function models
the thread ceiling only and documents that a caller answering "does this
DECLARED width fit" must additionally clamp by the device-reported block cap
(`CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR`, 24 on this device): at
width 32 the thread arithmetic predicts 48 blocks and 1536 resident threads
where the hardware delivers 24 and 768. Selecting the widest survivor never
reaches that regime, and the 512 this now picks sits at 3 blocks per SM, well
clear of the cap.

BREAKING: `vyre_driver::validation::LaunchGeometryLimits` gains a public
`max_threads_per_sm: u32` field. The struct is not `#[non_exhaustive]`, so every
struct-literal construction site must add it. Use `0` for a backend that does
not report a per-SM thread budget, which preserves prior behavior exactly.

### Fixed: cooperative preflight admitted grids the driver refuses (`vyre-driver-cuda`)

Two independent per-SM ceilings govern cooperative residency and
`cooperative_thread_residency_block_limit` respected only one. It derived
admissible blocks from the per-SM THREAD budget, `max_threads_per_sm /
workgroup`, while hardware separately caps BLOCKS per SM. At narrow widths the
block cap binds first: on an RTX 5090 reporting 24 blocks per SM, width 32 was
admitted at 1536/32 = 48 blocks per SM and 8160 blocks device-wide against a real
24 and 4080. The preflight answered "fits" and `cuLaunchCooperativeKernel` then
refused the launch, which is exactly the predicate-versus-driver disagreement
that giving the residency division one definition was meant to eliminate. It is
reachable rather than theoretical, because grid-sync programs are exempt from
launch-width tuning, so a declared 32 survives to launch.

The limit now clamps by a probed `CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR`.
Widths of 64 and up are unchanged on this device: 1536/64 = 24 is exactly the
cap, which makes 64 the narrowest width still reaching full occupancy and leaves
it no margin. A driver that does not report the attribute stores `0`, which reads
as unreported and applies no clamp, so behavior there is byte-identical to
before. A negative value is treated as unreported too, rather than cast into a
cap near four billion that would clamp nothing.

Measured, not calculated: per-width occupancy came from
`cuOccupancyMaxActiveBlocksPerMultiprocessor` on a real emitted vyre kernel at
each candidate width (10 registers per thread, zero static shared memory, element
count a multiple of every width so tail waste could not skew it). The table and
its method are recorded in the `vyre-driver-cuda::occupancy` module documentation
so the next reader finds measurements instead of re-deriving arithmetic: width 32
gives 24 blocks per SM and 768 resident threads, widths 64 through 512 all reach
1536, and width 1024 gives 1 block and 1024.

### Changed: the cooperative grid-barrier release order is now unrepresentable to get wrong (`vyre-driver-cuda`)

Four launch sites hand-wrote the same sequence: run the launches in a closure,
release the lease, then propagate the launch error. The order is load-bearing and
the reason is not local. `GridBarrierGuard` frees the gate on drop, including on
unwind, so the gate can never be permanently stranded; the hazard is the
opposite. Releasing through `Drop` instead of through the release path frees the
gate while SKIPPING both the stream synchronize and the arrival audit. The next
sequence then acquires the gate and zeroes `_vyre_grid_barrier` underneath a grid
that may still be running, whose remaining barriers wait for a release target
that can no longer be reached. That is a hang rather than an error, it reproduces
only under cooperative launch, and the edits that cause it (hoisting `launched?`
above the release, or deleting the closure so `?` returns directly) both compile
and keep every non-cooperative test green.

`GridBarrierLease::launch_then_release` now owns that order. It consumes the
lease, so a call site cannot skip the release, and it captures the launch
closure's error rather than letting it escape, so no error path can bypass the
synchronize. `release_after_launch` is private to the module, so an open-coded
release will not compile from another module. The release delegates to a small
ordering function whose synchronize and audit steps are closures, which lets unit
tests assert the gate is still HELD while the synchronize runs and freed only
after the release returns, pinning the ordering rather than the end state.

### Changed: the synthetic device profile no longer claims to be real hardware (`vyre-driver-cuda`)

`blackwell_sm120_caps()` named "NVIDIA GeForce RTX 5090" and its module
documentation called it a source of truth, while five of its fields disagreed
with that machine and the three substantial ones all OVERSTATED it: 2048 threads
per SM against a measured 1536, 256 KiB of shared memory per SM against 100 KiB,
and 128 KiB per block against 48 KiB, the last being unreachable even with the
99 KiB opt-in maximum. Every occupancy figure derived from it was therefore
optimistic, and a number that reads as measured and is not is the defect class
that also produced a test pinning a stale cooperative ceiling.

The values are unchanged, deliberately. The roughly twenty tests that use it are
correct arithmetic against a fixed envelope, and an estimator test needs a fixed
envelope rather than a true one: a test pinning `2048 / 256 = 8` checks division,
not any GPU, and rewriting those to chase real hardware would churn them again on
the next device. So the fix is the name and the claims. It is now
`synthetic_sm120_envelope`, its documentation states plainly that it is a test
fixture whose values are not this machine's, names the specific divergences, and
forbids deriving a hardware decision from it. Verified that no shipping path
consumes it: every caller in the workspace is inside `#[cfg(test)]` or under
`tests/`. It also gains a blocks-per-SM value, chosen so the cooperative clamp
above is exercised without a CUDA context.

BREAKING: `blackwell_sm120_caps`, `blackwell_sm120_caps_default` and
`BLACKWELL_SM120_DEFAULT_MEMORY_BYTES` are renamed to `synthetic_sm120_envelope`,
`synthetic_sm120_envelope_default` and `SYNTHETIC_SM120_DEFAULT_MEMORY_BYTES`.
`CudaDeviceCaps` gains a public `max_blocks_per_sm: i32` field; the struct is not
`#[non_exhaustive]`, so struct-literal construction sites must add it, and `0`
means unreported.

### Fixed: `Node::Return` was silently discarded instead of emitted or refused (`vyre-emit-ptx`)

The PTX emitter handled `Node::Return` with a comment and no instruction.
`finish_with_return` writes the single trailing `$L_exit:` / `ret;` at the end of
the kernel, so a `Return` nested in an `If` or a loop emitted NOTHING and fell
through, and the program kept running past its own exit. The branch target
already existed and `Trap` already branched to it, so this was a missing match
arm rather than a missing mechanism.

A dropped control-flow node is a correctness defect, not a performance one. It
happened to be survivable wherever the work after the exit was idempotent, which
is why every correctness test in the tree passed while it was broken: the answers
were right and only the work was wrong. A consumer whose loop body is NOT
idempotent after its exit condition got a wrong answer from an exit the emitter
quietly deleted, with nothing reporting it.

A `Return` now lowers to `bra $L_exit`, and the emitter REFUSES the cases it
cannot honor instead of dropping them. This half matters more than the branch. A
`Return` taken by only SOME invocations lets those invocations leave while the
rest continue, and the ones that left can never arrive at a later `bar.sync` or
cooperative grid barrier, so the ones that stayed block forever. Trading an
invisible slowdown for an invisible hang would not have been a fix, so the
emitter proves the exit is uniform across the grid or refuses at compile time,
naming the reason and the fix.

Uniformity is proven, never assumed: values built from literals, buffer lengths,
the subgroup size, and loads from global or constant memory at a uniform index
qualify. Anything derived from an invocation id, a workgroup id (uniform within a
CTA but not across the grid, and a whole CTA leaving early strands the others),
a subgroup op, shared memory, or an atomic's returned value does not. A loop whose
bounds are not uniform also counts as divergent, because invocations then leave it
on different iterations even with no conditional present. Unproven is treated as
varying, so the failure direction is a build error rather than a hang.

`vyre-emit-ptx/tests/nested_return_branch.rs` pins both halves, including a
control proving the asserted branch comes from the `Return` and not from the
entry's predicated element-count guard, which also branches to `$L_exit`.

### Added: `persistent_fixpoint_grid`, a grid-correct convergence loop (`vyre-primitives`)

`fixpoint::persistent_fixpoint` drives convergence from an in-kernel
`Node::Loop`. Lane 0 clears the single shared `changed` word once per
iteration, ordered only by a `MemoryOrdering::SeqCst` barrier, which is
workgroup scope, while every lane in every workgroup sets that same word with
`atomic_or`, and each workgroup gates its own `Node::Return` on it.

Above one workgroup nothing orders one group's clear against another group's
set. The severe face is a lost set: the clear erases a flag another group had
already raised, that group reads 0, returns early, and leaves its slice of the
state unconverged with no error. For a caller whose convergence means "no work
remains" that is a wrong answer, not merely wasted work. The mild face is a
false verdict: a downstream GPU tokenizer measured correct state and a correct
two-pass convergence with `changed` still reporting non-zero against a
fifteen-pass budget, which is indistinguishable from a real cap-out.

At ONE workgroup the same code is ordered and does not lose a set: the
sequence is clear, barrier, sets, barrier, barrier, read, so the conflicting
accesses to `changed[0]` are never concurrent and a CTA-scope fence is
sufficient. An intermediate revision of this entry claimed the clear made the
builder unsound at one workgroup as well, and that was wrong; it is corrected
here because a consumer selects this builder for its single-group path and the
claim would have implied a live exactness defect there.

`fixpoint::persistent_fixpoint_grid` takes the same positional parameters,
buffer names, bindings, and workgroup size, so selecting between the two is a
`match` on group count. It replaces the in-kernel loop with `max_iterations`
top-level waves separated by `MemoryOrdering::GridSync` barriers, the shape
`persistent_bfs_grid_sync_parallel` already uses for the same reason. Each wave
is five nodes: the caller's transfer body, a grid fence, the per-word compare
and ping-pong, a grid fence, and `if changed[i] == 0 { Return }`.

The early exit survives the grid barrier protocol because it is collective.
`changed` carries one word per iteration and is NEVER cleared, so a set cannot
be lost, and the word is read only after a grid fence, so every group computes
the same verdict and the whole grid leaves together or none of it does. Do not
collapse the per-iteration word back to one cleared word; that reintroduces the
race and turns the return into a stranding hazard in a single edit.

One ABI difference: `changed` is `max_iterations` words wide instead of 1 and
the caller must supply it zero-filled. In exchange the array decodes the pass
count exactly. `changed[i] == 1` iff wave `i` changed the state and the kernel
leaves at the first zero, so iterations entered is the index of the first zero
plus one, or `max_iterations` when no word is zero.

`persistent_fixpoint_grid` also carries a cooperative-residency ceiling that
`persistent_fixpoint` does not, because a `GridSync` lowers to a native
cooperative launch that needs every block co-resident. A dispatch path that
cannot provide it must refuse, naming the block count and the device limit;
`VyreBackend::cooperative_grid_sync_fits` is the preflight and
`VyreBackend::allows_host_grid_sync_split` says whether the kernel-split
fallback is permitted at all. A silent reroute there is a correctness failure.

`persistent_fixpoint` is unchanged: its emitted IR, signature, and first-zero-read
pass semantics all stay as they were, because downstream pass-count bounds are
denominated in them. Its module doc claimed convergence required `changed` to
read zero on two consecutive iterations, which the code never did; that text was
corrected to describe the first-zero-read exit it actually implements. The shared
`[256, 1, 1]` geometry both builders emit is now the exported
`PERSISTENT_FIXPOINT_WORKGROUP_SIZE`, so a caller derives its routing threshold
from the declared geometry instead of a literal.

`vyre-primitives/tests/persistent_fixpoint_grid_contracts.rs` pins all of it,
including a differential test that runs both builders through the reference
interpreter across four transfer bodies and every budget, and a probe that steps
the workgroups back to front: the grid builder returns the same state and the
same verdict in either order, while `persistent_fixpoint` at two workgroups
reports `changed == 1` forward and `changed == 0` reversed for the same input,
which is the race made deterministic.

The same-location property is pinned structurally by
`the_grid_builder_never_writes_changed_with_a_plain_store`, which asserts at
four budgets that no `Node::Store` targets `changed` and that each wave `i`
atomic-ors exactly word `i`, so the zero count means "all atomic" rather than
"nothing written". It has to be a structural assertion on the emitted IR
because the reference interpreter does not model L1 against L2 and cannot
reproduce the hardware race, so a reintroduced clear would keep every
value-level test green. The same test asserts that `persistent_fixpoint` still
shows exactly its one plain clear, which proves the probe detects a plain
store when one is present instead of matching nothing.

The collective exit is honored on PTX, which was NOT true when this primitive
was written. The emitter used to discard a nested `Node::Return`, so every
emitted wave ran regardless of how early the grid converged; that is fixed in
this same release (see the `vyre-emit-ptx` entry above) and a three-wave build
now emits three unpredicated `bra $L_exit` instructions, one per wave. The
`changed` decoding is unaffected either way, because a skipped wave and a wave
that changed nothing both leave their word at 0.

The exit saves LAUNCHES only under a native cooperative launch, which bounds
that guarantee. `GridSync` lowers either to a cooperative grid barrier or to a
kernel split, and under the split each wave is its own launch, so a
`Node::Return` in segment N returns from that launch alone and cannot stop the
host issuing segment N+1 (`vyre-driver/src/grid_sync.rs` dispatches every
segment in order). A run converging at wave 2 of a 16-wave budget still issues
all `2 * max_iterations + 1` segments. The ANSWER is unaffected on that path,
since a converged wave recomputes the same `next`, sets no flag word, and
copies idempotently, so only the saved work disappears and a device-side pass
counter reads the full budget instead of the convergence depth, which looks
like a cap-out and is not one. A downstream caller measured exactly that with
byte-correct state and a correct `[1, 0, 0, ...]` flag buffer. Read convergence
depth from `changed`, which is authoritative on both paths, never from a pass
or launch count. Pinned by
`the_split_path_launches_every_wave_because_return_is_per_segment`, which
asserts the segment count and that the exits are spread across segments rather
than concentrated in one that could short-circuit the host loop.

A budget sweep from 2 to 256 confirmed that the IR, the pre-lowering optimizer,
and the emitted PTX each preserve exactly one exit per wave at every budget, so
no stage drops the exit at any particular wave count.

This primitive also satisfies the emitter's new uniformity requirement by
construction: `changed[i]` is read from global memory at a literal index, which
is grid-uniform, and it is read after a grid fence, so every invocation computes
the same verdict. An exit gated on anything per-invocation is refused at compile
time rather than silently dropped.

Both builders now document a caller requirement that was previously implicit:
the transfer body must write EVERY word `w < words` of `next` on every
iteration. Violating it is a wrong-answer defect that reports success, so
nothing in the run looks wrong. The compare-and-copy step writes
`current[w] = next[w]` for every `w`, not only the words the body touched, so
a word the body never wrote overwrites `current[w]` with a stale `next`; the
buffers then agree everywhere and the loop exits converged on corrupted state.
Pinned by `a_transfer_body_that_skips_words_silently_corrupts_them`, which
asserts the exact bytes both ways: state `[9, 0, 0, 0]` from seed
`[9, 9, 9, 9]`, with `changed` reading `[1, 0, 0, 0]`, a converged verdict.
The docs first claimed this shape would fail to converge instead; the test
falsified that and both doc blocks were corrected to the measured behavior.

### Changed: `persistent_fixpoint` clears its flag atomically (`vyre-primitives`)

That clear was a plain non-atomic `Node::store` to a word every other write
reaches through `atomic_or`. It was correct, because the barriers around it
ordered the clear against the sets, but only for that reason, and the
dependency is invisible at the call site: weaken or move one of those barriers
and the program breaks without anything correctness-shaped being edited. This
primitive already has a failure mode that reports converged while being wrong,
so a write whose safety rests on an unstated ordering assumption is a poor
thing to leave in it.

The clear is now `Expr::atomic_exchange` writing 0 to the same location, so
every write to `changed` in both builders is an atomic. In the emitted PTX the
clear is an `atom.global.exch` instead of a plain `st.global.u32` against an
`atom.global.or.b32` at the same address. Cost is one lane one operation per
iteration. Values and pass counts are unchanged, which the existing
convergence-equivalence and both-builders differential tests confirm, so
callers denominated in this builder's pass counts are unaffected.

This does NOT make the builder multi-workgroup safe. The race is about barrier
SCOPE, not atomicity; above one workgroup use `persistent_fixpoint_grid`. The
property is pinned by `neither_builder_writes_changed_with_a_plain_store`,
which asserts no `Node::Store` targets `changed` in either builder and then
points the same predicate at `next`, which IS written by plain stores, so a
matcher that silently stopped matching could not make the test pass.

### Added: `FRONTIER_TO_QUEUE_WORKGROUP_LANES` (`vyre-primitives`)

`graph::csr_frontier_queue::frontier_to_queue` builds a deliberately
single-workgroup scan, so its declared workgroup size, the stride its lanes walk
`node_count` with, and the lane gate confining the scan to the first workgroup
must agree. They were separate literals, which is the shape that lets a fixed
workgroup declaration drift away from its lane gate and produce silent duplicate
coverage above one workgroup. They are now one exported constant.

### Fixed: a writable buffer declared without a count was mis-sized, and the CPU oracle accepted programs every real target rejected (`vyre-foundation`, `vyre-driver`, `vyre-driver-wgpu`)

A writable `BufferDecl` declared without `.with_count(n)` produced either a
zero-length buffer or a correctly sized buffer with a zeroed tail on the WGPU
backend, while CUDA and the CPU reference both sized it from the declared byte
range. `dynamic_element_count_from_bytes` and `output_binding_layout_parts` are
now exported from `vyre-driver` so the WGPU backend derives the element count by
that same shared rule instead of its own.

The worse half was a certification hole. A buffer the backend allocates itself
(`BufferDecl::output`, any `WriteOnly`, or a `pipeline_live_out` ReadWrite)
declared without `.with_count(n)` has no host bytes to take its size from, so
nothing can size it and the only correct answer is refusal. The CPU reference
instead answered a countless `BufferDecl::output` with `Some([])` while CUDA and
WGPU both refused it, and answered a countless `WriteOnly` with `Some([])` while
CUDA refused, so a program could pass the oracle and then be rejected by every
real target. `BufferDecl::require_static_readback_size` is now the single
refusal, called from both the execution planner and `vyre-reference`, so the
oracle refuses exactly what the backends refuse. It is driven by
`is_backend_allocated_output()` rather than the narrower `is_output()`, which is
what brings `WriteOnly` and `pipeline_live_out` under the same rule.

Reference, CUDA and WGPU now return byte-identical output for a countless
ReadWrite at every length from 1 to 4096, and all three refuse the
un-inferable forms naming `.with_count(n)`.

### Fixed: the CPU reference sentinel could fail open and return an empty result as success (`vyre-foundation`)

`is_cpu_reference_sentinel` identifies an op whose CPU lowering is only the
structured-reference sentinel, and that comparison sits in front of a refusal.
It compared function addresses, and a function's address is not a unique
identity: with more than one codegen unit the compiler may materialize a second
copy or a local thunk, so `fn_addr_eq` compares two different addresses for the
same function and answers `false`. The dispatcher then stopped refusing and
INVOKED the sentinel, which clears the output and returns `Ok(())`, handing the
caller an empty byte vector that looks like a successful CPU reference result.

The identity is now the exported `SENTINEL_CPU_REF` static, which holds a single
pointer resolved once, so a producer that stores it and a consumer that compares
against it compare the same bits by construction.

Two of this release's fixes share one shape, and it is worth naming as a class:
a refusal degrading into an empty output returned as success. The
countless-buffer defect above did it three ways (an empty readback, a zeroed
tail, and a reference oracle answering `Some([])`), and the sentinel did it by
invoking the very lowering it was meant to refuse. This class survives a test
suite because the call reports `Ok` and the output has a plausible shape, so
only an assertion on exact bytes catches it while a shape-only or `is_empty`
check passes happily. A refusal that stops refusing does not throw, it succeeds,
so a refusal path is covered only when the test asserts the refusal AND its text,
never merely that the call returned, with a counted control beside it so a
blanket rejection cannot pass as a fix either.

Asserting a zero length is NOT sufficient, which is the trap in the obvious
reading: a legitimately empty result is indistinguishable from this bug in
isolation. Only a contrast discriminates, the same declaration returning 0 bytes
for an empty seed and exactly 256 for a 256-byte seed, asserted together.

### Added: `FusionWorkgroupGeometryError` (`vyre-foundation`)

A fused launch takes the axis-wise maximum of its arms' workgroup sizes. That is
harmless for an arm whose invocations are independent, and unsafe for an arm that
synchronizes its workgroup or keeps state in workgroup memory: an arm written for
4 invocations guards its body so the other 252 skip it, which makes the workgroup
barrier non-uniform, and its workgroup buffers are sized for the narrow geometry.
The observed symptom was an inclusive prefix scan built for 4 elements, fused
behind a 256-wide elementwise arm, returning the wrong last lane on roughly one
dispatch in ten.

Fusion now refuses that pairing with a typed error naming the arm index, the
geometry it was built for, the geometry the fused program would run it under,
what makes the widening unsafe, and the fix.

## [0.6.5]  -  2026-07-13

### Added: C-frontend visible-type precompute wiring (`vyre-frontend-c`, `vyre-libs`)

- Complete the visible-type precompute path so the precomputed-context typedef annotator no longer drops the ordinary declarator flag for `T x;` where `T` is a typedef-name. `c11_precompute_vast_visible_type` resolves the per-node visible-typedef-name table once (after the decl-context table settles) and the annotate pass reads the bit; `c11_annotate_typedef_names_precomputed_context[_packed_haystack]` now take the table as a ReadOnly buffer at binding 3. The vyre-frontend-c pipeline gained a `vast_pg/visible_type.rs` stage (stage-pipeline cached) that feeds both the fused and unfused annotate dispatches, failing closed if the table is absent on the non-global path.

### Added: IR-parity + regression coverage sweep (`vyre-primitives`, `vyre-self-substrate`, `vyre-foundation`, `vyre-libs`)

- Add and extend reference_eval GPU-IR-vs-cpu_ref parity proptests and regression tests across graph/nfa/math/decode primitives, including signed fixed-point negative-intermediate coverage and sharding-decomposition property gates. Boundary anchors assert real values, not shape.

### Changed: signed fixed-point correctness + ONE-PLACE dedup (`vyre-primitives`, `vyre-libs`)

- Route weighted-Jacobi / AMG divides through `fixed_sdiv_by_positive_expr` so negative 16.16 intermediates no longer corrupt (validated by the new parity tests). Replace inline masked 256-table lookups with the canonical `crate::ir_safe::byte_table_lookup`. Add `dfa_compile_case_insensitive[_with_budget]`. New dev-only `vyre-test-support` crate holding the canonical registry/coverage closure gate.

### Added: interpreter op-counting + roofline operating point (`vyre-reference`, `vyre-bench`)

- Added `vyre_reference::count_ops`: a thread-local scope that counts the arithmetic IR operations (`BinOp`/`UnOp`/`Fma`) the reference interpreter executes during a closure, a backend-agnostic dynamic operation count for roofline / complexity analysis. Because the interpreter runs the same vyre IR with the same data-dependent control flow any backend does, its count for a `(program, inputs)` pair equals the GPU's dynamic IR-op count for those inputs (at vyre-IR granularity, coarser than hardware SASS). Counting is opt-in, a no-op thread-local read outside a `count_ops` scope, so ordinary interpreter use (all in tests) is unaffected (vyre-reference and vyre-primitives suites green). This closes the last non-root piece of the W3-6 roofline: the new `scan_roofline_operating_point_cuda` test measures the literal scan's operational intensity via `count_ops` on the CPU reference backend and its achieved bandwidth on the RTX 5090, placing the operating point on the roofline, intensity 13.77 IR-ops/byte, left of the 29.23 ops/byte ridge (memory-bound side confirmed), achieved compute ≈3.2 T-IR-ops/s under the 52-TOPS ceiling. The full roofline, both ceilings, ridge, both measured axes, and the bound verdict, is now complete and honest without root. A finer SASS-granularity count (`sm__inst_executed`) via Nsight-Compute would only refine the granularity and remains the optional root-gated step.

### Added: property gates for sharding decompositions (`vyre-primitives`, `vyre-libs`)

- Added 10k-case property tests hardening the two sharding decompositions shipped this cycle (Testing Contract: proptest per feature). `proptest_csr_frontier_shard` (vyre-primitives, 3×10k cases) proves the graph-frontier device-sharding invariant over random graphs, frontiers, and shard counts: sharded expansion always equals single-device expansion, the vertex partition is always disjoint+complete, and the OR-merge is order-independent and round-trips the frontier. `shard_assignment_is_a_valid_total_partition` (vyre-libs, 4k cases) proves the scan-sharding load balancer always produces a valid total partition for any window sizes / shard count / weights, one shard per window, all in range, byte-work conserved (nothing dropped or double-counted), and exact round-robin unweighted. These are the invariants the parallel sharded scan + graph frontier rely on to stay byte-identical to single-device regardless of work distribution.

### Added: device-sharded graph frontier expansion (`vyre-primitives`)

- Added `vyre_primitives::graph::csr_frontier_shard`: the W3-5 `graph-frontier-device-shards` decomposition. A forward `csr_frontier_step` expands only the vertices set in `frontier_in`, so the active frontier can be partitioned across device shards by vertex ownership (`partition_frontier_by_vertex`: disjoint, complete, contiguous vertex ranges) and the per-shard `frontier_out` bitsets OR-merged back together (`merge_frontier_out`: the cross-shard visited/frontier merge, a peer-transfer reduce on real multi-GPU, a host OR here). `frontier_step_sharded` runs one sharded expansion level given a per-shard `expand` closure (each shard dispatched on its own device), and fails closed on a zero shard count, a mis-sized frontier, or a wrong-sized shard output. Because per-vertex expansions are independent and the partition is disjoint and complete, the merged result equals a single-device expansion exactly, proven three ways: a hand oracle over a graph with cross-shard edges, a pure-Rust expansion oracle across 1–5 shard counts, and (the load-bearing proof) the real `csr_frontier_step` GPU program driven through the reference interpreter across 1–4 shards versus the single-device run. Device sharding therefore changes no reachability bit. Per-device concurrent dispatch reuses the `std::thread::scope` pattern already proven for byte-range scan sharding; only wall-clock multi-GPU speedup and the on-device peer-transfer merge need a second physical GPU.

### Added: roofline COMPUTE ceiling + full model (`vyre-driver-cuda`, `vyre-bench`)

- Added `CudaDeviceCaps::peak_compute_ops_per_sec()`: the compute ceiling of the W3-6 roofline, alongside the existing `memory_bandwidth_gbps()` memory ceiling. It is `SM_count × 4 warp-schedulers × warp_size × core_clock`, backed by a new `core_clock_rate_khz` device attribute (`CU_DEVICE_ATTRIBUTE_CLOCK_RATE`) joining the existing memory clock. The "4 warp schedulers per SM" factor is a universal published NVIDIA architectural constant (every SM from Volta through Blackwell is four processing sub-partitions, each issuing one warp-wide instruction per cycle), not a fabricated per-generation cores-per-SM table, so the ceiling is an honest analytical value with no invented device numbers. With both ceilings the roofline now has a **ridge point** (operational intensity where memory-bound flips to compute-bound). The new `scan_roofline_model_cuda` test assembles the full model on the RTX 5090: peak memory 1792 GB/s + peak compute ~52 TOPS + ridge 29227 ops/KiB + the scan's measured memory-axis point (218 GB/s achieved, 12% util) → the bound verdict (memory-side, launch/latency-bound, not compute-bound). A pure unit test locks the peak-compute formula (`170×4×32×2.41 GHz` ≈ 52 TOPS, asserted in the sane 40–80 TOPS Blackwell range). Only the scan's achieved *compute* operating point (executed op-count → arithmetic intensity) still needs Nsight-Compute instruction counters (admin-only here); both ceilings, the ridge, the measured memory-axis point, and the bound verdict are complete without root.

### Changed: cross-device sharded scan now dispatches in PARALLEL (`vyre-libs`, `vyre-driver-cuda`)

- `scan_sharded_core` (behind `scan_sharded_fused`/`_weighted`/`_timed`) now runs each device shard on its **own OS thread** via `std::thread::scope`, every shard prepares its own resident session and dispatches its assigned windows **concurrently** with the other devices, replacing the previous sequential shard loop. This is W3-5's "true cross-device PARALLEL dispatch (spawn per-device threads)." Aggregation stays deterministic despite the nondeterministic thread interleave: each thread globalizes into owned per-window blocks tagged with the global window index; the parent re-sorts by window index and concatenates presence in window order (byte-identical presence layout) while matches are gathered and canonically sorted by `finish_result` (order-free). It fails closed on a shard-thread panic (no partial cross-device result), each thread frees its own session before surfacing any error (one free path per thread), and `scope` guarantees all threads join so none leaks on the error path. The globalization logic is now shared between the sequential single-device paged driver and the parallel sharded core via extracted `window_presence_words` + `map_window_matches` helpers (ONE PLACE). As part of this, the CUDA resident-scan launch path (`dispatch_resident_via_borrowed_into`) now binds the device context on the calling thread (`warmup()`), it was the one resident entry point missing the bind its `batch`/`async`/`sequence` siblings already had, a latent foreign-thread `CUDA_ERROR_INVALID_CONTEXT` that per-device threading would otherwise trigger. Proven on the RTX 5090 (`parallel_sharded_dispatch_across_four_concurrent_handles_equals_single_shot_on_gpu`): a 32-file, ≥8-window corpus sharded across a four-handle set (four concurrent threads/sessions) is byte-identical to the single-device paged scan, with honest per-shard timing showing the work spread across all four shards. The existing 1-/3-device and throughput-weighted parity tests now also exercise the parallel path. True multi-GPU wall-clock speedup + peer-transfer aggregation remains gated on a second physical GPU; the parallel dispatch and deterministic aggregation are proven correct on one device.

### Added: stream-ordered `cuMemPool` device allocator (`vyre-driver-cuda`)

- Added `CudaStreamOrderedPool` (`backend/stream_ordered_pool.rs`), the stream-ordered device allocator half of W3-4. Where the synchronous bucketed `DeviceAllocationPool` recycles raw `cuMemAlloc_v2` blocks behind a host free-list (every acquire/release ordered by hand), this binds the device's **default** CUDA memory pool via `cuDeviceGetDefaultMemPool` (no private pool to create/destroy, no `Drop` hazard against context teardown) and drives it with the driver's stream-ordered allocator: `alloc_async`/`free_async` take a caller stream so an allocation and its free ride the same stream as the dispatch that consumes them, and the driver reuses a freed block for the next same-stream allocation with no host round-trip. Construction sets `RELEASE_THRESHOLD=u64::MAX` so freed physical memory stays **reserved** for reuse (the default 0 releases it on every sync, which would defeat a re-dispatch loop); `reserved_bytes()`/`used_bytes()` expose `RESERVED_MEM_CURRENT`/`USED_MEM_CURRENT`, and `trim(min_keep)` hands the reservation back to the OS. Proven on the RTX 5090 (`stream_ordered_pool_serves_usable_memory_and_reuses_reserved_blocks_on_gpu`): (a) a `memset(0xABCD1234)`→DtoH readback confirms the pool serves *usable* device memory; (b) freeing a block then re-allocating the same size leaves `reserved_bytes` **exactly unchanged**, the freed block is reused, not re-faulted; (c) `trim(0)` strictly *drops* the reservation. Hot-path integration (threading a stream through `DeviceAllocationPool::acquire`) is the tracked follow-up; this lands and proves the allocator primitive first.

### Added: roofline achieved-bandwidth evidence (`vyre-bench`)

- Added `scan_roofline_bandwidth_cuda`: the memory-bandwidth axis of the W3-6 roofline, sourced from vyre's own timing (no Nsight-Compute, which is admin-only here). A resident fused scan's achieved read bandwidth is `haystack_bytes / device_ns` (1 byte/ns == 1 GB/s), compared against the device peak from `CudaDeviceCaps::memory_bandwidth_gbps()` to place the scan on the roofline and state its bound. Measured on the RTX 5090: a 32 MiB scan runs at 235 GB/s against a 1792 GB/s peak (13% utilization → not-bandwidth-bound; this literal-set scan is latency/compute-bound with large DRAM headroom). The sanity ceiling allows for legitimate L2 over-DRAM-peak effects. An honest timing-sourced datum, explicitly not presented as Nsight counters.

### Added: scan-counter proxy capture (`vyre-bench`)

- Proved the `SCAN_COUNTER_EVIDENCE.toml` proxy counters are actually SOURCED from runtime telemetry (not just schema-declared) for the cuda backend, with a real-GPU `scan_counter_proxy_capture_cuda` test: it runs a live `GpuLiteralSet` scan through `CudaBackendRegistration` and captures `memory_bytes` (host↔device bytes), `occupancy_proxy` (the new `mean_occupancy_bps()`), `branch_divergence_proxy` (`logical_thread_waste_bps`), and `candidate_count` (match count, asserted against the planted total of 5). Measured: `memory_bytes=177612 occupancy_bps=10000 branch_divergence_bps=0 candidate_count=5`. The precise Nsight-Compute counters are admin-only on the host (`RmProfilingAdminOnly=1`), so the TOML cuda row now states that `unavailable_reason` and documents the runtime-telemetry proxy source + proving test, an honest counter source, not fabricated ncu values. The occupancy work above is what made the `occupancy_proxy` sourceable.

### Added: per-kernel occupancy evidence (`vyre-driver-cuda`)

- Every CUDA kernel launch now records its driver-measured achieved occupancy as telemetry evidence (W3-6). The launch path queries `cuOccupancyMaxActiveBlocksPerMultiprocessor` once per kernel shape and caches the result by `(function, threads_per_block)`, occupancy is constant per shape, so after the first launch it is a map lookup, never per-launch FFI (Law 7). The active-blocks count feeds a shared `occupancy_estimate_from_blocks` helper (extracted from the theoretical `estimate_occupancy` so both the register/shared-limit estimate and the driver measurement compute occupancy as the *same* fraction of `max_warps_per_sm`, ONE PLACE) and lands on `CudaTelemetrySnapshot` as `launch_occupancy_bps_sum` / `occupancy_measured_launches` / `occupancy_unmeasured_launches` with a derived `mean_occupancy_bps()` and four Prometheus series. A launch whose geometry or driver query is unusable is counted as *unmeasured* (loud), never silently dropped, so a partial mean is never mistaken for full coverage (Law 10). Occupancy recording never fails a launch (the kernel has already run). The single `cuOccupancyMaxActiveBlocksPerMultiprocessor` FFI is now behind one `query_active_blocks_per_sm_raw` helper shared with the cooperative-residency validator. Proven by telemetry unit tests (mean arithmetic, accumulate + reset) and a real-GPU `steady_state_launches_report_per_kernel_occupancy_evidence` test that runs a 256-thread dispatch loop and asserts every launch is measured, none unmeasured, and the mean is a real fraction in (0, 10000] bps consistent with the raw sum/count.

### Added: device-allocation-pool hit-rate telemetry (`vyre-driver-cuda`)

- Instrumented the transient `DeviceAllocationPool` with hit/miss counters (an acquisition served from the free-list is a hit; one that falls through to a real `cuMemAlloc_v2` is a miss) and surfaced them on `CudaTelemetrySnapshot` as `device_pool_hits`, `device_pool_misses`, and a derived `device_pool_hit_rate_bps()` (basis points, zero-safe, exact through a u128 intermediate), plus three new Prometheus series. The counters live on the pool, its only source of truth, since the caller cannot tell a hit from a miss, and are overlaid at the backend's `telemetry_snapshot()` boundary; `reset_telemetry()` resets them into the same epoch as the rest of the counters. This is the W3-4 "pool-hit-rate evidence" deliverable: a real re-dispatch consumer workload can now see whether the pool is actually serving from cache. Proven by a pure hit-rate-arithmetic unit test and a real-GPU `steady_state_redispatch_loop_reports_high_device_pool_hit_rate` test that runs a 32-dispatch identical-shape loop and asserts the steady-state hit rate is majority-hits (the pool working), with the rate exactly consistent with the raw counters.

### Added: paged corpus benchmark (`vyre-bench`)

- Added the `scan.literal_set.paged_corpus` benchmark case: it scans a multi-megabyte corpus split into thousands of small files with a window budget far smaller than the corpus (many windows) through both `scan_paged_fused` and `scan_paged_fused_async`, reporting throughput and the sync-vs-async pipeline overlap factor. Correctness is hard-gated two ways: the paged matches must equal an independent CPU `reference_scan` of the concatenated corpus, and the async result must be byte-identical to the sync result.

### Added: pattern-database sharded scanning (`vyre-libs`)

- Added `vyre_libs::scan::scan_pattern_sharded(shards: &[PatternShard], backends, haystack) -> Vec<Match>`: the W3-5 `pattern-database-replicated-shards` workload, it stripes the RULE database (not the haystack) across a device set. Each `PatternShard` is a sub-matcher over a disjoint rule subset plus a local→global pattern-id map; it runs on `backends[shard % n]`, its matches are remapped to the global rule numbering, and all shards merge into the canonical `(pattern_id, start, end)` report order. Because literal matching is independent per rule, the striped union equals the full un-sharded matcher's match set, the plan's replicated/striped parity policy. Fails closed on an empty device set and on a malformed shard map (a local id with no global mapping errors rather than dropping or mis-attributing the finding). Proven on the RTX 5090 (a 2-shard stripe over 1- and 2-device sets equals the full-database scan; malformed map errors).

### Added: multi-GPU sharded scanning (`vyre-libs`)

- Added `vyre_libs::scan::scan_sharded_fused(matcher, backends: &[&dyn VyreBackend], files, window_budget_bytes, max_matches)`: the W3-5 `regex-haystack-byte-range-shards` architecture. It distributes the corpus's byte-range window shards round-robin across a device SET (window `k` → `backends[k % n]`), each backend holding its own resident fused session, so on a multi-GPU host the shards run concurrently on distinct peer devices. The partition, halo (`L-1` overlap), and aggregation (host globalize + stable sort by `(region, start, end, pattern_id)`) reuse the exact `scan_paged_fused` helpers (ONE PLACE), so the sharded result is byte-identical to a single-shot scan for any device-set size, the plan's parity policy. Fails closed on an empty backend set; one ordered free pass so no resident session leaks. Proven on the RTX 5090 (1-device and 3-device sets both equal the single-device scan, boundary-spanning match survives sharding, empty set errors). On a single-device host the shards run sequentially; only cross-device parallelism awaits a second physical GPU.
- Added `scan_sharded_fused_timed(...)` (with `ShardTiming` / `ShardedScanTiming`): the per-shard-timed twin, an identical result plus a per-device breakdown of windows, byte-work, wall time, and device (kernel) time. This is the `per-shard-active-ns` signal the plan's `load_balance_policy` rebalances on: a skewed timing across shards under equal round-robin is the evidence to feed proportional `weights` into `scan_sharded_fused_weighted` next batch. Each shard's `device_ns` stays `Some` only while every window on it reported device time (loud `None` otherwise, never a fabricated 0); an idle shard reports `Some(0)`. Proven on the RTX 5090 (per-shard window counts and byte-work sum to the totals, each active shard reports real wall + non-zero device time, timed result == untimed).
- Added `scan_sharded_fused_weighted(matcher, backends, weights: &[u32], files, window_budget_bytes, max_matches)`: the throughput-weighted twin, cumulative byte-work per device tracks `weights[i]` (the plan's `device-throughput-weight` / `load_balance_policy`) via a deterministic greedy least-loaded-by-weight assignment shared with the round-robin path (ONE PLACE `shard_assignment`; zero weight treated as 1, never starved). Fails closed on a weights/backends length mismatch. Because aggregation is order-independent, the weighted result is byte-identical to round-robin and single-shot for any weights, only the work distribution changes. Proven by a pure host unit test (3:1 weight → 3 of 4 windows to shard 0) and the RTX 5090 parity test.

### Added: paged corpus scanning (`vyre-libs`)

- Added `vyre_libs::scan::paged_corpus::scan_paged_fused` (with `PagedScanResult` and `GlobalMatch`): scans a corpus of files that may exceed one resident window as a sequence of resident fused-window dispatches, returning the per-region presence bitmap in a single global region numbering plus every positioned match in u64 global coordinates. Files are planned into byte-budgeted windows at file boundaries with stable global region ids; each window runs as an independent local scan and is globalized on the host with `L-1`-byte overlap, a discardable dummy overlap region, and start-based dedup, so the result is byte-identical to a single-shot scan of the concatenated corpus (no boundary miss, no over-fire, no double count), while host RSS stays bounded by one window instead of the whole corpus. Proven on real GPU against a single-shot scan including a boundary-spanning match.
- Added `scan_paged_fused_timed` (with `PagedScanTiming`): the timed twin of `scan_paged_fused`, extending W3-3 "attribution everywhere" onto the paging path. It returns a result byte-identical to the untimed driver plus an honest aggregate over the per-window dispatches, window count, total own bytes scanned (overlap excluded, a valid throughput denominator), summed wall-clock time, and summed device (kernel) time. The device aggregate is `Some` only when every window reported a device timer; a single timer-less window collapses it to a loud `None`, never a fabricated 0 (Law 10). It differs from the untimed driver in exactly one call (`scan_into_timed` vs `scan_into`) and reuses the same shared staging/globalization helpers, so the paged result cannot drift. Proven on real GPU (timed == untimed, device time present and non-zero) plus an empty-corpus test locking the zero-window `Some(0)` aggregate.
- Added `scan_paged_fused_async`: the asynchronous twin that pipelines the windows (window `k+1`'s staging and upload overlap window `k`'s device execution, two dispatches in flight) via the borrowed async fused dispatch. It shares the exact overlap/dummy-region/dedup globalization with the synchronous driver, so its result is bit-for-bit identical (proven on real GPU (async == sync)).
- Added `scan_paths_paged`: the disk-backed paged scanner, it takes file paths and reads only one window's files into memory at a time, so host RSS stays bounded by the window rather than the corpus. It shares the same globalization as the in-memory driver, so its result is identical (proven on real GPU (disk == in-memory) plus a no-GPU test of the window disk-read + overlap prefix).
- Added `scan_paths_paged_prefetched`: the prefetching disk scanner, a background thread reads window `k+1`'s files while the GPU scans window `k`, so disk I/O overlaps device compute, with a depth-1 bounded channel keeping host RSS to at most two windows. Result is identical to the synchronous disk scan (proven on real GPU (prefetched == sync)).

### Added: fast-path corpus example (`vyre-libs`)

- Added `vyre-libs/examples/scan_corpus_fast_path.rs`: a runnable consumer example that coalesces a set of files (a real directory tree, or a built-in multi-file corpus) into a haystack plus `region_starts`, compiles the matcher once, prepares a resident fused session, and runs one timed dispatch producing both the per-region presence bitmap and the positioned matches, the runnable companion to the fast-path guide. With no GPU it falls back loudly to the portable `scan_all` on the CPU reference backend.
- Added `vyre-libs/examples/scan_paged_corpus.rs`: the disk-ingress companion for a corpus larger than one window. It materializes a multi-file corpus on disk (or pages a real directory-tree argument), plans windows under a deliberately tiny byte budget to force multi-window paging, and runs `scan_paths_paged_prefetched`, printing per-file presence and every positioned match in global (file-index, u64-byte) coordinates. A pattern that straddles a window boundary is reported exactly once. With no GPU it says so loudly and falls back to reading every file into memory plus `scan_paged_fused_async` on the CPU reference backend, surrendering the bounded-RSS property but yielding the same global match set.

### Added: fast-path scanning guide (`docs`)

- Added `docs/scanning-a-corpus-the-right-way.md`: the intended route through the resident/async/fused/count-then-collect APIs, a decision table for which API to use and a five-step fast path (compile once → prepare a resident session → overlap batches with the async twins → leave timed attribution on → let the device count with `scan_all`). Every signature is copied verbatim from the current `GpuLiteralSet` public surface, and the guide is listed in `docs/INDEX.md`.

### Added: head-to-head vs CPU aho-corasick benchmark (`vyre-bench`)

- Added the `scan.literal_set.vs_cpu_aho_corasick` benchmark case: it runs the same pattern set over the same consumer-shaped corpus through vyre's resident GPU literal-set scan (end-to-end, staging included) and the `aho-corasick` crate (built with `MatchKind::Standard` + `find_overlapping_iter`, the all-overlapping semantics vyre's DFA emits), and reports the end-to-end speedup plus the GPU device-vs-staging split. Correctness is a hard gate, the GPU matches must be byte-identical to the aho-corasick matches (a fast wrong answer fails), and the CPU baseline is pre-checked to reproduce the engine's `reference_scan` set exactly. The performance delta is reported, not gated: this is the standing head-to-head that makes the "beats the best CPU path end-to-end" claim (and any gap) visible per release.

### Added: consumer-shaped cold-start & decode-heavy benchmarks (`vyre-bench`)

- Added the `scan.literal_set.cold_start` benchmark case: it times the full cold-start path of a one-shot literal-set scan, building the matcher (`try_compile`), the first table upload, and the first dispatch with cold caches, against the warm steady-state per-dispatch cost, and reports the cold-start overhead factor plus the compile-vs-first-touch split. This is the cost a consumer that scans one corpus and exits actually pays, invisible to a steady-state loop.
- Added the `scan.literal_set.decode_heavy` benchmark case: it measures the decode-bound regime on a dense-match corpus (the shortest pattern tiled every 128 bytes, ~32k matches over 4 MiB) scanned through a resident session, so the immutable tables upload once and every dispatch is dominated by writing the match triples, reading them back, and decoding them on the host, reporting the device-vs-host-decode split. Both cases hard-gate correctness: the GPU matches must be byte-identical to the independent CPU `reference_scan` (Law 10), verified via exact-output comparison and, without a GPU, by `CpuRefBackend` unit tests.

### Added: async two-batch overlap benchmark (`vyre-bench`)

- Added the `scan.literal_set.async_overlap.2batch` benchmark case: it runs the asynchronous literal-set position scan over two distinct consumer-shaped batches both sequentially (submit → await → submit → await) and overlapped (submit A → submit B → await A → await B), and reports the overlap factor plus the sequential kernel-vs-host-staging split. Correctness is a hard gate: the overlapped matches must be byte-identical to the sequential ones for both batches (Law 10, overlap changes no result bit), verified via the case's exact-output comparison and, without a GPU, by a `CpuRefBackend` unit test. This is the quantitative companion to the existing `literal_set_async_two_batch_pipeline` correctness gate.

### Added: distinct regex-unsupported diagnostics (backreference / huge alternation / nested repeats / capture)

- The GPU-NFA regex frontend now DISTINCTLY detects four constructs that previously collapsed into a generic `Parse` or `TooManyStates` error, so a consumer can route each on its canonical `REGEX_UNSUPPORTED_DIAGNOSTICS.toml` code. Backreferences (`\1`, `\k<name>`, `(?P=name)`) are classified by an escaping-aware structured source scan (run only on parse failure, never by matching parser error text) and map to `VYRE_SCAN_UNSUPPORTED_BACKREFERENCE`. Over-budget alternations map to `VYRE_SCAN_UNSUPPORTED_HUGE_ALTERNATION_BUDGET` and nested bounded repeats whose unroll product exceeds the state budget map to `VYRE_SCAN_UNSUPPORTED_NESTED_REPEAT_BUDGET`, both detected before lowering collapses them into `TooManyStates`. The reclassification is sound: both budgets equal the state cap, so no pattern that compiled before now errors.
- Capture groups remain a successful whole-match compile (making them an error would regress acceleration); `CompiledRegexSet::captures_present` and `CompiledRegexSet::capture_extraction_diagnostic_code()` surface the `VYRE_SCAN_CAPTURE_EXTRACTION_REQUIRES_VERIFIER` signal for a consumer that needs submatch spans.
- New public API: `RegexConstruct` enum and `regex_construct_diagnostic_code`: the single owner of every regex-construct diagnostic code string, through which both `RegexCompileError::diagnostic_code` and the capture-signal path route.

### Added: resident fused presence+positions pipeline (`ResidentFusedRegionScan`)

- Added `GpuLiteralSet::prepare_resident_fused_scan` and the `ResidentFusedRegionScan` session it returns (`scan_into`, `scan_into_timed`, `max_regions`, `max_matches`, `haystack_capacity`, `free`): the resident twin of the FUSED per-region presence + positions scan (`scan_presence_and_positions_by_region`). It is the fusion of `ResidentPresencePipeline` (per-region presence bitmap + region controls) and `ResidentLiteralScan` (positioned match output), one all-resident dispatch of the 14-binding fused program produces BOTH outputs, uploading the immutable DFA + suffix-prefilter tables ONCE and re-staging only the haystack, region controls, and two zeroed accumulators (presence prefix + match counter) per scan. All 14 bindings are resident (incl. the two read-write accumulators and the `matches` output), so it runs on the CUDA backend; the fixed-size `matches` buffer fails CLOSED on overflow (Law 10), and an over-capacity haystack or over-cap region count fails closed before dispatch. Real-GPU parity: the resident presence bitmap AND triples are byte-identical to the borrowed fused scan across repeated re-dispatches.

### Added: resident position-scan pipeline (`ResidentLiteralScan`)

- Added `GpuLiteralSet::prepare_resident_scan` and the `ResidentLiteralScan` session it returns (`scan_into`, `scan_into_timed`, `max_matches`, `haystack_capacity`, `free`): the positioned-scan sibling of `prepare_resident_presence`. It uploads the immutable DFA transition/output/pattern-length tables and the three suffix-prefilter masks into backend resources ONCE, then re-dispatches the literal MATCH program across a corpus re-uploading only the per-file haystack and resetting a 4-byte match counter, eliminating the multi-MiB per-scan table re-upload the borrowed `scan_into` repeats on every file. All 11 bindings are resident (including the `matches` output buffer, which the resident dispatch resolves as an output and reads back), so it runs on the CUDA backend with no borrowed mix. The fixed-size resident `matches` buffer FAILS CLOSED when the device match count exceeds `max_matches` (never a silent truncated decode), and an over-capacity haystack fails closed before any upload. Measured 1.84× faster than borrowed across a 400-detector / 192-scan corpus on an RTX 5090.

### Added: attribution (`TimedDispatchResult`) twins for every literal dispatch path

- Added `GpuLiteralSet::scan_presence_by_region_timed` and `scan_into_timed`: timed twins of the hot region-presence and position (`scan_into`) paths, returning `vyre_driver::TimedDispatchResult` (wall / device / enqueue / wait) alongside the same result the untimed entry produces, so a consumer or benchmark can split per-scan cost between the GPU kernel (`device_ns`) and host staging/readback. The untimed hot paths are untouched and pay no timing cost; `device_ns` is a loud `None` on a backend without a device timer, never a fabricated zero.
- Added `GpuLiteralSet::scan_presence_timed` (global-presence path) returning `(bitmap, TimedDispatchResult)`, built on a new owned dispatch-staging path that reuses the shared immutable-table encoder so every presence path encodes byte-identical tables.
- Added `GpuLiteralSet::scan_presence_and_positions_by_region_timed` (fused presence+positions path) returning `(bitmap, TimedDispatchResult)` and decoding the `(pattern_id, start, end)` triples into a caller buffer, with the same fail-closed overflow contract as the untimed fused scan (a match count over `max_matches` errors, never a silent truncated decode).
- Added `GpuLiteralSet::scan_all_timed` (auto-resize complete-match path) returning `ScanAllTimed { timed, resized }`: the timing describes the dispatch that produced the returned matches, and `resized` loudly states whether that was the resize re-dispatch (the two-launch case is reported, never silently summed).

### Added: asynchronous (`PendingDispatch`) twins for every single-dispatch entry point

- Added `GpuLiteralSet::scan_presence_async` (→ `PendingPresence`), `scan_into_async` (→ `PendingMatches`), and `scan_presence_and_positions_by_region_async` (→ `PendingFusedRegion`): submit the GPU dispatch and return a handle immediately so callers can overlap host-side work with the in-flight scan, then decode via `await_words` / `await_into` / `await_matches`. Each retains its owned upload buffers until the decode and, on a non-pipelining backend, yields a trivially-ready handle whose result is byte-for-byte identical to the synchronous entry (no silent change on the degraded path). Together with the pre-existing `scan_presence_by_region_async` this covers every single-dispatch scan entry point.
- New public types: `ScanAllTimed`, `PendingPresence`, `PendingMatches`, `PendingFusedRegion`.

### Added: device-side per-region compaction primitive

- Added `vyre_primitives::matching::region::compact_first_per_region_pattern_flag_program` (op id `COMPACT_FIRST_PER_REGION_PATTERN_OP_ID`) and its CPU-parity oracle: a per-invocation first-occurrence kernel keyed on the `(region, pid)` pair that emits a survivor flag for the first match of each pair, so stream-compaction leaves exactly one positioned representative per pair, the positioned companion to the presence-by-region bitmap, computed on device with no host per-region group-by after readback. Completes the W2-5 device-side post-processing set (sort, dedup, per-pattern cap, per-region compaction).

### Added: grid-aware reference evaluation

- Added `vyre_reference::reference_eval_with_dispatch` / `run_arena_reference_with_dispatch`, which let a caller pass the true byte-scan grid (invocation count) so the interpreter covers what the real GPU dispatch would. `reference_eval` is unchanged (grid floor 0). This closes a silent under-coverage in the reference oracle where a byte-scan over a haystack larger than its max buffer element count skipped high positions on CPU-ref only (the GPU was always correct).

## [0.6.4]  -  2026-06-23

- Added `GpuLiteralSet::prepare_resident_presence` and the `ResidentPresencePipeline` it returns: a resident literal-set region-presence session that uploads the immutable DFA transition/output/pattern-length tables and suffix-prefilter masks into backend resources ONCE, then re-dispatches across a corpus's coalesced batches re-uploading only the per-file haystack and resetting the per-region presence buffer, eliminating the multi-MiB per-scan table re-upload the borrowed `scan_presence_by_region` path repeats on every file. All-resident so it runs on the CUDA backend.

- Added `ResidentPresencePipeline::scan_into_timed` returning `TimedDispatchResult` (wall / device / enqueue / wait nanoseconds) so callers can attribute a region-presence dispatch's GPU-kernel time separately from host staging and decode; `scan_into` now wraps it. Direct CUDA attribution on an RTX 5090 (8 MiB, 900 detectors) measured the region-presence kernel at ~41 µs (the borrowed path's cost is per-scan table re-upload, not the kernel).

- Made `prepare_resident_presence` fail closed at prepare time when the requested resident haystack capacity is smaller than the NFA program's statically-declared input buffer (binding 0), with an error naming the required byte count and the fix, instead of dispatching against an undersized resident buffer.

- Added `GpuLiteralSet::scan_presence_and_positions_by_region[_with_scratch]`, a single suffix3 dispatch that folds per-region literal presence and confirmed match positions into one GPU pass (previously two separate dispatches), with GPU-vs-exhaustive-CPU-reference differential coverage.

- Added row-strided queue-to-queue delta enqueue for skewed CSR fixpoint waves, wired IFDS queue closure to select it for high-degree rows, and refreshed public API snapshots for the exposed graph/frontier planning surfaces.

- Made the CUDA-resident C sparse lexer compact terminal path read back `out_counts` first and then download only the live dense token column ranges, cutting host transfer volume for sparse translation units without breaking the resident GPU chain.

- Sized C sparse-lexer compact outputs from the scanned token count instead of source byte count for staged and block-total compaction paths, reducing readback and downstream token-buffer pressure on whitespace-heavy translation units.

- Made budgeted resident CSR queue batches plan ordered chunks from each chunk's effective frontier popcount, so sparse runs before and after a dense outlier still pack tightly under the resident scratch budget.

- Clamped resident CSR frontier-queue dispatch capacity from in-domain frontier popcount, reducing graph-sized scratch allocation and overlaunch for sparse single-query and batched traversals while keeping caller queue capacity as a hard cap.

- Sized resident adaptive sparse-queue traversal from the active frontier popcount instead of graph node count, reusing larger queue scratch across smaller frontiers and preserving row-strided traversal for high-degree rows.

- Added 30,000 generated row-strided CSR queue primitive checks covering skewed graph traversal, caller-owned output reuse, malformed CSR rejection, and dispatch-grid coverage.

- Routed the IFDS skewed active-queue and queue-materialization benchmarks through the row-strided CSR queue consumer for high-degree rows, increased the benchmark fixture hub degree to 2,048 edges, and added telemetry proving when the strided traversal path is active.

- Added a row-strided CSR queue traversal primitive for skewed active frontiers, wired resident CSR and adaptive sparse-queue paths to select it for high-degree rows, and refreshed the `vyre-primitives` public API snapshot.

- Made the sparse C tokenizer's raw `U8` haystack runtime-sized, removing the host-side bucket padding copy before token classification while keeping bucketed GPU output shapes.

- Moved the full C comment/splice fallback to runtime-sized raw `U8` source buffers, removing the remaining padded splice-input staging from the byte-filter pipeline.

- Added a backend-extension gate proving new backends remain one crate plus `inventory::submit!`, and declared SPIR-V dispatch capability through the same inventory path as CUDA and wgpu.

- Hardened the base monument benchmark check so it proves the executable `vyre-bench` meta-harness, JSON registry, thesis workload IDs, and deep coverage dimensions instead of only checking for the PRD.

- Added a million-node graph frontier benchmark to `vyre-bench`, with exact CPU-oracle verification and release-suite thesis coverage contracts so benchmark evidence cannot regress to element-wise-only workloads.

- Added explicit graph launch sizing for CSR frontier degree-sum and refreshed the public API snapshot for the current graph/dispatch surfaces.

- Added explicit RLE segment-length dispatch sizing and multi-block CPU/CUDA parity coverage for packed decode workloads.

- Added explicit bigint add-carry dispatch sizing and multi-block CPU/CUDA carry-pattern coverage for large limb arrays.

- Added explicit union-find dispatch sizing through the self-substrate path and multi-block CUDA coverage for large edge batches.

- Added explicit d-DNNF evaluation dispatch sizing and multi-block CUDA coverage for literal-heavy knowledge-compile waves.

- Reworked Scallop single and wide lineage fixpoint kernels to preserve high-cell and high-word seed facts without CUDA grid-barrier races, with CUDA parity coverage for the exposed high-word case.

- Restored multi-block Scallop dispatch for large relation matrices through split-visible GridSync phases while keeping small matrices on the block-local persistent path.

- Packed `tensor_flow_forward` source-node dataflow lanes into 256-lane workgroups and added CUDA parity for context/field propagation past the first block.

- Made GPU region dedup cluster-aware for nested/touching scanner spans, added merged-end metadata for on-device compaction, and proved multi-workgroup CUDA parity.

- Added a 256-lane parallel `bracket_match` path when parser depth caps cannot affect output, with CUDA parity for large nested token streams and retained bounded-stack fallback for overflow-capped shards.

- Routed large adaptive sparse-queue traversal frontiers through the deterministic word-prefix queue materializer, with resident CUDA parity for a large sparse graph step and refreshed adaptive traversal program-cache identities.

- Replaced multi-block word-prefix queue scatter's per-word previous-block loop with an in-place block-offset scan and precomputed-offset scatter, with resident CSR/adaptive wiring and live CUDA coverage for generated multi-block frontier queries.

- Added a CSR-only resident adaptive sparse-queue graph upload and step path so sparse-queue workloads avoid dense adjacency allocation/upload, with live CUDA telemetry coverage and generated sparse-queue matrix coverage on the no-dense path.

- Added CSR frontier queue property gates covering 40,000 generated materialization, traversal, adversarial queue, and validation cases, and doubled live CUDA adaptive sparse-queue generated coverage to 1,024 resident steps per materializer.

- Removed the redundant resident atomic sparse-queue `queue_len` init dispatch from CSR and adaptive traversal paths, dropping small resident sparse-queue steps from four kernels to three while keeping queue length initialization inside `frontier_to_queue`.

- Added packed-`U8` line indexing, UTF-8 validation, and C line-splice classification for text scans, fixed CUDA/PTX byte and halfword memory ops, and covered the paths with generated reference parity plus live CUDA boundary matrices.

- Moved the C preprocessing byte filter to raw `U8` source buffers through preflight, line/block comment paths, full comment masking, and compact scatter, fixed literal-close handling before later comments, and added live CUDA generated-corpus coverage for the end-to-end filter.

- Moved the sparse C tokenizer pipeline to a raw `U8` haystack while preserving packed and expanded compatibility entrypoints, with reference-eval ABI checks and live CUDA generated-corpus parity for token and directive columns.

- Moved the C directive-metadata stage used by the preprocessing pipeline to raw `U8` source bytes while preserving the packed standalone ABI, eliminating another source repack between tokenization and directive classification.

- Moved fused `#define`/`#include`/`#undef` payload parsing in the preprocessing pipeline to raw `U8` source bytes while preserving packed standalone parser ABIs.

- Moved `#ifdef`/`#ifndef` and `#if`/`#elif` compatibility evaluators in directive extraction and live conditional re-evaluation to raw `U8` source rows and macro-name tables while preserving packed standalone evaluator ABIs.

- Removed the now-unused C GPU-preprocess U32 byte-padding staging helper so raw-byte directive and live conditional paths cannot route back through padded host macro-name buffers.

### New

- **`vyre-foundation`  -  effects-handler lowering is on the release path.**
  `PassScheduler` now has an effects-handler enforcement gate: rewrites may
  discharge existing effects, but any newly introduced effect row bit is
  reverted unless the pass declares it through `allowed_effect_additions`.
  Backend `pre_lowering::optimize` enables this gate beside cost-monotone
  enforcement, and pass metrics now expose before/after effect-row bits.
- **`vyre-foundation`  -  linear BufferAccess is on the release path.**
  `PassScheduler` now enforces `BufferDecl::linear_type` postconditions for
  backend pre-lowering: rewrites may repair existing violations but cannot
  introduce new linear/affine/relevant usage violations before lowering. Pass
  metrics expose before/after linear-violation counts.
- **`vyre-foundation`  -  liquid BufferDecl shapes are on the release path.**
  `PassScheduler` now enforces `BufferDecl::shape_predicate` postconditions for
  backend pre-lowering: rewrites may repair existing shape violations but cannot
  introduce new predicate/count contradictions before CUDA or WGPU lowering.
  Pass metrics expose before/after shape-violation counts.
- **`vyre-foundation`  -  liquid shapes now erase dynamic loop guards.**
  `loop_var_range_fold` consumes `ProgramShapeFacts` so comparisons between a
  loop induction variable and `buf_len(buffer)` fold when `ShapePredicate`
  min/max facts prove the branch true or false. Runtime-sized buffers with
  `AtLeast`/`Exactly`/bounded affine shape facts can now drop redundant
  per-iteration bounds checks before CUDA lowering.
- **`vyre-foundation`  -  wire parser adversarial properties run in normal CI.**
  Added generated `Program::to_wire`/`Program::from_wire` property coverage for
  10,000 generated programs, 10,000 arbitrary hostile byte blobs, 10,000
  truncations, and 10,000 digest-refreshed body mutations. The new tests found
  and fixed a decoder gap where tampered but checksum-correct bytes could
  produce zero workgroup dimensions; `from_wire` now rejects zero workgroup
  dimensions and invalid output byte ranges at parse time.
- **`vyre-foundation` / `vyre-driver-cuda` / `vyre-reference`  -  explicit
  single-rank collectives execute through one shared transform.** Added
  substrate-neutral lowering for `CommGroup::WORLD` `AllGather` and
  `ReduceScatter` into bounded copy IR while reducing single-rank `AllReduce`
  and root-0 `Broadcast` to identity semantics. CUDA dispatch, CUDA compiled
  pipelines, and the reference oracle now consume the same transform. Non-world
  groups and nonzero single-rank broadcast roots fail closed with actionable
  errors, so multi-rank transport is never silently emulated. New proptests
  generate 16,384 collective-lowering/reference cases and live CUDA tests cover
  host dispatch, native compiled pipelines, and adversarial root rejection.
  Capability scanning now distinguishes lowerable single-rank collectives from
  collectives that genuinely require transport, and the canonical pre-emit
  pipeline applies the same transform before descriptor lowering.
- **`xtask` / release gates  -  recursion thesis is load-bearing.** Repaired
  `recursion-gate` root detection for the standalone Vyre workspace, made it
  scan the current `vyre-self-substrate/src` tree recursively plus the primitive
  catalog surface, taught it to parse grouped Rust imports across newlines,
  excluded private helper modules from the public primitive inventory, and
  wired `scripts/check_recursion_gate.sh` into release signoff so missing
  self-consumers fail release validation. Added the self-substrate
  `data::parsing_dispatch_pipeline` so packed-AST constant folding and
  bytecode dispatch-table packing consume the parsing primitives on the
  production substrate path.
- **`vyre-foundation`  -  derived pass-order artifact.** Added
  `optimizer::derived_order` with a live inventory-derived pass order,
  declared requirement edges, causal invalidation adjacency, and
  adjustment-set back-door safety checks. Release pass-order validation now
  consumes this artifact instead of reconstructing an independent ordering.
- **`vyre-foundation`  -  planar rewrite batching on the optimizer execution
  path.** Added a foundation-owned non-overlap batch planner,
  `ProgramPass::batch_apply`, refusal-aware `try_batch_apply`, and scheduler
  wiring so high-candidate passes can apply disjoint rewrite waves instead of
  relying on one-candidate-at-a-time launches. The primitive reference oracle
  now delegates to the same planner, keeping CPU contracts and GPU primitive
  tests on one source of truth. The batch activation threshold is runtime
  configurable through `VYRE_PLANAR_REWRITE_BATCH_THRESHOLD`.
- **`vyre-driver` / `vyre-driver-wgpu`  -  natural-gradient launch resolver
  on release paths.** Exported the canonical workgroup candidate table and
  shared launch resolver, wired CUDA `LaunchPlan` and WGPU pre-lowering
  config through safe-gated natural-gradient cold-start workgroup selection,
  and cached the selected launch shape per program/element-count/limit tuple
  so the hot path does not rebuild policy vectors. CUDA timed dispatch now
  records real `device_ns` measurements back into the bounded launch cache,
  allowing later automatic launches to move away from the cold-start
  heuristic when hardware timing proves another candidate faster. WGPU timed
  dispatch now returns timestamp-query `device_ns` as structured
  `TimedDispatchResult` data and feeds it into the same launch-feedback path.
  Measured launch decisions now persist across process restarts through the
  existing bounded tuner TOML cache.
- **`vyre-primitives`  -  dominator-tree public primitive surface.** Added
  the registered graph primitive to the self-consumer catalog, moved its
  scale/VRAM benchmark into the central `vyre-bench` release harness, and
  refreshed the public API snapshot for the new graph contract.
- Document `vyrec` / `vyre-frontend-c` as beta active-development consumers
  rather than the core Vyre `0.4.2` release proof.
- [A06] Document workspace member listing convention (S13)
- [A11] Bulk-fill Jules ticket queue (fixture_sweep + cve_replay)
- [A05] Examples consume published crates via patch.crates-io
- [A03] Validator error code documentation (S8)
- [A02] Rename vyre-cc to vyre-frontend-c
- **`vyre-foundation`  -  `BinOp::MulHigh` IR primitive.** Widening unsigned
  32×32→64 multiply returning the upper 32 bits. Wire tag `0x21`.
  Full support: const-fold in `ir_eval.rs`, interpreter in `node_kind.rs`,
  wire encode/decode in `bin_op_tag.rs`/`bin_op_from_tag.rs`, and
  `Expr::mulhi()` builder. Required for Granlund-Montgomery division.

- **`vyre-foundation`  -  Granlund-Montgomery constant division.** Strength-reduce
  pass now rewrites `x / d` (for constant non-power-of-two `d`) into a
  `MulHigh + Shr` sequence using Hacker's Delight Algorithm D. Eliminates the
  ~70-cycle hardware division in favor of ~5-cycle multiply-shift. Exhaustive
  correctness tests cover all divisors 2–1000 plus extreme boundary cases
  (2³¹±1, 2³²−1). Located in `optimizer/passes/strength_reduce/arithmetic.rs`.

- **`vyre-driver`  -  `LoweringStrategy` trait + capability-driven selector.**
  Two-layer optimization architecture: Layer 1 (IR-level math rewrites in
  `vyre-foundation/optimizer/passes/`) is backend-agnostic. Layer 2 (backend
  lowering strategies in `vyre-driver/strategy/`) is target-dependent.
  Strategies declare capabilities via `BackendCapabilities` and are selected
  by priority. `select_strategy()` picks the highest-priority applicable
  strategy. See `docs/ARCHITECTURE.md § Two-layer optimization architecture`.

- **`vyre-libs`  -  `c_lower_ast_to_pg_nodes` Cat-A op.** Added registration for
  `vyre-libs::parsing::c::lower::ast_to_pg_nodes`, a pure-IR lowering from
  structural VAST rows to packed `PgNode` tuples
  `(kind, span_start, span_end, parent_idx, payload_lo, payload_hi)`.
  Added witness fixture, pure CPU reference oracle, WGSL emission smoke test,
  GPU dispatch parity sample, and adversarial coverage (60 fixtures + proptest).

- **`vyre-runtime`  -  persistent megakernel + `io_uring` NVMe streaming.**
  Persistent megakernel runtime loops on host-fed ring slots for typed
  Programs (not a general VIR bytecode interpreter). Linux-only NVMe
  zero-copy via raw `io_uring_setup` + mmap of SQ/CQ rings, with a
  `uring-cmd-nvme` feature for `IORING_OP_URING_CMD` passthrough
  (kernel 6.0+). Three-buffer layout (control / ring / debug_log),
  256-lane × N-workgroup sharding, opcode extension hook for vendor
  intrinsics, per-tenant authorization masks, atomic `done_count`
  counter, and a PRINTF debug channel.
- **`vyre-libs`  -  Category A composition ecosystem.** Pure-IR
  compositions over `vyre-ops` primitives (`math`, `nn`, `matching`,
  `crypto`). No raw shader source  -  every library function is a
  `Program` consumers can round-trip, validate, and inline.
  `substring_search` lands with a real byte-by-byte equality instead of
  the earlier LAW 1 placeholder.
- **10 io_uring + IR innovations.** `IORING_REGISTER_BUFFERS` +
  `READ_FIXED`, `IORING_REGISTER_FILES` + `IOSQE_FIXED_FILE`, GPUDirect
  Storage `GpuMappedBuffer::from_bar1_peer`, `futex_waitv` completion
  doorbell, per-workgroup slot sharding, ring-credit backpressure,
  opcode extension hook, tenant-mask routing, PRINTF debug channel,
  AF_XDP/RDMA ingress demonstrated via a TCP smoke test.
- **Error-code catalog grew a `P-*` family** for
  `vyre-runtime::PipelineError`.
- **Workspace docs pristine.** `cargo doc --workspace --all-features
  --no-deps` runs clean  -  zero unresolved intra-doc links, zero
  private-link leakage, zero output collisions.

### Fixed

- **Descriptor `identity_elim` fma-zero fold ignored inf/NaN**  -  it folded
  `Fma(a, b, c) → c` whenever a factor was a literal numeric zero, with no
  check on the other factor. vyre Fma is float-only and `0.0 * inf =
  0.0 * NaN = NaN`, so `Fma(0.0, inf, c)` is NaN, not `c`: the fold silently
  replaced a NaN with the addend. Now requires the other factor to be a
  *finite literal*, matching the foundation `simplify_fma` guard (one
  auditable contract via the new `ScalarLiteral::is_finite_numeric`).
  Regression test asserts `Fma(0.0, inf, c)` is not folded.

- **Descriptor LICM hoisted convergent subgroup collectives out of loops**  -
  `SubgroupBallot/Shuffle/Broadcast/Reduce` were classified hoistable. Their
  result depends on the participating-lane set, so lifting one out of a loop
  (execution count N → 1) changes that context and the result. Now fail-closed
  for the four collectives, matching the authoritative foundation
  `expr_is_observably_free` gate; `SubgroupLocalId`/`SubgroupSize` stay
  hoistable as per-lane loop-invariant constants. Regression test asserts a
  `subgroupAdd` of a loop-invariant value stays inside the loop.

- **Loop fusion fused across a compare-exchange `expected` cross-loop read**  -
  `collect_vars_in_expr` walked an atomic's `index` and `value` but dropped the
  CAS `expected` operand, so a fusion that reordered a scalar the `expected`
  reads was not blocked. Now walks `expected` (and is exhaustive over leaf
  variants); proven by a `reference_eval` oracle differential.
- **LAW 1 placeholder in `vyre-libs::matching::substring_search`**  -  the
  inner-byte check was `Expr::u32(1)` (matched every position); now
  `load(haystack, i+k) == load(needle, k)` routed through a select to
  stay integer. Gap L-7 closed with a structural regression test that
  fails if the compare ever collapses back to a constant.
- **LAW 9 evasion audit sweep**  -  removed all `// TODO` / `// FIXME`
  markers from shipped code. Subgroup intrinsics return a structured
  error pointing at RFC 0004 instead of a TODO; the autotune workgroup
  heuristic is documented as intentional default instead of a TODO.
- **Driver binary name collision**  -  `vyre-driver-wgpu`'s CLI bin
  renamed from `vyre` → `vyre-wgpu` so it no longer collides with the
  `vyre` lib target in `cargo doc`.
- **Workspace version drift**  -  `vyre-runtime` workspace dep bumped
  from `0.1.0` → `0.6.0` to match the crate's own manifest.
- **`vyre-libs::security::aliases_dataflow` RAW-hazard barrier gap.**
  The local `merge_programs` helper concatenated the seed / hop /
  merge / intersect / union sub-programs without inserting any
  `Node::Barrier`. Threads in later warps observed pre-seed
  `reach_x_buf` state and the BFS frontier silently dropped nodes
  past the warp boundary on every aliases-using rule. Routed
  through `vyre_foundation::execution_plan::fusion::fuse_programs`
  so RAW/WAR hazards get precise barriers. Local helper deleted.
  Two regression tests pin the structural barrier presence and
  unique non-Workgroup binding numbering in the fused output.
- **`vyre-libs::parsing::python` validator-rejected programs.**
  Lex-level `is_ident_start` / `prev_identish` lets stored bool
  exprs that the validator rejected when later compared with
  `u32(0)`; coerced through `select` so the bool→u32 lift happens
  at the let_bind. Structure / call / decorator extractors hoisted
  every cross-block name (`name_end`, `cursor`, `dot_pos`,
  `after_dot`, `target_tok`, `target_name`, `target_kind`,
  `async_def`, `after_decorator`, `after_type_params`, `after_params`,
  `decorator_end`) into the outer body so they outlive the
  if-then blocks that assign them, with new
  `search_next_token_into` / `find_matching_delimiter_into`
  assign-only helpers used inside if-blocks to skip the redundant
  outer let_bind. Closes 13 cascading V008 / V032 / undeclared-var
  validation errors that hid behind a single bool/u32 mismatch.
- **`vyre-primitives::reduce::workgroup_tree`** E0382 use-of-moved-
  value on `dtype: DataType` consumed three times in a single
  `Program::wrapped` BufferDecl block; first two uses now
  `dtype.clone()` so the third use lands on the still-owned value.
- **`vyre-primitives::effects::handler_apply::tests::from_bits_round_trip`**
  literal `0b101_0011` corrected to `0b0010_1011` (bits 0, 1, 3,
  5 = BufferWrite + Atomic + GpuDispatch + AsyncLoad). The pre-fix
  literal had bits 0, 1, 4, 6 set (Atomic + Barrier + Trap) but
  the assertions read GpuDispatch / AsyncLoad → guaranteed test
  failure regardless of the runtime behavior.
- **`vyre-libs::nn::attention::attention_reference_program`** signature
  drift: the function returns `Program` but the body used `?` /
  `Ok(...)`, which only compile under a `Result<…>` return.
  Reverted to panic-on-overflow (callers wanting the fallible path
  go through `try_attention_reference`, which already returns
  `Result<Program, TensorRefError>`).

### Changed

- **Driver boundary and shared-driver lifts.** Concrete backend crates now own
  concrete runtime/API names, while `vyre-driver` hosts shared AOT emitter
  registration, validation cache, binding/program walks, specialization maps,
  tuner framework, subgroup taxonomy, and cross-dispatch fusion decisions.
  Public API snapshots were refreshed for the resulting shared surfaces.
- **Frozen/public API snapshots refreshed.** Snapshots now reflect the
  intentional 0.6 contract surface for borrowed output reuse, borrowed async
  dispatch, subgroup visitors, required lowering implementations, categorical
  laws, and the current published public items for driver/wgpu/foundation/
  primitives/spec crates.
- **`vyre-foundation` program-shape analysis surface.** Public snapshots now
  include `program_shape_facts`, the reusable buffer-shape analysis used by
  optimizer passes and downstream cache consumers.
- **`Node::forever(body)`** helper in `vyre-foundation::ir::Node`. Linus
  principle  -  `forever` lowers to `Node::Loop { 0..u32::MAX, body }`,
  no new enum variant, no cascade of match arms. Persistent kernels
  use it.

## [0.6.0]  -  2026-04-19
(layered workspace: foundation → driver → ops; single inventory registration path)

### New in 0.6.0

- **Nine-crate layered workspace.** Extracted `vyre-foundation` (IR, wire format, visitor traits, extension resolvers), `vyre-driver` (registry, runtime, pipeline, routing, diagnostics), `vyre-driver-wgpu` (wgpu backend, buffer pool, bind-group cache, pre-recorded dispatch), `vyre-driver-spirv`, `vyre-ops` (stdlib dialects), from what was a single god-crate. `vyre` remains as a back-compat meta shim.
- **Machine-checked layer DAG.** `scripts/check_layering.sh` enforces R1–R3+R5 from `COMPUTE_2_0.md §3`: foundation has no driver/ops/backend deps, driver has no ops/backend deps, ops has no backend deps, reference has no backend deps. Cross-layer imports go DOWN only; violations fail CI.
- **True IR openness.** `Expr::Opaque` and `Node::Opaque` now round-trip through the wire format (tag `0x80`) via inventory-registered `OpaqueExprResolver` / `OpaqueNodeResolver`. Validator, optimizer passes, and visitor adapters all honour Opaque explicitly  -  no wildcard fallthrough remains in foundation transforms.
- **Single op registration path.** `inventory::submit!{OpDefRegistration::new(...)}` is THE way to publish an op. `OpSpec` surface is gone; `DialectRegistry` is the frozen index.
- **Zero-alloc dispatch hot path.** `bound_handles` returns `SmallVec<[_; 8]>`, bind groups cache keyed by bound-buffer identity, buffer pool recycles power-of-two allocations across dispatches.
- **`vyre-reference` Memory** replaced `HashMap<String, Buffer>` with `BufferMap` (`SmallVec<[(Arc<str>, Buffer); 8]>`)  -  branch-predicted inner-loop lookups, no per-access SipHash, no per-name `String` allocs. `LocalSlots` interns via `FxHashMap<Arc<str>, _>`.
- **Invariant catalog truthful.** Every descriptor in `vyre-spec/src/invariants.rs` now references a real file at `conform/vyre-conform-enforce/tests/invariants.rs`, enforced by `scripts/check_invariant_paths_exist.sh`.
- **Ratchet CI gates.** `scripts/check_no_string_wgsl.sh` caps Law-B string-WGSL violations at 54 and `naga::front::wgsl::parse_str` sites at 84. `scripts/check_warning_budget.sh` caps workspace warnings at 921. Each gate decreases only; regression fails CI.

### Breaking

- Op registrations must go through `vyre-driver::registry::OpDefRegistration`. Consumers using legacy `OpSpec` surface must migrate.
- `vyre-core/src/` is reduced to `lib.rs` (meta-shim re-exports). Files that reached into `vyre::ir::transform::...` etc. must import from `vyre_foundation` directly  -  the meta-shim still provides the `vyre::ir::X` paths for surgec/pyrograph/warpscan consumers.

## [0.5.0]  -  2026-04-19
(substrate-neutral IR: open extensions + conform certificates)

### New in 0.5.0 final

- **VIR0 wire-format spec published**  -  `vir0-spec.md` at repo root declares the wire format stable across 0.5.x, reserves the `0x80..=0xFF` tag range for third-party extensions in perpetuity, and documents conformance requirements for non-Rust bindings (Phase 22).
- **Bytes extraction validation**  -  `BufferDecl::with_bytes_extraction(true)` opt-in relaxes V013 on load/store of `DataType::Bytes` buffers for legitimate bytes-producing ops like `decode.base64`, `compression.lz4_decompress`, and the decoder family. `Signature` gained `#[non_exhaustive]` + `bytes_extraction` field + `bytes_extractor` constructor (Phase 3).
- **Canonicalized 7 primitive programs** to match the emit-asserted WGSL shape  -  `abs_diff` routes through `max(a,b) - min(a,b)`, `div` / `mod` wrap in zero-guard `select`, `logical_not` uses boolean-style `select(x==0, 1, 0)`, `negate` uses two's-complement `~a + 1`, and `shl` / `shr` zero-guard shifts `>=32` (Phase 2).
- **photonic backend crate** lives in `backends/photonic/` as a registered non-dispatching substrate with `supports_dispatch = false`  -  proves the three-substrate surface claim today, while photonic compute remains future work.
- **SPIR-V backend skeleton** in `backends/spirv/`  -  `SpirvBackend::emit_spv` consumes `naga::Module` built by the shared builder family and calls `naga::back::spv::write_vec`, giving vyre a second real compute-capable backend alongside wgpu (Phase 14).
- **Conform crates scaffolded**  -  `vyre-conform-spec` (witness sets + composition laws), `vyre-conform-generate` (proptest-style shrinking minimizer), `vyre-conform-enforce` (algebraic-law prover over witness pairs), `vyre-conform-runner` (CLI + Certificate schema) at `conform/vyre-conform-*` (Phase 17).
- **rules/op/ certificate library**  -  5 op certs (`decode.base64`, `compression.lz4_decompress`, `match.dfa_scan`, `string_matching.aho_corasick_scan`, `graph.bfs`) plus `SCHEMA.md` defining op_id / signature_blake3 / allowed_backends / witness_set_blake3 / laws metadata (Phase 4).
- **NFA bytecode micro-interpreter fully retired**  -  the remaining `nfa_scan` kernel was deleted in the 2026-04-19 zombie sweep, README/CHANGELOG/VISION cross-references scrubbed, scan and lexical ops now compose in vyre IR end-to-end (Phase 7).
- **Docs**  -  `docs/THESIS.md`, `docs/ARCHITECTURE.md`, `docs/memory-model.md`, `docs/targets.md`, `docs/wire-format.md` authored as load-bearing spec.

### Breaking

- `Signature` is `#[non_exhaustive]`  -  out-of-crate literal construction must move to `Signature::bytes_extractor(...)` or `Signature { inputs, outputs, attrs, ..Signature::default() }` equivalent.
- `BufferDecl` gained the `bytes_extraction: bool` field; source-compatible through the builder API (`::read`, `::output`, `::read_write`, `::storage`, `::workgroup`), but direct struct literals must set it.

### Fixed

- `all_primitives` arithmetic / bitwise assertions now see the canonical WGSL shapes emitted by `naga_emit`  -  `abs_diff`, `div`, `mod`, `logical_not`, `negate`, `shl`, `shr` all validate against the assertion set.
- V013 no longer blocks valid decode / decompress flows that read and write typed `Bytes` buffers.
- README no longer describes a bounded `nfa_scan` bytecode micro-interpreter; it was deleted.

### Substrate (Claude)
- core: structured `Diagnostic` API with stable `E-*` / `W-*` codes,
  rustc-style human render, JSON round-trip for LSP / CI integration
  (A-C1b).
- wire: rev 3 framing  -  schema version bumped to 3 with structured
  `Error::VersionMismatch { expected, found }` replacing string-based
  version mismatch (A-C2).
- dialect: op versioning + migration table (`Migration`,
  `Deprecation`, `AttrMap`, `Semver`) via `inventory::submit!`; chain
  resolution + deprecation diagnostics (A-C2b).
- perf: `BENCHMARKS.md` performance contract  -  10 targets, numerical
  stability per-op ULP bounds, regression gate spec (A-C14b).
- optimizer: `AdapterCaps` + `PassCtx` + `AnalysisCache`; typed-error
  conversion from `PassSchedulingError` to `Diagnostic` (A-C7b part 1).
- core: runtime introspection API  -  `dialects()`, `ops()`, `backends()`,
  `lowerings()`, `coverage_matrix()` (A-C11b).
- docs: op-id stability catalog + regen-on-demand gate
  (`docs/catalogs/op-id-catalog.md`); coverage matrix + regression gate
  (`docs/catalogs/coverage-matrix.md`) (A-B4d, A-C11b).
- scripts: layout / file-size / mod.rs-size / prelude / readmes CI
  law scripts under `scripts/laws/` (A-C11c part 1).

### Dialects (Gemini A)
- core: dialect foundation types  -  `OpDef`, `LoweringTable`,
  `DialectRegistry`, `InternedOpId`, `BackendRegistration` (A-B0).
- core: every Cat C intrinsic migrated to `naga::Module` builders  - 
  91 ops, zero shader assets remain in op trees (A-B1).
- core: primitive Cat A ops migrated; KAT coverage for 7 previously-
  missing programs (A-B2).
- core: `io` dialect  -  4 Cat C zero-copy intrinsics
  (`io.dma_from_nvme`, `io.write_back_to_nvme`, `mem.zerocopy_map`,
  `mem.unmap`) registered with no backend opt-in (B-B3 scope).

### Backends (Gemini B)
- wgpu: dispatch via `DialectRegistry.get_lowering`  -  `OpSpec::intrinsic`
  read path removed (B-B1).
- wgpu: `impl Executable` + `impl Compilable` for `WgpuBackend` with
  `WgpuIR` progressive-lowering artifact (B-B5).
- reference: `dialect_dispatch` module routes op ids through
  `DialectRegistry.get_lowering(CpuRef)` (B-B4).

### Performance (Gemini C)
- wgpu: lock-free `BufferPool` via crossbeam; `PrerecordedDispatch`
  pre-recording (C-B1).

### Pre-existing (landed earlier in the cycle)
- core: blake3 fingerprinting for IR stability and cache invalidation (MOD-008)
- core: arena-backed reference interpreter (P-2)
- runtime: zero-copy output-slice readback (P-5)
- runtime: streaming chunked dispatch (P-7)
- validator: tightened atomic indexes, fma/select typing, mixed arithmetic typing, and u64 bitwise-unary acceptance (VAL-001..004)
- conform: widened overflow-contract surface for primitive arithmetic regression coverage (CONF-001)
- conform: added build-scan regression coverage for generated operation metadata (CONF-002)
- wire: added depth-cap regression coverage for hostile nested IR blobs (EDGE-001)

### Changed
- `vyre-conform::specs::primitive` now walks `vyre::ops::registry` for every `primitive.*` op and builds specs from core metadata plus normalized `rules/kat/primitive/<family>/<op>.toml` vectors. Legacy per-op modules that were not present in the core registry, including `logical_and`, `logical_or`, `logical_xor`, `logical_nand`, `logical_nor`, `avg_floor`, `wrapping_neg`, and `popcount_sw`, were removed rather than kept as conform-only specs.

## [0.4.0-alpha.2]  -  2026-04-17

### Added
- Architecture and process contracts were formalized with `ARCHITECTURE.md`, `rules/SCHEMA.md#kat`, and `docs/PRIMITIVES.md`, giving a stable contributor contract for frozen traits, op classification, and community rulesets.
- New publishable package structure was established: `vyre-spec` (`0.1.0`) and `vyre-build-scan` (`0.1.0`) plus release-ready crate metadata for the workspace surface.
- Conformance foundations landed for this release with canonical `CpuOp` CPU reference plumbing in `core::ops::cpu_op`, `conform` pipeline cleanup, and the move of `reference` into `vyre` so evaluator semantics and wire-era tooling are co-located.
- Benchmark and evidence publishing pipeline landed: `primitives_showcase` entrypoint, `benches/RESULTS.md`, and synchronized benchmark presentation in README + book.

### Changed
- DeepPerf wave cleanup converted temporary tree-gen and generated-cruft artifacts into a stable one-file-per-op structure, including conform command/layout simplification and generated module deduplication.
- Core/conform import surfaces and type contracts were adjusted for category and registry stability, including `Category`/`IntrinsicTable` migration into `vyre-spec` and elimination of brittle cross exports.
- Documentation and validation semantics were tightened: `Fix:`-prefixed actionable diagnostics, contract-first doc language, and release-oriented invariant text for affected public surfaces.
- Package and build metadata was harmonized for publishability and release continuity.

### Fixed
- Fixed immediate compile/dependency coupling regressions from the prior refactor wave by removing dead or misleading generated surfaces and restoring stable compile boundaries.
- Fixed benchmark evidence drift by rebaselining published values from `benches/RESULTS.md` and aligning user-facing benchmark tables.
- Fixed stale release-state items by auditing all open coordination entries and refreshing statuses with explicit reopen criteria.

### Perf
- DeepPerf benchmark capture completed for primitive ops across 1K/10K/100K/1M element sizes with CPU and GPU end-to-end timings, crossover annotations, and the full 48-op table in `benches/RESULTS.md`.
- Preserved the end-to-end performance gate by excluding structural hacks and ensuring benchmark coverage remains tied to committed results data.
- Captured remaining hotspot context for future release polish (`gcd`, `lcm`, and uncovered KAT boundary classes) in coordination notes for targeted follow-up.

## [0.4.0-alpha.1]  -  previous

### Added
- Workspace merge of `vyre` core and `vyre-conform` into a single workspace.
- `SANTH_STANDARD.md` and `template_op.rs`  -  standardized contributor template for adding new ops (8fa6ab6, 436264b).
- `automod` wired across all op categories (bitwise, math, reductions, data_movement, string, scan, sort, encode, stats, buffer, compiler_primitives, rule, decode, match_ops, string_similarity, graph, workgroup, security_detection, hash) (c6af953, c4ab1f7, a39a9c5).
- CI workflow for check + clippy + doc (3c57a49).

### Changed
- Core consolidated from ~2000 files down to 1117 files with 0 compile errors (0956373, 5b6e1e5, 436264b).
- Conform merged and consolidated from 3645 files down to 883 files with 0 compile errors (09a6496).
- GPU feature gates stripped from conform; conform now assumes GPU is always available (ac760a8, b1b7991).

### Fixed
- Original 80-entry op registry restored after agent overwrites (b1b7991).
- Tree-gen damage consolidated and reverted where it broke the module graph (ade08d5, c91ad8c, 35f7342, dd71607).

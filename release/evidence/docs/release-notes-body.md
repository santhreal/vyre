# vyre-v0.7.2

Vyre 0.7.2 releases from candidate tag `vyre-v0.7.2-rc.1` and final tag `vyre-v0.7.2`.
Backend crates carried at that version: `vyre-driver-cuda@0.7.2`, `vyre-driver-wgpu@0.7.2`.

### Added

- The docs-register gate reads docs/REGISTER.toml and fails on a banned
  register phrase, an em dash, or a host-local build setting in any authored
  Markdown page, and on a repository-root page the documentation manifest does
  not declare.
- The `contract-in-source` gate reports a comment that defers its contract to a
  published document instead of stating it. A pointer costs a reader a second
  file and outlives the file it names: when the book those comments pointed
  into was deleted, every pointer became a pointer to nothing and no gate went
  red. The rule lived in a test in an unrelated crate and shelled out to git;
  it now reads the tracked source set through the scanner, reports the file and
  line, and is exercised on 4511 files.
- The CI registry keeps a row for every workflow path the tree carries or once
  carried, and the row says whether it runs, is paused with a way back, is
  superseded by the workflow and gate that run its checks, or leaves a
  verification class uncovered. Seven lanes were deleted in one commit and
  nothing went red, because the wiring is derived from the tree and a deleted
  lane derives to nothing: adversarial input generation, CODEOWNERS
  trust-boundary coverage, mutation coverage of the verifier and the bitset
  primitives, and repeat-dispatch determinism on CUDA are recorded as uncovered
  rather than gone, the catalog, book and randomized-order lanes name what runs
  their checks now, and a workflow deleted from here on leaves a row that fails
  the gate until someone records where its checks went.
- The gates sweep now reads the gate sources and fails a registered gate whose
  run path cannot construct a finding, so a gate that only prints notes can no
  longer pass as coverage. A gate whose honest output is a note is declared
  with the gate that carries its failing form, and the sweep also fails a
  declaration whose gate can find or whose name is not registered.
- The `ci-required` gate holds the required status contexts to the workflows
  that define them: every context resolves to a job by display name or job id,
  every workflow carrying one runs on pull requests and on pushes to the
  protected branch without a path filter, and a fan-in job that always runs
  reads a dependency result and exits nonzero. Those six assertions used to run
  only when an operator applied branch protection by hand, so they ran on no
  schedule at all. The fail-closed rule now reads the job that opted into
  always running rather than searching the whole file, where an unrelated build
  step that exits nonzero satisfied it. The operator script keeps the API
  mutation and runs the gate before it applies anything.
- The `frozen-contracts` gate reads the snapshot directory and reports a
  snapshot no frozen contract claims. The frozen set is a table in the gate and
  the snapshots are files on disk, so deleting a row left its snapshot behind,
  where it read as a frozen declaration that nothing compares against. The gate
  sweep reports a workflow step that names a script the checkout does not
  carry, which is what every script this campaign deletes leaves behind: a step
  that fails at run time under a name that still reads as coverage.
- The ci-steps gate resolves every package, test, bench, example, binary and
  feature token in the workflow steps and the scripts against the workspace
  manifests, so a step that silently selects nothing fails.
- The source-include-module gate reports include! of a tracked Rust file. A
  pasted file has no module path, so a name it defines cannot be qualified, its
  items sit in the including module's namespace, and the reachability rules
  read it as an orphan. The tree reached zero such includes by converting the
  files to modules and nothing held that line. A build script include of a
  generated path under OUT_DIR names no tracked file and is not a finding.
- hygiene-matrix bounds the panics nothing else answers for. A panicking call
  whose function documents a # Panics section is a contract, and one on a hot
  path is a release blocker; between them sat every panic that is neither,
  which no gate measured. docs/testing/PANIC_BUDGET.toml records a ceiling per
  crate, measured at 35 sites across seven crates, and the gate reads the
  population out of its own classification. Over the ceiling blocks. A crate
  carrying such a panic with no row blocks, so a new crate cannot arrive
  unmeasured. A ceiling left above a crate that reached zero blocks, because
  that row is what stands between the crate and the next panic added to it. A
  count under its ceiling and still above zero is a note carrying the number to
  write, since a gate that fails on the improvement it asks for is one that
  gets switched off. The conformance lenses spent the first three: an expect on
  a missing neutral builder, in three functions that already had a failure
  channel, now returns that failure with the operation named.
- The reference and SPIR-V backends carry a tracked hostile-input closure
  target, `vyre-driver-reference/tests/hostile_input_closure_contract.rs` and
  `vyre-driver-spirv/tests/hostile_input_closure_contract.rs`. Both backends
  previously held their adversarial case in a file name the repository ignores,
  so the assertion ran only where someone had generated it and no branch could
  review it. The obligations come from `vyre_driver::hostile_input_closure`,
  and the ignore rule that asked for a crate-owned name is now satisfied for
  these two crates.
- `vyre_test_support::binop_parity::assert_covers_every_synthetic_op` and
  `assert_covers_every_total_op` fail when a backend parity suite has no
  reference arm for an op the shared table declares, or names one the table
  does not. Before them each suite carried one hand-written test per op, so a
  row added to the table was dispatched by neither backend and nothing failed.
  The op set is read from the table at run time, so a row added tomorrow is
  required of every suite tomorrow.
- The vyre-safetensors adapter reads bounded safetensors headers and sharded
  indexes without reading tensor payloads, confines shard paths to the
  checkpoint root, rejects duplicate or unmapped tensors, validates
  caller-supplied dtype and shape requirements, and streams complete shards
  against an exact trusted BLAKE3 set before returning an immutable checkpoint
  identity.
- `vyre-libs` gates the row contract the C typedef-annotation family agrees on,
  deriving its member set at run time from the `pub use` re-exports of
  `vyre_libs::parsing::c::parse::vast::typedef_ann` so a new builder is red on
  its first run until someone records what it writes. No builder may declare a
  zero-length buffer for an empty node table, every builder must size its row
  tables by the one shared stride at several row counts, and every pass must
  carry every VAST field it does not declare that it writes, checked by running
  each program on a table whose carried fields all differ per row.
- `c_frontend::parity_matrix::assert_case_table_covers_fixture_file` reads a
  fixture family's own source at run time, collects the fixture builders it
  declares, and fails when the family's `CASES` table does not name one of
  them. Adding a construct and forgetting to enumerate it used to leave it
  proven on no backend and indistinguishable from a construct that passed,
  which is how GNU `__restrict` normalization stayed unproven on every device.
  A companion contract reads the fixture directory itself, so a new family is
  red until it either has a case table or is recorded as proven another way
  with the reason.
- `vyre_libs` runs every C-AST parity case on the reference interpreter through
  `vyre_driver_reference::CpuRefBackend`, dispatching the same programs the
  driver arms dispatch. A parity failure used to be observable only on a
  machine with a working adapter, so a case no arm named looked exactly like a
  case that passed. Splitting the arms separates a kernel that disagrees with
  its oracle, which fails here, from a device that disagrees with the kernel,
  which fails only there.
- The neural library now executes floating channel-major depthwise causal
  convolution with exact left padding, masks, bias, SiLU, F16/BF16 conversion,
  and output truncation. A short-chunk route emits an explicit next-state
  generation whose outputs and tail match full prefill across arbitrary token
  partitions and reset.
- The neural library now executes chunk-size-64 gated delta prefill with F32
  cumulative log-decay, a strict lower-triangular solve, initial-state
  correction, chunk output reconstruction, and explicit final matrix state.
  F16, BF16, and F32 inputs retain F32 internal math. Guarded rows in the final
  structural tile cannot read padding, change state, or appear in truncated
  output.
- xtask/ci-registry.toml declares which subsets hold each gate, which workflows
  run it, every check CI runs that is not a gate, and every paused workflow
  with the condition that ends the pause; the ci-registry gate holds that
  declaration to the registry, the subsets and the workflow steps in both
  directions.
- A gate resolves every name a `run:` step passes to cargo or to a shell
  against this tree: each `-p` package is a workspace member, each `--test` and
  `--bin` target is declared or auto-discovered by the package the same command
  line addresses, and each `scripts/` path is published. Steps are extracted by
  indentation and each is its own scope, so a target is never resolved against
  a neighbouring step's package. Nothing is listed in the test: a workflow
  added tomorrow is judged tomorrow, and renaming a target without updating its
  workflow is red locally instead of in CI.
- A gate over `vyre_libs::graph` rejects the argument-transposition class. It
  walks the module directory and parses every declared signature at run time,
  so a CSR entry point added later is covered without editing a list: closure
  entry points must receive the graph as a bundle and must not declare three or
  more consecutive parameters of one type, and the wider slice-taking family
  must give each role a single name across the tree.
- The neural library now executes a reusable dense gated-MLP ProgramGraph with
  learned RMSNorm, checkpoint-native output-major gate and up projections, F32
  SwiGLU math, output-major down projection, and residual addition. F16, BF16,
  and F32 storage use F32 normalization, projection accumulation, activation,
  and residual arithmetic with source-dtype boundaries.
- The docs-coupling gate holds an authored page to the code it answers for.
  Every current authored page row in docs/DOCS.toml declares covers, the source
  paths it states the content of, and both sides of the check derive from that
  declaration at run time: a page without covers is a finding, a covers entry
  matching no path is a finding, a repository path a page cites inside a code
  span or fence and the tree does not hold is a finding, and a diff that
  changes a covered path without changing its page, or without any changelog
  fragment, is a finding. It ran red on a real broken pointer on first use, a
  parsing chapter citing vyre-libs/src/parsing/vast as a directory when the
  module is a file. The gate is in the docs subset, which the gates workflow
  already runs, and its pinned count is zero. With no base ref and a clean
  worktree it reads an empty diff, so the coupling rule contributes nothing and
  a push event answers on the other three. A base ref the checkout does not
  hold, which is what a shallow clone gives a pull request, is one finding
  naming the ref and the fetch that fixes it, rather than a gate that could not
  run: that would take a whole workflow red over a fetch depth, and an empty
  diff would leave the coupling rule unenforceable in the only environment it
  exists for.
- Four subsets group the gates no workflow named: `composition`, `structure`,
  `docs` and `ir`. The gates workflow runs each as its own step, so a red gate
  is addressed to the owner of that domain instead of arriving inside one
  whole-registry log. The whole-registry sweep stays as the backstop for a gate
  that belongs to no subset.
- Every workspace member carries a SPEC.md naming what it owns, what it must
  never contain, the edges that cross it, the direction that may not reverse,
  its invariants and the gates that enforce them, and a README.md; the
  crate-pages gate derives the roster from the workspace members and fails on a
  missing page, an undeclared edge or a module map naming a path the crate does
  not have.
- The `example-capability` gate builds every crate under `examples/` outside
  the workspace and runs what it asserts. Subjects come from the tracked tree,
  so a new example is covered when it is added: a directory with files and no
  manifest is reported, a manifest without its own `[workspace]` table is
  reported, a standalone crate needs a committed lockfile and passes `cargo
  test --locked`, a crate with a binary is also run, and a `Cargo.toml.liquid`
  scaffold is rendered against this checkout and its tests run. A template
  placeholder the gate has no value for is reported instead of substituted.
- A source-inspecting test is informational only when
  `docs/testing/STRUCTURAL_GATES.toml` declares it by file and test name with a
  reason, and a row the tree no longer backs is itself a release blocker.
  Twelve gates assert a property with no run-time witness, such as which crate
  owns a symbol or that no second file spells a constant; they cannot be
  rewritten as behaviour tests, and blocking on them permanently would have
  been answered by deleting them. Keying on the pair means a reviewed
  declaration exempts the gate it names and not the next one added to the same
  file.
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
- The IR now supports first-class Tile values with explicit element type,
  static extents, layout, and hardware residency, introducing dedicated
  TileLoad, TileStore, TileMatmul, TileReduce, TileElementwise, and TileDecl
  statement nodes, capability validation, reference interpreter execution, and
  VIR0 schema version 7 serialization.
- `vyre_foundation::fp_parity::max_output_ulp` reports the largest ULP distance
  over a program's declared F32 output slots. Output-slot alignment is one
  internal walk shared with the tolerance comparison. The conform ULP audit
  carried its own copy of that walk: it decided per slot from
  `program.buffers()` while reading `output_buffer_indices` for the mapping, so
  a program whose outputs were not declared in slot order was audited against
  the wrong element type, and a saturated distance read as a measurement rather
  than as an incomparable pair. The gate and the audit now read the slots
  through one owner, and the one place they legitimately differ is documented
  on the function: an audit treats two NaNs as agreeing because a backend
  chooses its own payload, and the gate does not.
- The neural library now provides gated RMSNorm with float32 accumulation,
  source-dtype rounding, learned scaling, float32 SiLU gating, and exact
  last-dimension row isolation. The reference interpreter now executes
  canonical F16 and BF16 loads, stores, and F32 conversions with
  round-to-nearest-even semantics while preserving raw element-byte APIs.
- `vyre_test_support::case_table::ArmCoverage` reads a shared case table's
  declared group names at run time, records which groups a crate's arm actually
  asserted, and fails naming every declared group that crate has no branch for.
  A corpus shared through an include has a hole a per-crate copy also had: the
  table declares a group, one crate grows an arm, the other does not, and the
  crate without the arm still passes because nothing in it mentions the group.
  Group-count and case-count floors make a collapsed table fail rather than
  report a clean sweep of an empty set. Both the dense-matvec and exploded-IFDS
  tables are enrolled, in four arms across two crates.
- `graph_primitive_binding_contracts` reads the operation registry at run time
  and asserts that every registered graph primitive declares bindings that are
  the contiguous range from zero, and that a `changed` convergence flag sits
  read-write directly above the read-write frontier it reports on. A count of
  zero is deliberately not a signal: it is the documented sentinel for a
  runtime-sized storage buffer.
- Grouped affine INT4 linear now provides a typed batched program builder that
  dequantizes each immutable weight tile once and reuses it across independent
  resident batch rows. Release evidence measures normalized per-inference
  latency.
- Five operations now own the pieces the Jacobi eigensolver used to spell
  inline: `givens_rotate_pair` rotates one strided element pair,
  `jacobi_apply_rotation` applies one rotation at a pivot to a matrix and its
  accumulator, `matrix_identity_fill` seeds a rotation accumulator,
  `matrix_diagonal_extract` reads a diagonal out, and `eigenvector_column_sign`
  fixes the sign of every eigenvector column. `symmetric_eigen_jacobi` composes
  them and emits identical numerics; the three column, row and accumulator
  rotations inside one Jacobi step were byte-identical five-node loops
  differing only in base offset and stride, and are now one builder with two
  address parameters.
- The neural library now provides last-dimension L2 normalization for grouped
  query and key heads. It accumulates sum-of-squares in F32, applies epsilon
  inside the canonical inverse-square-root contract, isolates rows exactly, and
  converts output once to F32, F16, or BF16.
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
- `loop_legality_collector_closure` plants a scalar read, a buffer load, a
  store and a binding in every operand and body slot of every declared `Node`
  variant and asserts each read set reports it. The variant set comes from
  `NODE_VARIANT_NAMES` through the shared IR fixtures, so a new variant is red
  before any of the three questions is asked.
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
- The neural library now composes F32 query and key normalization,
  cache-position partial rotary embedding, explicit query-to-KV head grouping,
  and dynamically bounded causal attention in one typed ProgramGraph. Prompt
  and cached-decode routes exclude future cache rows and support configurable
  head ratios, head widths, and rotary dimensions.
- `docs/book.toml` builds `docs/` as one mdBook. `src` is the documentation
  directory itself, so the navigation mdBook reads is the `SUMMARY.md` that
  `xtask docs-check` renders from `DOCS.toml`, and there is no second table of
  contents to keep in step. `create-missing = false` turns a chapter the
  navigation names and the tree lacks into a build failure instead of an empty
  page. `docs/archive/0.7-2026-08-15/` holds the frozen 0.7 book: the two
  authored pages it still carried, plus the thirty pages deleted on 2026-08-12
  recovered from history, each registered `status = archived` with `nav =
  false`, so the record is one directory rather than a directory plus a commit
  range. The documentation gate already excludes archived pages from navigation
  and from link resolution.
- The documentation is one book of short chapters under docs/, each stating one
  concern with its examples first: guide/install, guide/first-program and
  guide/backends for a caller; architecture/crates, architecture/artifact,
  architecture/compile-search and architecture/parsing for a contributor;
  reference/values, reference/operations, reference/diagnostics and
  reference/wire-format for the API surface; extending/operation and
  extending/backend for an out-of-tree author; conformance/program and
  release/process for the two lifecycle contracts. Every claim in them is read
  off the current source: the backend acquisition example now names
  vyre_driver::acquire rather than a pub(crate) module path, the wire envelope
  is recorded as the VYRE magic tag at schema version 6 reading versions 4
  through 6, and the validation catalog is recorded as 96 rules across eight
  phases.
- The crate-structure gate now fails when a src/ module file sits beside a
  directory of its own name, and when a module or binary name states no
  contract (common, core, helpers, misc, types, utils, or an _ext suffix). It
  judges every crate in the checkout, found by walking for a manifest that
  declares a package, so a crate outside the workspace roster is judged too. A
  module the committed public-API snapshot publishes keeps its name while it
  stays published, because renaming it renames a path consumers import; a
  binary root has no module path and is judged by the name a reader types to
  run it.
- `xtask public-api-paths` measures, per crate, how many items are published at
  more than one path, and pins the number. A crate that declares `pub mod
  inner` and re-exports what it holds publishes every one of those items twice;
  both paths compile, both are documented, and nothing says which one a
  consumer should write. The first measurement over the committed snapshots is
  4064 such items across 26 crates, with `vyre-foundation` at 789,
  `vyre-runtime` at 735 and `vyre-driver-cuda` at 642. A crate with no row is a
  finding, so a newly published crate is red rather than unjudged, and
  `--write` lowers a recorded number to what it measured and never raises one.
- `vyre-foundation` has a criterion benchmark over the whole optimizer pass
  pipeline: the eight release corpus families, wide kernels at 16 and 64
  buffers, and a 4x8 loop nest. The pipeline had per-pass timing inside the
  scheduler report and no measurement of the pipeline as a whole, so a rewrite
  of the walk every pass runs through had nothing to be measured against.
- `memory_pass_alias_owner` reads the pass set from the inventory registry at
  run time, keeps every pass in the memory phase, and asserts none of them
  changes how many times a program reaches a buffer across a gap node the alias
  owner reports as interfering. Each probe carries a control with a harmless
  gap that at least one pass must rewrite, so a pass that simply had nothing to
  do cannot pass for one that consulted the owner. Registering a third memory
  pass with its own copy of the analysis turns the suite red on the day it is
  registered rather than on the day someone diffs two files.
- `the_optimizer_expression_rewrite_reaches_every_operand_slot` puts the
  optimizer's expression rewrite inside the owner-closure suite that previously
  checked only the three reference-mode walks. It plants a uniquely-bound
  literal read in one operand slot at a time, taking the slot set from the
  variant registry rather than from a list in the test, and requires the
  registered propagating pass to fold it. Reintroducing the pre-collapse
  private walk fails it at the async-copy offset, which is the position the two
  copies actually disagreed about.
- Every published FNV-1a64 program builder, the slot-precise `arg_of_slot`
  traversal, and both queued-row CSR traverse delegating forms now have parity
  coverage. The FNV-1a64 and delegating-form member sets are derived from the
  source at run time, so a new member fails the suite instead of shipping
  unproven.
- `scan_prefilter_width_closure` reads the `PrefilterWidth` variants out of the
  width table's own source at run time and fails when a width has no recorded
  dispatch ABI, when a recorded row names a width the source no longer
  declares, when a shipped program's bindings disagree with the row's counter,
  mask and sink layout, when a declared mask is never read by the emitted body,
  and when the widths stop forming a mask prefix chain. An empty or unparsable
  member set panics rather than passing, because an empty set makes every
  closure assertion vacuous.
- ProgramGraph now derives one versioned, domain-separated BLAKE3 identity from
  canonical topology and Programs, typed port contracts, artifact schema,
  validated model configuration, exact symbolic bindings, and verified
  immutable-weight identities. Mutable sequence-state contents are excluded, so
  cache growth reuses compiled artifacts while any executable or provenance
  change invalidates the key.
- The neural library now executes recurrent gated delta attention with F32 Q/K
  normalization, grouped heads, scaled queries, exponential decay, sigmoid
  beta, F32 matrix-state continuation, source-dtype output, and explicit
  validated prior and next state generations.
- `vyre_reference::reference_eval_lane_rotated` executes a program with the
  workgroup and invocation step order rotated left by a caller-chosen amount.
  Reversal is a symmetric permutation, so an implementation that confuses lane
  identity with step position can be made reversal-symmetric and stay wrong; a
  rotation separates the two and catches a subgroup collective that resolves
  its peers by physical step position.
- A registration behind a feature the registry walker does not enable is
  invisible rather than absent, so nothing was red when one fell out. Two rules
  in `xtask-registry` close that class, both derived from the tree at run time
  rather than from a list: every file that submits a registration is reachable
  through the walker's declared feature selection, resolved transitively
  through the source crate's own feature table, and every feature that gates a
  registration is one the source crate's widest aggregate turns on. A new
  domain feature carrying registrations turns both red until the aggregate
  covers it, and a consumer naming the aggregate then needs no change.
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
- `xtask-registry` asserts that every crate submitting an operation
  registration in source contributes at least one operation to the live
  registry. Dropping one of those links left the registry answering with
  hundreds of ids from the crate that was still linked, every count agreeing
  with itself, and a whole tier missing from the catalog. The expected set is
  read from the sources when the test runs, so a third registering crate is
  covered the day it registers. This replaces a smoke test that grepped
  `list-ops` output for one crate's prefix from another crate's test target.
- The runtime now owns immutable resources, reusable artifact instances, and
  mutable leased state through one budgeted residency boundary. Cold and warm
  admission, rollback, cancellation, generation-checked reset, completion,
  eviction, and manager destruction release resources without exposing stale
  state.
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
- `vyre_foundation::algebra::composition::single_invocation` and
  `single_invocation_region` build the entry of a serial kernel: one anonymous
  composition region whose body runs on invocation zero of axis zero. The shape
  was written out by hand in every serial primitive.
- `structure-gate` derives the set of crates that submit `inventory`
  registrations from the tree and rejects any `use <that crate> as _;` in
  workspace sources, naming `vyre-registry-link` as the way to read the
  registry instead. The submitting set is read at scan time rather than listed,
  so a new submitting crate is judged the moment it submits, and a companion
  contract fails when the scan finds no submitters, which is the state that
  would accept every discarding import in the tree.
- `docs/lego-block-rule.md` is back, rewritten against source. It owns the four
  things nothing else states: the discovery step, the Category A and Category C
  placement test, the promotion criteria, and the Gate 1 budget in prose. The
  mdbook deletion took it out while `check-tier-deps` still read it for the
  cross-crate promotion contract, so the gate reported five findings against a
  file that was not there, and the workspace rules named required reading that
  did not exist. Every claim in it was checked: the discovery commands are the
  subcommands that exist, the budget numbers are the constants the gate
  compiles, the promotion contract forbids the compatibility shim this
  workspace does not ship, and the worked example names the primitive that
  survived it. `gate1` states the countable half and points at the policy for
  the rest, instead of recording that the policy was deleted.
- The `ci-matrix` gate reports a hosted CI matrix that has lost a platform or a
  toolchain, and a device escape hatch inside it. The rule it replaces asked
  whether the workflow file contained the word `stable` anywhere, which a step
  name satisfies, and whether it contained `macos-latest`, which a
  commented-out axis satisfies, so an axis could be deleted and the check
  stayed green. The gate reads the axis values out of the matrix block, which
  is what expands into jobs.
- vyre-libs gains an llm composition layer behind the llm feature.
  paged_kv_gather and paged_kv_append address a block-table paged key-value
  cache through the attention layout base, so a paged read is the same index
  map as every other layout move and a paged append is its scatter.
  logit_adjust applies a CTRL-style repetition penalty and the temperature
  divide in one pass over the vocabulary, nucleus_select walks the kept-mass
  prefix of a top-k candidate list, and sample_token fuses the three arms with
  softmax_top_k into one registered program whose intermediates are demoted
  after fusion. Program attribution moves to one owner in
  plumbing::program::attribution, which names the composition that selected a
  built Program and wraps an invocation-gated arm so fusion does not run it
  under the widest geometry; math::scan::prefix_sum builds on that owner and
  its IR is unchanged.
- `scripts/check_branch_accounting.py` derives the campaign's own branch and
  worktree state from git at run time and fails when it is inconsistent: a
  branch no owner branch holds and no worktree carries is work nobody is doing
  and nobody is merging, and a branch an owner already holds while a worktree
  keeps it alive is a source tree every scan walks for nothing. Owners are the
  integration branch and the subsystem tier, and a branch never accounts for
  itself. The derivation refuses to run against a repository with fewer than
  two local branches or no integration branch, because a check that silently
  derives nothing is the same defect as no check.
- `xtask` declares `default-run`, and a tree contract derives every `cargo run`
  in the workflows and scripts at run time, derives the binary targets from
  every member's manifest and source layout, and fails when an invocation does
  not resolve to exactly one target. A package that ships more than one binary
  and declares no default makes a bare `cargo run -p <package>` exit 101 before
  running anything, which is a property of the manifest rather than of any line
  of code: adding a second file under `xtask/src/bin` failed nineteen hosted
  jobs at their first step and nothing in the workspace could see it. The sweep
  already asserted that every registered subcommand is named by a workflow;
  this is the other direction. An invocation form the scan cannot attribute is
  an error rather than a skip, because a form nobody checked is how this
  reached CI.
- `vyre_foundation::visit::try_for_each_node` walks every node in a body and
  every nested body, stopping at the first `Break`, and `for_each_node` now
  delegates to it. A short-circuiting scan outside the crate previously had to
  implement the abstract-by-default `NodeVisitor` and write a no-op body for
  every variant it did not care about, which is the cost that made one scan
  hand-roll its own descent with a catch-all arm instead. `node_buffer_refs`,
  `expr_buffer_ref` and their result types are public for the same reason: a
  lowering crate answering "what does this statement do to a buffer" now reads
  the exhaustive owner rather than restating it.
- ProgramGraph now composes reusable Programs through canonical typed value
  identities, explicit consumer and output ports, symbolic or concrete shapes,
  access and lifetime contracts, and validated state transitions. Its bounded
  VGR0 wire format embeds existing VIR0 Programs and rejects implicit casts,
  rank drift, alias conflicts, dangling state, malformed framing, and hostile
  counts before mutation.
- vyre_libs::prelude names every item one dialect uses from another. The three
  dialects that reached into a sibling module tree now import from it, so the
  set of cross-dialect edges is one list in one file instead of a property of
  scattered use statements.
- A gate takes the published `min_` key set from the serialized
  `WallClockMinima` at run time and fails unless each key is either a sample
  floor or a member of the aggregate percentile field list. A minimum added to
  the record that no reader checks now fails the suite by name instead of
  reaching the artifacts unexamined.
- ProgramGraph now validates complete compositions and derives canonical
  topological schedules, inclusive value liveness, and deterministic
  interval-colored allocation plans. Invocation-local values reuse only
  nonoverlapping slots, while immutable weights, sequence-state generations,
  and caller-visible outputs retain dedicated storage.
- A CI step that passes a subcommand to `xtask`, `xtask-registry` or
  `xtask-evidence` is resolved against the registered subcommand table. A
  renamed or deleted row left the step naming a command that no longer exists,
  and nothing local went red: the package existed, the binary existed, and the
  name was one shell token in a `run:` line. `xtask gates` judges the other
  direction, that a registered row is wired into CI, so the pair is now closed.
  Names are extracted per step, including every command in a joined `run:`
  block, and resolved when the test runs rather than listed in it.
- `xtask --help` is checked against the registered subcommand table rather than
  against four names written into a test. A help route that prints the header
  and a truncated table still exits zero and still contains `SUBCOMMANDS:`, so
  the previous route check passed while commands were unreachable.

### Changed

- DeviceProfile::from_backend is the one spelling of the neutral device
  profile. The backend trait default and the Metal runtime override now take it
  and restate only the fields the backend knows better, instead of each writing
  out all forty.
- Barrier lowering follows the IR memory ordering. Acquire, Release and AcqRel
  name global-memory visibility and lower to a storage fence, PTX membar.gl and
  WGSL STORAGE. SeqCst stays a full workgroup barrier and its WGSL flags are
  the address spaces the barrier body actually touches, so a workgroup-scratch
  reduction round no longer requests a device-scope storage fence and a global
  release fence no longer converges the whole workgroup. The four strong
  orderings used to collapse onto one construct.
- The workspace denies the unexpected_cfgs lint instead of warning. A cfg
  attribute naming a feature its own crate does not declare removes the code
  under it, and at warning level that signal is one line in a build that emits
  many. The one allowance stays: external_ifds_engine names a bridge that
  cannot be a Cargo feature here.
- The heuristic audit reads an author's note, not any sentence that names a
  policy. A marker now has to open a plain comment: a doc comment describing a
  cache's eviction policy to its caller, a term used mid-paragraph while
  explaining an allocation, and a marker inside a string literal or an inline
  test module are no longer reported as hand-rolled heuristics. Three of the
  four findings it reported were descriptions, one of them on a test, and two
  sat in crates that cannot depend on the composition the fix names.
- A derived coverage set reads every declaration form. The two tests that
  enumerate an enum from its own source each recognised one variant shape, one
  struct-like and one unit, so a variant of the other shape was absent from the
  derived set and the comparison agreed with itself while the axis had grown;
  both now call one parser that accepts struct-like, tuple and unit forms, and
  a case states it against all three. The fused sampler names its three
  intermediates __vyre_llm_sampling_adjusted, __vyre_llm_sampling_candidates
  and __vyre_llm_sampling_weights instead of unscoped literals, which is the
  naming every other shared intermediate in the crate already uses. The
  registered online-softmax witness supplies its three logical inputs and no
  output placeholder, so the reference interpreter allocates the output the way
  it does for every other library registration instead of taking its legacy
  input path. The paged key-value append and gather document the block-table
  range precondition alongside the injectivity one: an entry naming a block at
  or past the block count addresses past the cache, and neither guard can bound
  it.
- A file name states what the file holds, and the gate now judges every tree a
  crate compiles rather than src/ alone. The prohibition on names like common,
  support, helpers, types and utils was written for library modules and never
  reached test-adjacent files, which is where the population had moved: 15 of
  the last 16 offenders were tests/common/mod.rs or tests/support/mod.rs. The
  same words as a suffix are the same dumping ground with a qualifier bolted
  on, so spec_types is now op_signature, backend_impl is backend_dispatch,
  test_support is test_parity_oracles, and each crate's shared test module is
  named for what it provides: harness, program_fixtures, wire_words,
  fixture_backend, gate_fixtures, workspace_sources. Two further classes are
  rejected: sibling files told apart only by a number, where the number carries
  the distinction and the name carries none, and a file that repeats the
  directory holding it, which is the directory's mod.rs. A number inside a name
  that means something has no numbered sibling, so crc32 and flash_attention_2
  are untouched. A module stays exempt only while the committed public-API
  snapshot publishes it, so vyre_libs::parsing::core and vyre_driver_wgpu::ext
  keep their names until the snapshot stops publishing them.
- Constant propagation, dead-branch elimination and loop-invariant code motion
  moved from vyre_pass_engine::optimizer to vyre_foundation::transform, and the
  cross-scope CSE hoist moved into vyre_pass_engine::optimizer::cse_via_encoded
  beside the canonical ids it consumes. The pass engine keeps only the passes
  it dispatches as vyre Programs.
- Every authored page row in docs/DOCS.toml carries a covers list naming the
  source paths the page states the content of. docs/lego-block-rule.md also
  records both anonymous generator prefixes: it named only anonymous:: while
  vyre_foundation::composition::ANONYMOUS_GENERATOR_PREFIXES has held inline::
  as well, so a reader checking a region against the gate saw half the rule.
- xtask gates --write-baseline records a measured finding count only when it is
  at or below the pin already recorded, and fails naming every gate that
  reports more, so a run can no longer legalize a red gate.
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
- `examples/external_backend_extension` registers a dispatch backend from
  outside the workspace. It described `vyre_driver::VyreBackend` as sealed
  against outside implementations and built a program instead, which stated a
  limitation that does not hold: `vyre_driver::sealed` is exported, so the
  crate now implements the trait, submits a `BackendRegistration` and a
  `BackendCapability`, and asserts that `vyre_driver::acquire` serves it and
  that a dispatch through it returns the expected bytes. Execution stays inside
  `vyre-reference`.
- The architecture coherence check and the published-reference check are
  registered gates instead of Python scripts. `architecture-contract` holds
  `docs/ARCHITECTURE.md` to the workspace roster, the release train, the
  generated operation schema, the backend evidence, the ownership registry and
  the documentation manifest, and `docs-references` resolves every path-like
  code span and command argument in a published document against the listed
  tree. Both are in the `docs` subset, both carry a pinned baseline, and both
  report every failure in one run rather than exiting on the first. The
  operation schema version is now pinned by two constants in one build instead
  of a number compared as text across two languages, so the drift that once
  shipped a generator on 3 against a checker demanding 2 cannot compile.
- The two CSE passes in `vyre-pass-engine` that walk IR while counting
  expression-arena positions share one cursor, `optimizer::arena_cursor`. Each
  had its own copy of advancing an index per node, remembering a position to
  rewind to, and skipping a nested body's worth of ids, and the nesting each
  skipped was written out variant by variant, so a new statement-carrying
  variant would have misaligned an arena verdict against the node it was
  computed for without any error. `ArenaCursor` takes its nesting from
  `visit::child_bodies` and stays in the pass engine, because the numbering is
  the encoder's and not the IR's. The hoisting decision and the same-scope
  let-dedupe decision stay separate: they are different decisions over the same
  walk.
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
- An artifact-inspection gate is declared, not hand-written. Eleven gates
  across xtask, xtask-registry and xtask-evidence each spelled the same struct,
  the same four trait methods and the same call into `settle_inspection`,
  differing only in a name, a help string and one inspection expression.
  `xtask::artifact_gate!` now owns that shape and the eleven declarations state
  only what differs. Gates whose `run` does real work, such as
  release-conformance argument parsing, keep their own implementation.
- Gate execution now derives names, owners, areas, artifact paths,
  prerequisites, and mutation proofs from one authoritative descriptor
  registry. Every gate reports non-empty subject coverage, generators declare
  and report exact workspace-relative outputs, only artifact owners receive
  `--write` authority, and the complete sweep rejects unowned workspace
  mutations. README generation has one owner: `crate-readmes` refreshes both
  crate-contract and CLI-contract sections.
- Batched dispatch readback has one owner and one allocation for its rows.
  `vyre_driver::BatchOutputs` stores every dispatch's output row end to end in
  a single buffer and hands rows out borrowed, replacing the
  `Vec<Vec<Vec<u8>>>` returned by `vyre_driver_wgpu::pipeline::compound` and
  `vyre_driver_wgpu::engine::graph`. The middle level of that type held exactly
  one output slot at every call site, so an n-dispatch batch allocated 2n+1
  vectors to carry n byte rows and copied each row out of the mapped staging
  range into a vector of its own.
  `vyre_driver::program_walks::enforce_output_budget` now owns the
  `DispatchConfig::max_output_bytes` policy that a second wgpu-local
  implementation had restated with its own wording, and
  `vyre_driver_wgpu::pipeline::WgpuPipeline::dispatch_coalesced_borrowed`
  absorbed the private function whose arguments it forwarded unchanged. A new
  `xtask` tree contract, `nested_byte_row_types`, parses every crate that
  implements the backend trait and fails on any three-deep byte-row vector;
  scope comes from the root manifest and a parsed trait scan at run time, so a
  backend crate added later is covered the day it lands. The existing ratchet
  counted the legitimate two-level per-slot output in one crate only, which
  left a three-deep copy in any other crate unobserved.
- Artifact materialization has one skeleton for every backend.
  `vyre_driver::materializer_passthrough` answers
  `ArtifactMaterializer::device` from the acquired descriptor and, given a
  backend field, forwards the four resident-resource methods, which two
  backends had written out as eight identical bodies differing only in a field
  name. The Metal and SPIR-V instances now run
  `vyre_driver::materialize::InstanceCore::execute_modules` instead of each
  carrying its own module loop: both had rebuilt the same config clone, grid
  override, binding-plan build, input gather and output absorb, and both
  discarded the device timing that loop reports, so a dispatch that measured
  elapsed device time was reported as untimed. SPIR-V also stops tolerating a
  dispatch that returns fewer buffers than its plan declares, which silently
  kept the previous value for a declared output, and its third spelling of the
  unmapped-output rejection is gone in favour of the one its own message record
  already declared. Metal's private copy of the unbound-input rejection is
  replaced by `vyre_driver::materialize::unbound_input`, so all four backends
  refuse an unbound canonical value in the same words.
- `vyre_test_support::cast_parity::NARROWING_CASES` holds the four `u32`
  narrowing casts as data, and each backend arm keeps only its dispatch and
  walks the matrix, so a case added there is dispatched by both without editing
  either arm. `NarrowingCase::assert_target_words` checks the pin before
  comparing a target's words, so a drifted oracle is reported as a drifted
  oracle rather than as a device divergence. The wgpu arm also stops
  open-coding the elementwise program, the little-endian packing and the
  readback decode that `vyre_driver::parity_harness` owns and the CUDA arm
  already used.
  `the_matrix_covers_every_narrowing_target_the_cast_table_admits` reads the
  integer-like scalar predicate out of `vyre_foundation::validate::cast` at run
  time and requires a case for every admitted integer under 32 bits, so
  admitting a narrower integer there is red until both arms dispatch it;
  enumerating the `DataType` enum instead would demand a case for the sub-byte
  quantization storage families that predicate deliberately excludes.
- Every dispatch trait requires the borrowed-row form and defaults the owned
  one over it. `vyre_driver::VyreBackend::dispatch_borrowed`,
  `vyre_driver::backend::CompiledPipeline::dispatch_borrowed` and the
  C-preprocess `ProgramOracle` are now the methods an implementor writes;
  `dispatch` is a default that borrows the rows it was handed. It was the other
  way around, so a backend that binds caller memory received rows it had to own
  first and `clone_borrowed_inputs_for_dispatch` copied every input byte on the
  path whose purpose is not to. That helper is deleted, the four backends that
  carried a reverse shell of the same body no longer declare one, and the
  oracle's owned entry point is gone rather than kept as a second door. The
  inherent `dispatch` methods the wgpu, CUDA, SPIR-V and reference backends
  published leave the public surface with it.
- The vyre-bench duplication pin records the measured tree: 2485 to 2179
  duplicated lines, with total_lines corrected from 37430 to the measured
  37504.
- The seven `foundation.*` micro benchmark cases are rows on one owner,
  `vyre-bench/src/cases/micro.rs`, which states the program, fixture, reference
  and work accounting each case differs in and nothing else. `gpu_case.rs` is
  absorbed into it, and its metric sweep now covers both work-accounting arms
  across every lane rather than one.
- The resident work-queue buffer set the megakernel benchmark cases dispatch
  over has one owner, `vyre-bench/src/cases/resident_queue.rs`. Each of the
  three cases carried its own copy of the four-buffer control/ring/debug/io
  encoding, its own post-dispatch transfer accounting and its own resident-pool
  metric point. Each case keeps what it measures: its own ring builder, its own
  CPU reference, and its own grid override, because the latency case drains a
  single workgroup while the condition case drives the full grid and that
  difference is the measurement.
- The three benchmark checks are registered gates. `bench-baselines` requires a
  published `benches/RESULTS.md` section for every package `cargo bench` can
  run a target for, derived from the manifests rather than from a directory
  name, so a crate whose `benches/` holds only prose owes no invented number
  and a nested target is named as `-p` accepts it. `bench-smoke-runtime`
  enforces the wall-clock budget declared in `contracts/perf_targets.toml`
  rather than a number of its own. `bench-coverage` holds each measured
  dimension to a registered case and resolves every `--case` a workflow or
  evidence manifest names against the live registry. Each reports every failure
  in one run.
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
- Benchmark case declarations have one owner, `vyre_bench::cases::harness`.
  Each of the eight honest cases open-coded the same ten-method `BenchCase`
  block, and the copies had drifted: `search.binary.u32.1m` omitted the smoke
  suite from its private list and so ran in no smoke suite, and
  `regex.backtracking.adversarial` inherited a byte-accounting default that
  reported reading and writing nothing because its prepared payload was not a
  bare program. A case is now a static `WorkloadDescription` plus a `CaseOps`
  record of the operations a description cannot carry, and it reports which
  owner built it through `BenchCase::declaration_owner`.
  `vyre_bench::cases::harness::HONEST_SUITES` is the single honest suite list,
  replacing two verbatim copies and one per-case spelling. The two YARA-like
  condition workloads likewise share one owner each for the nine per-rule
  parameters, for the five-condition conjunction their device programs are
  scored against, and for the four IR predicate blocks, in
  `vyre_bench::cases::conditional`; both previously held a private copy of all
  three, so a predicate dropped from one copy of the host oracle would have
  read as a device correctness violation rather than a host bug. Callers still
  concatenate the IR blocks in their own order, so both programs keep their
  recorded fingerprints. A new gate walks the inventory registry at run time
  and fails when an honest case reports no declaration owner or an owner
  serving only itself.
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
- The bracket matching kind constants published by vyre-primitives::matching
  are BRACKET_KIND_OPEN, BRACKET_KIND_CLOSE, BRACKET_KIND_OTHER and
  BRACKET_MATCH_NONE, and the CPU oracle is bracket_match_cpu_ref /
  bracket_match_cpu_ref_into under its own name rather than an aliased cpu_ref.
- Scan products now return the foundation-owned `ByteRange { tag, start, end
  }`. The deprecated `Match` and `LiteralMatch` surfaces and the duplicate
  primitive range type are gone.
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
- The C11 hostile-parser fixtures are spelled through
  `c_frontend::spelling::c_rows` rather than one `TOK_` constant per line. Six
  distinct constructs shared long runs of `LBRACE DOT IDENTIFIER ASSIGN` and
  `LPAREN IDENTIFIER RPAREN`, which read as copied text and buried each
  construct in thirty lines of scaffolding plus an identical three-line span
  tail. A contract test pins the token count and the load-bearing rows of every
  fixture, because a spelling is a second way to say the same thing and a
  mistyped kind name would retarget a positional contract at another token
  while every assertion still passed. The union field designator both
  initializer-designator families index is one stream in
  `tests/support/c_frontend/fixtures/initializer_designator_streams.rs`, under
  one name in both.
- The WGPU C-frontend token tests share the fixture builder, the CPU pipeline
  runner, and the row readers with the CPU-side tests instead of carrying a
  second copy of each. The support file became a directory module, because a
  top-level file under `tests/` is its own test binary and therefore cannot
  reach the shared support module, which is what forced the copy. The shared
  fixture builder now models the lexer for whitespace and comment lexemes: they
  contribute source text and no token row, so a preprocessor fixture states
  real source layout instead of inventing rows for trivia.
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
- C typedef row phases remain canonical callable operations. The operation
  matrix marks them as inlined callees whose execution coverage belongs to
  fixture-backed parent operations.
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
- A changelog fragment is one file, `release/changes/unreleased/<id>.toml`,
  holding `category` and `text` with the file name as the id. Every fragment
  used to be a `[[fragments]]` table appended to one file, and a three-way
  merge of two branches that each appended one matched the shared blank line
  and `[[fragments]]` header between the two sides, left them out of the
  conflicting region, and resolved only the differing tails. The `merge=union`
  attribute then concatenated those tails under one header, so one fragment
  carried two ids and the file stopped being valid TOML; without the attribute
  the same merge stops on a conflict instead. The attribute is gone with the
  file it named, and `release-docs` reads the directory, rejects an unknown key
  and refuses a fragment whose text is empty. A regression test builds two
  branches that each add a fragment and merges them, so the fusion cannot come
  back quietly.
- The command-line contract no longer renders a book page.
  `scripts/cli_docs.py` kept generating `docs/CLI.md`, which is deleted, so the
  gate compared a generated document against a file that is not there while its
  real verdicts went unread: the manifest schema, the README each binary names,
  the declared binary set against cargo metadata, every help route exiting zero
  with bounded non-empty output, and the xtask help table against the
  registered subcommands. Those verdicts stay and the CLI section of each crate
  README is still generated. The contract test reads the subcommand count back
  out of the generated README blocks instead of out of the deleted page, and
  the build it runs no longer forces one codegen job.
- PTX f32 canonicalization now uses native flush-to-zero multiplication plus
  NaN selection, preserving signed zero and canonical NaN semantics with fewer
  instructions and registers.
- The whole-program compiler validates and ranks against live device facts.
  CompileRequest carries a DeviceFacts snapshot, a program that needs a
  capability the device lacks fails at compile, a whole-grid fence without
  cooperative launch fails at compile, the candidate cost model prices
  launches, materialized bytes and the occupancy cliff a fusion group crosses,
  and a measurement budget is spent through compile_measured with
  device-timestamped launches instead of being ignored.
- Persistent-kernel residency and launch geometry are compiler decisions. The
  dispatch policy bundle carries the artifact's ExecutionMode instead of
  deciding residency from a batch shape, so a Static artifact can never
  dispatch as a persistent kernel, and the driver launch resolver treats a
  recorded artifact geometry as authoritative over the launch tuner and
  dispatch overrides. A target descriptor that records no workgroup geometry
  fails the launch instead of falling back to a declared or tuned width.
- Two concepts that a consumer crate had re-derived now have one owner.
  `vyre_test_support::ir_regions` owns the three helpers that slice a stretch
  of generated IR out of a program and compare it against a sibling, which
  `vyre_primitives` and `vyre_libs` each wrote out; a comparison helper decides
  what its test can see, so a widened slice in one copy weakened an assertion
  in a crate whose author never read the change.
  `vyre_libs::solvers::bellman_tn_order` no longer re-proves the shortest-path
  relaxation of `vyre_libs::math::bellman_shortest_path`: what it owes is the
  routing assertion it already carries, that its composition emits the
  primitive program unchanged.
- Shared composer types have one public path under `vyre-libs`: CSR types under
  `csr`, elementwise types under `elementwise`, contraction types under `gemm`,
  state-machine types under `state_machine`, and grid types under `stencil`.
  The duplicate crate-root and semiring-module re-exports are removed.
- Eighteen composition domains moved out of vyre-primitives into vyre-libs:
  bitset, decode, fixpoint, geom, graph, hash, label, math, matching, nfa, nn,
  opt, parsing, predicate, reduce, text, topology and visual. vyre-primitives
  now holds marker types, the wire encoding, the dispatch grid owner, the IR
  safety helpers, the hardware intrinsics and the virtual file system, and its
  feature list is default, gpu, cpu-parity, vyre-foundation, hardware and
  inventory-registry. Every path of the form vyre_primitives::<domain> is now
  vyre_libs::<domain>; no compatibility re-export is left behind. Registered
  operation ids are unchanged, so built IR and the operation catalog keep the
  same names. Each moved domain has a feature of its own in vyre-libs that
  names only the domains its own source reaches.
- Sequential composition witnesses now live only in
  `vyre-reference::composition_witness`. `vyre-libs`, `vyre-primitives`, parity
  suites, and release benchmark adapters no longer retain independent host
  implementations; test and benchmark callers delegate to neutral
  reference-owned witnesses, while production compilation and dispatch remain
  GPU-only.
- `vyre-conform` depends on `vyre-libs` with `features = ["full"]` rather than
  restating ten of the aggregate's members by hand. A hand-kept list of
  aggregate members drifts silently against the aggregate, which is how the
  same shape on the registry walker's primitives edge lost four dialects and
  made three operations invisible.
- Four duplication pins record what the tree measures: `conform` 765 to 367
  duplicated lines, `vyre-spec` 244 to 84, `vyre-foundation` 4342 to 4127, and
  `vyre-libs` 11446 to 11425, each with `total_lines` measured. A pin with room
  under it hides the next copy.
- The standalone `vyre-harness` package is gone. Semantic operation identity,
  tier classification, and registration now live in `vyre-foundation`; library
  fixture views live in `vyre-libs`; conformance execution and parity policy
  live in `vyre-conform`; self-substrate behavior tests live with their owner.
- The cross-backend u32 parity suites read one op table instead of one
  hand-written test per op. `synthetic_binop_parity`, its CUDA twin, and both
  `div_zero_shift_mask` suites loop over the shared table with a per-backend
  reference arm and a per-backend divergence note, and each opens with the
  coverage assertion so a missing arm fails before a device is acquired. The
  reference arms stay per backend by design: the naga multi-step synthesis and
  the PTX instruction selection are unrelated lowerings of one contract and
  each owes its own live proof against the CPU reference. What is shared is
  which ops exist.
- CSR closure entry points take the graph as one named bundle.
  `vyre_libs::graph::csr_closure_inputs` owns `CsrGraphView { node_count,
  edge_offsets, edge_targets, edge_kind_mask }` and `CsrClosureInputs { graph,
  allow_mask, max_iters }`, and every closure entry point in
  `vyre_libs::graph::csr_bidirectional`,
  `vyre_libs::graph::csr_forward_or_changed`,
  `vyre_libs::graph::csr_backward_or_changed` and
  `vyre_libs::graph::persistent_bfs` now receives them instead of seven or nine
  positional slots. `CsrGraphView` has no constructor: a struct literal is the
  only way to build one, so a transposed buffer is a compile error rather than
  a wrong closure, and `CsrClosureInputs` provides
  `CsrClosureInputs::allow_all` for unrestricted edge filtering. The
  dispatcher-backed consumers in
  `vyre_libs::graph::dispatch::csr_bidirectional`,
  `vyre_libs::graph::dispatch::csr_forward_or_changed` and
  `vyre_libs::graph::dispatch::persistent_bfs` take the same bundle, so the
  flat shape no longer exists at any layer.
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
- `vyre_libs::graph::csr_closure_inputs` owns two things the CSR closure tests
  restated at every case: `CsrClosureInputs::allow_all` names the all-ones edge
  filter a case picks when the filter is not what it is testing, and
  `graphs::CHAIN_4` and `graphs::DIAMOND_4` name the two small graphs whose
  closure is known by inspection. Fifty-six call sites across
  `vyre-primitives`, `vyre-libs` and `vyre-driver-cuda` restated the filter as
  a bare `0xFFFF_FFFF` beside a four-field graph literal, and seventeen more
  restated one of those two graphs inline, so two cases could not be told apart
  by their graph and rustfmt put every field on its own row. The redundant
  `name: name` half of eighty-four fields the positional-to-named rewrite left
  behind is gone as well.
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
- The CUDA C-preprocessing parity tests share their two dispatchers. The
  payload, tokenize, macro-expansion and filter files each declared a private
  reference dispatcher that evaluates the program on the host and a private
  CUDA dispatcher that runs it on the device once owned and once borrowed, four
  copies of each. `vyre-driver-cuda/tests/common/c_preprocess_oracles.rs` now
  owns `ReferenceOracle` and `CudaOracle`; the parity assertion each file makes
  stays in that file, because the two arms being compared are the point of the
  test.
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
- The CUDA backend names `vyre_driver::input_identity::exact_input_key` and
  `ExactInputKey` directly. `vyre_driver_cuda::input_identity` was a module
  whose entire body was a re-export of those two items, and
  `vyre_driver_cuda::pipeline::materialized_cache` re-exported them again under
  second names, `materialized_input_key` and `MaterializedInputKey`, so one
  hash envelope and one key type were reachable under four spellings and the
  materialized cache read as if it owned a key format of its own. Both
  re-export layers are deleted and every use site names the owner.
  `vyre_driver_cuda::pipeline::tests::input_key_owner_contracts` replaces a
  test file that restated the envelope's own tuple-boundary and
  single-byte-sensitivity properties over 8192 generated cases under the alias
  name, proving nothing about CUDA while the question that mattered went
  unasked: it now asserts that a stored entry's key equals the shared
  envelope's answer for the same inputs, and that a resident-cache
  domain-separated key cannot reach materialized replay outputs for inputs that
  still hit under the plain envelope.
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
- `vyre_driver_cuda::optimizer` is `vyre_driver_cuda::resident_dispatcher`. The
  module holds no optimization pass: it holds `CudaProgramDispatcher`, the
  resident buffer pool, the static-upload cache and their eviction policy.
  Optimization semantics live in `vyre-foundation`, so an `optimizer` module
  inside a concrete driver crate named the caller rather than the contents, and
  four private helpers were named after the file instead of their work. The
  `CudaProgramDispatcher` re-export at the crate root is unchanged.
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
- `vyre-test-support` states which flat `DataType` forms exist in one ungated
  module, `data_type_elements`. The list sat inside `data_type_variants`, which
  is gated behind `ir-fixtures`, so a suite that wanted the flat element list
  had to enable a feature that pulls `vyre-foundation` into its dev graph; for
  `vyre-spec`, a leaf crate that declares `DataType` in the first place, that
  meant building the whole compiler to read a list of discriminants.
  `vyre-spec` is now a plain dependency of `vyre-test-support` and
  `ir-fixtures` means `vyre-foundation` plus `smallvec`. The IR fixture table
  builds its flat leaves from the same module, so the two tables cannot
  disagree about which element types exist.
- The `DataType` wire-tag table has one owner,
  `vyre_foundation::serial::wire::tags::data_type_tag`. Two copies were reading
  against it: the dense memory-region encoder, which now states only its
  narrower domain, and `vyre_driver::specialization`, whose copy had already
  lost `DataType::Quantized` and therefore filed every quantized specialization
  under the shared unknown-variant sentinel, so two element types could draw
  one cache key and one of them the other's compiled shader. Fixtures for the
  enum live in `vyre_test_support::data_type_variants`, checked against the
  declaration at run time.
- The host dataflow closures are read from their owner. `vyre-libs` carried a
  module whose six public functions each did nothing but call the identically
  named function in `vyre_foundation::pass_substrate::semiring_closure`, and it
  had grown a copy of the owner's whole test module: five assertions verbatim,
  plus two that could not fail because they compared the forwarder against the
  function it forwards to. The forwarders are gone and every caller names the
  owner. What remains is the one thing this crate adds, the call counter, in a
  file named for it.
- The decode-scan fusion pass takes its workgroup promotion budget from the
  caller's capability record instead of a fixed constant. `run`,
  `count_opportunities` and `candidate_handoffs` take an `AdapterCaps`, and
  `DecodeScanFuse::transform_for_adapter` promotes against a named target while
  `DecodeScanFuse::transform` keeps the conservative profile, so the default
  path is byte-identical. Previously every target was capped at the lowest
  reported shared-memory figure, which refused a handoff a reported budget
  allows.
- `xtask::delegate::run_delegated_main` owns what a delegated binary's `main`
  does: answer `--help` from the callee's own dispatch table, reject a missing
  subcommand, then resolve the name. `xtask-registry` and `xtask-evidence` each
  wrote that out, including the exit code for an unimplemented subcommand, so
  the two entry points could disagree about a contract that `xtask` and CI read
  as one. Each `main` is now a single call naming the package and why `xtask`
  routes there, and the `dispatch` wrapper each crate exported for its own
  `main` alone is gone. Help text, error text and exit codes are unchanged.
- The internal dependency version rule, the benchmark coverage rule, and both
  sweep runners derive their crate, case, and test rosters from tracked
  manifests and sources instead of hardcoded lists, no longer write outside the
  repository, and fail on a cargo failure instead of reporting a clean tree.
  The volume sweep runner had a three-crate list that left one tracked volume
  wave in no shard, and a shard index outside the shard count selected nothing
  and exited 0.
- User-facing crate READMEs, `docs/ARCHITECTURE.md`, `THESIS.md`,
  `CONTRIBUTING.md`, and the ownership/guide registries follow the workspace
  `README.md` charter. `vyre-libs` owns every composition, including
  compiler-internal domains. `vyre-primitives` owns only uncomposable
  intrinsics. Persistence is selected at compile time. Unmeasured selections
  are never called autoroute.
- Documentation pages now declare audience, owner, authority, kind, and
  generated/manual ownership. Crate dependency records declare purpose,
  features, target conditions, visibility, and destination seam, and optimizer
  pass reference pages are generated from the live pass registry.
- The documentation authority check is a registered gate rather than a Python
  program a gate shells into. `xtask docs-check` now reads `docs/DOCS.toml`
  itself, validates every owner and page row, renders `docs/SUMMARY.md` and
  `docs/INDEX.md` under `--write`, and resolves every outbound link in the
  published navigation, reporting a link that escapes the repository root,
  names no such path, or resolves to a path the repository excludes.
  `scripts/docs_manifest.py`, `scripts/check_docs_index.sh`,
  `scripts/check_docs_links.sh` and `scripts/test_docs_manifest.py` are
  deleted, and the validator's behavior contracts are unit tests beside the
  gate. The published set is read from the working tree instead of the git
  index, so a new page is unclassified on the commit that writes it rather than
  on somebody else's. The generated index no longer restates the workspace
  package and target counts: `cargo metadata` owns those, and a documentation
  index that repeated them was a second owner nothing reconciled.
- Doc CI builds the workspace documentation through the `workspace-docs` gate.
  The shell wrapper it replaced built with `--keep-going` and no feature flags,
  and its changed-only mode was reachable from no caller; the gate builds every
  crate with all features and reports one finding per diagnostic.
- The dominator-tree fixpoint composes two registered operations instead of two
  anonymous inline phase generators.
  vyre-primitives::graph::dominator_tree_depth writes the depth of every node
  from an immediate-dominator forest, and
  vyre-primitives::graph::dominator_tree_intersect_step walks predecessor edges
  once and reports whether the forest moved. Both are callable on their own and
  the composed fixpoint builds the same IR as before.
- The duplication baseline records the measured tree for three more crates:
  vyre-driver 1112 to 1082 and vyre-driver-wgpu 5710 to 5709 after their lanes
  merged, and vyre-foundation 5443 to 5339. Duplication is cross-file, so a
  crate's count moves when a sibling lands or drops a copy of its text, and
  each pin is measured against the merged tree rather than carried over from
  the branch that lowered it.
- `InstanceCore::absorb_outputs` and `InstanceCore::resident_completion` take
  the dispatch result by value and move each output buffer into the value map
  instead of cloning it. Callers owned the result and dropped it, so the copy
  protected nothing and cost one full pass over every output byte on every
  dispatch of every module. Output slots are assigned per buffer index in
  `binding.rs`, so a repeated slot is a plan defect: the move refuses it
  through the omitted-output rejection rather than serving the previous
  buffer's bytes a second time.
- The duplication pins for the four driver crates in this change record what
  this branch measures in isolation: vyre-driver-cuda 3238 to 3187,
  vyre-driver-wgpu 4017 to 3918, vyre-driver-spirv 194 to 118,
  vyre-driver-reference unchanged at 23 with its stale total corrected. Every
  `total_lines` is measured, not carried forward. Duplication is cross-file, so
  collapsing one side of a copy moves the other: vyre-driver-metal fell 251 to
  175 and vyre-driver 1082 to 1069 without either being edited, and both sit
  below pins this change does not touch.
- The hostile-input closure obligations every backend owes are stated once, in
  `vyre-driver/src/hostile_input_closure.rs`. Each backend crate had written
  the probe programs and the assertion text again against its own backend type,
  so the wording drifted per crate while the contract did not. A backend's
  adversarial target now supplies the backend and names itself. The reference
  backend keeps the hostile-bytes and trailing-input obligations it had; the
  SPIR-V backend keeps zero-workgroup and trailing-input, and reports rather
  than fails when no Vulkan device is present.
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
- `vyre-driver` no longer enumerates `Node` for itself. Indirect-dispatch
  discovery, the launch-geometry scan, and backend support validation each
  descend through the owning enumerations in `vyre-foundation`, so a new
  statement variant reaches them without an edit. The three hand-written walks
  ended in a catch-all arm and `Node` is `#[non_exhaustive]`: an unsupported
  operation nested inside an unrecognised variant validated clean, and a kernel
  that read launch geometry inside one read as not reading it.
  `visit::any_expr_in` is public, as the composition of the node, operand, and
  expression owners that a scan over both namespaces needs.
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
- The duplication pins for the four emitter crates now sit at what the tree
  measures: vyre-emit-metal 29 to 22, vyre-emit-naga 404 to 221, vyre-emit-ptx
  295 to 105, vyre-emit-spirv 49 to 43. A pin that passes with room under it
  hides the next copy, so each is lowered to the measurement rather than left
  as headroom. vyre-megakernel stays at 60: both of its duplicated files
  partner only with crates outside the emitter boundary, so nothing there could
  be collapsed from this side.
- Five enforcement sites state the contract they enforce instead of citing a
  document. `structure-gate` names the composition rule at `CATEGORY_A_CRATE`
  and the one-owner rule at `substrate_home_failures`; `gate1` states the
  countable half of the composition budget it enforces and leaves the reuse
  criterion to the policy an author applies; the CLI surface contract names the
  generated README block it counts; and the tree-contract link unit states why
  two suites stay separate targets. A rule that lives only in a document stops
  being enforced the day the document is deleted, and four of these cited
  documents that already were.
- Every library composition in vyre-libs sits behind a feature. The text,
  representation, parsing and graph module trees were declared with no cfg, so
  forty-two files submitted an operation registration in every build that
  linked the crate, and the default selection did not compile at all because
  those trees reach bitset, fixpoint and the parser kernels without a feature
  that enables them. Each tree now carries the gate its registrations route to,
  the parsing substrate that registers nothing stays available to a
  kernels-only build, and the two shared builder child regions register behind
  builder-ops. The manifest gained representation and builder-ops, full names
  both, and matching-dfa names text because the scanner post-process folds the
  reference byte histogram into its entropy oracle. A default build no longer
  exposes vyre_libs::graph, vyre_libs::parsing or vyre_libs::representation; a
  consumer names the tree it composes. The host reference oracles under text
  stay behind cpu-parity rather than behind text, so a default build that
  reaches text through decode carries no CPU classifier, and the integration
  targets that compare against them name the feature they need.
- Seven release gate scripts run `scripts/lib/<name>.py` instead of piping a
  Python program into an interpreter through a heredoc. A heredoc hides a whole
  second language from review, lint, and syntax checking, and these were the
  last seven. `scripts/cli_docs.py` invokes the workspace wrapper through
  `scripts/lib/cargo_runner.py`, the Python twin of the shell runner that
  already owned that decision.
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
- `vyre_foundation::optimizer::passes::algebraic::strength_reduce` recognizes
  four constant-divisor shapes before it lowers them. A divisibility test `x %
  d == 0` becomes Lemire's `rotate_right(x * inverse(odd(d)),
  trailing_zeros(d)) <= limit`, which reads the operand once and emits two
  operations where lowering the remainder emitted five. A common factor between
  a dividend's multiplier and the divisor cancels, a constant division chain
  fuses into one division, and a nested modulus narrows, each guarded by a
  range proof from a new provable-upper-bound analysis so a rewrite that would
  only hold without 32-bit wrapping is declined. Chained shift fusion no longer
  folds an over-width total to zero, which was a miscompile for a signed right
  shift, where the sign bit replicates instead. Rewrites that re-evaluate their
  operand now clear one duplication budget owned by `strength_reduce`, so a
  remainder of a buffer load no longer emits three loads of the same address.
- `scripts/check_feature_msrv.sh` printed a hardcoded 19-entry matrix and
  exited 0 without compiling anything. It now derives one entry per publishable
  member per declared feature from tracked manifests and compiles each alone on
  the toolchain `[workspace.package].rust-version` names, which must be
  installed.
- Compiler finalist evaluation binds representative workload inputs rather than
  zero-filled buffers so valid production traps do not abort compile-time
  device timing. `CompileRequest` owns representative bytes keyed by
  `GraphValueId`, and `RequestIdentity` commits to their ordered digests and
  lengths. `DeviceFinalists` fails closed on missing or mismatched host
  resources. `ProductionSession::compile_with_representative_inputs` and
  `ExecutionRoute::open_with_representative_inputs` accept Program host-input
  order. Runtime-sized host buffers keep dynamic target IR while their
  representative bytes establish exact artifact resource counts.
- Harness selection and convergence-flag width for the persistent fixpoint are
  one decision with one owner, `routed_persistent_fixpoint`, which returns both
  halves together. The grid form indexes `changed[iteration]`, so taking the
  harness without its flag width was an out-of-bounds atomic write on the
  second iteration.
- The `vyre-foundation` duplication pin sits at what the tree measures, 5125
  down to 3333 duplicated lines, with the total line count corrected from a
  stale 95984 to a measured 94304. It was lowered four times as each owner
  landed, because a pin with room under it hides the next copy.
- A changelog fragment is its own file under `release/changes/unreleased/`,
  named for its id, carrying a category and text. Ten times in one week a merge
  ate a `[[fragments]]` header: every fragment opened with that identical line,
  so diff3 aligned it as common context between two appended blocks, only the
  differing lines reached the merge driver, and the second block kept `id`,
  `category` and `text` while losing the header that separated it. Its keys
  then parsed as a second `id` in the fragment above, so the file stopped being
  valid TOML and every release document stopped regenerating behind a parser
  position that named no cause. A `merge=union` attribute could not fix it,
  because there was no conflict to resolve; `.gitattributes` carried that
  attempt and is deleted with the file it named. Two fragments now share no
  line, no header and no identity, and a merge that keeps both files keeps both
  fragments. A file in the retired shape is rejected by name rather than folded
  into its neighbour.
- Five shapes the gate crates repeated per gate have one owner each.
  `xtask::toml_text` renders a TOML basic string and a one-line string array,
  which four generators spelled with two different escapers. `xtask::tree_walk`
  is public, so a gate outside `xtask` walks the tree through the one prune
  rule instead of a hand-rolled `read_dir` recursion: dependency drift, the
  heuristic audit and the lego audit cross-dialect scan all read the same set
  of directories now, and the three that pruned only `target` and `.git` no
  longer read `target-codex`, `target_tests` or `.cargo-target`.
  `xtask::output_arg::cargo_runner` is the one resolver for the bounded cargo
  wrapper. `xtask_registry::corpus::selected_cases` selects release corpus
  cases for the compile and shrink gates, which carried the same filter and the
  same empty-selection error.
- `xtask gates` no longer accepts a failing gate. `xtask/gate-baselines.toml`
  pinned both a `status` and an `owner` sentence per row, and the sweep treated
  `status = "red"` as the expected result whenever the row named an owner, so a
  gate could fail indefinitely while the sweep reported that every gate held
  its baseline; three did, for a fortnight. Both fields are gone from the
  schema and a row that still carries either fails to load rather than being
  ignored, because a default derive would accept and discard it. What remains
  is a finding ratchet: `output_lines` may fall and be lowered, may not grow,
  and a nonzero exit is a failure whatever the pin says. `--write-baseline`
  refuses to record a run in which any gate failed.
- The gate sweep's wiring check now runs from gate to workflow as well as from
  workflow to gate. A registered gate that no workflow names, directly or
  through a subset a workflow runs, fails the check instead of sitting in the
  registry as decoration. `file-size` was red on fourteen source files while
  nothing in CI named it, so its judgement reached nobody who could act on it.
  Wiring failures are not pinnable.
- Five checks that no workflow invoked now run in
  `.github/workflows/gates.yml`: the transitive layer closure, benchmark
  registry coverage, internal dependency versions, the feature-isolated MSRV
  sweep, and the oracle-matrix sweeps with their volume waves. Each step names
  the class it closes and why no other workflow sees it.
- The generated CSR sweep shape stream has one owner: five copies of the same
  seeded generator across the primitive and substrate volume matrices are
  replaced by a single declared shape table with named hostile groups, and a
  run-time contract fails by name when a crate draws none of a declared group.
  The masked forward-step reference oracle is owned once as well, so the two
  crates that claimed independent oracles no longer share a byte-identical copy
  of one.
- The dense byte-tile Four-Russians matvec corpus has one owner,
  `tests/support/dense_matvec_cases.rs`, with one arm per crate:
  `vyre_libs::bitset::four_russians` pins its byte-LUT builder, word-count
  helper, CPU reference and dispatch Program, and
  `vyre_libs::encoding::bitset_transform_pipeline` pins its own sizing, LUT
  builder, parity oracle and composed Program. The corpus generator, the
  frontier masking and the naive boolean-semiring oracle were written twice,
  and the two copies swept different bounds: 0..=18 byte tiles by 1..=5
  destination words on the primitive side, 0..=24 by 1..=4 on the substrate
  side, so 384 cases in the union were exercised by neither. Both arms now run
  the union, plus a saturated-frontier group neither had, which is the only way
  the all-ones LUT row of every tile is reached at once.
- The dirty-output contract for a dense byte-tile matvec Program is asserted
  once, by `tests/support/dense_matvec_cases.rs`, which drives the reference
  interpreter with the all-ones output buffer and takes the LUT builder and
  Program builder as arguments. The two arms ran byte-identical interpreter
  setups and differed only in which builders they named, which is why they were
  the largest cross-crate duplicate pair in the repository. The primitive arm
  passes
  `vyre_libs::bitset::four_russians::four_russians_dense_matvec_byte_lut` and
  the substrate arm passes
  `vyre_libs::encoding::bitset_transform_pipeline::four_russians_dense_matvec_program`,
  and the failure message names which arm failed.
- The exploded-supergraph (IFDS) CPU-reference corpus has one owner,
  `vyre_test_support::exploded_ifds_cases`, which declares the cases and owns
  what a correct CSR for them is. `vyre_libs::graph::exploded` pins its
  allocating, fallible and workspace-reusing builders against it, and
  `vyre_libs::graph::dispatch::exploded` pins its host reference, its
  node-count helper and its dispatched path. The mixed intra/inter/GEN/KILL
  case stream was written twice and the copies ran different counts, 1024
  against 512, so the dispatched path was never asked about the upper half of a
  corpus its own file defined. The rule semantics that were hand-checked per
  suite, KILL suppression, GEN injection and inter-edge fact propagation, are
  now declared as dense edge expectations both arms are held to.
- The masked forward-closure reference oracle used by the persistent-BFS sweeps
  has one owner instead of a verbatim copy in each of the two crates that
  claimed to check each other with independent references.
- Six IR and serialization files are split along the seam their contents
  already had. `BufferDecl` keeps its own file and the two records it carries
  move out with the tests that own their grammar, `linear_type` and
  `shape_predicate`. Program metadata separates the bounded wire-hash fallback
  and the canonical buffer key, which decides program equality, from the
  `Program` methods that call them. Type checking separates the one static type
  walker from what each operator accepts in each operand position. The wire
  encoder and decoder each grow a `buffer_table` module holding the
  buffer-table and memory-region halves, which mirror each other and were the
  two largest blocks in either file. `Ident` moves out of the expression
  module. No signature changes and no re-exports removed.
- vyre-grammar-gen publishes each generator through its own module (dfa, lr,
  c11_lexer, wire, host_preprocess, lex_c11_max_munch, chunk_lexer_cpu,
  max_munch_cpu) instead of a flat crate-root re-export, because two sibling
  tables both define Action and a flat name cannot say which one it is.
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
- Three duplication pins record what the tree measures after the owners above:
  `vyre-driver` 1047 to 994 duplicated lines, `vyre-emit-naga` 221 to 155, and
  `vyre-reference` 743 to 676, each with `total_lines` measured. A pin with
  room under it hides the next copy.
- The hygiene scan's pattern names are declared where the scan emits them.
  `xtask::gates::hygiene_matrix` exports the hidden-fallback, resource-bound
  and cargo-wrapper pattern lists, and the release-evidence check that requires
  each family to have been covered now references them instead of restating
  nineteen names. Adding a pattern to the scan previously left the gate
  accepting evidence that never looked for it.
- The persistent collections come from `imbl` 7.0.1 instead of `im` 15.1.0.
  `im` carries RUSTSEC-2026-0248 as unmaintained, pulls `bitmaps` 2.1.0 which
  carries RUSTSEC-2026-0247, and its `OrdSet` insertion has an aliasing
  violation, RUSTSEC-2023-0126; `cargo deny check advisories` failed on all
  three. `imbl` is the maintained fork with the same API. The structural
  sharing is why a std map is not the answer: the DCE live set is cloned at
  every branch arm and every loop body, and the lowering variable scope
  snapshots its binding map per scope, so a std map would deep-copy every
  identifier where the persistent map shares its structure.
  `vyre_foundation::optimizer::passes::fusion_cse::dce::LiveSet` is new and is
  the one place the live-set type is named, replacing the concrete set spelled
  in four files.
- Items no consumer can reach are no longer declared public: the value-range
  body walk, the shared-memory promotion budget entry, and the pipeline blob
  ceiling are private to their crates, and the deprecation warning code has one
  published path at vyre_driver::DEPRECATED_OP_CODE.
- The IR-shape proptest corpus generates all twenty `Node` variants, up from
  eight. Async transfers and waits, traps and resumes, the four collectives,
  indirect dispatch, regions and opaque extensions were never produced, so
  every property that walks the corpus was only ever compared on stores, lets,
  assigns, ifs, loops, blocks, barriers and returns. A gate enumerates the
  variant names the IR registry declares at run time and fails until each is
  either generated or recorded with a reason, so a variant added later cannot
  silently escape the corpus. Case counts on the two cross-walk properties are
  raised to match the wider shape space.
- `bellman_shortest_path` and `sinkhorn_iterate` take a named binding record
  plus an extents record instead of nine and thirteen positional arguments. The
  IR-parity test for Sinkhorn had been passing its ten same-typed buffer names
  in binding order rather than parameter order, so it emitted a program that
  named its kernel matrix `u_curr` and its convergence flag `kv`; a record
  forces each name to be spelled at the call. `bellman_tn_order_program` and
  `sinkhorn_full_clustering_program` forward the same records.
- The Jacobi eigensolver (`symmetric_eigen_jacobi`) distributes independent
  identity seeding (`matrix_identity_fill`), eigenvector sign canonicalization
  (`eigenvector_column_sign`), and diagonal extraction
  (`matrix_diagonal_extract`) across declared workgroup lanes (`LANES = 64`),
  preserving sequential Givens rotation on lane 0 behind workgroup barriers.
- `reasoning::finite_category` states each construction once. Left and right
  Kan extension were four functions differing in whether the fold summed or
  multiplied and whether it ran per object or over a table; they are now
  `kan_extension_at` and `kan_extension_table` taking a `KanDirection`, over
  one shared fold. `is_adjoint_pair` is `adjoint_pair`, because it returns the
  pair and its witness rather than a bool, and `yoneda_natural_iso` is
  `natural_transformation_count`, because a count is what it returns.
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
- Launch geometry is a lowering decision produced by backend GeometryStrategy
  from neutral GeometryRequirements rather than hardcoded in library
  operations.
- `scripts/check_layering.sh` discarded `cargo tree` stderr, so a cargo that
  could not resolve the workspace printed a green result and exited 0. It now
  derives every workspace member, requires a `docs/CRATE_OWNERSHIP.toml`
  decision for each, and holds every transitive internal edge to the declared
  closure.
- The transitive layer closure is a registered gate.
  `scripts/check_layering.sh` and `scripts/lib/check_layering.py` are deleted
  and `layering` reads the graph from the manifests and `Cargo.lock` instead of
  from `cargo tree`, so it produces a verdict on a workspace cargo cannot
  resolve, which is the state the rule exists to describe. Member edges are
  read with the member's own default features activated, the edge set `cargo
  tree --edges=normal` prints; third-party edges come from the lockfile, so a
  substrate-neutral crate that reaches a backend API only through another
  third-party crate is caught rather than missed. A finding names the chain it
  found, not just the endpoints. An unregistered member, a layer with no
  neutrality decision, a decision no member uses, and a backend API name absent
  from `[workspace.dependencies]` are all errors rather than findings, because
  each of them makes the neutrality rule answer for a roster nobody reviewed.
- The `self-substrate-adapters` feature of `vyre-driver` and `vyre-runtime` is
  now `libs-compositions`. It was named for `vyre-self-substrate`, which is now
  `vyre-pass-engine`, and it never depended on that crate; it selects the
  `vyre-libs` composition domains behind driver cache invalidation, substrate
  telemetry, and the megakernel planner's program builders. Pre-1.0 rename with
  no alias.
- `vyre-libs` feature `full` is every dialect the crate ships, meaning every
  domain that submits an operation registration: it gains `logical`, `visual`,
  `security`, `parsing` and `matching-nfa`. It stops short of the compiler's
  own self-use composition domains, which submit nothing. Before this, `full`
  was a subset and two consumers each carried the same ten-name list to reach
  the rest.
- Two duplication pins record what the tree measures: `vyre-libs` 11425 to
  10453 duplicated lines and `vyre-primitives` 6981 to 6384, each with
  `total_lines` measured. Collapsing one side of a copy lowers both crates'
  counts, and a pin with room under it hides the next copy.
- The `vyre-libs` duplication pin records what the tree measures rather than
  leaving room under it, since a pin with slack hides the next copy.
- `vyre_libs::graph::dispatch::traversal_dispatch_pipeline` names the
  `vyre_libs::graph` program builders it composes instead of re-publishing
  them. Twenty-one wrappers forwarded every argument and returned the result
  unchanged, so each body was a restatement of the primitive's parameter list,
  free to drift from the list it forwards to and unprovable from either side.
  Nothing outside the module's own tests called any of them. This follows the
  same removal already applied to the sibling `structural_kernel_pipeline`.
- Three shapes repeated across the `vyre-libs` test corpus have one owner each.
  `vyre-libs/tests/support/optimizer.rs` owns `assert_optimizer_is_idempotent`
  and the debug first-difference reporter that names which node moved when the
  optimizer fails to reach a fixed point; three idempotence contracts carried
  their own copy of both. `vyre-libs/tests/support/ir_fingerprint.rs` owns the
  pinned-fingerprint family guard that proves a collapsed clone family still
  emits what every former copy emitted. Seven files that reimplemented
  little-endian `pack`/`unpack` read `vyre_primitives::wire::pack_u32_slice`
  and `decode_u32_le_bytes_all`, the crate that owns that layout.
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
- The duplication baseline pin for vyre-lower records the measured tree: 279 to
  205 duplicated lines, with `total_lines` measured at 13187. The header block
  listing rows pinned below their measurement is gone with its only entry.
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
- Target-payload admission has one owner for the last five clusters the four
  driver copies still shared. `MaterializerDevice::admit_modules` runs the
  admit-size-decode loop and each backend supplies only its dialect decode;
  `InstanceCore::submit_host_only` is the whole submission path for a backend
  with no resident route, and it reads the refused backend off the recorded
  device generation instead of a constant restated at the call site;
  `InstanceCore::ordered_resident_resources` owns the resident handle lookup,
  so the two backends with a resident path no longer each project buffer names
  onto canonical values; `materialize::omitted_output` owns the omitted-output
  rejection, whose sentence six copies had written out and one had already lost
  the recompile instruction the other five give; and `executable_module!`
  answers the two `ExecutableModule` methods every backend stored under the
  same two field names.
- Megakernel builder and scheduler body inspection now walks nested node bodies
  through one shared preorder helper built on the foundation-owned child-body
  enumeration, instead of two hand-written recursive matches that each had to
  list the nesting variants.
- The megakernel empty-template cache and the resident launch recommendation
  cache now share one bounded least-recently-used map, so eviction order and
  tick saturation are decided in one place instead of two.
- Persistent, finite and JIT megakernel lane bodies now share one assembly path
  and one published-slot claim body, so the node reservation bound, the
  slot-base binding order and the IO polling block are decided in a single
  place. Emitted IR and reservation error text are unchanged.
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
- `vyre-driver-metal`'s `backend_metric_snapshot` builds its scalar counters
  from a name-and-accessor table rather than fourteen hand-written pushes. A
  counter added to `MetalMetrics` and forgotten in the snapshot was previously
  invisible, because the push list carried no relation to the struct. The
  resident-buffer branch moved to `push_resident_table_metrics` next to the
  lock helper it shares a failure mode with, and the comment explaining the
  poison sentinel states the reason instead of citing an internal rule number.
- The programs the native Metal tests dispatch have one owner in
  `vyre-driver-metal/src/tests/fixtures.rs`. The single WriteOnly `u32` output
  word, declared with count 1 and an output byte range of 0..4, was restated at
  nine call sites across four modules, and the
  ReadOnly-input-plus-WriteOnly-output pair at four more, twice byte for byte
  inside one file. The declared element count and the output byte range
  together decide how many bytes the backend collects, so every test asserting
  a single little-endian word was asserting against a shape it also restated:
  one edited count would have retargeted the assertion at a different number of
  bytes in one place and left the others agreeing with each other. Each module
  now imports what it uses once at module scope instead of repeating the same
  two `use` lines inside every test function.
- The native Metal against wgpu-on-Metal comparison lives in
  `vyre-driver-metal/src/tests/wgpu_differential.rs`. It was the second test in
  the one-shot dispatch module, whose subject is what Metal does with a program
  rather than whether two backends agree.
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
- The structure gate rejects concrete backend vocabulary in production source
  of a substrate-neutral crate, reading the word list from
  structure-gate/backend-vocabulary.toml at run time. The roster is every
  workspace member whose layer in docs/CRATE_OWNERSHIP.toml is neutral, so a
  new neutral crate is scanned without a rule edit. Documentation and comments
  in vyre-driver, vyre-lower, vyre-runtime, vyre-spec, vyre-reference,
  vyre-libs, vyre-foundation and vyre-primitives name capabilities and target
  dialects instead of vendors and emitter crates.
  vyre_runtime::uring::gpudirect renames its stats reader and byte cap to
  read_gpudirect_stats and MAX_GPUDIRECT_STATS_BYTES.
- Shared crates describe launch planning, bitset compression, frontier encoding
  and scan admission in neutral terms instead of naming one vendor backend.
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
- `vyre_libs::solvers::numerical_kernel_pipeline` re-exports the Sinkhorn entry
  points of `vyre_libs::math::sinkhorn_iterate` instead of wrapping them. Seven
  wrappers forwarded every argument and added nothing, so their whole body was
  a restatement of the primitive's parameter list, up to twelve positional
  arguments, free to drift from the list it forwards to and unprovable from
  either side. A composition names the primitive it composes. The public names
  are unchanged.
- `MapResult` has one definition. `vyre-driver-wgpu` spelled `Result<(),
  wgpu::BufferAsyncError>` three times: the public alias in the readback ring
  plus a private copy in the readback and timestamp recorders, one of which
  imported the public alias in the same file it redefined. Both recorders use
  the published alias.
- Partial RoPE is emitted by the attention layout base, so one builder now owns
  every guarded element move in the attention and paged-cache families.
- ExternalIfdsSecurityBuffers borrows its ten buffer names and publishes them
  as ExternalIfdsSecurityBuffers::CANONICAL. The ten-argument positional
  constructor is gone; build the record from CANONICAL or from a struct
  literal. ExternalIfdsSecurityDispatch and
  route_security_taint_through_external_ifds carry the buffer lifetime.
- The `hot-path-nested-rows` gate reads the trait that returns nested byte rows
  rather than counting the text of the type in one crate. A dispatch trait that
  returns `Vec<Vec<u8>>` must also declare a form that fills slots the caller
  keeps, and such a form must fill them rather than assign fresh rows through
  the parameter. The counted spelling condemned the shared dispatch ABI at one
  backend while the caller-owned slot machinery that answers it read as a
  finding too, so the pin could only be met by rewording.
- Every registered xtask subcommand is now a gate answering one contract: it
  returns findings and notes instead of printing, and the runner decides what
  that means. The `Kind` enum is gone, so no check is exempt from the sweep by
  category. The `check-cat-a` and `release-gate` composites are named subsets
  of the registry, `xtask gates --subset cat-a` and `--subset prepublish`, and
  the cargo invocations they drove are gates of their own: `workspace-check`,
  `workspace-clippy`, `workspace-tests`, `workspace-docs` and `lockfile-clean`.
  `scripts/check_op_names.sh` and `scripts/check_parity_testing_not_leaked.sh`
  became the `op-names` and `parity-testing-isolated` gates.
  `xtask/gate-baselines.toml` pins `findings` rather than output lines, one row
  per registered gate, and the sweep enumerates the registry at run time so a
  gate without a row and a row without a gate both fail. A gate that owns a
  generated artifact checks it against the tree and rewrites it under
  `--write`, so regeneration is never a subcommand of its own.
- The lint policy is declared once, in [workspace.lints], and every workspace
  member inherits it. Two members declared their own tables and twenty crate
  roots overrode the policy with an inner attribute, so vyre-driver-metal
  allowed unsafe_code outside the reviewed budget and vyre-grammar-gen held
  missing_docs at warn while the workspace denied it. Blanket allows that warn
  by default moved into the workspace table with their justifications; the
  pedantic and restriction names beside them were suppressions of lints no
  member enables and are gone. The new lint-one-policy gate reads the member
  roster from the workspace manifest at run time and reports a member that
  declares its own table, omits the inheritance, or sets a level at its crate
  root; the only accepted crate-root exception is allow(unsafe_code) alone,
  which xtask/unsafe-budget.txt already reviews. It replaces
  lint-missing-docs-override, which read one lint out of that population.
  Deleting vyre-driver's crate-wide allow(unused_imports) exposed twelve unused
  imports, and one crate-root override was hiding a missing crate document.
- The host and resident execution paths of a materialized artifact instance
  have one implementation, `vyre_driver::materialize::MaterializedInstance` and
  `ResidentInstance`, instead of one per driver. Four backends wrote the same
  two bodies around the same `InstanceCore` calls and the same submission
  routing; what differs is the launch, the handle order a resident launch
  reads, and the rejection text, so those are the hooks and everything else is
  a default. A backend now supplies a launch and its own wording.
- Each decode codec is one module with one registered op id. base64, hex and
  inflate had a builder and a registration on both sides of the crate boundary,
  so a built program carried a registered region nested inside a second
  registered region naming the same work. The surviving ids are
  vyre-libs::decode::base64, vyre-libs::decode::hex,
  vyre-libs::decode::inflate_stored_block and
  vyre-primitives::decode::ziftsieve_literal_copy. The ids
  vyre-primitives::decode::base64_decode, vyre-primitives::decode::hex_decode,
  vyre-primitives::decode::inflate_stored and vyre-libs::decode::ziftsieve are
  gone, and programs from the three collapsed codecs lose one level of region
  nesting. The inflate trap diagnostics now name
  vyre-libs::decode::inflate_stored_block. ziftsieve_literal_copy_with_op_id is
  removed; it existed only to stamp a caller-supplied op id across the crate
  boundary the collapse deleted.
- Every module directory under a `src/` tree is entered through its own
  `mod.rs`. The workspace carried both layouts at once: 363 directories had a
  `mod.rs` while 113 modules were a file sitting beside the directory holding
  their children, so where a module began depended on which crate you were
  reading. The 113 files moved into their directories, and the 189 `path`
  attributes that had been pointing the compiler back out of those directories
  are deleted, because the default resolution now finds every child. No item
  moved between modules and no public path changed.
- The `all-lego` feature of vyre-primitives is gone. It aggregated the
  composition domains that have since moved to vyre-libs, so it had become an
  alias for `hardware` that gated no source line, while three manifests still
  justified requesting it in terms of bitset, decode, graph, geometry and
  optimization operations the crate no longer carries. Consumers name
  `hardware`, which is the one domain the crate declares. The conform workflow
  was naming twelve vyre-primitives test targets that moved to vyre-libs with
  their domains; it now runs each target against its owning crate with the
  features those targets require.
- One online-softmax core serves the whole attention family in vyre-libs. The
  scalar flash-attention kernel, flash_attention_2 and the reference copy each
  carried their own running-maximum recurrence, so a fix to one left the others
  wrong; the scalar path is now the tiled core at tile_size 1 and
  nn::attention::tiled_online_softmax is the single registered owner of the
  recurrence. The layout family collapses the same way: layout_permute,
  head_to_token, token_to_head and kv_cache become one index-map base that
  names its move, and the typed twins beside them are gone. Public API break:
  flash_attention_2_reference is removed. It was a scalar copy of the
  recurrence under test, so it agreed with the kernel by construction rather
  than by being right; compare against nn::attention::attention_reference, the
  offline three-pass schedule, which is the oracle every parity test in the
  family already uses. The pinned IR fingerprints for flash_attention move,
  because the scalar plan now stages scores through the shared score tile and
  renames its accumulator scratch. Both plans now report the shared-memory
  figure their program declares: the scalar plan counts the score tile and the
  accumulator alongside the query scratch instead of the query scratch alone,
  and the tiled plan drops the split-reduction state, which combines across
  workgroups and is therefore never allocated as workgroup memory.
- The gate tooling reads and writes a JSON document through one module.
  xtask::json_document owns both directions; the package readiness matrix, the
  release benchmark metrics and the release backend suite all read through it
  instead of each spelling its own bounded read and serde parse.
- A gate that reads a string field out of a TOML row now calls
  xtask::toml_text::string_field. The CLI documentation generator and the
  documentation checker had the same row-to-scalar closure.
- Eleven builders computed their own buffer cell count and wrote their own
  overflow message. `vyre_libs::math::matrix_cells` and `square_matrix_cells`
  now own both: the count of a `rows x cols` operand, the `n == 0` rejection
  for a square one, and one sentence naming the caller and the shape that did
  not fit. The messages change text. They previously named a domain noun the op
  id already carries, and eleven copies had drifted to eleven phrasings of the
  same fact.
- vyre_foundation::transform::grid_sync_split owns the whole-grid fence walk,
  hoist and segmentation. The compile-time planner cut and the dispatch-time
  split in vyre-driver both call it, replacing the copy that lived in
  vyre-driver. Hoisted let bindings are collected in sorted order, so a split
  segment's entry sequence no longer depends on a hash seed and the compiled
  artifact digest is reproducible.
- Scope truncation at the first Return has one implementation,
  vyre_foundation::transform::rewrite_walk::reachable_prefix. The pass engine
  encoder and the foundation DCE both read it instead of carrying their own
  copy.
- Every `Node::Region` in `vyre-primitives` and `vyre-libs` is now built by
  `vyre_foundation::algebra::composition::wrap_anonymous_region` or
  `wrap_child_region`. 188 hand-written struct literals restated the same three
  fields, and each one was a place where a generator name could be spelled
  without the `anonymous::` prefix the audit gates read, or a child region
  could be attached with no parent. The literals carry no information the two
  constructors do not, so they are gone.
- The line scanner that separates code from a comment, counts nesting, and
  decides which lines belong to an inline test module lives in the scan module
  that already owns what is not code. The hot-path scan held the only copy, so
  the heuristic audit could not tell a test fixture from production debt.
- The batched packed-activation INT4 matmul and its top-1 routing variant
  carried the same nine-node inner product, the same nibble select, and the
  same row-and-batch scale product, differing only in the prefix on every
  binding name. Both now build them from one parameterized helper in the
  quantized expression module. Both emitted programs are unchanged. Six
  duplicated doc summaries in the same file are gone.
- vyre-aot, vyre-debug, vyre-driver-spirv, vyre-driver-wgpu, vyre-emit-naga,
  vyre-grammar-gen, vyre-megakernel, vyre-reference and vyre-runtime publish
  each item at one path; submodules that exist because a file was split are
  private and their owning module re-exports what it holds.
- A composition region is wrapped through vyre_foundation::composition and
  nowhere else. vyre_libs::region re-exported the three wrappers under shorter
  aliases, vyre_libs::operation_catalog re-exported those aliases again, and
  vyre_primitives::hardware::region reimplemented all three over Node::Region,
  so one fact had four names and two implementations. Both modules are deleted
  and every caller in vyre-libs and vyre-primitives names wrap_region,
  wrap_anonymous_region, wrap_child_region, tag_program and
  reparent_program_children directly, as the rest of the workspace already did.
  The dialect template, the authoring guide and the gate fix messages name the
  same path.
- The benchmark harness delegates the source fingerprint and the dirty worktree
  digest to xtask::source_provenance, the one producer. Two implementations
  kept in agreement by a test were one duplication; the test now proves the
  delegation and goes red if a second implementation returns.
- vyre_driver::materialize owns the projection from a binding plan to the
  resident buffer names a dispatch must supply, and the unbound-resident
  rejection. The CUDA and Metal materializers each carried the same filter; the
  scratch-exclusion proof moved to a test beside the projection.
- Every `vyre-foundation` module has one public path. `algebra`, `analysis` and
  `dispatch` were grouping directories, each holding one or two unrelated
  modules, and each needed a crate-root re-export because callers named the
  short path: `vyre_foundation::composition` and
  `vyre_foundation::algebra::composition` were the same module reached two
  ways, as were `graph_view`, `dialect_lookup` and `extension`. The wrappers
  are gone and the five modules sit at the crate root, which is the path 200
  files already used. The crate-root item re-exports of `from_graph`,
  `to_graph`, the graph types, and the operation signature types are gone with
  them; two callers now name `dialect_lookup` directly. `visit` split into
  `node`, `expr` and `walk` while publishing all three, so every traversal
  answered to two paths; the submodules are private and the re-export at
  `visit` is the one path, which is what its own module documentation already
  claimed.
- Every item the composition move brought into vyre-libs has one public path,
  and that path names the module that owns the item. A codec is reached through
  its own module, so vyre_libs::decode::hex::hex_decode replaces
  vyre_libs::decode::hex_decode. The encoding-classification constants are
  reached through text, the adaptive traversal selectors through
  graph::adaptive_traverse, the d-DNNF gate encoding through
  graph::knowledge_compile, the region triple through matching,
  gaussian_weights through math::conv1d, and the semiring through
  math::semiring_gemm.
- Every item a vyre crate publishes is reachable at one public path. A
  submodule that exists because a file was split is now private to its crate
  and the owning module re-exports what it holds, so vyre-foundation,
  vyre-libs, vyre-driver, vyre-driver-cuda, vyre-lower and vyre-spec no longer
  publish the same item under both a flat name and a deep module path. Measured
  second paths fall from 3331 to 493 across the 26 committed snapshots.
- Every resident work queue item is published at one path. The parent module
  blanket re-export of 174 names from its public submodules is gone, so a
  caller names the submodule that owns the item, and the crate duplicate-path
  count drops from 735 to 119.
- Every runtime type, function and constant has one public path: the module
  that owns it. The crate root re-exported about thirty items that were already
  public at `vyre_runtime::replay`, `::tenant`, `::pipeline_cache`,
  `::artifact_admission`, `::persistent_executor` and `::recovery`, so the
  published surface carried 122 duplicate entries and a reader had no way to
  tell which path was canonical. The root re-exports are gone, every caller in
  the tree names the owning module, and the `vyre` facade re-exports from those
  module paths rather than from a second index.
- scallop_join takes the words-per-cell width. It is now scallop_join(state,
  next, join_rules, changed, n, w, max_iterations), and w = 1 emits the
  single-word bodies the old signature emitted. The separate scallop_join_wide
  op, its dispatch grid and its CPU oracle are removed: they were the same
  Lineage-semiring Datalog fixpoint with a second registration, a second parity
  oracle and a second set of trap messages, and the width was already a
  parameter of the shared bodies in scallop_persistent. semiring_gemm_wide
  moves to its owning domain as
  vyre_primitives::math::semiring_gemm::semiring_gemm_wide. The dispatch grid
  is documented as one lane per relation cell, with the lane walking the w
  contiguous words of its cell, so it does not scale with w.
- The two grouped INT4 linear lowerings, the lane-predicated one and the
  weight-tile-reuse one, shared five stages by copy: the workgroup lane
  decomposition, the packed-column index, the nibble select, the affine
  dequantization, and the warp reduction with its lane-zero biased store. Each
  is now built once in the grouped layout module and used by both, so a change
  to the packed weight layout or the reduction cannot land in one strategy and
  miss the other. Both emitted programs are unchanged.
- The host-side IR rewrites the resident pipeline runs are declared once in
  vyre_foundation::transform::HOST_REWRITES, in pipeline order, and the
  pipeline walks that table instead of naming each function. A rewrite absent
  from the table does not run, and the firing-case suite reads the table rather
  than scanning the source directory at test time.
- Test suites that reach only a crate's public API now live in that crate's
  tests/ directory. A suite that stays beside the code it exercises states
  which crate-private item it covers. Shared test fixtures that had been copied
  per target now have one owner. Test counts are unchanged.
- IR traversal has one module. vyre-foundation published two modules named
  visit, each with its own ExprVisitor and NodeVisitor trait: one the
  exhaustive per-variant contract, one a pair of one-method callbacks a walk
  pushes into. A crate cannot say which ExprVisitor a reader means, and the
  exhaustive contract already imported child_bodies from the other. The
  traversals, the per-variant decisions they are written against (node_parts,
  expr_parts) and the visitor contracts now live under vyre_foundation::visit,
  and the walk callbacks are named for what they are: NodeSink::accept_node and
  ExprSink::accept_expr. Lowerable and Evaluatable moved into files named for
  the contract each holds instead of a module named traits.
- The release-workload-matrix gate is the only writer of
  release/evidence/benchmarks/release-workload-matrix.json, vyre-bench
  release-matrix prints and no longer writes, and a test compares the committed
  body against what the case registry derives.
- `vyre_lower::op_facts` was both a module and a function inside it, so a doc
  link to either was ambiguous and `crate::op_facts` resolved to whichever
  rustdoc preferred. The function is `facts_for`, reached as
  `vyre_lower::facts_for` or `vyre_lower::op_facts::facts_for`. Five call sites
  and the crate root re-export name the new spelling.
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
- One test-only opaque extension pair serves every wire suite.
  `vyre-foundation/tests/support/opaque_echo_extension.rs` owns `EchoExpr`,
  `EchoNode`, their kinds and both `inventory` registrations; the opaque round
  trip, the adversarial wire cases and the round-trip property included their
  own copies. The pair is the contract under test, since an extension whose
  `wire_payload` and registered `deserialize` disagree makes all three suites
  pass against a resolver that does not round-trip, so one definition means one
  place that can be wrong. The copies were not identical after all: one carried
  a resolver that refuses a payload beginning `0xDE 0xAD`, which is the only
  reason the adversarial suite could prove `Program::from_wire` reports a
  refusal as a structured error rather than panicking past it. That rule is now
  declared as `REFUSED_NODE_PREFIX` beside the pair and applies to the
  statement half only, because the round-trip property builds one program
  holding every expression variant with payloads it does not choose. Each
  consumer includes the file with `#[path]`, since the resolver table is per
  test binary.
- Three optimizer hot paths stopped allocating per sample.
  `HotPathHints::record` allocated the key on every call including a repeat
  sample, and its LRU eviction cloned every key in the map to find the oldest;
  it now takes a `get_mut` fast path and clones the one key it evicts.
  `reaching_def_propagate` keyed its propagatable-let map and its
  shadowed-binding set by `String`, so every binding heap-allocated and
  memcpy'd its name twice per scope walk; both are keyed by `Ident`, which is
  an `Arc<str>` refcount bump and hashes through the same `str`, so `&str`
  lookups are unchanged. `loop_fission` walked its body building a new `Vec`
  node by node whether or not a fissionable loop existed; it locates the loop
  first and copies the prefix in one `extend_from_slice`.
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
- `vyre_foundation::optimizer`'s per-pass fixed-point contract is measured
  against every registered pass, discovered through
  `vyre_foundation::optimizer::registered_pass_registrations` and scheduled
  with the passes it declares a requirement on. The seven pass names it
  hardcoded were a fifth of the registry and could not go stale loudly: a pass
  registered afterwards was never held to the contract and nothing said so. The
  entry-point half is now a declared table, so `canonicalize_engine::run` and
  `optimize` are each held to the union of what the two suites separately
  asserted, three runs compared structurally and on the wire with
  reference-interpreter parity checked on every run.
- The generated-program corpus and the run-then-compare scaffold the optimizer
  contract suites draw from have one owner,
  `contract_cases::optimizer_program_corpus`. Two suites carried a copy each,
  down to identical recursion depth and branch weights, with two names for the
  same single-store program builder. Two copies of a generator is not a
  duplicated helper: the property a suite proves is only as wide as the
  programs it draws, so a generator that drifts in one file narrows one suite's
  claim while both stay green.
- `vyre_foundation::optimizer::passes::loops::substitution` no longer
  re-exports `vyre_foundation::transform::subst` under a second path; the loop
  passes name the owner. A module whose body is a re-export is not an owner,
  and it leaves a reader with a question that has no answer: which of the two
  paths is the real one.
- Foundation now exposes IR-specific `IrError` and `IrResult` contracts instead
  of a cross-domain error sink. Reference interpretation, backend execution,
  WGPU device selection, and runtime framing return owner-local typed failures.
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
- The pass engine has one scope walk. `optimizer::rewrite_walk` owns the
  encoder's reachable-prefix truncation and the borrow-preserving rebuild, and
  const propagation, cross-scope CSE, encoded CSE and the encoded-order rewrite
  call it. Four copies of the same loop stood beside each other and three of
  them discarded the walk's unchanged answer, so a pass that propagated nothing
  still deep-copied every nested body it visited. Constant propagation also
  stopped enumerating `Node` itself: it is a policy over the one structural
  node rewrite, so a body-bearing variant added to the IR is descended into
  instead of switching propagation off for everything inside it.
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
- Comments and doc comments in `vyre-foundation`, `vyre-primitives`,
  `vyre-runtime` and `vyre-pass-engine` state the hardware fact instead of
  naming a backend product. A load past a buffer end is undefined behaviour on
  a real GPU whichever driver reached it, a nested `Return` lowers to an exit
  branch in every machine-code emitter, and a launcher refuses a zero grid
  extent everywhere. The old wording pinned a general rule to one vendor, so a
  reader on another backend had to guess whether the rule applied to them.
  Backend-specific text now lives only in the crate that owns that backend.
- `vyre-primitives` classifies every Cargo feature in one place,
  `src/organization.rs`. Marker types and `hardware` belong. The other domain
  features are compositions parked pending a move to `vyre-libs`.
  `tests/feature_classification.rs` fails if `Cargo.toml` grows a feature that
  is not on exactly one of those lists, or if `hardware` is no longer the only
  intrinsic domain.
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
- `Program::canonicalized` reports what canonicalization changed instead of
  rebuilding the tree to discover it changed nothing. An already-canonical
  program keeps its entry body, its buffer table, and its memoized fingerprint
  rather than being replaced by an equal copy, and a program that only needs
  its buffer table sorted keeps its entry body. Canonicalization stays a fixed
  point in one application and a semantic no-op, both now asserted over the
  shipped release corpus and over shapes the corpus does not generate: a
  `Block` that owns a binding is a scope, so it survives, while a `Block` that
  owns none is spliced into its parent at every body position.
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
- The `vyre-reference` and `vyre-runtime` public-API snapshots record two moves
  already in the tree. Every `LocalSlots::visit_*` signature names
  `vyre_foundation::ir_inner::model::expr::ident::Ident`, which is where the
  identifier type lives since the expression module was split by what it holds.
  `ReplayFailureEvidence` and `ReplayFailureClass` are named from the
  `vyre-runtime` crate root, and the replay log's record and capacity constants
  `HEADER_BYTES`, `LOG_MAGIC`, `LOG_VERSION`, `MAX_REPLAY_RECORDS` and
  `RECORD_BYTES` are public, so a consumer sizing a replay log reads the same
  numbers the writer uses.
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
- README.md is a landing page: what vyre is, how to install it, one example,
  and links out. The crate-by-crate boundary manual it carried moved to
  docs/architecture/crates.md and the megakernel and persistence model moved to
  docs/architecture/artifact.md, both verified against the manifests and the
  compiler source. Its previous example did not compile against any released
  version: it called vyre::backend::select, which the facade does not export,
  and passed three arguments to a composition that takes six.
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
- The flat-evaluator test harness in `vyre-reference` has one owner,
  `vyre-reference/tests/flat_expr_eval/mod.rs`. The adversarial proptest, the
  adversarial gap suite and the subnormal flushing contract each carried their
  own wrapper program, zero invocation, expression-evaluation entry, literal
  `BinOp`/`UnOp` builders and float-bit extractor, and the copies had already
  drifted in panic text. The IEEE-754 canonicalization the sweeps compare
  against stays an independent restatement rather than a call into the
  interpreter, so the oracle still does not check itself.
- The reference interpreter now consumes foundation-owned IR, diagnostics, and
  operation metadata directly instead of depending on the public `vyre` facade.
- The duplication baseline pins for vyre-reference and vyre-spec now record the
  measured tree: 971 to 743 and 446 to 244 duplicated lines. Both total line
  counts are corrected to the measured values.
- Node::Region cites its parent composition through ir::Ident. The GeneratorRef
  wrapper around String is deleted, so the IR carries one name type and a pass
  that rebuilds a region shares the interned name instead of allocating a copy
  of it.
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
- The seventeen registry-linked checks answer the gate contract. Each one
  returns findings instead of an exit code, so the sweep counts what it found
  and pins that count. catalog, list-ops and optimization-docs now own
  docs/generated/catalog.toml, docs/generated/op-inventory.toml and
  docs/generated/optimizer-passes.toml, compared on every run and regenerated
  with --write. lego-audit runs its repo-context checks unconditionally and has
  no report-only mode, lego-quick scans the whole tree unless --staged narrows
  it, heuristic-audit has no advisory mode, compile and shrink run over the
  generated release corpus, and verify-rewrite-proofs fails when no solver can
  discharge an obligation.
- Release benchmark commands now run 300 warmup samples before measurement so
  accelerator clock preconditioning is explicit and reproducible.
- The changelog and the release notes body are generated by the registered gate
  `xtask release-docs`, not by a Python script beside the registry. The gate
  reads the release train and one fragment file per change, renders the
  unreleased section, and writes both documents under `--write`; without it, a
  document that disagrees with the fragments is a finding. It reports what the
  script reported and three rules the script did not: the number of
  approval-gated external actions is taken from the launch contract rather than
  restated as a literal, a fragment is read from the directory so one written
  but not yet staged still reaches the changelog, and every finding names its
  file and its corrective action instead of raising on the first error.
- The release evidence readers in `xtask-evidence` read each shared concept
  through one owner. The eight wall-clock minima are a single `WallClockMinima`
  record flattened into the three artifacts that publish them, the schema-2
  JSON read, the workload matrix, the `u64` field read and the array length
  read are one function each, the optimization family list is
  `vyre_foundation::optimizer::corpus::RELEASE_OPTIMIZATION_FAMILIES`, and the
  benchmark case fixtures are built rather than restated. A copy that a reader
  forgets to update is a gate that passes on evidence nobody checked.
- `vyre-release-gate` judges prepublication evidence by default and takes
  `--launch-complete` for the post-ship mode. The default was launch-complete,
  which demands `public_launch_complete`, all three approval-gated external
  actions complete and zero blockers, so the gate every sweep runs with no
  arguments could not pass before the version it guards had shipped. The
  removed `--prepublish` flag is rejected rather than ignored.
- Release tooling now has one owner per shared shape. Subcommand dispatch and
  the agreement check between the assignment table and each delegate crate live
  in `xtask::subcommands`; the operation table cells shared by the catalog and
  the operation inventory live in `xtask-registry::docs::schema_cells`; the
  optimizer catalog row published by both the integration matrix and the corpus
  pass manifest lives in `xtask-registry::release::optimizer_pass_rows`;
  bounded file reads route through `xtask::output_arg::read_text_bounded`. The
  xtask integration tests link as three binaries instead of ten and the
  registry tests as one instead of two.
- The resident asynchronous overlap contract has one owner,
  `tests/support/resident_async_overlap_contract.rs`. Two backends had carried
  it verbatim apart from acquisition, message text and one assertion.
  Device-measured time is now a parameter rather than a shared assumption, so a
  backend without a device timer on that path is still held to exact per-slot
  outputs and to enqueue and wait time being separated.
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
- The resident work queue program builders are reached as
  vyre_runtime::resident_work_queue::build_program... instead of through a
  builder submodule path.
- `vyre-runtime` builds pipeline-cache test artifacts from one fixture module,
  `src/pipeline_cache/test_artifact_fixtures.rs`, which the fingerprint suites
  include by path. The two suites previously built the same artifact
  independently and `artifact_for_program` looked only at the first buffer, so
  a program whose second buffer had a different access produced a fingerprint
  the fixture could not distinguish. It now walks every buffer with that
  buffer's own access.
- The duplication baseline pins for vyre-driver and vyre-runtime now record the
  measured tree: 1463 to 1112 and 520 to 412 duplicated lines. The vyre-lower
  total line count is corrected to the measured value.
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
- `vyre_libs::math::scallop_persistent` owns `lineage_fixpoint_program` and
  `accumulate_lineage_words`. `scallop_join` and `scallop_join_wide` each
  carried their own copy of the persistent program envelope and the lineage
  word join, so a change to the convergence protocol had to be made twice and
  the two spellings had already drifted in how they reported change. Both now
  call the owner.
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
- Repository checks that lived in shell scripts are registered gates with
  pinned finding counts. The frozen contract snapshots, the backend extension
  rule, the readback ring routing, the program wire field classification, the
  source file size ratchet, the device loudness scan, the cross-crate
  unification rows, the release evidence path oracle, the invariant test
  citations and the doc claim manifest now run from the gate sweep and hold to
  `xtask/gate-baselines.toml`. Snapshot regeneration is `--write` on the gate
  that owns the snapshots. Three device loudness patterns had been unreachable
  since they were written, because a malformed expression made the search exit
  with an error that read as no match.
- Floating-point parity now uses one foundation-owned comparison contract for
  semantic operation witnesses, including each operation's declared tolerance.
  The library operation catalog distinguishes the complete semantic inventory
  from its deterministic executable-fixture projection.
- Five helpers that sat at a crate root now sit under the concern they serve.
  The signed fixed-point pair fixed_mul_16_16_expr and
  fixed_sdiv_by_positive_expr is vyre_libs::math::fixed. fixed_u32_matmul and
  chebyshev_recurrence are vyre_libs::math. nodeset_filter is
  vyre_libs::label::nodeset_filter, under label rather than predicate because
  predicate already reaches label and the reverse edge would invert that
  closure. demote_intermediate_outputs is
  vyre_libs::plumbing::program::outputs. lane_grid has one owner reachable as
  vyre_primitives::lane_grid and the graph re-export of it is deleted. The
  unsigned helper in the visual dialect that shared the name
  fixed_mul_16_16_expr is now fixed_mul_16_16_unsigned_expr; it delegates to
  wide_mul_shr_u32 and builds the same nodes as before. The graph scratch
  reservation helpers fold into the crate scratch owner as reserve_items,
  reserve_capacity, reserve_items_with and resize_vec.
- Six source files over their measured ceiling are split along the boundaries
  already inside them, and each new file is named for its contents.
  `vyre-runtime/src/tenant.rs` became
  `tenant/{error,quota,handle,registry}.rs`;
  `vyre-primitives/src/matching/dfa_compile.rs` became
  `dfa_compile/{wire,compile}.rs`; `vyre-driver-wgpu/src/buffer/handle.rs`
  became `buffer/{handle,staging,bind_group_cache}/`;
  `vyre-primitives/src/math/sinkhorn_iterate.rs` became
  `sinkhorn_iterate/{program,reference,reference_f64}.rs`;
  `vyre-primitives/src/text/utf8_validate.rs` became
  `utf8_validate/{program,sequence_rules,reference}.rs`; and
  `vyre-driver-wgpu/src/runtime/readback_ring.rs` became
  `readback_ring/{slot,stats,capacity,ring,ring_set}.rs`. Every public path is
  unchanged. Each test module moved next to the code it proves, which is what
  keeps `lock_inner` and `lock_cache` private to the bind-group cache instead
  of widening them for a test in another module.
- Duplication pins for three crates now record the measured tree: vyre-macros
  89 to 41, vyre-lints 66 to 0, vyre-debug 54 to 9. Their total line counts are
  corrected to the measured values, which were already stale before these
  changes. A no-op test whose body was a comment claiming to pin a test count
  is deleted, and the two `raw_ir_in_libs` case files, named for a helper and
  for one of their own tests, are now named for what they hold.
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
- The `CollectiveOp` proptest generator in `vyre-spec` has one owner,
  `vyre-spec/tests/spec_variants/mod.rs`, replacing the copy in the op-wire
  sweep and the copy in the collective sweep. The frozen collective wire-tag
  table now appears only in `vyre-spec/tests/collective_op_contracts.rs`, which
  owns the round trip, the unassigned-tag rejection and the world comm-group
  constant; the subset copy in the extension-category test is gone.
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
- `vyre-driver-spirv` answers the seven capability queries it does not probe
  from the `VyreBackend` defaults instead of restating `false` for each. The
  Vulkan compute path probes four limits and claims nothing else, and the trait
  default is already the conservative answer, so a capability added to the
  trait no longer needs a hand-written denial here to keep the backend honest.
- `vyre-driver-spirv` builds its `DeviceProfile` from
  `DeviceProfile::conservative` plus the four limits it probes, instead of
  restating all thirty-two fields. The driver already owned the
  unprobed-capability defaults and nothing used them; a new capability field
  had to be answered by hand in `vyre-driver-spirv/src/lib.rs` and the answer
  was always the conservative one.
- The element-wise programs the SPIR-V dispatch tests and the Vulkan probe
  example run have one owner in
  `vyre-driver-spirv/tests/support/elementwise.rs`, shared by path include. The
  add, output-first add, and multiply-add programs plus the `u32` byte
  conversions were written twice, once in the test and once in the example, so
  the example could drift from the shape the test pins. The test also grew a
  reference-output helper and a lane comparison, which replaces two copied
  assertion loops and three copies of the little-endian input encoding.
- `vyre-driver-spirv` uses the neutral instance-rejection record instead of a
  local copy of five entries. Three of its overrides restated the neutral
  requirement without naming a corrective action, one restated the neutral
  text, and one is now the neutral text. The crate carries no rejection wording
  of its own for the shared materializer contract, so the words a caller reads
  do not depend on which backend rejected the call.
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
- The structural graph, causal and logic kernels are reached at
  `vyre_libs::graph`, not through a second set of names in `vyre-libs`.
  `graph::dispatch::structural_kernel_pipeline` held sixty-six wrappers across
  a `dispatch` module and a `references` module, one per primitive builder and
  one per CPU oracle, each restating the primitive's parameter list verbatim
  and calling it with those parameters in that order. Nothing outside the
  module's own tests called one, so the layer bought a second signature to keep
  in step with the first and no behaviour, and a parameter added on one side
  was a compile error at sixty-six call sites or a silent divergence. What
  survives is the module's test, which pins the primitive contracts the graph
  dispatch layer relies on. Two ceilings on the Newton-Schulz IR shape now come
  from `vyre_foundation::visit::walk_exprs` instead of a hand-rolled counter
  over `#[non_exhaustive]` enums whose catch-all arm read an unlisted variant
  as a leaf, so a tree that grew through a new variant counted as small.
- A test-only IR extension payload is declared through
  `vyre_test_support::test_expr_extension!` or `test_node_extension!` instead
  of six hand-written trait methods. Five of the six are the same in every test
  that needs an inert opaque leaf: report success from validation, hand back
  `self` for downcasting, and answer the identity questions from a literal.
  Only the kind string, the debug identity, the result type, the CSE answer and
  the fingerprint byte differ, so those are the arguments. A test that needs
  reachable structure, a wire body, or a validation failure still writes the
  impl out, because those are the cases where the boilerplate is the contract.
- The `tests/SKILL.md` contracts for `vyre-spec`, `vyre-foundation`,
  `vyre-macros`, `vyre-reference` and `vyre-primitives` now state the category
  contract where it is read instead of pointing at a file the tree does not
  contain, and every claim in them is checked against source. They previously
  routed op semantics to `vyre-ops`, which is not a package here; listed bench
  and fuzz targets for directories four of the five crates do not have; named
  `--test adversarial`, `--test property`, `--test gap` and `--test
  integration` where only two of those targets are declared anywhere; and
  described `vyre-primitives` as marker types with no runtime behavior and no
  bench, while it owns the Tier 2.5 substrate and declares the
  `wire_throughput` bench.
- The command-line contract is a registered gate. `scripts/cli_docs.py`
  declared the binaries, ran every help route and generated the CLI section of
  each crate README, and it read the xtask command set by matching a
  `SUBCOMMANDS` constant that the registry function replaced, so the comparison
  found nothing and the whole check failed on the constant instead of judging
  the tree. `xtask cli-docs` runs the routes, derives each command list from
  the help text a reader sees, and compares the xtask half against the live
  registry in process. It asks the workspace wrapper for the target directory
  rather than bare cargo, which is what made it run another checkout's
  executables. `docs/CLI.md` is deleted: a generated transcript of help output
  whose generator was retired is stale by construction, and the README sections
  plus `--help` are the live surface. `scripts/cli_docs.py` and
  `scripts/lib/cargo_runner.py` are gone.
- The duplication scanner lists source files through the shared tree scanner
  instead of invoking git itself, so one rule decides what counts as a source
  file in the tree.
- A frozen-contract snapshot records the declaration and nothing else: the item
  line, its signatures and its closer. Default method bodies, doc comments and
  blank lines are left out. Before, a refactor inside a default body or a
  repointed doc link reported a semver-major contract change, and the fix text
  told the reader to bump the major version for it, so the seven snapshots had
  drifted on work that changed no contract. Braces are counted on code alone,
  because a brace in a comment or a string literal nests nothing. `VyreBackend`
  reads as 182 lines of signatures where it was 940 lines of bodies.
- vyre_foundation::ir publishes each IR type once. The facade previously
  re-exported the whole private model tree, so every expression, node, and
  program type answered to both ir::Name and ir::model::<submodule>::Name.
  Callers use the flat ir:: path; node_op_id, Scope and the program-graph
  identity types joined it.
- The vyre_foundation::ir facade no longer re-exports items another module
  owns. Validation is reached at validate::{validate, ValidationError,
  LimitState, DEFAULT_MAX_CALL_DEPTH, DEFAULT_MAX_NESTING_DEPTH,
  DEFAULT_MAX_NODE_COUNT}, call inlining at transform::inline, the text wire
  format at serial::text, and the CSE and DCE pass internals at
  optimizer::passes::fusion_cse. MemoryOrdering is published once, at
  vyre_foundation::ir::MemoryOrdering, beside the node variants that carry it.
- The Metal MacBook benchmark gate invokes the benchmark through the cargo
  runner instead of locating a binary under a target directory it named itself,
  and its remote setup exports no build configuration into the remote shell.
  The VYRE_MACBOOK_CARGO_TARGET_DIR option is gone: the remote checkout
  declares its own build directory.
- The vyre-libs crate contract advertises examples/prefix_sum_megakernel.rs,
  which builds scan_prefix_sum and compiles it through vyre-megakernel. The
  previous example called a vyre-primitives builder and executed it on the
  reference interpreter, so it demonstrated neither a libs composition nor the
  compiler.
- The runtime module that owns persistent slot residency is
  `vyre-runtime/src/resident_work_queue/`, and its lane, its test files, and
  its architecture page carry the same name. The directory was `megakernel/`, a
  name the crate's own types never used: nothing in it is named `Megakernel`,
  the artifact compiler `vyre-megakernel` is a different crate that
  legitimately keeps the word, and the module was reached through a path
  attribute pointing at a directory the public paths did not mention. The 24
  `megakernel_*` test files, the `runtime_megakernel` ownership lane in
  `docs/optimization/OWNERSHIP.toml`, the hot-path and hygiene rows, and the
  regenerated testing guides now all name the resident work queue.
  `vyre-runtime/ARCHITECTURE.md` described three types that do not exist and a
  directory that no longer did; it now describes the real submodules and the
  real public surface.
- The marker trait that seals the driver, optimizer and registry traits lives
  in a module named sealed rather than private, and vyre-driver publishes it at
  vyre_driver::sealed. A module name states what a module contains, not who may
  reach it.
- `vyre_foundation::pass_substrate::dataflow_fixpoint` is now
  `pass_substrate::semiring_closure`. Two modules in one crate carried the name
  `dataflow_fixpoint`: this substrate, which closes a semiring matrix product
  to a fixpoint, and `transform::compiler::dataflow_fixpoint`, the live
  compiler primitive. A reader who found one had no way to know which one, and
  a caller who imported the wrong one got a type error rather than a name
  error. The substrate is named for what it computes, its twelve callers name
  it, and `pass_substrate/mod.rs` carries a header and one documented line per
  module instead of an allow for missing documentation.
- The recurrent and chunked gated delta schedules no longer carry private
  copies of the parts they share. The head partition, the matrix-state copy,
  the key and query L2 normalizers, and the normalized key and scaled query
  operands are built once in nn::attention::gated_delta_spec and used by both.
  Only the schedules differ: the recurrent form advances the state one token at
  a time and needs no tile scratch, so it keeps its 64 head slots per
  workgroup, while the chunked prefill keeps its cumulative-decay triangular
  tile. The chunked program is unchanged. The recurrent program derives its
  value head with a remainder, normalizes the key and the query in two loops
  instead of one fused loop, and orders the operands of each scaled product to
  match the chunked form, which changes its emitted representation without
  changing any value it computes.
- Three oversized vyre-libs sources become directory modules named for their
  concerns. matching/nfa_to_dfa.rs splits into error, state_set, subset and
  dedup; graph/motif.rs into pattern, layout, plan, program, cpu_ref and
  registry; graph/toposort.rs into error, csr, edge_list, plan and program,
  each with its contracts under tests. Every body moved verbatim, so the
  emitted programs and the operation catalogs are unchanged. The motif host
  reference carried the cpu-parity gate once per item; the gate now sits on the
  module declaration, where an omission cannot ship a host classifier into a
  device build.
- `recurrent_gated_delta` and `chunked_gated_delta` take `&GatedDeltaSpec`,
  which is now public, and the chunked schedule is exported from the module
  that builds it. Each entry point had restated the same sixteen positional
  parameters and then rebuilt the same spec literal field by field, so
  eighty-two lines existed only to pass a value straight through, and any two
  of the eight buffer names could be swapped at a call site without a type
  error. The two builders that consume the spec are the entry points, so there
  is no wrapper left to drift. `nn_attention_clone_family_ir_invariance` pins
  the built IR of both schedules at both dtypes by hash, and those hashes are
  unchanged.
- The generated crate README contracts invoke the workspace wrapper directly
  instead of prefixing `CARGO_BUILD_JOBS=1`. Build configuration is declared
  once, in the cargo config and the root profiles, where it is reviewable and
  applies to every build equally; a per-invocation override makes each reader's
  build a different build, and thirty-six generated documents were teaching
  exactly that.
- The thirty-six generated testing guides, their `docs/testing/TESTING.toml`
  command rows, the contributor gate list, and the performance notes all invoke
  `./cargo_full` with no `CARGO_BUILD_JOBS=1` prefix. The wrapper already
  exports that variable, so every prefixed command was a second copy of one
  setting, and the copies taught a rule the repository forbids: a
  build-affecting variable belongs in the cargo config or the wrapper, not on a
  command line. The generator and the two contract tests that pinned the
  prefixed text move together, and the contributor guide no longer claims a
  `rustc-wrapper` the cargo config does not declare.
- Two coverage gates that derive their member set from source are declared in
  `docs/testing/STRUCTURAL_GATES.toml`: the resident queue materializer variant
  scan and the published quantized entry-point scan. Each asserts the absence
  of a case or a row, which no execution of the covered code can witness.
- The tooling and workspace-contract source walks have one owner each.
  `tests/support/source_scan.rs` owns the workspace Rust walk, the brace
  matcher and the comment/string masker the two workspace contract scanners
  each carried; `xtask/src/tree_walk.rs` owns the prune rule that seven xtask
  WalkDir walks each spelled differently, so a build directory added to one is
  now skipped by all of them; `xtask/src/manifest_walk.rs` owns the
  read-parse-name sequence for a package manifest that the feature and metadata
  generators each repeated; `xtask::release::conformance_evidence_semantics`
  owns the conformance read bound and its error context, previously copied per
  module; `xtask/tests/common` owns the `path:line:column` violation format.
  `vyre_test_support::ir_variants::single_u32_output_program` owns the
  validator program fixture that three validation suites each built by hand.
- `vyre_foundation::algebra::composition::trap_program` builds the program a
  builder returns when its inputs cannot produce a valid one. `vyre-primitives`
  and `vyre-libs` each had their own copy, differing only in whether one output
  buffer of count 1 is declared, and a trap program is IR composed out of IR,
  so neither crate is its home. The declared output is an `Option` argument
  because that is the whole difference between the two former copies.
- The 3-bit KV scoring op carried two schedules for the same function: a fully
  unrolled single-invocation form for sequences up to 8 and head widths up to
  16, and a lane-parallel loop for everything else. Every test and the
  registered fixture fell inside the unrolled bound, so the schedule that runs
  on real shapes was the untested one. The unrolled form is gone. One schedule
  remains and the existing fixtures now exercise it.
- `rule_tree` is a module of the xtask library. Two binaries each declared it
  with a `path` attribute pointing into `src/bin/`, which compiled the same
  layout rules twice and put a shared module in the directory reserved for
  binary roots. Both binaries now use `xtask::rule_tree`.
- The C11 typedef annotation path now composes registered row phases instead of
  inlining them: c11_identifier_row_hash,
  c11_identifier_row_hash_packed_haystack,
  c11_enclosing_function_lparen_for_row and
  c11_builtin_declaration_kind_for_row are registered operations with reference
  oracles, and one shared owner emits the phase program buffer contract.
- Five domains left `vyre-primitives` for `vyre-libs`, because an operation
  belongs in `vyre-primitives` only when it cannot be composed, meaning it
  needs its own backend emitter arm and its own reference-interpreter arm. None
  of these did: `cat` is now `reasoning::finite_category`, `zx` is
  `reasoning::zx_diagram`, `dnnf` is `reasoning::dnnf`, `types::linear_check`
  and `types::shape_smt` and the whole `effects` domain collapse into
  `analysis`. The `cat`, `zx`, `dnnf`, `types` and `effects` features are gone
  with them, and `vyre-libs` reasoning now depends on `vyre-primitives/graph`
  alone.
- `vyre-runtime/src/uring/` exposes its flat parent re-exports and nothing
  else. Its six submodules were public alongside those re-exports, giving every
  type two paths, and the io_uring ABI structures and opcodes were public from
  a crate that never intends a caller to build a submission queue entry by
  hand. Every caller in the workspace already spelled `uring::X`, so the
  submodules are private and the ABI items, `get_sqe`, and `peek_cqe` are
  `pub(crate)`.
- `vyre_runtime::uring::IoUringState::submission_entries` and
  `AsyncUringStream::submission_entries` publish the submission entries the
  kernel allocated for a ring, and `UringPump::new` sizes its iovec scratch,
  free list and pending queue to that number. A submission needs a
  submission-queue slot, so the ring depth is the hard bound on anything
  tracking one submission each; the three queues previously started empty and
  grew during a scan.
- The validation rule registry is one table in
  `vyre-foundation/src/validate/catalog.rs`, carrying each code's phase,
  invariant and corrective action. `docs/generated/error-codes.toml` is
  rendered from it and `vyre-foundation/tests/validator_error_docs.rs` fails on
  divergence, reporting one finding per divergent code. `ValidationCode` gained
  `phase`, `invariant` and `corrective_action` accessors. The hand-maintained
  table in `docs/error-codes.md` was a second copy that had already drifted:
  four codes carried the placeholder text `Program validation error 0NN` with
  the corrective action `See diagnostic output`, which the catalog replaces
  with the invariant each emission site actually checks.
- `visit` is split by what is being visited. `node` owns the per-variant `Node`
  decisions a traversal cannot re-derive safely - which bodies a variant nests,
  which scalar name it binds and what it does to that name, which operands it
  evaluates, which buffers it names and in which direction - `expr` owns the
  same for the value namespace, and `walk` owns the traversals, which are
  written entirely against those two and restate neither. Every item is
  re-exported from `visit`, so no caller changes. The file was 1789 lines
  against an 829-line cap, and every match in it is exhaustive with no
  catch-all arm on purpose, which is the mechanism that makes a new IR variant
  a compile error rather than a silent leaf classification; one file that size
  hides which of those decisions a reader is looking at.
- `vyre-driver/tests/async_dispatch_contract.rs` is the sole owner of what the
  default async dispatch adapter guarantees: error propagation before any
  await, a ready handle that never blocks, independent handles per dispatch,
  object safety through both awaits, and the default `dispatch_borrowed` and
  resident-async paths. The counting, failing, and resident backend fixtures
  are declared once. `async_dispatch_always_nonblocking.rs` carried
  byte-identical fixtures and three duplicate tests and is gone; every
  assertion it made is made here, and the borrowed-dispatch case now checks the
  forwarded payload rather than only that the call succeeded.
- The `VyreBackend` trait contract has two owners in `vyre-driver`'s test
  surface instead of three overlapping files. `tests/backend_trait_contract.rs`
  owns the minimal and fully overriding backend fixtures and asserts that every
  capability default is conservative, that every override is observably
  different from its default, that lifecycle hooks succeed by default and
  receive every call once overridden, that the blanket `Backend` impl exposes
  the driver identity, and that the trait surface stays object safe and `Send +
  Sync`. `tests/backend_registry.rs` owns what the registry reports when
  nothing is registered, including that preferred-dispatch acquisition fails
  closed and never advertises a host fallback. The previous
  `backend_contract.rs`, `backend_capability_negotiation.rs`, and
  `backend_trait_compatibility.rs` each declared their own copy of the same
  fixtures and asserted overlapping subsets of the same contract, so an
  override that silently returned its default value was invisible in all three.
  That case is now a single invariant rather than two mirrored capability
  lists.
- `vyre-driver`'s dispatch policy bundle is covered by a test that calls each
  policy directly and compares the verdicts to the bundle's, so the bundle
  cannot drift from the policies it composes. A test drives every policy at the
  extremes of its integer domain and asserts the resulting verdict, replacing a
  case that bound the results to `_` and only proved absence of a panic. The
  module and field documentation names the concern each policy decides instead
  of an internal plan label.
- `vyre-libs` no longer reaches across dialect boundaries in private code.
  `telemetry` has one owner rather than a one-file directory, because counters
  instrument every dialect and belong to none; the scratch reservation, the
  host program cache and the dispatch byte marshalling sit together under
  `plumbing::host`, because host dispatch plumbing is not a dialect either. The
  `analysis::dataflow_fixpoint` re-export of the foundation substrate is
  deleted, so the closure family has one path instead of two, and every caller
  names the owner. Composition that crosses dialects names the module that owns
  what it composes: `nn::linear` names `math::linalg` for `MatmulBiasTiled`,
  and `reasoning` names `analysis::dataflow_fixpoint` for
  `reachability_closure_via_into`.
- The `vyre-libs` duplication pin records what the tree measures after the
  width table lands: 10649 to 10598 duplicated lines, with `total_lines`
  measured. What remains between the three match-emitting entry points is their
  frozen positional signatures and the shared value they each construct from
  them, which no owner can absorb without changing the public ABI.
- The vyre-libs crate root is a table of contents. Fourteen loose files sat
  beside twenty-five dialect directories, so the root did not say what the
  crate is. Each one now sits under the concern it serves: buffer names, tensor
  references and operand shape under plumbing::operand, program introspection
  and fused-output demotion under plumbing::program, type signatures, contract
  presets and the catalog view under plumbing::registration, operand
  marshalling, scratch reservation, the shape-keyed program cache and the
  composition counters under plumbing::host, and the byte-range ordering
  predicates and the builder registrations under builder. Every path that was
  public is still public at the same spelling, and no operation id,
  registration or built program changed.
- `vyre_lints::read_source_bounded` is public surface with its bound stated,
  and `vyre_lints::LINT_SOURCE_READ_CAP` is public beside it so the number a
  caller is refused at has one home instead of a constant and a sentence that
  can disagree. The `vyre-lints` binary is a separate compilation unit and
  reads the workspace manifest through the same reader, so making it
  crate-internal would have meant a second reader with a second cap in the
  binary. The public API snapshot records both.
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
- The wgpu runtime cache publishes AccessTracker and the tiered cache types;
  IntrusiveLru, AccessMeta and DEFAULT_INTRUSIVE_LRU_CAPACITY are
  crate-internal because no consumer outside the crate could reach them.
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
- The WGPU target compiler is declared as a
  `vyre_driver::target_dialect::TargetDialect` value instead of a hand-written
  implementation. The shell it duplicated (the compiler struct, the profile
  constructor, the factory, and a local word-comparison helper) is gone; what
  remains is the dialect record and the WGSL module emitter, so the device
  limits and payload format a backend reports are stated once in the same place
  every other backend states them.
- The `vyre` facade wire suites share one program fixture in
  `vyre/tests/support/mod.rs`. The round-trip suite proves a valid program
  survives an encode and a decode byte for byte and the malformed-wire suite
  proves hostile bytes never decode into one, so two copies of that program let
  the two halves of the claim drift. In `vyre-grammar-gen` the LR table the
  wire round trip exercises is the sample grammar `vyre-grammar-gen/src/lr.rs`
  already owned, not a second copy of it. Pins lowered to the measured tree:
  vyre 22 to 0, vyre-grammar-gen 137 to 119.
- The wire-format round-trip and corruption assertions live in
  vyre_foundation::serial::wire_round_trip instead of a test_helpers module
  nested inside envelope.rs. vyre_primitives::serial_data re-exports the new
  module.
- Every witness input expansion routes through `WitnessInputPlan`. Three copies
  of the planner existed beside the owner: the per-op ULP audit and the
  cross-backend parity matrix each carried a full reimplementation differing
  only in the prefix on each error string, and a rename layer aliased the
  owner's type and forwarded its functions under the copies' names. The owner
  gained the two operations only the copies had, `buffer_indices` and
  `plan_witness_inputs_owned_into`, and each copy's unique test moved to it.
  Three expansions of one fixture is three chances for a gate to compare
  against an input stream the dispatcher never saw.
- The workspace contract suite no longer compiles into `vyre-foundation`'s test
  target. `tests/contract/mod.rs` was included by
  `vyre-foundation/tests/contract_workspace.rs`, so a contract that judges the
  whole workspace could not run until the compiler built, and a crate that has
  nothing to do with the routing rule decided whether the rule ran at all. Each
  contract now sits with its subject: the device-only routing rule and the node
  child descent rule are `structure-gate` tests, which is the crate that reads
  source text and depends on no vyre crate; the validator rejection contract is
  a `vyre-foundation` test target; the public-API snapshot check joins the
  other snapshot contracts in `xtask`. `structure_gate::source_scan` owns the
  workspace source walk, the brace matcher and the comment and literal masker,
  and the masker is built on the same `opaque_span` the registration parser
  uses, so a masker and a parser can no longer disagree about whether a raw
  string or a nested block comment holds code.
- Three duplication pins record what the tree measures: `xtask-evidence` 475 to
  91 duplicated lines, `xtask-registry` 58 to 49, and `vyre-foundation` 4127 to
  4028, each with `total_lines` measured. A pin with room under it hides the
  next copy.

### Removed

- The use-path classifier no longer special-cases a test_helpers file stem. No
  file in the tree carries that name, and the shape it described is one the
  tree rejects.
- `vyre-runtime/src/resident_work_queue/scaling.rs` is gone. It declared no
  item of its own: every line was a `pub use` of a `planner` or `policy` item,
  so each of those 77 items had two public paths and a reader had to pick. Its
  one caller, `telemetry.rs`, imports from `super::policy` directly.
- Fifteen point-in-time reports under `audits/` are gone: five critique and
  audit narratives, five `V7_*.toml` finding tables, a 416 KiB filtered unwrap
  inventory, a pre-sweep error dump, and a closure report whose every row read
  completed while citing ten audit files the tree no longer carries. No gate,
  workflow, script, test or manifest read any of them.
  `audits/lego-composition.tsv` stays, because `lego-audit` reads it as the
  checked-in bootstrap composition baseline. The `audit-status` gate goes with
  them: it judged files carrying a `Status legend:` line, its subject class is
  now empty, and its own corrective action named deletion as the alternative to
  inventing one.
- Two demonstrators under `examples/` are gone. `three_substrate_parity` held a
  README and a manifest with no code, naming published reports under a
  `docs/parity/` tree that no longer exists. `wgpu_readwrite_count_repro` is
  superseded by
  `conform/vyre-conform/tests/countless_readwrite_output_parity.rs`, which the
  conformance suite runs. `external_ir_extension`, `external_backend_extension`
  and `libs-template` stay, and each is now built and run by the
  `example-capability` gate.
- `SlotState` is gone from the readback ring. It named the four slot lifecycle
  states a second time, as a public enum nothing in the workspace constructed,
  matched, or returned; the ring stores its state as a `u8` and compares
  against the `SLOT_*` codes. The codes are the single naming and are private
  to the ring.
- The C frontend has moved to a project of its own. `vyre-libs/src/parsing/c`
  and everything that existed only to build, test, benchmark or document it are
  gone: 675 files, 30 registered operations, the `c-parser` feature and every
  consumer that enabled it, the wgpu and cuda C parity suites, the
  `vyre.frontend.c.*` optimization documents, and the
  `parser.c_lexer.small_state_transition.4k` benchmark case. The
  language-neutral parsing substrate stays: `parsing/core` (AST node kinds,
  shunting yard, binding, blocks), `parsing/go`, `parsing/python`,
  `parsing/lr_tables` including the C11 expression table, the source cache, the
  parallel parse driver and the VAST wire. The shunting-yard AST builders were
  gated behind `c-parser` and are now ungated, reading token ids from
  `vyre-spec::c11_token`, which `vyre-grammar-gen` and the wgpu loop-carrier
  dispatch test already own. The three regressions that used the C lexer only
  to have a real program to lower now use the Python 3.12 lexer, which has the
  same signature and the same loop-carrier structure, and `vyre-debug`'s
  function-locals contract derives its expected names from the descriptor it
  walked instead of pinning one lexer's variable spelling.
- Neural operations and opaque-payload helpers now use their category-owned
  module paths. Flat compatibility re-exports and the `matching::ops` shim are
  gone; unclassified backend failures use `BackendError::Other`.
- The macro crate now exports only the production-used AST registry and
  semantic pass registration generators. Test-only operation registration,
  algebraic-law derive, no-op builder marker, and generated decoder stubs are
  gone.
- `vyre_foundation::transform::compiler` is deleted, with the
  `dataflow_fixpoint`, `recursive_descent`, `string_interner`, `typed_arena`
  and `visitor_walk` specs it held. Each published a workgroup size, an
  algebraic-law list and a Program builder that no backend lowered and no
  operation registered, so the module was 1736 lines of spec whose only reader
  was the gate that checked the specs against each other. The
  `security_dataflow` optimization lane no longer claims write scope over the
  deleted fixpoint file.
- `vyre_conform::fp_parity` is gone. It re-exported eleven names from
  `vyre_foundation::fp_parity` verbatim and added nothing, so the parity policy
  had two paths and a reader could not tell which was the owner. Every caller
  inside and outside the crate imports `vyre_foundation::fp_parity` directly;
  no alias remains.
- Three checks that could not fail are gone. Release hygiene named an xtask
  command module `release_gate` that has no source file, beside the
  `vyre_release_gate` that does. `scripts/architecture_docs.py` guarded a
  superseded-RFC contract behind `if the file exists`, and the RFC is deleted,
  so the whole block was unreachable; its fixture also wrote three pages the
  checker never opens, and the test that proved the stale-version rule poisoned
  one of them. `xtask/feature-isolation.toml` recorded five `vyre-primitives`
  feature selections - cat, dnnf, effects, types, zx - that no manifest
  declares since those domains moved to `vyre-libs`, and
  `scripts/docs_manifest.py` claimed provenance for a `RELEASE_CHECKLIST.md` no
  generator writes.
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
- Twenty-seven files under `scripts/` are gone. Each was a check whose rule a
  registered gate already owns, a helper only such a check called, or a
  baseline nothing read. Nothing in the tree invoked any of them, so they were
  a second statement of a rule that could drift from the gate that enforces it,
  and two pinned baselines that no reader compared against. The script
  assertion ledger now records only scripts that are still on disk, and states
  that a row leaves it by the script being deleted.
- Forty declared dependencies that no source file references are gone, across
  sixteen crates. `cargo deny`'s sibling gate, `cargo machete`, had never run
  to completion in CI because it could not install under the workspace
  toolchain pin, so the declarations accumulated unchallenged:
  `vyre-driver-wgpu` alone declared ten it never used, including `tokio` twice,
  `tree-sitter`, `tree-sitter-go`, `crc32fast`, `libm`, and dev-dependencies on
  `vyre-emit-metal` and `vyre-emit-spirv`. The `self-substrate-adapters`
  feature in `vyre-driver` and `vyre-runtime` pulled `dep:vyre-pass-engine`
  without either crate naming it. Four flagged entries were kept because they
  are used through workspace-root shared test files included by path, which a
  crate-scoped search cannot see.
- The `check_warning_budget` gate script is gone: its baseline lived at a
  gitignored path with no file, so it exited before measuring anything, it
  counted cargo summary lines as warnings, and a trailing `|| true` reported
  zero warnings for a failed build. `strict.yml` already builds the workspace
  with all targets and all features under deny-warnings, so the enforced
  ceiling is zero. The `check_tier_b_rule_contracts` gate script is gone: it
  required a repository-root `rules/` tree that does not exist here, and its
  second rule scanned a directory that does not exist behind a suppressed
  error.
- Backend registration is now consumed from `vyre-driver`, the `ReferenceKind`
  alias is gone in favor of `vyre-spec::CpuFn`, and `gpu_int_literal_scan()` no
  longer accepts an ignored source-length parameter.
- `vyre_primitives::effects` is gone. `EffectRow`, `EffectKind`, `Handler`,
  `handler_apply` and `handler_compose` restated the seven effect kinds and
  three bit operations that `vyre_foundation::lower::ProgramEffects` already
  declares, on the same bit assignments, and a kind added to one would not
  appear in the other. The one live consumer is
  `vyre-libs::analysis::effect_signature`, which takes `ProgramEffects`
  directly: `check_signature` returns the unpermitted row as a verdict and
  `residual_effects` names what stays open after a handler discharges. Handler
  composition is row union, so it needs no function of its own.
- vyre_lower::emit_adversarial_corpus no longer exports EmitAdversarialBackend,
  REQUIRED_BACKENDS or required_backends. Each emitter test asserted only that
  the hand-written list named its own variant while already running the corpus.
  The backend-extension gate now derives the roster from the vyre-emit-*
  workspace members and requires each one to consume emit_adversarial_corpus,
  and fails when the roster is empty.
- vyre-grammar-gen is deleted. It generated the host-side C11 lexer DFA and
  LR(1) tables for the C frontend that is leaving vyre, and its src carried CPU
  implementations of C preprocessing and lexing, which no crate outside
  vyre-reference may hold. Nothing consumed its output:
  vyre-libs/src/parsing/lr_tables/c11_expr.rs is hand-written Rust built from
  pack_shift and pack_reduce and takes its terminal ids from
  vyre_spec::c11_expr_token, not from a blob this crate emitted. The neutral
  DFA and LR machinery stays in vyre-libs/src/parsing/lr_tables with its
  contract suite, and the C11 token id tables stay in vyre-spec. The workspace
  member row, the c-grammar-gen dependency alias, the path-only dev-dep in
  vyre-driver-wgpu, the structure-gate member roster row, the duplication pin
  and the crate documentation pages are removed with it.
- `vyre_primitives::types` is gone. `LinearDiscipline` declared the same four
  substructural disciplines and the same `forbids_drop` and `forbids_reuse`
  predicates as `vyre_foundation::ir::LinearType`, and `check_linear_use` was a
  second implementation of the per-buffer decision
  `vyre_foundation::validate::linear_type` already makes and reports with a fix
  hint. `ShapeFormula` aliased `vyre_foundation::ir::ShapePredicate` and its
  evaluator forwarded to that type's own method. Both had no callers, and the
  foundation owner's own tests already assert every boundary the copies
  asserted.
- The mdbook is deleted: 134 files, every `.md` under `docs/` plus `book.toml`.
  It described a tree that had moved out from under it, and a document that
  contradicts source is worse than no document, because a reader who checks it
  is misled and a reader who does not is unserved. What remains under `docs/`
  is machine-readable contract data that gates read directly: the `.toml`
  policies including `OWNERSHIP.toml`, `HOT_PATHS.toml`, `OP_MATRIX.toml`,
  `TESTING.toml` and `STRUCTURAL_GATES.toml`, the per-crate public-API `.txt`
  snapshots, `OP_SCHEMA.json` and `architecture.svg`. `README.md` and per-crate
  `README` files remain the documentation surface. `scripts/release_docs.py`
  accordingly generates `CHANGELOG.md` alone, and the release train's required
  artifact tokens are generated into its unreleased section from
  `release/release-train.toml` rather than checked against prose that could
  disagree with it.
- One NFA transition-table layout remains.
  `vyre-libs::scan::nfa::build_transition_table` and `subgroup_nfa::nfa_step`
  both use the state-major `[num_states x 256 x LANES]` shape indexed `src *
  256 * LANES + byte * LANES + dst_lane`. The transposed
  `build_transition_table_lane_major` packer, its fallible counterpart, the
  `bench` feature and the `scan::nfa::bench` module that was its only non-test
  consumer are gone.
- `scripts/check_platform_consumer_docs.sh`,
  `scripts/check_parity_testing_not_leaked.sh` and
  `scripts/check_primitive_contract.sh` are gone, along with
  `vyre-pass-engine/tests/platform_doc_consumer_boundary.rs`, which existed to
  give the first one a cargo entry point. The `platform-consumer-docs` and
  `parity-testing-isolated` gates own the first two rules; the third was a
  shell adapter in front of the registered `primitive-admission-gate` and its
  only assertion was that it refused path arguments.
- `scripts/check_doc_claim_to_test.sh`, `scripts/check_op_names.sh`,
  `scripts/check_invariant_paths_exist.sh` and `scripts/check_repo_hygiene.sh`
  are gone. The `doc-claims`, `op-names`, `invariant-paths` and `repo-hygiene`
  gates own those rules. The claim manifest is parsed with a TOML parser rather
  than awk, so a phrase carrying a quote or spanning lines is read correctly,
  and the op-name scan no longer splits a path on whitespace or skips files by
  a hand-typed filename list.
- `scripts/check_evidence_paths.sh` is gone, and with it
  `vyre-pass-engine/tests/release_evidence_path_contract.rs`, which existed to
  give the script a cargo-visible entry point. The `evidence-paths` gate owns
  the rule. Its four fail-direction fixtures moved into the gate: a citation
  that resolves to nothing is reported at its route, a cited path that exists
  but is gitignored is reported for reachability rather than absence while a
  committed path stays clean, every placement a narrower filter once missed is
  read, and a version, schema id, operation id, fingerprint, command or ratio
  is not read as a citation.
- `scripts/check_max_file_size.sh` is gone. It was a second implementation of
  the per-file line cap with its own table of exemptions, invoked by no
  workflow and no gate, so the two owners of one rule could disagree without
  anything noticing. The `file-size` gate is the only owner; the script's
  section left `xtask/script-assertion-ledger.md` by deletion.
- `scripts/check_trait_freeze.sh` is gone. It was a second implementation of
  the frozen-declaration check with its own table of seven contracts, their
  source files and their keywords, invoked by no workflow, so the two owners of
  one rule could disagree without anything noticing. The `frozen-contracts`
  gate is the only owner.
  `vyre-foundation/tests/ci_script_frozen_contract_coupling.rs` held a third
  copy of the same table and is gone with it; the workflow-reference rule it
  also carried now lives in the gate sweep, where the registry, the baseline,
  the subsets and the workflows already have to agree. The script's section
  left `xtask/script-assertion-ledger.md` by deletion.
- `scripts/check_no_hot_path_inventory.sh`,
  `scripts/check_no_hot_path_vec_vec.sh` and the shared
  `scripts/lib/source_scan.sh` they read the tree through are gone. The
  `hot-path-inventory` and `hot-path-nested-rows` gates own those two rules and
  read the tree through `xtask/src/gates/scan.rs`, where a failed scan is an
  error by type and a scan path that does not exist is a finding rather than a
  clean count.
- Three shell copies of the lint hygiene rules are gone:
  `scripts/check_expect_has_fix.sh`, `scripts/check_unsafe_budget.sh` and
  `scripts/check_unsafe_justifications.sh`. The `lint-expect-fix`,
  `lint-unsafe-budget` and `lint-unsafe-justification` gates own those rules,
  read tracked files rather than the working tree, and cannot mistake a failed
  search for a clean one. The expect ratchet in particular took its ceiling
  from `VYRE_EXPECT_BASELINE`, so any caller could raise it from the
  environment; the pin now lives in `xtask/gate-baselines.toml`, which no
  caller can override.
- `scripts/check_no_string_wgsl.sh` and `scripts/check_gpu_test_loudness.sh`
  are gone. The `shader-source` and `gpu-loudness` gates own those rules. Both
  scripts had branches that could not fire: the shader guard compared a count
  against zero with `-lt`, and three of the loudness patterns required a
  literal backslash before `[cfg` in Rust source, so the cfg classes its own
  header claimed to cover were never checked.
- `scripts/check_unification_baselines.sh`,
  `scripts/check_every_source_file_is_reachable.sh` and
  `scripts/lib/check_every_source_file_is_reachable.py` are gone. The
  `unification` and `source-reachability` gates own those rules. The
  unification gate gained the two fixtures the shell version never had: a
  second owner of a unified surface is reported with its count and its ceiling,
  and a row whose scanned path is missing is reported instead of scoring zero,
  which is how three of the five shell rows passed by measuring nothing.
- The buffer-name form of each classic Aho-Corasick program left the published
  surface of vyre-libs. The `build_*`/`try_build_*` entry that binds the pinned
  ABI names is the one published path per program. The legacy buffered
  inflate-then-scan builder is deleted, and the tile-width form of the fused
  stored-block scan is internal to its module.
- The vyre-libs published surface no longer carries the parity fixed-point
  oracles. FIXED_ONE, to_fixed, from_fixed, fixed_mul, fixed_matvec,
  fixed_sdiv_by_positive, signed_fixed_17, signed_fixed_18, signed_fixed_19,
  xorshift32, u32_bytes and wrap_program_sequence were test scaffolding shipped
  as library API; a consumer reaches the same oracles at
  vyre_test_support::fixed_point, which is a dev-dependency and not part of any
  shipped binary.
- Retired `GOAL.md`. Its roadmap and compiler boundary rules are canonically
  owned by `docs/ARCHITECTURE.md`, `docs/CRATE_OWNERSHIP.toml`, and crate
  architecture documentation.
- Routing has no host arm. `PolicyRoute::CpuSimd`, `RoutingDecision::CpuSimd`,
  `ExecutionPolicy::use_cpu_fast_path` and the two host fast-path thresholds
  that only fed it are gone, and `ExecutionPolicy::route` no longer takes a
  byte count the deleted predicate was its only reader of. Vyre executes
  compute on a device; the only host arithmetic in the workspace is
  `vyre-reference`, which is a parity oracle and not a route. No executor arm
  ever served the variant and the standard policy rewrote every suggestion to
  the persistent megakernel regardless, so what the declaration bought was a
  caller that believed a degradation path existed. The routing contract test
  asserted that one plan did not pick that route. A workspace gate now
  recognises a routing enum by its own variants, wherever it is declared, and
  fails on a variant naming host execution or a route with no recorded
  executor, so a route added back under a fresh enum name fails on arrival.
- Self-substrate no longer publishes source-text validators for deleted
  C-frontend test files or parser release artifacts. Diagnostic and
  preprocessing conformance now belongs to the live frontend and conformance
  paths.
- `vyre_libs::prelude` is gone. It re-exported forty items that each already
  had a path, so `TensorRef`, `BuildOptions` and every built-in builder were
  reachable at two paths and some at three. It was declared as the one seam a
  dialect crosses through, but one module in the crate reached through it and
  one template outside the crate glob-imported it, so the seam described a
  discipline no code followed. Callers name the module that owns the item:
  `reasoning::do_calculus_change_impact` names `analysis::dataflow_fixpoint`,
  and the dialect template names `vyre_foundation::ir`, `vyre_libs::region` and
  the crate root.
- `scripts/check_feature_msrv.sh`, `scripts/lib/check_feature_msrv.py` and
  `scripts/lib/cargo_runner.sh` are gone. The MSRV class is `feature-isolation
  --sweep --msrv`, which reads `[workspace.package].rust-version`, installs
  that toolchain when rustup does not carry it, and compiles an axis derived
  from the same manifests: every member rather than only publishable ones, plus
  the cross-member edge selections the retired matrix never built. The workflow
  step that read the manifest with a second reader is gone with it, and each
  script that resolved its own cargo now runs the tracked `./cargo_full`
  wrapper that every workflow already runs.
- CpuOracleDispatcher is deleted. It matched Region generator ids onto
  vyre-primitives CPU functions, which made vyre-libs a second host execution
  path beside vyre-reference. Every parity suite that used it now dispatches
  through vyre_driver_reference::ReferenceEvalDispatcher, which moved out of
  vyre-libs into vyre-driver-reference.
- vyre_foundation::ir::ExprArena, ExprRef, ArenaProgram and Program::with_arena
  are gone. Nothing in the workspace ever constructed one: it was a second
  expression arena beside the live flat arena in optimizer::expr_arena,
  published under a colliding name, and it carried the only unsafe code in
  vyre-foundation. The crate no longer allows unsafe outside one test that
  frees leaked static keys, and no longer depends on bumpalo.
- The WGPU host-ingress and raw persistent-kernel compiler routes are gone.
  Persistent product execution uses authenticated artifact sessions; concrete
  pipeline compilation remains available only as a hidden oracle helper for
  driver cache tests.

### Fixed

- The conformance runner exercises each backend through the route its
  registration declares. A backend that registers a target compiler and a
  materializer takes the production artifact route, as before; the reference
  interpreter registers neither and is now dispatched directly, instead of
  failing every one of its 356 pairs on a materializer its registration says it
  does not have. The dispatch route submits under the invocation grid the
  program needs, because a neutral program dispatched under the default grid
  executes one invocation and leaves every other output element at zero. A pair
  whose backend is the reference interpreter and whose operation records no
  expected outputs now fails with that reason, because comparing the
  interpreter against itself is not evidence.
- The benchmark harness builds its compile request from the probed backend
  profile instead of unknown device facts, so a case that uses subgroup
  intrinsics is measured on a device that has them rather than recorded as a
  validation failure.
- `vyre-primitives::reduce::multi_block_prefix_scan_inclusive_sum` now declares
  a workgroup every registered target admits. Its default builders previously
  used 1024 lanes on the false claim that every target admitted that width, so
  a target profile capped at 256 refused the payload with `target workgroup
  extent 1024 exceeds profile limit 256`. The default block builders now read
  the single target-neutral floor,
  `vyre_foundation::ir::PORTABLE_WORKGROUP_INVOCATIONS`, directly; no operation
  declares a crate-local geometry alias. Backend-specific lowering can still
  select a wider admitted launch through `LaunchGeometry`. The cooperative
  block width is load-bearing for the scan, so the block was rebuilt at the
  selected width rather than clamped after construction.
- The wrapper rule reads an instruction, not any sentence containing the word
  cargo. A comment is a finding when a run verb comes before the command and
  the command is quoted as code, because prose says a full cargo build while an
  instruction quotes what to type. Sentences describing what a build sees were
  previously findings whose only repair was to describe the build less
  precisely.
- The four surviving citations of the deleted optimization control-plane page
  now name the file that holds the claim: the conservative shared-memory cap in
  the CUDA driver profile, the external-operation contract page, the hot-path
  and benchmark target policies, and the optimization ownership data in the
  documentation evidence map.
- A crate document resolves a module path against the crate src directory, so
  an architecture heading naming a module is read as the module it names.
- Each crate README links the testing guide rendered for that crate instead of
  the data file every guide is rendered from. The generated Testing section
  pointed at `docs/testing/TESTING.toml`, which sends a reader to a table of
  every crate to find the rows describing one, and the generated per-crate page
  is what answers the question. The section now links `docs/testing/<crate>.md`
  and names the TOML as its authority.
- Each crate SPEC.md links the crate-boundaries chapter by a path relative to
  the page, so the link resolves from the crate directory it is read in.
- The crate pages gate refuses a placeholder where a crate states what it must
  never contain, so a body reading none recorded, none, n/a, tbd or nothing
  fails the same clause an absent section fails. A boundary with only an
  inclusion half admits everything, and a non-empty check passed on the page
  that stated no exclusion at all. The bound on the output a failed benchmark
  child contributes to a report applies to the joined text, so a command
  writing on both streams no longer contributes twice the bound.
- A registration is rejected when the tier it declares is one the crate that
  minted its id cannot carry, and when the id names no crate at all.
- A default build of vyre-libs emitted programs whose child regions named
  builder operations that build was not compiling. The two shared builder child
  regions register behind builder-ops, and the dialects that emit them,
  math-dialect, math-linalg, nn-activation, nn-norm, nn-attention and llm, did
  not name that feature, so thirteen catalog entries lowered to a region
  reference no registry in that build could resolve. Each composing dialect now
  names builder-ops, and a test walks every catalog entry under the selection
  it is compiled with and fails on any region naming an operation the running
  registry does not hold.
- An op the whole-program compiler splits into more than one fusion group can
  build a target payload. `selected_resource_bindings` resolved a descriptor
  binding name against the entry ABI, which lists what the host binds and
  therefore excludes a value one group writes and another reads, so every such
  op failed target compilation with `fusion group N descriptor binding `X` has
  no canonical artifact resource`: `vyre-libs::security::aliases_dataflow`,
  `flows_to_to_sink`, `flows_to_with_sanitizer`, `integer_overflow_arith`,
  `sink_intersection`, `taint_pollution` and
  `vyre-primitives::text::line_index`, on every dispatch backend. The lookup is
  now built from the artifact resource set, which is the owner of value
  identity, and one artifact that gave two values the same resource name is
  refused by name rather than resolving to whichever came first. Payload entry
  identity moved from the entry symbol name to the canonical node for the same
  reason: every fusion group emits its own module image with its own entry
  symbol, so two groups both naming their entry `main` is correct, and
  requiring the names to be unique across a payload rejected every artifact
  with more than one group. A value an entry produces without reading it is no
  longer requested from the caller as a host input, because an inter-group
  intermediate is device state rather than a caller buffer.
- The cross-dialect reach-through audit asks whether the library source root
  carries Rust source instead of whether its directory exists, because a
  directory outlives the deletion of every file in it.
- Validating an operation schema document now reports what the document gets
  wrong even when the live registry does not assemble, and names the field that
  differs rather than one sentence about the whole file.
- The docs-references gate now reads every tracked Markdown file rather than
  the root, .github and docs pages only, resolves a relative path against the
  crate that owns the document as well as the workspace root, and exempts
  historical records; the crate CONFIG, ARCHITECTURE, AUTHORING and README
  pages that cited a deleted script, an op corpus that never existed, or a
  module that moved are repointed at what the checkout carries.
- feature-isolation compiles the library and binaries only, so a dev-dependency
  edge can no longer hand the feature resolution the union the gate exists to
  see past.
- A backend feature marker is scored against implementation text: line,
  trailing, block and doc comments are removed, `#[cfg(test)]` and `#[test]`
  items are removed, and string literals are kept because emitted target text
  is implementation. The scan removed whole-line comments only, so four tokens
  were satisfied by prose that outlives the feature it claims: `cp.async`
  appeared in the async staging pattern only in its header comment, and the
  no-CPU-fallback marker scored the words never, cpu and fallback against a
  test file. The staging marker now names `supports_async_copy`,
  `supports_ldmatrix` and `KernelOpKind::StoreShared`; the fallback marker
  names the adapter selector that refuses a CPU adapter; the batched megakernel
  dispatch marker the release gate required and nothing emitted is declared
  again against the persistent pipeline. The required marker ids and the marker
  floor are read from the declarations that produce them instead of from two
  hand-written lists, one of which required a marker no longer emitted and
  floors that had drifted to 12 of 17 and 7 of 9.
- Three integration suites declared the feature they import through. The
  scope_rewrite_owner_contract and encoded_rewrite_walk_contract suites in
  vyre-pass-engine reach vyre_pass_engine::optimizer, and the
  registry_link_rules suite in vyre-registry-link reaches
  vyre_registry_link::operation, so a selection without optimizer or operations
  failed to build the crate's test targets. Each suite now carries
  required-features for its prerequisite, which makes cargo skip it for a
  selection that cannot host it instead of failing the build. Both features are
  default, so a plain test run still executes all three. The feature-isolation
  sweep found the eight affected selections; they had been recorded as
  compiling.
- The CI step gate joins a YAML block scalar the way the runner receives it, so
  a workflow command written as a folded `run:` block across several lines is
  checked whole. The reader joined shell continuations only, and every package,
  feature and test target named after the first line of a folded step was
  discarded unread: 226 selectors across 108 commands now resolve against the
  manifests where 159 did before, in both the unresolvable-selector direction
  and the silently-skipped-target direction. A paused workflow that names a
  script the checkout does not carry is reported, and a parked file still
  credits no check. The registry names `ci-registry --write` as the writer of
  the declaration it reads.
- A gate asked for `--help` answers with its usage and reads nothing.
  `bench-crossback --help` read 35 measurements across 18 cases and reported a
  clean gate, which is the check running on the caller who asked what the check
  takes. The dispatcher answers a leading `--help` from what the gate declares,
  a delegated gate answers from the package that implements it, and the answer
  travels back as report notes so the report protocol holds. Every option a
  gate names in its help line is declared as a usage line, and the rule reads
  the gate table at run time, so a gate registered later with an option it
  never names goes red.
- The lego-quick gate answers from source text and no longer links the
  operation registry: it moved to xtask, which is where both crates already say
  a source-text gate belongs, so a pre-commit run stops building the registry
  crate. Its rule against sibling-dialect imports in vyre-libs is replaced by
  one against an import the manifest does not declare, because composing
  another dialect is what the composition policy asks for and undeclared
  coupling is what it forbids; dialect membership is now derived from the cfg
  attributes in vyre-libs/src/lib.rs instead of a hand-kept list of five names,
  and the fix text names the shared builder and descriptor modules. Two checks
  are gone: a large-file advisory that was a third owner of a measurement the
  file-size cap and the composition audit already own, and a ban on IR
  construction under vyre-libs, which forbade the one thing a Category A
  composition is defined to do and whose 203-row path exemption list had rotted
  to 59 dead rows because no path list survives a file split.
- The release-benchmarks gate returns a report for every invocation, including
  --help, and captures the output of the commands it spawns, so a delegated run
  no longer fails to parse a formatted table as a report.
- The command-hygiene scan reads authored documents only. CHANGELOG.md and the
  release notes beside it are generated from release/changes, and a released
  entry records what a version did rather than telling a reader what to run, so
  twenty bare-cargo mentions in frozen history were recorded with line numbers
  that every added fragment moved. The evidence artifact went red for documents
  nobody had edited, and the text could not be fixed where it was reported.
- `ArtifactSession::host_bindings` and `submit_host_inputs` ask for one buffer
  per graph external. The set was the union of every artifact entry's inputs,
  so a program the compiler split across fusion groups demanded a buffer for
  each inter-group intermediate as well, and the count refusal named a number
  the caller could not derive from the graph it authored. A value some entry
  produces is device state; a value no entry produces has no source but the
  caller. A slot whose value is missing from the artifact resource set is now
  refused instead of bound at zero bytes.
- hot-path-scan matches a pattern at a path boundary, so SmallVec::new is no
  longer counted as Vec::new and one FxHashMap::new is one finding.
- A package error behavior override no longer stands in for a missing layer
  profile, and an override declared empty is reported instead of rendering a
  heading with nothing under it.
- A layout move in vyre-libs launches over the elements it moves. Every move
  guards on an element count and declares its buffers, and a launch geometry
  inferred from those buffers takes the largest one, which is the output for a
  gather and the whole cache for a scatter. The paged key-value append is the
  first scatter: it moves one decoded token chunk into a cache that is
  deliberately much larger, so an inferred launch fired one lane per cache
  element on every decode step and let the guard discard the rest.
  nn::attention::attention_layout_dispatch_grid and
  ATTENTION_LAYOUT_WORKGROUP_SIZE are now the owners of a layout move launch,
  llm::paged_kv::paged_kv_dispatch_grid sizes both paged moves from the token
  count they touch, and the grid and the guard read the same element count so a
  launch cannot cover a different domain from the one the move admits.
- The four Scallop lineage fixpoint bodies take one LineageFixpoint value
  instead of restating a nine-argument wiring list each, and the cell count,
  word count and per-lane chunk count are derived there. A relation cell is
  owned by one lane which walks the w words inside it, so the launch grid
  covers n*n*w words with n*n lanes, and the grid-sync fences the large-matrix
  path emits are cut into sequential dispatches rather than reaching an
  emitter.
- docs-references now resolves markdown link targets, so a page reached by a
  link is held to a published path the same way a code span is.
- docs-references reads every markdown link target as a path claim, so a target
  with no file suffix and a target naming a directory are resolved instead of
  skipped.
- The reader that finds which file defines an operation no longer loses every
  literal after a char literal holding a double quote, so a lexer is placed
  where it is registered.
- The operation placement reader parses the workspace manifest as a TOML
  document rather than as a single TOML value, and reports an unreadable,
  unparseable or crate-rootless manifest by name, so one broken file is no
  longer answered as every registered operation having no definition site.
- The benchmark harness names its build profile from the optimization level
  cargo compiled it with and the assertion cfg together. It read the assertion
  cfg alone, so a profile that turns assertions off without optimizing reported
  itself as release, and the guard that refuses an unoptimized release
  measurement let it through. The recorded environment carries the level as
  `build.opt_level:<n>`.
- The shared module walk reads a cfg(not(test)) declaration as production
  source rather than as test source, so a module carrying it stays on every
  feature route it belongs to.
- Seven modules whose source file lived outside the directory of the module
  that declared it are moved inside it. A `path` attribute was carrying each
  one across a directory boundary: the typecheck critical-contract tests, the
  resident work queue telemetry tests, the C preprocessor GPU byte filter and
  its eleven program files, the validate rule-pipeline tests, the fusion tests,
  the resident planner contracts (which were named `core_tests` two directories
  up), and the reference round-robin node stepper. Each now sits under the
  module it belongs to and is declared without an attribute. The fusion tests
  had two declaring owners, the parent module and the file they prove; only the
  file that proves them declares them now.
- The shared module walk records each file it has read, so a symlinked cycle in
  a crate source tree ends the walk instead of running forever.
- The digraph resolution phase of the C11 lexer no longer borrows another crate
  operation id for its name. That child region was labelled with the utf8
  validation id and then, when the constant went crate-private, with the line
  index id: two different primitives for one body that resolves digraphs and
  splices lines and calls neither. It is a phase boundary inside one operation,
  so it carries the anonymous prefix and names what it does. Nineteen pinned IR
  fingerprints across the lexer and parser walk families are re-recorded with
  the change that moved each of them: this rename, the earlier rename of every
  child region that had derived its name from its parent operation, and the
  collapse of the C declaration prefix walk onto one owner whose disqualifier
  token set differs from the copy the annotate builders had been reading.
- The contract that a failing xtask command says why it failed judges the exit
  itself. It used to match one shape of the original defect, an `if` whose
  condition named a blocker and whose branch exited, and the gate architecture
  removed every member of that set, because a gate now returns findings and
  only the dispatcher exits: the rule matched nothing and passed by judging
  nothing. It now derives every nonzero `process::exit` in the xtask crates at
  run time and requires an enclosing block to write the cause on either stream,
  so a silent exit added anywhere in the tooling fails it.
- The Metal parity gate resolves vyre-conform and the driver crate to their
  workspace member directories instead of joining the package name onto the
  checkout root.
- The contributor and crate documentation names the tree as it stands:
  CONTRIBUTING.md carries no host-local build settings, the placement charter
  points at docs/architecture/crates.md, and the vyre-primitives page lists
  only the paths and features the crate still declares.
- The operation placement reader reports the declared package name rather than
  the workspace member directory, so a crate under a grouping directory is
  named by something a reader can pass to cargo.
- The single-workgroup prefix scan sweeps each element a bounded number of
  times instead of writing every lane on every round. The workgroup is capped
  at 256 lanes and a longer input gives each lane a run of elements, so the
  scratch traffic no longer carries a log2 factor: at 1024 elements the
  dispatch executes 1544 workgroup-scratch stores where the previous form
  executed 31745. Sizes above the single-block limit are the multi-block chain,
  and vyre-libs scan_prefix_sum is the one builder that picks between the two.
- The artifact route returns a Program's writable buffers in the order the
  Program declares them. `ArtifactSession::ordered_outputs` projects canonical
  ABI slot order, which numbers graph values, and a graph lifted from one
  Program mints an external value for every retained read-write buffer before
  the node that produces the declared outputs. Slot order is therefore
  retained-then-output, so a Program declaring an output buffer before a
  retained one had its outputs returned transposed: `vyre-libs::decode::base64`
  returned its 4-byte decoded length where the caller reads the 24-byte
  payload, `decode::inflate_stored_block` 4 against 40, `matching::emit_hit` 4
  against 64, and `nn::ln_scale_backward` returned `grad_scale` where `grad_x`
  was read, which read as an f32 value disagreement rather than a
  transposition. Slot order cannot express declaration order, because an output
  value does not exist before the node that produces it, so the projection back
  onto declaration order is now `ArtifactSession::program_outputs`, keyed on
  the canonical resource names, and every caller holding the Program takes it:
  `vyre-conform`'s production session and all three `vyre-bench` artifact
  submissions. `ordered_outputs` keeps its slot-order contract for a caller
  that authored the graph and has no Program to declare an order. Closes 7
  diverging (backend, op) conformance pairs.
- The release benchmark generator builds vyre-bench with --release, vyre-bench
  refuses to measure the release suite from an unoptimized build, and every
  release benchmark artifact records the build profile that measured it, so a
  debug-build CPU baseline can no longer inflate a published speedup.
- A cfg attribute quoted inside a comment or a string literal no longer gates
  the item written after it, so a file that documents the scanner keeps its
  production code in the scanned text and its test fixtures out of it, and a
  malformed attribute no longer leaves every later test item classed as
  production.
- Three file-size ratchet rows whose files moved from vyre-primitives into
  vyre-libs follow the code to its new path at the measured count instead of
  lapsing to the flat cap.
- The crate README generator renders nothing while the ownership registry, the
  crate guides or the release train disagrees with the workspace, and it
  refuses to write a generated contract that itself claims a retired release,
  while a retired claim in the crate own prose is reported and no longer blocks
  the regeneration of the generated region.
- A per-backend conformance artifact recorded under an older shape is now
  reported as stale, naming both the version it carries and the version the
  reader holds, instead of as unparseable JSON. Three committed artifacts
  carried a row-count field under its former name and read as corrupt files.
  The shape version is raised, and a test pins the recorded field set to it so
  a rename turns the suite red until the version records that the shape
  changed.
- The repository hygiene gate read the instruction redirects case-sensitively
  and reported both CLAUDE.md and GEMINI.md as policy files, because each opens
  its sentence with a capital letter. It also reported its own rule table and
  the loudness gate's, since the language a silent-skip rule forbids is a
  string literal and so is the table that spells it. The two rule sources are
  exempt by path, and a test requires each to exist and to still carry the
  language, so a stale exemption is red rather than a silent widening.
- The compiler-grade thesis axes and the CPU-SOTA 100x contract page name the
  workloads the suite measures: AST motif traversal replaces the deleted C
  parser axis, e-graph saturation points at workload 17, and the required 100x
  case list matches the ten cases the release thresholds declare. The two
  superseded e-graph artifacts are deleted and the cross-backend comparison
  table is regenerated from what remains.
- The release workload matrix calls a CUDA command reproducible only when
  --release reaches cargo. Everything after the argument separator goes to the
  benchmark harness, which builds nothing, so a --release written there
  described an optimized build that never happened.
- The metal-parity device run quotes the remote checkout path for the remote
  shell and refuses an ssh destination that opens with a dash, so neither value
  can carry a command.
- The crate graph renderer reports a dependency that carries no registry record
  by name instead of panicking on the lookup.
- The required CI document names each gates workflow job by its display name,
  which is the status context branch protection waits for.
- The required-context gate accepts only the status context a job reports,
  which is its display name when it declares one, so a required check named by
  job id is a finding.
- A row in .github/CI_REQUIRED.md naming a workflow the checkout does not carry
  fails the ci-required gate, and the workflows whose path filters and
  directories no longer exist are deleted rather than parked.
- A retired release claim is any dotted number that starts with the retired
  train, so a four component version is reported, and the same digits inside
  another train version or inside a hash are not.
- The composition audit reports an unreviewed shape pair rather than a
  duplicate. A shape verdict cannot tell a shared algorithm from a shared IR
  idiom: a guarded lane index, a row-major loop nest and straight-line unrolled
  arithmetic have one fingerprint whatever work they do. The reviewer records
  one of two outcomes, a shared builder or a reviewed pair with the reason the
  shape cannot express, and every row of both lists is checked against the live
  registry.
- A gate that detects a stub may spell it. A code-call pattern found only
  inside a string literal is a rule definition, the same reason a doc comment
  was already exempt, so a pattern table row reading text equals todo-open no
  longer reports the gate that owns the rule. A real call still blocks the
  release. The two lists of files that own a rule are named once and a test
  requires every row to resolve, after two rows had outlived the tree they
  named.
- nucleus_select built a program that read below the start of its candidate
  buffer when it was given no candidates. The prefix walk kept nothing, the
  fallback set the kept count to the candidate count, and the draw indexed one
  element before that, so a zero count addressed u32::MAX. It now returns a
  trap program naming the parameter, which is the shape the other count-taking
  sampler stages already used.
- A scanning rule whose skip predicate excludes every file under its roots now
  fails with the scope and the number of files the skip removed. A missing root
  was already fatal for the same reason: a rule that reads no file reports
  success forever. The lego audit also holds its shared-plumbing directory rows
  and its cross-dialect rule to the same standard, and the cross-dialect rule
  reports a violation instead of a note.
- The operation-schema contract tests read the validator's report. A delegated
  gate binary returns findings as JSON and exits zero, because the dispatcher
  treats a non-zero exit as a gate that could not run, so eleven mutation tests
  asserting a non-zero exit could never fail and passed against a committed
  schema that already disagreed with the live registry. The schema is
  regenerated from the 359 live registrations.
- The ci-steps gate reads every script under scripts/, including the shared
  shell functions and readers in scripts/lib that the registry declares and
  every script sources; a single-level read skipped them while reporting the
  directory covered. Cargo invocation resolution moved into that gate, which
  reads workflows and manifests rather than Rust source text, so its
  structural-gate row is deleted rather than re-pointed.
- The sharded release conformance run waits for one worker at a time, so a
  clean run exits zero and a failed shard is counted once.
- The elementwise map builders live in the builder module rather than under
  math. A shared skeleton hosted inside one gated dialect is unreachable from
  the others: logical depended on math-broadcast only to reach the builder, and
  nn quantization could not reach it at all and carried its own copy of the
  unary map. The logical feature no longer pulls in math, and int8 packing, the
  skip gate and the EMA update route through the shared builders.
- The one-public-path gate reads the crate source to tell an item at two paths
  from a name several sibling modules each declare. A snapshot line is
  identical for both, so a terminal id table per grammar, a per-op identifier
  and a lint entry point named for the scan it runs all read as one item
  published four ways, and a quarter of the measured count could not be closed
  by deleting anything. A name the crate declares twice is two items; only a
  name declared at most once can be one item at two paths. The count of shared
  names is reported and left unpinned, because a module is what disambiguates
  two grammars naming the same bracket.
- The exemption list shipped with vyre-lints drops 59 rows that named a file no
  longer in the tree, and a test now fails on any row that names nothing. An
  exemption keyed on a path silently exempts nothing after a rename or a split,
  so the count it holds back grows by every construction site in the files that
  moved. The header states what the file is: the default configuration of the
  shipped lint, read by no workspace gate.
- The source scanner reports the length of a non-code span as a non-zero value,
  so a scanner that skips one cannot fail to advance and every caller drops its
  own guard against a zero-length answer.
- The routing-contract closure test reads a split op as one module. An op whose
  program moved into its own file registered in the tests module beside it,
  which the test looked for only under a directory named for the file, so a
  routed convergence op read as unregistered while its registration sat two
  lines away. A registration in a neighbouring file counts when it names the
  op, so a dialect directory shared by several ops still cannot lend one op's
  registration to another.
- The sentence that declares a recorded artifact stale is now formed and
  recognised through one constant in xtask::source_provenance, so the four
  producers and the recogniser cannot drift apart.
- A workflow step that runs the gate sweep with --subset credits the gates in
  that subset and no others. Recording the bare sweep for it gave every
  registered gate a workflow, so a gate no workflow selects could not be
  reported and the rows that were reported named the wrong file.
- A test that compiles a scratch crate builds it in the cargo build directory.
  vyre_test_support::monorepo::cargo_target_directory reports where cargo is
  writing this run's artifacts, resolved from the running test binary, so no
  test reads or sets CARGO_TARGET_DIR. The feature-boundary fixture had
  compiled the substrate and the foundation under the temp filesystem, which is
  capped, and it no longer clears its own build tree between the two consumers
  it checks.
- The walker that checks every registration is linkable now reads production
  text only, so a registration quoted inside a test module is no longer
  reported as one the build cannot reach.
- An artifact whose math reaches `Exp`, `Log`, `Sin`, `Cos` or `Tanh` compiles
  for CUDA. The PTX emitter refused to lower those ops to a native instruction
  unless `PtxEmitOptions::ulp_budget` was positive, so 21 registered ops failed
  target emission on the production artifact route with a refusal naming a
  setting no caller on that route could reach. The refusal described a choice
  that does not exist: PTX has only an approximate `tanh`, `ex2`, `lg2`, `sin`
  and `cos`, so there was no strict form to fall back to and nothing for a
  budget to select. It is gone, and those ops now emit their one instruction
  unconditionally. The budget still governs `InverseSqrt` and `Reciprocal`,
  where PTX does offer both a `.approx` and a `.rn` form, and `Sqrt` was always
  exact. `vyre_foundation::fp_parity::f32_ulp_tolerance` remains the owner of
  the acceptance window the reference comparison judges output with.
- The tail of a failed child command is cut at a character boundary. A cut that
  landed inside a multibyte character fell back to the whole stream, so the
  byte bound did not hold for the output it exists to bound.
- A whole-grid fence is cut into sequential dispatches during compile-request
  validation instead of failing at emit. Program fusion writes
  MemoryOrdering::GridSync inside one node body, so a single-node graph had no
  fusion pair to reject and every wgpu compile of a fused security flow such as
  taint_pollution failed in the WGSL emitter. The node now becomes one node per
  segment, ordered by an explicit retained-state succession that fusion
  legality cannot contract back together. A fence inside a loop body has no
  correct cut and is refused with the loop named.
- A workflow step that runs a repository script is reported, since every
  continuous integration assertion is a registered gate and a script invocation
  carries an assertion the registry, the baseline and the subset roster cannot
  see.
- A standalone workgroup reduction is built from a WorkgroupReduction value
  naming one fold, so the identity, the combine and the tree sweep are derived
  together instead of arriving as three arguments that could disagree. The
  nine-parameter builder and its too_many_arguments allow are gone, as are
  three further allows on Scallop provenance dispatch entry points that were
  already under the argument threshold.
- The one-implementation rule for target-payload admission recognizes the
  descriptor form. Every concrete backend now routes through
  `TargetDescriptor::admit_modules`, which calls the shared `admit` and decodes
  each admitted module in the backend's own dialect, but the rule still looked
  for a literal `materialize::admit(` call and so reported all four backends as
  hand-rolling admission. It now accepts either spelling and additionally
  rejects a backend that defines `admit` or `admit_modules` itself.
- A benchmark artifact session is cached against the program fingerprint and
  the device signature together. The key was the program alone, so a context
  repointed at another backend served an artifact compiled for the previous
  device facts, and the report named a device the measured artifact was never
  built for.
- The workspace-wrapper hygiene rule reads a diagnostic that names a cargo
  command as a sentence rather than an invocation, so a gate spawning through
  the one cargo resolver is no longer release-blocking, while a spawn naming
  its program and a message printed for a reader to type stay findings.
- Gate fixtures that created an empty directory to stand for a crate now write
  Rust source into it, so a rule that reads whether a directory carries source
  is proved in both directions.
- The hot-path scan measures the path a successful call takes. An allocation
  that builds an error is excluded and counted separately, because this
  workspace requires an error to carry context and a fix, so the CUDA host
  dispatch surface read as nineteen per-launch allocations when every one of
  them was a format inside a return Err. Each file's note now ends with its
  error-path count, and every hot-path budget in
  docs/optimization/HOT_PATHS.toml is lowered to the measurement that remains.
- The composition audit reports an exemption row that matches nothing. The
  phase-marker list and the declared Tier-3 leaf list are named once each and
  every row is checked against the live registry, so a row naming a renamed or
  deleted op is a finding instead of reading as coverage. Two rows were already
  in that state and are gone.
- An f32 literal reaches a materializer with every bit intact.
  `LiteralValue::F32` was written as a JSON number inside the target-module
  bundle, and JSON has no non-finite number, so a lowering that seeds a running
  maximum with negative infinity produced a bundle carrying `null` where the
  literal belonged and every backend refused it with `invalid type: null,
  expected f32`. That took `vyre-libs::nn::top_k` and `nn::softmax_top_k` out
  of the conformance certificate. A non-finite literal now carries its IEEE-754
  bit pattern in hex; a finite one is still a plain number, byte for byte what
  it always was, so no other surface carrying a descriptor changes shape. The
  escape is asked for only where the format is self-describing: a compact
  format carries all 32 bits in a number and answers no `deserialize_any`, so a
  dumped descriptor stays decodable and a descriptor hash taken over that
  encoding keeps its value. `TARGET_MODULE_BUNDLE_SCHEMA_VERSION` moves to 3
  because the payload can now hold values version 2 could not represent.
- Every command this workspace tells a reader to run names the wrapper. The
  dispatcher usage text, its rebuild and help messages, the scaffold and audit
  binaries, the structure gate header, the error catalog regeneration note, the
  conformance witness fix, the benchmark crossback header and the optimization
  docs note all spelled a bare cargo invocation, which builds with a different
  configuration than the wrapper in the same checkout.
- The generated operation schema reads the tier and the enabling features each
  operation registers instead of deriving them from the identifier. Reading the
  vyre-primitives and vyre-libs prefixes reported 163 intrinsics against 164
  compositions where the tree has 9 intrinsics and 318 compositions, and a
  table of 18 domain names supplied a feature for every operation whether or
  not one gated it. The defining crate of an operation is now the crate that
  registers it, found by walking module declarations from each crate root, and
  the number of registrations that record no enabling feature is a pinned count
  rather than a fabricated one.
- An operation tier that reaches no owner rule is reported as the id that
  carries it, so a tier added to the registry cannot take the empty owner list
  a foundation operation gets.
- The testing guide orphan scan is suppressed only when a row never reached the
  render set, so a finding about one member no longer hides every leftover
  guide in the directory.
- The operation matrix reads the owner directory for a domain from the tree, so
  an id whose domain moved under another module, such as the optimizer and
  quantization compositions under nn, names the directory that carries its code
  instead of a top-level path with no source in it.
- The op matrix reads each operation owner directory relative to the checkout
  root and resolves each namespace and domain once, so the generated rows no
  longer depend on the directory the run started in.
- Every CODEOWNERS row names a path this tree carries, and a tree contract
  fails when one stops resolving. Nine of fourteen rows named an older layout,
  including `/spec/src/lib.rs`, `/conform/src/reference/`, a threat-model
  document that is not in the tree and two scripts described as future files,
  and GitHub ignores a pattern that matches nothing, so the review requirement
  on the algebraic laws and the CPU reference interpreter had been off for as
  long as those directories had their current names. The rows now name
  `/vyre-spec/`, `/vyre-reference/`, `/conform/`, `/xtask-evidence/` and
  `/release/evidence/`.
- The operation placement reader reports a source file it could not read or
  that exceeds the read cap, so the registrations that file holds are no longer
  silently missing from the schema.
- A tracked gate source the working tree cannot read is a gate canon finding
  naming the file rather than an error that aborts the gate, so the rest of the
  registry is still judged and a ratchet constant nobody could read is not
  silently unjudged.
- The feature-isolation sweep no longer reports a selection that the record
  does not mention as a blocked row that has started compiling. A missing row
  is the agreement half's finding and carries the fix that closes it, so
  counting the same omission twice sent a reader to a row that does not exist.
- An unrelated finding no longer hides an orphaned testing guide; only an empty
  crate record set suppresses the orphan scan.
- `abstraction-gate` no longer demands an operation registration for a region
  that names no operation. Two prefixes mean the same thing: `inline::`, minted
  by `reparent_entry_node` for a body the composer reparented onto its caller,
  and `anonymous::`, written by a builder that needs a named phase boundary
  inside one operation. The gate knew only the first, and fell back to
  `source_region.is_some()` for the second, which proves nothing because
  composition stamps `source_region` onto every entry region it reparents.
  Seven regions were reported as unregistered building blocks that must not be
  registered. `vyre-foundation/src/algebra/composition.rs` owns the answer as
  `ANONYMOUS_GENERATOR_PREFIXES` and `is_anonymous_generator`, and the gate's
  fix text now names the rename. The gate also descends through
  `vyre_foundation::visit::child_bodies` rather than its own list of node arms,
  so a new nesting variant cannot hide a region from it.
- Architecture guides now use the generated 36-crate dependency graph, joined
  operation registries, CUDA-first backend evidence, typed cross-program
  composition, and explicit runtime/compiler/driver megakernel boundaries. The
  earlier device-bytecode-interpreter RFC is retained as superseded rationale.
- Artifact materialization now maps target resources to Program inputs by
  canonical identity instead of backend descriptor position, excludes
  backend-allocated read-write outputs from prior host inputs, and rejects
  conformance fixture byte lengths that disagree with the canonical Program ABI
  before device measurement.
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
- Validation rejects a program whose async copy may still be in flight where
  the invocation ends (V133), so a transfer nothing waited for is caught before
  a backend reads a destination no wait ordered.
- Validation rejects an async copy that restarts a tag still in flight (V131)
  and a wait with no transfer to wait for (V132), so a multi-stage pipeline
  that reuses one tag is caught before a backend copies over a live
  destination.
- Validation refuses an async transfer whose destination resolves to a buffer
  the program declared unwritable (V134), which the target compilers would have
  lowered to stores through a read-only binding and which loop-invariant
  hoisting was already assuming could not happen.
- Runtime barrier elision now takes child descent from the foundation rewrite
  walk and buffer-effect descent from visit::child_bodies instead of two
  hand-written matches ending in a catch-all, so a Node variant added with a
  body is descended into on the commit that declares it rather than treated as
  a leaf, and an unchanged scope is no longer rebuilt.
- A strong barrier is narrowed to the address spaces its body touches, and that
  decision was covered only by two pinned backend corpora. One of them was
  stale: the WGSL golden carried the narrowed storage barrier while the SPIR-V
  golden still pinned the memory-semantics word for the wider fence, so the
  SPIR-V byte-stability test failed on a change nobody had made. The mapping is
  now tested where it is decided, including the storage-only, scratch-only,
  both, neither and nested-body cases and the refusal of relaxed and grid-wide
  orderings, and the SPIR-V corpus is regenerated to the narrowed word.
- Megakernel fusion admits two programs that synchronize at one shared
  workgroup geometry, and reports MKL006 only when the geometries differ and
  one program reasons about the size of its own workgroup.
- `vyre_libs::math::bellman_shortest_path::BellmanBuffers` publishes
  `CANONICAL` and `TERSE` binding-name sets, matching the
  `SinkhornBuffers::CANONICAL` it already had. Four sites spelled the same six
  names, one of them in another crate, while the record documented itself as
  the only place they were spelled. Every field is a `&str`, so a full
  re-spelling is a positional list wearing field names: a transposed
  `src`/`dst` or `dist`/`next_dist` compiles, looks deliberate from either
  copy, and silently relabels which buffer the relaxation reads. The two sets
  stay disjoint so an assertion that reads a name back out of a program cannot
  pass by matching the other set.
- `scripts/check_bench_baselines.sh` requires a `benches/RESULTS.md` section
  for every crate that owns a bench source file, not for every directory named
  `benches`. The directory search demanded a measured section for
  `vyre-grammar-gen`, whose `benches/` directory holds documentation and no
  target, so the gate could not pass without an invented number.
  `benches/RESULTS.md` now carries the criterion medians of `cargo bench -p
  vyre-primitives --bench wire_throughput` and `cargo bench -p vyre-bench
  --bench release` with the machine, GPU, CPU, toolchain, and commit they were
  measured on.
- Benchmark case declaration and span timing each have one owner.
  `api::metric::elapsed_ns` narrows every measured nanosecond count, replacing
  three spellings across 35 sites in 28 files: a bare `as u64` cast, a
  `min(u64::MAX)` clamp, and a `try_from().unwrap_or()`. The bare cast wrapped,
  so a span past about 584 years reported as a short one and the slowest sample
  read as the fastest. `cases::honest_case` owns the suite list, metadata and
  memory floor every honest case declares; `search.binary.u32.1m` had omitted
  the smoke suite from its own copy of that list and was never smoke-tested.
  `cases::reference_sample::run_against_reference` accounts both halves of a
  reference comparison, closing two records that published a baseline carrying
  only a wall time and one that read the baseline's written-byte total off the
  device output. `cases::release_workloads::resident_batch` owns resident batch
  dispatch and its metric points, replacing two hardcoded reset-byte constants
  with the uploaded payload length and routing the metadata condition workload
  through the shared run assembly it had bypassed, which is where it had been
  silently omitting its throughput metrics. `api::case::prepared_as` and
  `api::case::prepared_as_mut` own borrowing a prepared payload as its own type
  and own the wording when it is the wrong type. Sixteen cases hand-rolled that
  downcast; the read-only and mutable flavours meant nine copies survived a
  first collapse, and the message had drifted into three unrelated sentences
  for one condition. An `Option`-returning downcast is a different operation
  and stays where it is. `case_declaration_contracts` derives its coverage from
  the case registry and a walk of the crate source, so a new case or a
  reintroduced narrowing fails by name rather than going uncovered.
- The timed CPU reference in `vyre-bench` has one owner,
  `vyre-bench/src/cases/reference_sample.rs`, and it saturates. Eleven cases
  hand-rolled the same timer and seven of them cast `Duration::as_nanos()`
  straight to `u64`, so a reference slower than roughly 584 years was reported
  as a small number instead of a large one, inverting the speedup it was
  compared against.
- A benchmark performance contract may only name a CPU baseline this checkout
  can run. Ten release workload rows advertised tree-sitter, libclang,
  Hyperscan, ripgrep, egg and unnamed SIMD or optimized CPU implementations,
  bigint.modexp.4096 named rug 1.27 with a GMP backend, and interpreter named a
  hand-tuned C threaded interpreter with computed goto. None of those were
  linked: every one of those cases timed an in-tree scalar Rust routine. The
  labels now come from one owner that describes the routine actually timed, and
  a new contract test resolves every named crate against the workspace members
  and the vyre-bench dependency tables read at run time, so a case that invents
  a competitor fails instead of shipping. The baselines that do run, faer,
  openssl, hashbrown, pcre2 and rayon, are unchanged.
- Enforced benchmark contract failures now retain correctness, timing metrics,
  device identity, and measured speedup in the failed case report instead of
  collapsing into an unprobed error shell.
- Removed the unreachable `vyre-bench` dataflow baseline module whose
  undeclared feature could never be enabled and whose engine dependency does
  not exist in this workspace. Benchmark feature guards now have a manifest
  agreement gate, so a hidden undeclared case cannot recur.
- `BinOp::result_class` is the one answer to what a binary operator's result
  type is, and `BinOp::takes_numeric_operands` is derived from it.
  `validate::typecheck` asked that question twice and wrote its own operator
  list each time, once to give an expression a static type and once to decide
  what its operands must be. The lists had already drifted on `AbsDiff`, which
  is in the operand list and not the type list; that happens to be correct and
  nothing said so, and nothing would have caught the reverse. Both lists ended
  in a catch-all, and `BinOp` is `#[non_exhaustive]`, so a new operator took
  whatever the last arm held. The match now lives in `vyre-spec` beside the
  enum with no catch-all arm, so adding an operator fails to compile there
  instead. `BinOpResult` is closed on purpose: a new operator is additive, a
  new RESULT CLASS is a new answer every consumer has to decide for.
- A dispatch that wrote its outputs into caller-owned slots was invisible to
  dispatch telemetry. Both borrowed-input `dispatch_borrowed_into` defaults
  recorded output-slot pressure and skipped `record_dispatch_io`, so
  `vyre_driver_dispatch_launches_total`, the input byte count and the output
  byte count only ever advanced on the resident path. Every backend that does
  not override the method went uncounted.
- The buffer-set equivalence property reads the declared owners instead of
  restating them. `referenced_buffers` answers from `ProgramFacts`, whose SoA
  extraction fills its `buffer_refs` column from its own exhaustive match, and
  nothing made that column agree with `visit::node_buffer_refs` and
  `visit::expr_buffer_ref`, which are the declared owners of which variant
  names a buffer. The oracle that checked it was a third enumeration ending in
  `_ => {}`, so a variant added to both real enumerations was reported as
  naming nothing by the oracle and the property passed while the two sides were
  free to disagree about it. It now composes the two owners, which makes it an
  oracle for the agreement rather than a second opinion about the variant list.
- The radix prefix table, digit-value decode, type-suffix set and digit
  accumulator of a C integer literal are owned by
  `parsing::c::preprocess::c_int_literal_grammar`. The standalone scanner and
  the inline scan in the `#if` evaluator each carried a copy and the copies had
  drifted: the evaluator accumulated with wrapping `u32` arithmetic, so a
  literal above `u32::MAX` could carry to zero and flip a conditional, and it
  consumed a type suffix after a radix prefix with no digits after it. The
  scanner and the CPU `consume_integer` saturate and reject, and that is now
  the only spelling.
- `vyre_libs::parsing::c::parse::vast::build::vast_row_fields` clamps the index
  before loading a prior row kind. The subtraction was unclamped, and because a
  select in this IR evaluates both arms, row 0 wrapped its index to near
  `u32::MAX` and the untaken arm addressed a row far past the end of the table.
  Three of the five hand-written copies of the read already clamped. The
  global-typedef fast pass also read its forward neighbour with no out-of-range
  fallback while every sibling and the CPU oracle substituted the sentinel;
  both reads now come from the shared owner.
- Release operations now use one runbook and one generated checklist derived
  from release-train versions, repositories, package groups, tags, approval
  actions, and validated changelog fragments. The guarded launcher pushes
  candidate tags before publication, final tags afterward, and records
  completion only after external actions succeed.
- The `cargo_full` wrapper no longer exports `CARGO_BUILD_JOBS`. An environment
  variable overrides `.cargo/config.toml`, so the wrapper's default of `1`
  silently replaced the declared `[build] jobs = 16` on every invocation, and
  every build, test run, and gate in this workspace ran one codegen job at a
  time regardless of how many cores the host had. Parallelism is declared once,
  in the config file, where it is reviewable and applies to every build
  equally.
- Test fixtures now resolve host-owned Cargo target directories from either
  Cargo cache marker. Older shared target roots that lack `CACHEDIR.TAG` but
  contain Cargo's `.rustc_info.json` no longer fail before feature-boundary
  builds can run.
- The asm-alias, mixed-initializer and incomplete-initializer classification
  contracts live with the rest of their family in `vyre-libs`. Nothing in them
  dispatches anything, so holding them in a driver crate's tests made the CPU
  classification of three C constructs depend on that driver compiling and left
  a package-scoped `vyre-libs` test run unable to check them. Deleted with the
  move: a second fixture builder for `typedef int (*fn_t)(int); fn_t f;`, whose
  four assertions were already made against the byte-identical
  `c_frontend::fixtures::declarator_matrix_constructs::fixture_nested_typedef_complex_declarator`.
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
- All remaining compile-time checkout root derivations in `vyre-foundation`,
  `vyre-megakernel`, and `vyre-driver-cuda` resolve repository and fixture
  paths at run time through `vyre_test_support::monorepo` delegations
  (`vyre_workspace_root` and `vyre_crate_directory`). `structure-gate` checkout
  provenance gates enforce runtime path derivation across all member sources
  and test binaries, catching both `env!` and `option_env!` variants without
  waivers.
- The workspace cargo runner no longer exports a workstation-specific target
  directory, so hosted CI and new checkouts use their own writable Cargo
  configuration.
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
- `workspace-clippy` passed `--message-format=json` after the `--` that hands
  the rest of the line to the lint driver, so clippy-driver answered
  `Unrecognized option: 'message-format'` for every crate and the only gate
  that judges lints could not run. The flag now sits on cargo's side of the
  separator.
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
- Region fusion reads which buffers each side touches, and both of the walks
  that answered it enumerated buffer positions per node variant and ended in a
  catch-all. Neither had an arm for the four collective statements, which name
  their operands as buffers and carry no operand expression at all, so an
  all-reduce, all-gather, reduce-scatter or broadcast was reported as touching
  nothing; an atomic read-modify-write was reported as a pure read. The
  positions are now owned exhaustively in
  `vyre-foundation/src/visit/node_parts.rs`, where adding a variant fails to
  compile, one walk collects both directions, and a node carrying an opaque
  payload reports the sets as a lower bound rather than as empty, which
  declines the fusion instead of guessing at it. The test that should have
  caught this compared two inline copies of the walk against each other and
  never called the production one.
- The megakernel launch floor and traffic rate are derived from the recorded
  benchmark files at run time, so a weight that drifts from the recording it
  cites fails the suite.
- The registry-closure coverage corpus counts only test-gated source. It had
  treated every byte after the first `#[cfg(test)]` marker as test text, so a
  production re-export list covered 174 symbols by naming them, and a test
  module written in its own file counted as production code. A crate whose only
  builders are test fixtures now reports zero builders and is held honest by a
  production-file guard instead of a floor.
- CPU-parity integration tests now declare their required Cargo features.
  Default-feature `vyre-libs` test builds no longer import reference-only
  symbols that are intentionally absent from production builds.
- Workspace crate ownership now comes from one manifest-checked registry. The
  tier gate rejects missing crates, undeclared production edges, and stale
  generated graph or ownership guides, while planned compiler boundaries stay
  visibly separate from current workspace members.
- The cross-backend comparison gate derives its table from the committed
  release benchmark evidence and records it under release/evidence/benchmarks/.
  It used to time an unrelated loop inside the build-task crate, record no
  backend measurement at all, and write the result into a gitignored directory,
  so a fresh checkout reported the table missing and one local regeneration
  turned the gate green on a table nobody could read. A table carrying no
  measured backend row, a measurement arriving without its commit, source-tree
  fingerprint or device signature, and a recorded table that disagrees with the
  evidence are findings now, and a backend a case contract declares without a
  measurement is listed in the table instead of dropping out of it.
- `vyre_libs::graph::csr_backward_or_changed` named the per-edge kind array
  `masks` and the scalar edge-kind filter `edge_kind_mask`, inverting the roles
  every sibling CSR module gives those two names. A call repointed from a
  sibling module fed the scalar where the per-edge array belonged. Both
  parameters now carry the tree-wide role names.
- The CUDA e-graph device-image upload contracts build their snapshots from
  named fixtures, and `assert_span_matches_foundation` has one definition.
  `upload_layout_contracts.rs` redefined that helper identically to the
  parent's, so the parent's copy was shadowed and a correction to it would have
  silently missed one file. Seven inline snapshot literals across four files
  were three distinct shapes, now `shared_eclass_add_snapshot`,
  `duplicate_add_snapshot` and `distinct_add_snapshot`, each documented with
  the property it exists to exercise.
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
- Every CUDA graph, executable graph, stream and device pointer guard keeps a
  reference to the context it destroys its handle against. The guards took
  their context liveness from a sibling field of the enclosing struct, and both
  enclosing structs declare that field first, so the context was released
  before the destroy calls ran. cuGraphExecDestroy then read freed driver
  memory and blocked forever on a lock word with no owner: a conformance
  dispatch stopped with no CPU use, no device work and no error, and the
  cross-backend parity matrix never reported a summary.
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
- The public-API snapshot for the CUDA driver names the crate its adapter
  signatures actually come from. Ten of its public items take or return
  `FrontierTypedPlan`, `DeviceResidentTokenFactGraph`, `MegakernelScaleSample`,
  and `MegakernelScheduleError`, which moved to `vyre-libs` when the pass
  engine narrowed to pass execution; the snapshot still spelled them under the
  pass engine, so the stability gate reported drift on a surface nobody had
  changed and would have gone on reporting it in front of every real change to
  that crate.
- A CUDA device now reports a device trap instead of returning wrong data.
  `Node::trap` is how an op refuses an input outside its declared domain, and
  `vyre-emit-ptx` lowered `KernelOpKind::Trap` to a source comment and a branch
  to the kernel exit: the trapping lane left and nothing was recorded, so the
  host could not tell a refused dispatch from a completed one. Both CUDA
  capability records nevertheless reported `supports_trap_propagation: true`,
  which is the flag `vyre_foundation::program_caps::check_backend_capabilities`
  admits a trap-declaring program on, so every guard in nine trap-declaring
  source files was a guard that did not exist on that backend. The emitter now
  declares a four-word module-scope trap record, claims it with one
  `atom.global.cas.b32` so exactly one trapping lane writes it, and stores the
  address operand, the trap tag code, and the reporting lane in the same word
  layout the secondary text emitter uses. The host resolves the symbol when it
  loads the module, parses the code-to-tag table out of that module's own text,
  zeroes the record once per launch sequence, and reads it back after the
  stream synchronize; a written record refuses the dispatch and names the
  address, code, lane, and tag. Trap tag numbering has one owner,
  `vyre_lower::descriptor_trap_tags`, replacing three copies of the same walk.
  Because the record is per-module, a launch that holds it serializes against
  other launches of the same module through the gate that already serialized
  cooperative launches, and CUDA graph capture refuses a trap-declaring program
  up front because a capture cannot synchronize and so could not read the
  record back.
- Compiled CUDA pipelines route trap-declaring and cooperative programs to
  direct stream-ordered dispatch instead of CUDA graph capture, preserving
  fail-closed trap readback without graph capture replay failures. Canonical
  text line indexing declares exact portable workgroup geometry requirements
  (256 invocations), validates explicit block lane powers of two, and
  propagates lowering widths uniformly across flag and scan passes so fusion
  never fails with workgroup geometry mismatches.
- The declaration-prefix walk in front of a VAST row is owned by
  `parsing::c::parse::vast::declaration_prefix_scan`. The precomputed-context
  declaration classifier walked the prefix forwards with no delimiter depth, so
  the `int` inside a cast reached the identifier after it and `(int) a;`
  classified `a` as an ordinary declarator; the self-contained classifier
  skipped balanced paren and brace groups. Both call the same walk.
- Four token tables that decide whether an identifier is a declarator name are
  single-owner in `parsing::c::parse::vast::token_grammar::declarations`. The
  declarator-follower set no longer admits `]`, which had made the bound in
  `int a[N];` a declarator on the precomputed path only. The declaration-prefix
  set gained `auto` and `register`, and the precomputed path now reads it
  instead of a 23-entry copy that omitted every C23 scalar type, `_Alignas`,
  `typeof` and the GNU specifiers. A new matrix reads the token vocabulary out
  of `vyre-spec` at test time and fails when any kind is classified differently
  by the two paths.
- Driver decorators now preserve the concrete backend device profile, including
  device-timestamp capability and timing quality.
- The two benchmark pins that named the three `frontend.rust.*` cases record
  their departure. Those cases left with the Rust front end, so the registry
  stopped publishing them and both pins had been red since; the thesis-workload
  list now names the C parsing workloads that remain, and the release suite
  detects parsing evidence from a workload's own `parser`/`ast` tag rather than
  from an id prefix that no longer matches any case. The enumeration pin
  reports which ids appeared or vanished instead of printing two truncated
  lists.
- The lane-count-to-dispatch-grid ceiling division has one owner,
  `vyre_primitives::dispatch_grid::lane_grid`, ungated at the crate root and
  re-exported from `vyre_libs::graph` so every existing call path resolves
  unchanged. It previously lived inside the `graph` domain, where a domain that
  does not enable `graph` cannot see it: `decode` does not, and
  `vyre_libs::decode::rle_segment_lengths` therefore split its lane count into
  whole blocks plus a tail block and floored the sum, its own fourth spelling
  of the same arithmetic. An owner a caller cannot reach is not an owner, and
  the caller that cannot reach it writes the copy. Routed onto the owner with
  it: `vyre_libs::graph::union_find`,
  `vyre_libs::graph::persistent_bfs::layout`, `vyre_libs::math::scallop_join`,
  `vyre_libs::math::scallop_join_wide`, `vyre_libs::math::bigint_add_carry` and
  `vyre_libs::math::scallop_persistent`, which carried a private ceiling helper
  of its own. The persistent-BFS copy spelled it `((count - 1) / width) + 1`,
  which underflows at zero and was safe only because its one caller guarded
  zero separately.
- The documentation reference rule read an absolute path in a code span as
  workspace-relative, so a document mentioning `/dev/null` was reported as
  claiming a missing `dev/null` in this repository. An absolute path outside
  the checkout is no longer treated as a claim this repository can satisfy; one
  inside the checkout is still checked. The feature matrix for `vyre-libs` also
  named nine scan modules by a path that resolved nowhere, and one of them, the
  regex compiler, has been a directory rather than a file since it was split.
- Workspace documentation now resolves NFA conversion and megakernel table
  links.
- The documentation matrix now covers every indexed public document and
  workspace crate README. Each row records audience, owner, authority, source
  artifacts, verification date, executable examples, version coherence, support
  status, and claim-evidence blockers.
- Documentation coverage now reports measured gates instead of universal
  completeness. Public guides distinguish generic consumers from named
  integrations, and the documentation gate rejects missing or gitignored
  repository inputs hidden in code spans and shell examples.
- Every current public guide is now revalidated for Vyre 0.7.2. Historical
  architecture, migration, release, operation, and testing documents are
  explicitly archived or superseded, generated views identify their source, and
  crate-local paths remain reproducible in a clean checkout.
- The backend error-code catalog is generated from the enum that emits it.
  `ErrorCode` owns `ALL` and `summary`, and a const assertion makes a variant
  missing from the catalog a compile error. The previous markdown table, and
  the seven-variant list that checked it against a nine-variant enum, are gone.
- The test guides for `vyre-driver` and `vyre-driver-wgpu` describe what the
  crates contain. Both previously pointed at a category contract file that does
  not exist, listed bench and fuzz targets for directories neither crate has,
  and gave `cargo test --test` invocations for target names (`adversarial`,
  `property`, `gap`, `integration`) that were never declared. `vyre-driver`'s
  guide also claimed sealing through a macro name that is not in the source and
  invariants for a `DialectRegistry` the crate does not define. Every invariant
  now names a construct that exists, and the sealing entry names
  `backend::private::Sealed`.
- Parity suites, backend materializers and the gate tooling each take their
  shared routine from one definition instead of a per-file copy.
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
- The GPU e-graph mirror is split into the refusals, the columnar snapshot, the
  device image, the row signature, the merge and the measured bridge, and its
  suite moved to an integration test.
- The async rule roster derives its covered set from the validation catalog and
  the suite sources at run time, which closed a gap where the empty-tag rule
  V128 had no test case in the tree.
- Device waits are bounded. Stream and event synchronization poll the CUDA
  driver with a spin window and capped sleep instead of blocking without a
  deadline, and a conformance compile, dispatch or session drop that outlives
  its step deadline returns an error naming the operation, the backend and the
  ceiling. VYRE_CUDA_DEVICE_WAIT_TIMEOUT_SECS sets the driver-level ceiling,
  which defaults to 300 seconds.
- Every intra-doc link in the workspace resolves, so `cargo doc` builds with
  `broken_intra_doc_links` denied. The regex DFA module pointed readers at
  `crate::pattern::RegionEvidencePipeline`, a type that exists nowhere in the
  tree; it now names `crate::pattern::regex_anchored_window`, the module that
  actually consumes candidate origins from a prefilter, and disambiguates
  `nfa_to_dfa()` from the module of the same name. A module header comment
  resolves its links in the scope of the parent that declares the module, so
  the telemetry and security family-mask headers name their items by full path
  instead of by bare name.
- Every `vyre-primitives` feature compiles alone. The operand-shape guards
  `matrix_cells` and `square_matrix_cells` lived behind the `math` feature
  while `graph` used them, and `math` already enables `graph`, so the missing
  edge could not be added without a feature cycle, and every selection naming a
  domain failed to build. The guards now live in
  `vyre_primitives::operand_shape`, compiled unconditionally because a shape
  check is not a domain.
- The release gate reports a requirement that reaches no semantic evidence
  check instead of passing it, and judges the documentation authority the docs
  evidence map requires.
- An artifact under `release/evidence` is written only when the tree it records
  can be identified. The recorder captures one `git:<commit>:dirty=false` or
  `git:<commit>:dirty=true:worktree=<digest>` fingerprint per run, stamps it at
  the head of the artifact, and refuses to write when git names no commit,
  cannot tell whether the tree is dirty, or cannot digest a dirty worktree.
  Those three states used to be recorded as `unknown` inside an otherwise
  well-formed fingerprint and left for a later reader to discover. Comparison
  reads the body under the stamp, because the tree an artifact was recorded
  from is not a divergence from the tree reading it, and regenerating an
  unchanged body keeps the fingerprint it already carried. The judge that names
  an imprecise fingerprint has one home in `xtask::source_provenance`, used by
  the recorder before it writes and by every reader of recorded provenance.
- `docs-check` and `feature-matrix` pass. The generated testing guide for
  `vyre-registry-link` had no row in the documentation manifest, and
  `vyre-test-support` declared features with no explicit default policy. A tree
  contract now derives the workspace member list at run time and fails when a
  member has no classified testing guide row, or when a row names a crate the
  workspace no longer has.
- The unbounded-read rule in `xtask::gates::hygiene_matrix` matches a call to
  `fs::read` rather than the text `fs::read(`, which also matched
  `BufferRefs::read(count_buffer)` and reported a graph accessor as an
  unbounded filesystem read. The release-tooling scan reads `.py` alongside
  `.sh` and `.yml`, so a rule that a shell script cannot evade cannot be evaded
  by moving the body into a Python file beside it.
- Every evidence subcommand prints its blockers to stderr before exiting 1,
  through the one owner of that epilogue,
  `xtask::output_arg::report_evidence_artifact`. It wrote the artifact and
  exited on a non-empty blocker list without naming a single entry, so nine
  gates reported a bare exit code and the cause was readable only by opening
  the JSON. `xtask::release::release_conformance` returned a count of failing
  backends instead of the failures; it now returns each one prefixed by its
  backend id. `xtask_registry::release::conformance_matrix` kept a private
  write-then-exit epilogue beside the owner and is routed through it.
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
- `release/evidence/metadata/metadata-matrix.json` and `feature-matrix.json`
  still listed `vyre-frontend-rust` after the workspace evicted it. An artifact
  naming a crate the workspace does not have shifts every row after it, so one
  deleted member reports as a whole tail of drift.
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
- The workflows named "Generate release evidence", "Generate measured
  evidence", "Generate release conformance evidence artifacts" and "Generate
  release benchmark evidence artifacts" pass `--write`. A gate only touches the
  tree under that flag, so every one of them judged the artifacts already
  committed and uploaded them unchanged, which is indistinguishable from
  generating them until the evidence goes stale. The measured-evidence job also
  stopped forcing `CARGO_BUILD_JOBS=1`; parallelism is declared in
  `.cargo/config.toml`.
- Command-line documentation now inventories and executes all 12 workspace
  binaries and 84 subcommands, publishes exact help, exit-code, environment,
  configuration, hardware, and failure contracts in crate READMEs, and gates
  drift in documentation CI. The vyre-wgpu demo is documented and exercised on
  the real GPU lane, while helper --help routes are side-effect free.
- Expression type inference has one owner,
  `vyre_foundation::validate::typecheck::expr_type`, and the environment its
  consumers supplied differently is the
  `vyre_foundation::validate::typecheck::TypeEnv` trait: a scalar lookup, a
  buffer element lookup, and a hook that observes the type of every
  subexpression a walk resolves.
  `vyre_foundation::optimizer::fact_cache::type_facts` and the reverse-mode
  forward pass in `vyre_foundation::transform::autodiff::grad` each carried a
  second `Expr` walker, and the three answers had drifted. `Expr::BufferRef`
  was a word in both copies where validation reports nothing on purpose, so a
  buffer name could pass an operand typecheck it must never pass; the
  validator's answer wins. Arithmetic in the fact cache was the left operand's
  type falling back to the right, so a mixed-width expression took the left
  operand's width; it now unifies, and an unknown or mismatched pair answers
  `u32`, which is what validation already assumes of the same expression, so an
  optimizer fact cannot contradict a validation decision. `BitAnd`, `BitOr`,
  `BitXor`, `AbsDiff`, `WrappingAdd`, `WrappingSub`, `MulHigh` and `Shuffle`
  were typed from their operands in the fact cache and are integer-typed by the
  owner; logical `And` and `Or` and the comparisons stay `bool`, as all three
  copies already had them. Autodiff typed `Expr::SubgroupShuffle` and
  `Expr::SubgroupReduce` as words, so the adjoint of an f32 moved between lanes
  was refused as a non-differentiable cast; both now report their value
  operand's type. Autodiff also inherits the iterative walk, so a deep forward
  expression can no longer overflow the stack while its locals are typed. The
  deliberate i32 `AbsDiff` rejection and the unknown result type of
  `Expr::Call` are unchanged. A contract reads the `Expr` enum's own source at
  run time and fails when a variant has no recorded answer, or when a second
  file in the crate defines an `expr_type` walker.
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
- The feature-msrv gate writes only the advertised toolchain version on stdout
  when invoked with --print-toolchain, returning cleanly without printing notes
  or finding counts.
- The shared gate fixture checkout states the corrective action when a
  temporary directory or git is unavailable, and the recorded backend
  feature-marker matrix matches what the tree produces.
- The frontier leaderboard reads a metric percentile through the same reader as
  every other artifact inspector. Its private copy accepted only an integer
  p50, so an artifact recording a float percentile was reported as missing the
  metric entirely.
- Materialized artifact execution preserves every mutable carrier across
  multi-segment dispatches and resolves fused-module resources by authenticated
  identity. Schema version 7 records canonical retained-predecessor lineage
  plus named entry input and output bindings. `InstanceCore` derives module
  resources from exact target binding metadata, updates antecedent retained
  values after dispatch, and rejects missing or mismatched resource identities
  instead of falling back to positional descriptor order.
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
- Every cargo-fuzz job now installs and selects nightly, matching the sanitizer
  flags the libFuzzer runner passes to rustc.
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
- A delegated gate ran the binary at the shared `target/debug/<package>` path,
  which every checkout building this workspace overwrites, so a sweep that took
  minutes could execute another checkout's gate code against this tree. The
  child now runs from a copy made when the build finished.
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
- Every workspace crate README now carries a manifest-backed contract for
  purpose, boundaries, a runnable example, features, errors, testing, release
  status, and ownership. Retired 0.4.x package claims and README drift fail the
  documentation gate.
- `docs/CLI.md`, `docs/CRATE_GRAPH.md`, `docs/OWNERSHIP.md` and the 35
  per-crate testing guides are present again. The book deletion removed them
  without retiring the generators that own them, so `crate_ownership.py`,
  `testing_guides.py` and `cli_docs.py` each failed `--check` against a tree
  their own `--write` reproduces, and `docs/DOCS.toml` listed two pages out of
  forty. `docs/optimization/OWNERSHIP.toml` also assigned two scopes that match
  nothing: a gitignored `BACKLOG.md`, which made the architecture-docs verdict
  depend on an untracked file, and a deleted `docs/catalog/**`.
- The generated documentation navigation regenerates over its own staleness.
  `docs-check` resolved every link before rendering and returned on the first
  finding, so deleting a page from the manifest left the stale summary linking
  a file that no longer exists, and `--write` reported that link instead of
  writing the summary that no longer carries it: the only way out was to
  hand-edit a generated document. Links are now resolved after the navigation
  is written, a manifest that does not hold still blocks the render, and a
  check run reports the drift and the dead links together instead of stopping
  at the first.
- Testing guides are now generated for all 36 workspace members from Cargo
  features and targets plus maintained hardware, evidence, skip, and failure
  metadata. The documentation gate rejects missing, orphaned, or stale guides.
- The `graph-dispatch` feature now enables the `encoding` domain that owns its
  GPU reduction-metric dispatchers. A production `vyre-driver-cuda` build no
  longer fails when motif existence and participation adapters import reduction
  metrics without the owning module.
- `vyre-primitives` feature `graph` enables `fixpoint`.
  `graph::persistent_bfs::program` reads
  `fixpoint::persistent_fixpoint::grid_sync_barrier`, the single owner of the
  grid-wide fence node, so the crate did not compile with `graph` alone.
  Nothing was red because every aggregate that exercises graph also turns
  `fixpoint` on.
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
- Megakernel admission contracts cover both whole-grid fence outcomes: a fence
  the planner cuts compiles into fence-free segments, and a fence that survives
  the cut is refused on a device that cannot launch a cooperative grid.
- The grid-sync split fixtures read a segment body through
  `visit::child_bodies` instead of a hand-written match with a catch-all arm.
  The walk applies each literal store a test backend stands in for, so a
  nesting form it does not descend into makes a store invisible and a split
  that dropped a write looks correct. The catch-all meant a statement variant
  that gains a body would have been skipped silently; the nesting is now stated
  once, in the crate that owns the IR.
- Grid-fence counting for wave-structure assertions descends through the
  workspace's single exhaustive owner of node nesting. Seven copies each
  re-derived the walk with a trailing catch-all arm, which classifies an
  unrecognised nesting variant as containing no fences and would let an
  under-fenced program pass its own structure test. The grid-fence builder, the
  launch-width reader, and the declared-flag-width reader also had three copies
  each.
- Grouped-query attention now composes the canonical max, normalization-sum,
  and weighted-write primitives with explicit KV-head bases. Overflowing row or
  element counts fail with a sharding error before buffer declarations are
  built.
- `xtask heuristic-audit` now resolves both standalone Vyre checkouts and the
  enclosing Santh workspace without duplicating the Vyre path.
- The hot-path roster names the three modules the columnar fact view split
  into, and the lowering boundary budget drops to the zero it measures now that
  its two format calls are gone.
- `hot-path-scan` counts runtime code only. It claimed to skip `#[cfg(test)]`,
  calling those "intentional dev-only lines, not runtime cost", but an
  attribute annotates the item that follows it and the scan skipped the
  attribute line alone. Every `panic!`, `format!` and `.to_string()` inside a
  `mod tests` body was reported as hot-path cost and weighted per kLOC, and the
  same bodies inflated the `count_code_lines` denominator, so a file's real
  density read lower than it is. Seventeen of 125 findings were test code, one
  `mod tests` spanning 575 lines of a backend dispatch file. The scan now
  tracks the item a `#[cfg(test)]` annotates to its closing brace, or to its
  semicolon when it has no body, with a brace counter that ignores braces
  inside string and character literals and a `'` that opens a lifetime rather
  than a literal. The `hot-path-scan` pin falls from 170 to 154 and
  `abstraction-gate` from 32 to 23.
- Call inlining expands a call in every operand position. The caller side and
  the callee side each enumerated `Node` themselves and had diverged: the
  caller copied the `AsyncLoad` and `AsyncStore` offset and size and the `Trap`
  address verbatim, so a call written there survived a pass whose contract is
  that no call reaches a backend. Both sides are now policies over one
  expansion walk, `transform::inline::expand_walk`, which descends through the
  structural node rewrite. A call inside a callee body was also expanded under
  the caller's substitution, which renames locals but does not bind callee
  parameters or retarget buffer-reference arguments, so `call
  outer(BufferRef(data))` with a nested call in `outer` produced a program
  reading a buffer only the callee declares. Nested-call arguments are expanded
  under the callee's policy.
- The neutral rejection record in `vyre-driver/src/materialize.rs` owns the
  corrective action for a foreign artifact binding and for a completion
  consumed twice. Both were routed through `invalid_module`, whose suffix tells
  the caller to recompile the target payload from the neutral artifact, which
  is wrong advice for either: a binding aimed at another instance's digest is
  repaired by binding against the right digest, and a completion consumed twice
  is repaired by consuming it once. `vyre-driver-cuda` and `vyre-driver-spirv`
  had each independently overridden the foreign-artifact string with the same
  replacement text, which is the evidence that the action is not
  backend-specific.
- The INT4 conformance requirement is a gate blocker instead of dead code.
  `conformance-matrix` carried a private routine that required every INT4
  quantization op to be registered with fixture inputs and expected outputs,
  and to be present in the op matrix catalog. Nothing called it, so no INT4 op
  was ever held to it. The requirement now runs inside the gate, computed from
  the same entries the evidence document reports, and a missing fixture or a
  missing catalog row is reported as a blocker. The live registry satisfies it,
  so the gate reports no INT4 blocker today.
- Megakernel selection prices a launch at the device's measured per-launch
  overhead and traffic at the recorded 3788 bytes per nanosecond, so a fusion
  is ranked against what a launch costs on the target host instead of against a
  rate a thousand times below the one the benchmark recorded.
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
- The `vyre-libs` operation-registry target declares the features its
  assertions need. It pins the numeric tolerance of thirteen registered
  compositions, eleven of which are only compiled under an `nn` or `math`
  feature, and it had no manifest entry, so `cargo test -p vyre-libs` built it
  with default features and it failed on a missing registration rather than on
  a wrong tolerance. It now requires `math`, `math-linalg`, and the four `nn`
  domains that own those ops.
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
- Loop-invariant hoisting moves a load out of a loop only when the header
  proves the body runs, so a read is never issued for an iteration space that
  may be empty.
- Loop fusion no longer skips a fusable pair after a refusal. When two adjacent
  loops could not be fused, the walk advanced its cursor by two instead of one,
  so the pair formed by the second loop and its successor was never considered;
  a comment claimed the scheduler retried the skipped pair, and nothing did.
  The pass now bails before cloning a body that holds no fusable pair at all,
  rather than deep-cloning the whole body and discarding it.
- Loop fusion rewrites the retired induction variable inside async copy offsets
  and sizes and inside trap addresses, so a fused body no longer reads a
  variable that no scope binds.
- The loop restructuring passes ask three questions before they reorder
  statements, and each now has one owner.
  `vyre_foundation::optimizer::passes::loops::var_reads`, `touched_buffers` and
  `bound_names` are public, and `vyre_foundation::visit::node_bound_name`
  answers which statement binds a name with no catch-all arm. The walks they
  replace named their own variants and ended in `_ => {}`, so a `Var` read in
  `Node::Trap.address` or in an async copy's `offset` reported ABSENT:
  `loop_fusion` fused two loops across a scalar one of them assigns, which
  silently changes the values the program computes, and
  `legality::bindings_flow_across` weakened the capture guard for both fusion
  and fission. The rematerialization pass asked the same question through a `_
  => false` arm and could inline a stale definition across a rebinding.
- Loop guard elision, range folding, software pipelining and unrolling rewrite
  every nested body the IR node declares, taking the slots from the shared
  owner instead of a per-pass list.
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
- `vyre-driver-metal`'s poisoned-resident-table test poisons the lock. It was
  named for the poison path and documented the sentinel, then contained a
  comment-only block whose note explained that `MetalBackend` does not expose
  `resident_buffers`, and asserted only that a healthy snapshot carries both
  resident keys with a count of zero. It could not fail on the bug it existed
  for. It now allocates a resident buffer, poisons the table from a thread that
  panics while holding the lock, and asserts the count and byte total are both
  `u64::MAX`, that `metal_resident_buffer_error` is 1, and that the unrelated
  counters survive. The healthy test asserts the error key is absent, so the
  pair separates poison from emptiness instead of proving one state twice.
  `resident_buffers` is `pub(crate)` because poisoning the lock is the only way
  to observe the contract.
- Six intra-doc links were unresolved or pointed at a private item, and each
  was reported only by the public-API gate's rustdoc pass rather than by a
  build. A module carrying both an outer doc comment on its `mod` declaration
  and an inner `//!` header has the two merged and resolved in the PARENT
  scope, which is why `vyre-driver`'s `TargetDialect` link failed from inside
  the module that defines it; that one is now fully qualified. The five links
  naming a private item are plain code spans, because a link to an item a
  reader cannot reach is not a link.
- Weighted paged-corpus scans now expose per-device timing and byte balance.
  The physical two-adapter benchmark verifies exact single-device parity and
  records paired end-to-end speedup, topology, staging overhead, and raw
  samples.
- The pre-emission scan that decides which buffers Naga emits as `atomic<...>`
  and which keep `BufferAccess::ReadWrite` takes descent, operand positions and
  per-node buffer direction from the exhaustive owners in
  `vyre_foundation::visit`, and returns both sets together. The write half was
  a hand-rolled recursive descent ending in `_ => {}`, so the four collective
  variants were reported as writing nothing: a buffer written only by an
  `AllReduce`, `AllGather`, `ReduceScatter` or `Broadcast` was auto-downgraded
  to `ReadOnly` and emitted as `var<storage, read>`. The atomic half restated
  `node_operands` as fifteen `NodeVisitor` method bodies. `node_buffer_refs`
  disagreed with the old scan about `Node::IndirectDispatch`, which names its
  count buffer as a read because the host writes it and the shader only reads
  it; the scan now agrees, and no emitted shader changes because the Naga
  emitter rejects `IndirectDispatch` before producing WGSL.
- A region that names the operation behind it counts as composed. Making every
  library operation tag its own entry region added one node that the
  composition measurement read as own work, so the measured fraction of every
  operation that took that fix went down and two ops reported a composition
  regression they had not suffered. The trend baseline is unchanged; the
  measurement is.
- The substrate-neutrality rule can see the dependency form this tree uses.
  `scripts/check_architectural_invariants.sh` matched `^name =` against
  manifest text, so a dependency written `name.workspace = true` never matched,
  and the forbidden-edge half of the rule was unreachable for its whole life.
  The `neutral-crates` gate reads the parsed manifest, so both spellings are
  one entry, and it reports the edge the shell form could not: `vyre` depends
  on `vyre-runtime` in `[dependencies]`. Retired crate names are checked over
  every tracked manifest instead of through a ripgrep whose nonzero exit was
  read as no hits, and nothing is written outside the repository. The script is
  deleted, architectural-invariants.yml names `neutral-crates` and `layering`,
  and the workflow no longer sets `CARGO_BUILD_JOBS`, which is build
  configuration and belongs in `.cargo/config.toml`.
- No gate sets a build-affecting variable on a command line. The shared cargo
  runner, the release shard prover, the Metal MacBook gate and both the
  conformance and release-evidence workflows exported CARGO_BUILD_JOBS, and
  three of them read or exported CARGO_TARGET_DIR, so each one built a
  different build than a bare cargo invocation in the same checkout and none of
  them shared a compiled artifact with it. Job count and build directory are
  declared once, in .cargo/config.toml.
- Adding an IR `Node` variant can no longer be handled by a catch-all arm
  nobody chose. The AST registry macro emits `NODE_VARIANT_NAMES` and
  `node_variant_name` from the declaration site,
  `vyre_foundation::visit::node_shape` records for every variant whether it
  nests statements, carries operand expressions, or holds an opaque payload,
  and `child_bodies` is the one exhaustive owner of child enumeration. Two
  traversals that re-derived that list were wrong: the reference interpreter's
  barrier scan claimed an exhaustive match but let `Node::Region` fall into its
  default, so a barrier inside a region body read as absent; and `walk_exprs`
  skipped the `offset` and `size` operands of asynchronous copies and the
  `address` operand of a trap, hiding those buffer references from every
  analysis built on it. Loop unrolling's local-declaration check now also
  treats a region body as scope-transparent. Tail duplication's read check now
  answers yes for a statement form it does not recognise instead of no, so an
  unfamiliar tail costs a missed duplication rather than code sunk past a live
  read. The duplicate visitor implementations in `vyre_foundation::visit` now
  delegate to that owner rather than restating the variant list, and the
  descendant scan uses an explicit worklist so a deep tree cannot overflow the
  native stack.
- One resolver answers which cargo a gate starts, so a build a gate spawns
  cannot compile a different workspace than the one being judged: the exported
  wrapper wins, then a wrapper beside the workspace root, then the toolchain
  that started the process, then the bare name.
- No crate publishes an item at more than one path. The atomic and bit-count
  compositions in vyre-libs published each op both flat and through a generated
  module, vyre-primitives carried a prelude and a facade module for another
  crate's wire envelope, vyre-lower kept two import-boundary modules that only
  re-exported the fact modules beside them, the wgpu pipeline re-exported the
  bind-group statistics and the persistent dispatch item, the driver root
  re-exported one validator out of the validation module that owns it, and the
  reference crate re-exported its op counter at the root. Every second path is
  deleted and every caller names the owner.
- The crate ownership registry, the crate README contracts and the 34 per-crate
  testing guides are held by registered xtask gates instead of three Python
  generators under scripts/. crate-ownership, crate-readmes and testing-guides
  each render the same documents byte for byte, report every violation instead
  of raising on the first, and carry a pinned baseline; check-tier-deps no
  longer shells into python3 to validate the registry, so one owner answers for
  the contract.
- Loop-invariant hoisting has one owner, so the resident pipeline no longer
  hoists a binding whose name a sibling loop also binds and produces a program
  the validator rejects, and a load from a read-only buffer now leaves the loop
  under the pass engine as well.
- The f32 parity canonicalizer has one owner.
  `vyre_foundation::fp_parity::canonical_f32` states the rule once: a NaN
  becomes the canonical quiet NaN and a subnormal becomes a zero of the same
  sign. Three production copies of that body existed, in the foundation scalar
  operations, the foundation IR evaluator, and the reference float operations,
  so a change to the parity contract had to be made in three places to take
  effect. The wire encoder keeps its own canonicalizer because it enforces a
  different contract: it also collapses negative zero and flushes subnormals to
  positive zero, which loses sign. The two test-side restatements also stay,
  each now saying why: a judge that calls the code it judges proves nothing.
- The Gate 1 budget is measured once, by `gate1`, over
  `xtask-registry::gates::composition_budget`. Two gates walked the region tree
  with their own copy of the count and disagreed about what composition is: one
  credited every node inside a region carrying a `source_region`, which a phase
  wrapper around inlined code also carries, so
  `vyre-primitives::graph::dominator_tree` read as 91.1 percent composed there
  and as 0.0 percent in the other walk. The reading that could not fail was the
  one wired to the pin, so nine operations over the loop and node budget stood
  green. A node now counts as composed when it is a call to another registered
  operation or sits inside one, `abstraction-gate` keeps the boundary questions
  and reports the budget no longer, and the shared walk takes its child bodies
  from `vyre_foundation::visit::child_bodies` rather than a hand-written
  variant list that counted a new nesting variant as a leaf.
- Three megakernel suites build the producer and consumer graph from one shared
  fixture instead of three copies of the builder, which removed 114 duplicated
  lines from the crate.
- The random-IR corpus and the wire property suite draw operators from one
  strategy owner, so a builtin table and its opaque arm weighting cannot differ
  between them.
- The release gate judges the evidence census through the module that writes
  it, so the schema, the required generators and the artifact rules are stated
  once. The second reader still asked for a per-command spawn status and two
  counters retired in schema 5, which reported every generator as failed and
  printed an absent counter as the u64 maximum.
- The VIR0 wire round trip is held to one oracle: every suite now asserts
  non-empty bytes, an equal decode, and byte identity across a re-encode,
  instead of three partial readings of the same contract.
- Every gate, doc generator and evidence run resolves cargo through
  `xtask::cargo_runner::runner`, which reads `VYRE_CARGO_RUNNER`, then a
  `cargo_full` beside the workspace root, then that name on `PATH`. Nine call
  sites plus the delegated-gate builder resolved it three different ways; two
  of them read `CARGO`, which cargo sets to the plain binary and never to the
  wrapper, so a child build used a different target directory and job count
  than the build that started it.
- The workspace state the crate ownership contract is judged against rejects a
  member listed twice and two members declaring one package name, instead of
  letting the surviving row decide every answer.
- vyre-macros keeps its unit tests in one placement beside the parser they
  cover, and each surviving inline module states why no integration test can
  reach it.
- Every assertion the repository shell scripts made is a registered gate. The
  CUDA and SPIR-V parity budgets, the feature-and-MSRV axis, the oracle and
  volume sweeps, the Metal counter roster, the wire determinism diff, the crate
  ownership registry, the crate READMEs and the per-crate testing guides are
  subcommands with a pinned baseline, a subset and a workflow step; the fifteen
  scripts that carried them are deleted. A workflow step may no longer name a
  script by glob, which is how a deleted assertion used to stay in a workflow
  that still read as coverage.
- The crate ownership registry stores every declared directory slash-separated,
  so a row written with backslashes owns its own directory and not only the
  files beneath it.
- A binary-operator rewrite rule is a table row over one owned shape instead of
  twelve lines written out per rule, and the CSE rule body imports the literal
  body it builds on.
- Tiled reductions share one program skeleton. reduce_mean, rms_norm,
  layer_norm and softmax each hand-built the same reduce-then-publish shape
  (bind the lane, accumulate with a stride, reduce through workgroup scratch,
  publish from lane zero of workgroup zero, stream the normalized output back),
  so a change to the barrier placement or the publish guard had four places to
  be made. The skeleton lives in the unconditional builder module, above the
  separately gated math and nn dialects that reach for it, and the emitted IR
  is unchanged. A publish that is the last thing a program does no longer emits
  the barrier that fenced it, because nothing reads the published scalars.
- Every instruction that names the build wrapper names it as ./cargo_full. A
  bare cargo_full is not on the search path, so a fix line quoting it told the
  reader to run a command that does not resolve. The generators that embed
  those lines emit the same spelling, so the artifact and its generator agree.
- The vyre-release-gate now collapses every stale-source benchmark verdict into
  one finding that states how many verdicts over how many artifacts stand
  behind it and names the command that re-measures each backend, with the
  verdicts kept as notes, because one commit invalidates every recorded
  benchmark at once and each invalidated artifact then failed several checks.
- Acquiring the wgpu backend from two threads at once no longer kills the
  process. Every adapter query and device request built its own wgpu instance,
  and two overlapping instance constructions raced inside the Vulkan loader:
  one thread negotiating an ICD in vkCreateInstance left the loader dispatch
  table half written, and the other called through a null function pointer from
  vkEnumerateInstanceExtensionProperties. Instance construction is now
  serialized process-wide, and the instance stays per acquisition because the
  GLES backend inside it owns a thread-current EGL context.
- Op matrix rows no longer carry a bench_targets field that no producer fills;
  the benchmark roster owns the mapping from a target to the case it measures.
- OP_MATRIX owner paths point at a directory that exists for the matching
  domain. The generator derives a vyre-libs owner from the operation id as
  `vyre-libs/src/<domain>`, with named exceptions for optim, quant and builder.
  Every `vyre-libs::matching` operation lives under `vyre-libs/src/scan`, so
  four families named `vyre-libs/src/matching`, which is not a directory, and
  `op_matrix_covers_every_registered_op_once` failed on the missing path. The
  domain is now mapped the way optim and quant already are.
- The op matrix carries only rows that resolve to a live operation
  registration, the integer strength reduction rewrites are named by the
  optimizer pass catalog that owns them, and three rewrites that had no catalog
  id now have one.
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
- Operation documentation now has one generated JSON authority covering every
  linked library, primitive, intrinsic, and runtime dialect operation.
  Schema-derived inventories and subsystem catalogs expose exact tiers,
  categories, program or dialect signatures, Cargo feature routes, oracles,
  backend support evidence, algebraic laws, composition chains, and counts.
- The pass-family benchmark manifest records a family as covered only when the
  case proving it passed. It used to push each specification's declared
  families whether or not the artifact behind them could be read, and the
  specification list is the same list as `required_pass_families`, so
  `uncovered_pass_families` was structurally always empty and the release gate
  demanding it be empty could not fail on the one thing it exists to catch: a
  missing or blocked artifact. A test asserts that state, so the unconditional
  form turns the suite red.
- Optimizer type facts now bind and restore loop induction variables, discard
  stale assignment types, and traverse tile elementwise bodies.
  Constant-division duplication budgets count every dividend copy. Batched
  dispatch output reservations reserve requested increments from the current
  length. Portable `vyre-conform --no-default-features` builds no longer link
  default GPU drivers.
- Inlining reaches a call inside a subgroup operand.
  `vyre_foundation::transform::inline` enumerated `Expr` itself and classified
  `SubgroupBallot`, `SubgroupShuffle` and `SubgroupReduce` as carrying nothing,
  so a call in one of those operands was handed back verbatim: the program kept
  an `Expr::Call` that inlining exists to refuse, and where unresolved calls
  are kept deliberately, that call's own arguments were never inlined either.
  Operand positions now come from the one owner. Because that walk is
  bottom-up, a call site is reached with its arguments already inlined, so the
  argument loop that walked every argument a second time is gone.
- The buffer-interference proof the memory passes need before they may rewrite
  across a gap has one owner,
  `vyre_foundation::optimizer::passes::memory::alias`. `dead_store_elim` and
  `store_to_load_forward` each carried a full node-by-node copy of it, both
  exhaustive and both tested, and they disagreed: a compare-exchange against
  another buffer, which is how a lock is taken, blocked the dead-store proof
  and did not block the forwarding proof, so a load across a lock acquire was
  replaced with the value stored before it. Both copies also inspected only the
  one buffer `Node::Trap` and `Node::IndirectDispatch` name, while both pass
  module docs promised the node blocks outright; a host effect handler and a
  launched grid may touch any buffer, and a grid-synchronizing collective is at
  least a fence. The owner answers all of that once, and what stays per-pass is
  the one bit that genuinely differs: whether a write to the buffer interferes
  or only a read. Node descent and buffer naming come from
  `vyre_foundation::visit`, so a new IR variant fails to compile here rather
  than defaulting to harmless.
- `vyre_foundation::transform::rewrite_walk::rewrite_node` is the only
  rewriting enumeration of `Node`, which it already claimed to be.
  `vyre_foundation::optimizer::rewrite` carried a second exhaustive match over
  every variant, and the pair had diverged: the owner descended into an async
  copy's `offset` and `size` and the copy did not, so every pass routed through
  the optimizer's whole-program expression rewrite left those two expression
  positions alone. The by-value copies in
  `vyre_foundation::optimizer::passes::algebraic::const_fold::reaching_def_propagate`
  and `vyre_foundation::optimizer::passes::loops::loop_lower_bound_normalize`
  are gone as well; each ended in a catch-all arm, so a `Node` variant added
  later would have been walked as a leaf and its children never substituted,
  which for a substitution is a stale variable reference rather than a missed
  optimization. Lower-bound normalization now calls
  `vyre_foundation::transform::subst`, which already owned that rewrite and
  additionally refuses to substitute into a nested loop that rebinds the name.
- Strength reduction folds a chained shift whose counts reach the register
  width to zero again. It had been changed to leave both shifts standing, on
  the reasoning that a right shift on a signed operand replicates the sign bit,
  but V094 rejects a shift whose operands are not `u32`, so that operand cannot
  reach the pass. The fused shift is not an alternative: the target text masks
  a shift count with `& 31`, so emitting `x << 32` would emit `x`.
- Package readiness now validates unpublished, version-matched release
  dependencies through local registry patches after Cargo normalizes path
  dependencies for packaging. Cross-repository `weirflow` archive evidence now
  records its real files, examples, Rust sources, and file-list digest.
- The encoded pattern-match pass is split into the action encoding, the
  analysis programs, the two rule bodies, the decoder and the driver, and it
  reads the shared arena workgroup size instead of declaring a second copy of
  it.
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
- The persistent-fixpoint routing contract is asserted in one place,
  `vyre_libs::fixpoint::routing_contract`, over a description of what an op
  builds on either side of one workgroup width. Each routed convergence op
  previously carried its own copy of the same four obligations, and the copies
  had drifted in what they accepted: both pinned the grid fence count to a
  literal 8 without stating that the count is two fences per wave, so neither
  would have caught a build whose wave count changed. The contract now checks
  all four at four iteration budgets, and both ops register every way their
  dispatch outgrows one workgroup rather than only the widest state buffer:
  `vyre_libs::math::bellman_shortest_path` registers node-widened and
  edge-widened spans, and `vyre_libs::math::sinkhorn_iterate` registers
  scaling-widened and kernel-widened spans. The span is the widest declared
  buffer once a program carries atomics, so four nodes with 257 edges, and a
  17-by-17 kernel over two 17-element scaling vectors, both cross the threshold
  while the ping-pong state still fits one workgroup.
- The optimizer compiles for the adapter it was given. `Autotune` and
  `DecodeScanFuse` both read device facts, and both hardcoded
  `AdapterCaps::conservative()` in their `ProgramPass::transform`, so every
  program that went through the standard pipeline was tuned for a device with
  no optional features whatever the real adapter reported; the caps-aware
  `transform_for_adapter` existed the whole time and only a caller who already
  knew to ask could reach it. Nothing failed, which is why it lasted: the
  pipeline produced a valid program and a slower one. `ProgramPass` now
  declares `transform_for_adapter`, whose default discards the adapter because
  an IR-only rewrite is the same program on every device, and
  `#[vyre_pass(adapter_dependent = true)]` is how a pass says its output moves
  with the device. `PassScheduler` carries the adapter it was built for,
  `PassScheduler::for_adapter` and `optimize_for_adapter` are the entries a
  backend uses once it has probed one, and a scheduler built without an adapter
  states the conservative fallback at the place that chose it instead of
  leaving it to whichever pass reached for a profile first.
- Single-program artifact graphs now classify every backend-allocated buffer,
  including read-write pipeline live-outs, as an output. Artifact submission no
  longer asks callers to provide internal fused-pipeline storage as a host
  input.
- The `types` feature of `vyre-primitives` now depends on `vyre-foundation`,
  which its shape-predicate evaluator has always aliased. Enabling only that
  feature against the published crate failed to compile; in-workspace builds
  hid it because another member always enabled a feature that pulled the
  foundation in.
- The columnar program fact view is split into the tags, the fact table and the
  build walk, and its suite moved to an integration test that reads only the
  public surface.
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
- The library composition provenance gate no longer pins a hand-measured
  operation count. The population is the registry the build linked, which
  changes with the enabled dialect features, so the pinned floor of 100 was red
  under the default feature set that registers 97. It now requires that the
  registry linked at all, that no operation reached the check without a program
  builder, and that no exemption row names an operation which already stamps
  its own id or an id no source registers.
- The public-API snapshot gate reports a crate it could not read instead of
  skipping it. Two paths dropped a package out of the comparison without a
  word: a publishable package whose `src` directory was missing, and an
  extraction that succeeded but returned nothing. No committed snapshot is
  empty, so an empty surface is a truncated rustdoc or a zero-byte `.rmeta`
  from a parallel build sharing one target directory, which is exactly the
  state in which a crate most needs to be checked. Both are findings now, and
  an extraction that exits nonzero prints what `cargo public-api` said rather
  than a bare sentence naming the crate.
- The public-API snapshot fixture no longer forces `CARGO_BUILD_JOBS=1` on the
  refresh it runs. Parallelism is declared in `.cargo/config.toml`, and an
  environment variable overrides it, so the fixture rebuilt one codegen job at
  a time regardless of the host.
- Public API checks now discover every committed crate snapshot, parse exact
  package names, use dependency-noise-free output, and reject ordinary snapshot
  updates that remove or change an existing item.
- The public-API snapshot is taken with every feature enabled. It read the
  default feature set only, so every feature-gated public module was outside
  the file that claims to pin the public API: the whole `graph` surface of
  vyre-primitives, including an exported macro, was unmeasured, and a change
  that deleted three public items there left the gate green. The check is now
  the registered gate `public-api-snapshot`, which derives the publishable
  roster from the manifests, reports a snapshot naming a package that no longer
  publishes, refuses an extraction with no items instead of skipping the crate,
  and prints the added and removed items of every snapshot it writes so an
  unintended bless of somebody else's in-flight surface is visible.
- The committed public-API snapshots for `vyre-driver`, `vyre-driver-spirv`,
  `vyre-libs` and `vyre-pass-engine` record the surface those crates publish
  now. The identifier type moved to
  `vyre_foundation::ir::model::expr::ident::Ident` when that file was split,
  the driver publishes `ErrorCode::summary`, `ErrorCode::ALL`, the
  `error_catalog` module and `migration::DEPRECATED_OP_CODE`, the SPIR-V
  registration no longer answers seven capability questions one method at a
  time, and the tiled matmul builders are published by `math::linalg`, which
  owns them. A stale snapshot fails the drift check for every crate at once,
  which hides the next real change behind noise.
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
- Public API snapshots now cover every workspace package whose Cargo manifest
  permits publication, including CUDA and every emitter/runtime library. The
  manifest-derived gate rejects both missing snapshots and stale snapshots for
  packages that no longer publish.
- Empty QK-gain tensor shapes now declare a zero-byte output range instead of
  an unknown-size backend allocation, while overflowing positive shapes fail
  closed with an actionable trap program instead of wrapping their element
  count.
- The registered witness programs for
  `vyre-primitives::math::quantized::i4x8_matvec_f32_scaled` and
  `i4x8_batched_matvec_f32_scaled` bound their output buffer to a slot named
  `vector_scale` and sized it for one f32, so every lane above the first wrote
  out of bounds on the declared fixture. The registry safety rules read that as
  an out-of-bounds access on valid input, an out-of-bounds access under grid
  over-fire, and a cross-lane write-write race, three failures for one wrong
  buffer. Both now name the output `out` and declare only the three input
  buffers, matching the other packed-INT4 registrations.
- The reduction benchmark now measures atomic-scalar and workgroup-tree sums on
  the same GPU at 32 and 1,048,576 elements. It verifies both routes exactly,
  selects the measured winner per size, and records contention and barrier
  counters. NVIDIA idle clocks no longer invalidate a cold, low-utilization
  microbenchmark as thermal instability.
- The `vyre_reference::workgroup::Invocation::bind` and `bind_loop_var`
  documentation examples name `vyre_reference::ReferenceError` instead of
  `crate::ReferenceError`. A doctest compiles as its own crate, so the old path
  resolved to nothing and both examples failed to build.
- Regex DFA replay now gives open-ended repetitions an explicit finite policy
  instead of treating their minimum as a maximum. Whole-buffer variable-length
  matches derive exact starts from candidate origins, and region evidence
  returns one longest extent per pattern and origin.
- Both validator walks agree that `Node::Region` scopes its body. The legacy
  multi-walk arm recorded no scope log for a region body, so a `let` inside a
  region was never undone and stayed live past the region, past an enclosing
  `If` branch, and on through the rest of the program, which let it shadow or
  collide with a later binding of the same name without a diagnostic. The pass
  that flattens a small region into its parent sequence re-wraps it in a
  `Node::Block` on a name collision, and that is sound only while a region
  body's bindings die at the region boundary.
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
- The crate that exists to link every operation registration linked vyre-libs
  on default features, so 78 of the 327 registered operations never reached the
  registry it publishes: the geometry, optimization, topology, logical,
  succinct, algebra, attention and quantization registrations were absent from
  every walk that read it. Its per-source floor could not see this, because a
  source linked with a narrow feature selection still contributes more than
  nothing and every count shrank together. vyre-registry-link now names
  vyre-libs feature `full`, and a new rule reads the operation ids out of the
  generated catalog at run time and fails when any of them is missing from the
  live registry.
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
- The release macro benchmarks no longer time a CPU baseline that rebuilds its
  own input. `synthetic_cpu_count` regenerated every record from its index
  inside the timed region, twelve to twenty-four rotate-multiply rounds per
  column, while the GPU side was handed pre-materialized, pre-uploaded buffers.
  The recorded evidence read 5508x to 6729x for the count patterns and 928x for
  the one case that already read materialized bitmaps, and the gap was the
  generator, not the device. `synthetic_cpu_count_over_inputs` now counts over
  the same host buffers the device reads and the generator cross-check happens
  outside the clock. Re-measured on the shipping path,
  release.condition_eval.1m reports 165.708x with a 4422427 ns baseline p50,
  and no speedup pin had to move.
- Four release surfaces named documents the book deletion removed, and each
  failed open. `release_contract_path` pointed at `docs/RELEASE.md` and now
  names `release-train.toml`, the surviving authority for versions, tags,
  package membership and the approval-gated actions. `hygiene-matrix` listed
  `docs/RELEASE.md` three times and skipped every entry that is not a file, so
  it reported clean while scanning none of the documents it names; a listed
  document that is absent is now a finding. The version matrix resolved release
  notes to `docs/release/v<version>.md` and repeated one path twice, so its
  tag-command scan spent its blockers on unreadable files and double-counted
  the one file it could read. `scripts/final-launch.sh` passed that same
  missing page to `gh release create --notes-file`, so the last outward step of
  a release would have failed after the crates were published;
  `scripts/release_docs.py` now generates
  `release/evidence/docs/release-notes-body.md` from the fragments the
  changelog is built from and `--check` holds the two together.
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
- `vyre-release-gate` implements the `Gate` contract. The registry assigned it
  to `xtask-evidence` and no implementation existed, so that crate's own table
  check failed. Options now parse from the runner's argument offset instead of
  argv position 2, the default manifest resolves under the context root instead
  of through a second checkout probe, and the two exit paths become one
  `Report` whose findings are the blockers and whose note carries the
  requirement count and the scope. `--help` left the option parser because
  `help()` answers it, and `GateOptions` derives `Debug`, without which the
  crate's lib test did not compile.
- The release gate no longer replays an aggregate's recorded freshness verdicts
  as live findings. Thirty-seven of the thirty-eight blockers stored in
  `cuda-release-suite.json`, `bench-release-axes.json` and
  `cpu-only-100x-proof.json` embed a hash labelled "current workspace source"
  that was current when the aggregate was written, so the gate printed a value
  no reader can resolve and that disagrees with the one the same run computes.
  Freshness is recomputed against the tree the gate is running on; every other
  recorded verdict is replayed unchanged.
- The release hygiene scan named `release_gate`, a module deleted when the
  composite became the `prepublish` subset, so one entry in its list scanned
  nothing while reading as coverage. It reads `lockfile`, the gate that
  inherited the lockfile step.
- The `ci-required` gate resolves every workflow file name
  `.github/CI_REQUIRED.md` quotes, not only the ones under a blocking heading.
  The contexts under a heading were resolved against real workflows and a file
  name written as prose was read by nothing, so the deep-gate rows went on
  naming two lanes after both had been moved to `.github/workflows-paused`. A
  quoted name that is in neither workflow directory is now a finding, and a
  paused workflow named as a blocking section is a second one, because a paused
  workflow cannot report a context branch protection waits for.
- Resident throughput batches preserve complete device-timestamp totals and
  normalize them per logical item. String bitmap scatter uses subgroup ballots
  to materialize 16 independent output rows in one resident dispatch, with
  exact CPU-oracle parity.
- Ring occupancy sums report an overflowing slot count instead of saturating to
  a plausible total, so a launch recommendation over an impossible decoded ring
  now fails with that reason instead of running on ratios derived from a
  saturated total, and autotune cost selection carries the empty candidate set
  in its return type instead of a debug assertion absent from release builds.
- The root README now derives every workspace crate's publication and support
  status from manifests and maintained metadata. Operation tier counts come
  from the canonical operation schema, backend claims come from executable
  backend evidence, and the architecture identifies Metal as Apple-active
  instead of planned.
- The reviewed workspace roster no longer lists `vyre-frontend-rust`, which
  left the workspace when the Rust frontend became its own product. The
  frontend owner table keeps its rust row on purpose: the owner ships outside
  this workspace, so no member matches it and any workspace crate that grows
  rust frontend stages is a second frontend.
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
- The scalar storage-graph matrix declares every operation the scalar oracle
  defines. It declared 40 and the oracle defines 76: the four bit-unpack ops at
  both 32-bit integer widths, the whole bitwise set at `i32`, and the
  transcendental, rounding and classification set at `f32` were all undeclared,
  so the matrix asserted that the oracle refuses operations it evaluates. The
  row that reports an undeclared operation now reports every one of them in a
  single run instead of the first, which is what turned a 36-item finding into
  one measurement rather than 36 rebuilds.
- `vyre_foundation::scalar_ops` is the single owner of scalar operator
  semantics. The literal folder in `vyre_foundation::ir_eval` and the
  storage-graph interpreter in `vyre_foundation::ir_inner::model::node_kind`
  each carried a full per-width operator table, and the two tables disagreed on
  more than thirty (operator, width) pairs: the folder retyped integer
  transcendental, rounding, classification, `Abs` and `Sign` expressions to
  f32, folded i32 `AbsDiff`, `And`, `Or`, `RotateLeft` and `RotateRight`, f32
  `Mod`, and bool `BitXor`, all of which `vyre_foundation::validate` rejects,
  while the interpreter had no answer for u32 `Negate`, the unpack operators,
  i32 `BitNot`, `Popcount`, `Clz`, `Ctz` and `ReverseBits`, the f32 unary math
  set, or f32 comparisons. The validator decides every row. f32 `Div` and i32
  `Shl`/`Shr` are now total, matching IEEE-754 division and the
  shift-count-modulo-width rule the backends lower to, instead of bailing on
  NaN, zero or a negative count.
- The scan conformance matrix is judged against the regex compiler instead of
  against itself. `vyre-libs/tests/scan_conformance_matrix.rs` held three
  constant lists and asserted them against each other: it ran no engine, and
  its `expected_output_hex` values decoded to labels like
  `leftmost:pattern0:0:1` that no engine emits. Seven rows of
  `SCAN_CONFORMANCE_MATRIX.toml` and five `scan_construct` proof gates cited it
  as evidence, and the release gate accepted the citation because it only
  checked that the file exists. The suite is gone, the invented output bytes
  are gone, and the release conformance gate now reads the construct-to-code
  mapping the regex compiler owns and refuses any row naming a code the
  compiler never emits, which caught `VYRE_SCAN_UNSUPPORTED_CAPTURE_GROUPS`:
  the real code is `VYRE_SCAN_CAPTURE_EXTRACTION_REQUIRES_VERIFIER`. A cited
  path must also name the matrix, so a citation cannot point at source that
  never reads the row. `RegexConstruct::ALL` is the enumerable construct list,
  closed by an exhaustive match in its own crate, so a construct added later
  reaches the gate instead of hiding behind `#[non_exhaustive]`. The engine
  support map is now marked for what it is: a declaration, since nothing in
  this workspace can run hyperscan, vectorscan, or a Metal device.
- The pass scheduler judged a rewrite against facts taken from the wrong
  program. A pass that reported no change while rewriting the program had its
  effect, linearity, shape, and cost certificates recorded against the program
  it returned rather than the program it was handed, so the next pass was
  measured against a certificate describing its own input and a cost regression
  was accepted as flat. Facts are now derived once per program, keyed on the
  program fingerprint, and reused across every gate and every fixpoint
  iteration, which also removes one deep clone of the program per running pass.
- The self-exclusive region scan descends through `visit::child_bodies` instead
  of its own exhaustive `match node`. A node variant that carries a body would
  have had to be added to both lists, and the scan's copy is the one a reader
  would not think to check when adding one.
- Restored public Target export in vyre-driver, scan database wire header and
  budget types in vyre-foundation, lower and WORKGROUP_SLOT_BASE in vyre-lower,
  and public submodule paths in vyre-spec to maintain SemVer compatibility with
  published releases.
- Megakernel selection charges a workgroup tile once per fusion group instead
  of once per member, so a group whose members share a tile by name is no
  longer pushed over the device scratch budget and ranked below the pair it
  beats.
- Chained shift fusion has one owner and one answer. Constant folding and
  strength reduction each carried the rule, and they disagreed: folding
  declined a pair whose counts reach the word width and left the double shift
  for the backend, reduction folded it to zero. Both were also wrong about a
  count above 31, which the target text masks with `& 31`, so `(x << 40) << 1`
  shifts by nine and not off the end. `algebra::shift_fusion::reduce_shift`
  masks each count, folds to zero once the sum reaches the width, and fuses
  otherwise, checked against an evaluation of the chain it replaces for every
  count pair up to 40.
- The WGPU stream-sharding error is now nameable as
  `engine::multi_gpu::StreamShardError` without changing existing signatures.
- The `vyre-primitives::hardware::subgroup_add` intrinsic performed no subgroup
  operation. It summed thirty-two memory neighbours in a serial loop, so every
  lane re-read and re-added its whole subgroup out of storage, while
  registering the hardware semantic for a subgroup add and documenting itself
  as mapping to one. It now builds `Expr::subgroup_add`, the reduction the IR
  already carries and all three emitters already lower, with the lane value in
  a guarded local and the collective in uniform control flow so the
  participating-lane set is defined. Values are unchanged: the reference oracle
  and its three boundary cases are untouched and pass, including two subgroups
  of thirty-two.
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
- The panic budget reads a module gated behind cfg(test) as test code, derived
  from the declarations the tree writes, so a fixture module no shipped build
  compiles no longer counts against its crate.
- A publishable crate's src tree is held to product. The
  test-material-placement gate reads every source file whose name says fixture,
  oracle, mock, stub, sample, golden or harness, and reports the ones a default
  build compiles that only test code refers to, or that nothing refers to at
  all. A module behind a test-only cfg, a module product code calls, and a
  module behind a feature that is off by default all pass, so no frozen public
  path has to be renamed. No dependency outside dev-dependencies may name the
  test support crate.
- eigenvector_column_sign declared the matrix it rewrites in place as an output
  buffer. An output buffer is not a witness input, so the caller matrix never
  reached the program: the operation read zeros, wrote zeros, and its recorded
  expected outputs were unreachable. The buffer is read-write, which is what
  the body does.
- The consumer-neutrality gate no longer carries a roll call of seventeen
  documents, fifteen of which the documentation collapse deleted, each reported
  as a finding on every tree. The scanned set is enumerated from the tree,
  which already covered the two survivors and covers every document added
  since, so the gate reaches zero by describing what is there rather than by
  mourning what is not.
- The delegating-form closure check reads the source of the crate that owns it
  instead of a crate named in a literal, so the two prefix-scan facades in
  vyre-libs solvers are compared against the builders they forward to. Both
  pass the op id and both buffer names through unchanged, across the empty,
  single-workgroup and multi-block regimes.
- The scaffold under `examples/libs-template` renders into a crate that builds.
  It pinned `thiserror` to an exact version the workspace had moved past, so
  the rendered crate could not resolve, and it passed the `Arc<[u32]>` shape of
  `TensorRef::shape` where `TensorRefError::ShapeMismatch` takes `Vec<u32>`, so
  it could not compile. Its shape and element-count checks now call
  `check_same_shape` and `checked_element_count` instead of restating them, and
  its conformance test no longer uses a crate it did not declare.
- The feature-isolation gate no longer stores a compile outcome. Every row of
  xtask/feature-isolation.toml recorded whether the selection compiled and
  whether a sweep had ever measured it, so a fact that goes stale the moment a
  feature edge moves lived in a tracked file, and the file could not tell a
  measurement from a value someone typed. A row now declares the selection and,
  when it cannot compile, the constraint that exempts it. Whether a selection
  compiles is produced by the run that compiles it and is never deserialized, a
  sweep narrowed by member or by unrecorded selections reports every selection
  it left unmeasured and exits non-zero instead of reporting agreement it did
  not observe, and a data file that still carries the measured key or an
  outcome of compiles is rejected with the reason the column moved into the
  run. The probe also resolves cargo through the one owner of that decision
  rather than reading CARGO itself, which is unset on a CI step that does not
  run under cargo.
- The per-file line ratchet holds ceilings for files that exist. Forty-eight
  rows named files that had already been split away, one row excluded the whole
  resident runtime tree while its restructure was pending, and twenty-three
  audit rows were dead or shadowed by a tighter core row; a ratchet row on a
  missing file holds no ceiling and reports nothing. The stale rows are
  deleted, the tree exclusion ended with the restructure it waited on, and the
  six files split in this change are replaced by rows measuring their largest
  children, so each one is now held to a tighter number than the file it came
  from.
- The file-size gate test that claims the core ratchet beats the audit ceiling
  now proves it. It named a path only the core table listed, so it asserted a
  core cap and never exercised the precedence; cap_from takes both tables, and
  the test injects one path into both and asserts the tighter number wins.
- The gate-canon gate holds the registry, the pinned baselines and the subsets
  to each other and fails on the seven shapes that soften them: a baseline
  count that rises, a floor constant that moves up, a weakened target, a gate
  removed while its baseline row survives, a baseline row with no gate, a
  registered gate in no subset, and a lowered floor whose doc comment records
  no measured count. The rules are derived from the registry and the baseline
  file at run time, so a new gate is covered without being listed anywhere.
- Bitset equality and subset now reduce through the grid-stride reduction
  primitive instead of a second copy of it. The copy was missing the
  first-workgroup guard the owner documents, so the relation ops wrote their
  result from every workgroup. The owner takes an input list and a value
  expression, which is what the relation ops needed and what the copy existed
  to provide.
- The release hygiene scan reads every xtask source file. It named thirteen
  command modules by hand, so a release command added beside them was never
  scanned and a renamed module kept its row while resolving to nothing, which
  reads as coverage. The set is the tree now, and one owner answers whether a
  path holds test source for both the workspace walk and the tooling walk.
- Module layout and file naming rules moved out of the structure gate crate
  root into their own module, bringing the root back under the per-file line
  cap the workspace holds every file to.
- vyre-registry-link enables every domain of vyre-primitives instead of the
  hardware domain alone. Linking one domain left the operation registry partial
  in every binary that reads it: the release conformance run covered 356 of 359
  registered operations and reported the geometry and optimization operations
  the op matrix requires as missing, when they were only unlinked.
- The feature-msrv sweep compiles each selection through the workspace cargo
  with a leading toolchain argument, so the sweep builds into the target
  directory the checkout owns.
- The hand-written node descent scan reads its child-slot vocabulary from the
  Node enum at run time. It carried four field names written into the test
  file, so a variant declaring a body under a new name was invisible to it: the
  descent would bind a field the scan did not know, the block went unreported,
  and the roster stayed green while the shape it exists to find sat in the
  tree. Named fields whose type holds nodes and variants carrying a body in a
  tuple position are now both read from the declaration, a pattern that renames
  the field it binds is followed to the binder, and a constructed node is told
  apart from a destructured one so a rebuild is not reported for iterating what
  it just built. The wider vocabulary found one site the old one could not see:
  node_reads_any in the tail_duplication pass derived Block children by hand
  behind a catch-all, and is now exhaustive like its neighbour, so a new
  variant stops the pass at compile time instead of being refused by an arm
  nobody wrote.
- The crate ownership registry records the feature selection each dependency
  edge is built with. The `xtask-registry` to `vyre-libs` row named no features
  while the edge enables `full` and `matching-regex`, so the derived crate
  graph described a build nothing performs. The row now names both, and the
  graph is regenerated from it.
- The cross-backend parity matrix measures every registered operation on the
  reference backend and on every linked backend in one run, and reports each
  operation it could not measure with its stage, its backend and its detail. It
  aborted on the first missing fixture or refused dispatch, so the summary
  counters described a sweep that had stopped early.
- The runtime publishes 4 items at more than one path, down from the recorded
  119, and the pin records it. Deleting the re-export-only `scaling` module and
  making the uring submodules private removed 115 second paths; the committed
  snapshots are refreshed to the surface that remains.
- The script assertion ledger is generated where it counts and checked where it
  claims. Its totals and its two derived lists are rendered from the rows and
  from the tracked files, the prose above them states no number, and every row
  is held to two facts: a row may claim its script is present only while the
  tree carries it, a row whose script has left the tree must name the
  registered gate that carries its assertions and the injection that proved
  that gate red, and a row whose script is still tracked must record it as an
  operator action.
- The sweep runner's name has one owner. `gates` was a literal in the
  dispatcher, in the generated help and in the check that every subcommand a
  workflow names is dispatchable, and that check compared against the gate
  registry alone, so it reported every workflow step that runs the sweep as an
  unregistered subcommand: the whole tree-rules job and the release-evidence
  workflow. The name is now `xtask::gates::sweep::RUNNER`, the check accepts
  every gate plus the runner, and it still fails on a workflow step naming a
  subcommand nothing dispatches.
- The crates whose suites workspace-tests runs are read from the ownership
  registry by layer at run time, so a crate added to a contract layer is tested
  instead of silently uncovered, and a layer that names no crate is a finding
  rather than an empty roster.
- Three gates that reported `could not run` now run and report a number.
  `list-ops` and `catalog` failed while building the canonical operation
  schema: five registered `vyre-primitives::math` operations, the
  symmetric-eigen Jacobi family, had no backend support row, because
  `docs/optimization/OP_MATRIX.toml` was generated before they were registered.
  Regenerating it gives each a row, and both gates then reported the artifact
  they own does not exist; `docs/generated/op-inventory.toml` and
  `docs/generated/catalog.toml` are now committed. `verify-rewrite-proofs`
  needed a solver on PATH and discharges all ten shipped rewrite obligations as
  unsatisfiable. A gate that cannot run records no finding, and a baseline that
  stores its zero is a baseline that reports a defect as an achievement.
- The gate crates resolve the checkout they report on from the working
  directory at run time, through `structure_gate::workspace_root`, and no
  checkout-identifying variable is declared in the cargo config. A
  `VYRE_CHECKOUT_ROOT` with `relative = true`, read with `env!` so its value
  entered each crate's dep-info, was tried as the way to stop one checkout
  being handed a binary another compiled; cargo does not export such a variable
  to the process it runs, so every gate fell through to its compiled-in value
  and the shared binary reported a worktree's numbers as this tree's. The
  declaration is rejected rather than absent, and the comment saying so is the
  only thing that stops it coming back.
  `structure-gate/tests/checkout_provenance.rs` rejects both spellings,
  assembling each from parts at run time so the gate does not report itself and
  needs no exemption.
- The reference interpreter evaluates `BinOp::WrappingAdd` and
  `BinOp::WrappingSub` on `u32` and `i32` operands. Both are aliases of `Add`
  and `Sub`, which the interpreter already evaluates with wrapping arithmetic
  and which the constant folder and every emitter lower identically, but only
  the `u64` branch listed them; a graph that named either on a 32-bit width was
  refused with `unsupported u32 binary operation` instead of evaluated. Four
  generated per-width sweeps covered the same operation table and none of them
  swept the pair, so the gap was invisible to all four.
- `release/changes/unreleased.toml` parses again. Two fragments had been
  appended without their `[[fragments]]` header, so the second `id` key
  overwrote the first and every release-docs command failed at the TOML parser
  before it reached a verdict.
- V055 now accepts a post-barrier loop exit only when its full return path is
  workgroup-uniform. It derives same-address loads from an acquiring barrier
  and rejects intervening writes, divergent indices, atomics, and
  lane-dependent guards. The DCE fixpoint loop therefore removes one redundant
  barrier per iteration without weakening the unsafe-exit rejection.
- Validation now propagates nested-loop exit uniformity to a fixed point,
  applies subgroup capability decisions consistently, accepts IEEE-754 floating
  division by zero, rejects ordered Boolean comparisons and invalid wrapping or
  multiply-high operands, preserves same-scope duplicate-binding errors under
  nested shadowing, reports specialized expression errors at the current node,
  and publishes the emitted V020–V022 meanings.
- Validator and runtime diagnostics build their fix text without a formatting
  call that has nothing to format, the dead-store pass filters its surviving
  nodes instead of mapping a boolean, and the runtime module headers link items
  by their crate path so rustdoc resolves them.
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
- IR variant descent has one owner per reference mode.
  `vyre_foundation::visit::child_bodies_mut` is new and owns the body slots of
  a node held by unique reference, alongside `child_bodies` for a shared read
  and `vyre_foundation::transform::rewrite_walk::rewrite_node` for a
  borrow-preserving rebuild. `vyre_foundation::visit::node_map::map_body` took
  its slot list from a hand-written match ending in a catch-all that returned
  the node unchanged, so a body-bearing variant the list had not been told
  about made every pass composed on it a silent no-op inside that variant,
  including `rematerialize_cheap_let` and the pass engine's constant
  propagation. The scalar namespace also has one owner,
  `vyre_foundation::visit::node_scalars`, reporting the bound name, what the
  statement does to it, and the operand expressions in one record;
  `node_operands` and `node_bound_name` are derived from it, and
  `vyre_foundation::visit::bound_names` no longer classifies an unrecognised
  variant as binding nothing. `vyre_foundation::optimizer::cost` folds the
  divergence dimension over the owning descent instead of a search that stops
  at the first match, and `vyre_foundation::transform::autodiff::grad` reads
  its forward types and adjoint targets from the owner and reports an
  unsupported node by its registry variant name rather than by a `Debug`
  rendering truncated at sixty characters.
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
- The workspace lockfile now records the TOML parser already declared by
  vyre-libs, so locked builds resolve without modifying the checkout.
- `vyre-lints` resolves the workspace root by walking ancestors for a manifest
  that declares `[workspace]`, and `--print-default-roots` prints those roots
  relative to it. A run from inside a member directory previously enumerated no
  members at all, because the default root is the current directory and a
  member manifest declares no `workspace.members`. Manifest reading also goes
  through `toml::from_str::<toml::Table>` rather than parsing a whole document
  into `toml::Value`, which the pinned toml release rejects at run time with
  `unexpected content, expected nothing`, and source reading goes through the
  one bounded reader `vyre_lints::read_source_bounded` so a single cap covers
  every file the lint walks.
- The hand-written-descent waiver roster holds each row to the owner
  docs/CRATE_OWNERSHIP.toml declares for the crate containing the file, instead
  of a compiled-in list of three names the tree never used.
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
- WGPU resident dispatch now splits `GridSync` programs at launch boundaries
  before compilation, preventing oversized resident fixed-point grids from
  deadlocking inside a software global barrier.
- `vyre-driver-wgpu`'s loop-carrier scope-latch dispatch test reads
  `TOK_LBRACE` and `TOK_RBRACE` from `vyre_spec::c11_token` instead of
  declaring its own pair with the values 1 and 2. Those two declarations were
  the only `TOK_`-prefixed constants outside `vyre-spec` in the tree, so
  `vyre-spec`'s `no_file_outside_the_owner_declares_a_token_id` failed on this
  file. The test pins the shape the c-parser scope walker emits, and it now
  pins that shape against the real numbering the walker actually sees: the
  canonical ids are 12 and 13. Nothing observable moves, because the program
  compares `scope_kind` for equality against the same constants the fixture
  inputs are built from, and every assertion is on `scope_open` and
  `scope_depth`.
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
- The WGPU release suite now completes on the portable target limits it
  records. The metadata workload uses a 256-lane workgroup, the grouped INT4
  workload keeps its one-dimensional dispatch below 65,535 workgroups without
  changing its per-item mathematics or release threshold, and
  `release-benchmarks` declares ownership of the WGPU suite and all seventeen
  WGPU workload artifacts.
- The framing module doc in vyre-foundation named the envelope tag VIR0 in two
  places while the constant beside it is VYRE, so a reader inspecting a blob
  byte by byte compared against a tag no encoder has written.
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
- The workgroup Blelloch sweep now ends on a barrier, and its final store is
  bounded by the lane count. Callers read a slot the sweep wrote from another
  lane: `frontier_word_block_offsets_single_workgroup` reads `scratch_a[lane -
  1]` on the statement after the sweep returns, so a lane could observe its
  predecessor's slot before that lane's inclusive add landed and take a block
  offset short by the previous block's own count. Publication is now the
  sweep's contract rather than something each of its five call sites has to
  remember.
- The workspace member roster has one reader,
  `structure_gate::workspace_members`. It was parsed out of the root manifest
  three times, once inside `structure-gate` and once in each of two gates, and
  every copy carried its own filter over the member list, so a member added
  under a path one copy accepted and another skipped would be gated by one and
  invisible to the other. The CI reference gate resolves a package's directory
  through `structure_gate::member_directory` for the same reason instead of
  walking the roster itself.
- `docs/CRATE_OWNERSHIP.toml` declares `xtask-evidence`'s production edge on
  `vyre-foundation`, the `foundation-ir` seam it reads the release optimization
  family list across. The manifest gained the dependency without the
  hand-maintained ownership contract gaining the row, which fails both the
  ownership and the crate-README gates.
- Twelve clippy findings in `xtask` are gone. Two were a loop body that pushed
  the same item on every iteration, which reads as a copy-paste defect in the
  rule and was a bounded filler run in a test; both are now a resize. One was a
  `for` loop that could not iterate more than once, now an `if let`. The rest
  were a redundant trim, two complex types that wanted an alias, a manual
  character comparison, a block that wanted the question-mark operator, a
  `vec!` that never needed to allocate, and a `&PathBuf` parameter that only
  ever read a path.
- `vyre_libs::decode::ziftsieve` publishes `ZiftsieveBuffers` and
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
- ZX identity removal no longer splices out a phase-zero spider whose two edges
  are self-loops. The rule only checked that the spider had degree two, so a
  spider wired to itself twice satisfied it, was deleted, and left the loop
  edge naming an index past the end of the spider list. It now requires both
  neighbours to be other spiders of matching colour. `simplified_diagram` runs
  fusion and identity removal to a joint fixpoint, which terminates because
  every firing removes exactly one spider.

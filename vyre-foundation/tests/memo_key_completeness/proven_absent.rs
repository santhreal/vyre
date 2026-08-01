use super::*;

// ---------------------------------------------------------------------------
// 4. Proven-absent results
// ---------------------------------------------------------------------------

/// PROVEN ABSENT. A poisoned fact cache does NOT change what `optimize`
/// produces, for a fixture whose fact table is provably wrong.
///
/// Why this exists: it is the difference between "this cache can be poisoned"
/// and "this cache changes compiled output", and reporting the first as if it
/// were the second is how a correct probe produces a wrong severity. The
/// previous test proves `build_cached` hands out a table describing a different
/// tree. This test then asks the only question that decides severity: does that
/// wrong table survive into the optimizer's output?
///
/// Method: run `pre_lowering::optimize` on the same program twice, once on a
/// thread whose fact cache has been deliberately poisoned with a
/// fingerprint-colliding program, and once on a freshly spawned thread, which
/// necessarily has an empty thread-local cache. Compare the raw `Debug`
/// rendering of the resulting entry trees, not `==`, because `Program`'s
/// equality is itself blind to some differences.
///
/// The two arms agree. Two reasons, both worth knowing: `optimize` canonicalizes
/// as its very first step, so the tree the passes see already matches its own
/// canonical form, and every consumer reads canonicalization-invariant
/// projections as documented above.
///
/// What breaks if this regresses: a failure here means a poisoned cache now
/// reaches generated code, which promotes the finding above from LATENT to LIVE
/// and makes it a release blocker. This test failing is the alarm.
#[test]
fn optimize_output_is_stable_under_a_poisoned_fact_cache() {
    let warm = {
        let _poison = ProgramFacts::build_cached(&indexed_primer());
        vyre_foundation::optimizer::pre_lowering::optimize(indexed_target())
    };
    let cold =
        std::thread::spawn(|| vyre_foundation::optimizer::pre_lowering::optimize(indexed_target()))
            .join()
            .expect("cold arm must not panic");

    assert_eq!(
        format!("{:?}", warm.entry()),
        format!("{:?}", cold.entry()),
        "Fix: optimize() output now depends on thread-local fact-cache state, which promotes the \
         build_cached key gap from LATENT to a LIVE wrong-output defect. Do not silence this by \
         clearing the cache in the test."
    );
    assert_eq!(
        warm.fingerprint(),
        cold.fingerprint(),
        "warm and cold arms must agree on the canonical form too"
    );
}

/// PROVEN SOUND. The per-`Program` value memos are keyed by NOTHING, so they
/// structurally cannot serve another program's artifact, and they are cleared
/// on mutation rather than going stale.
///
/// Why this exists: the enumeration that produced this file found that most
/// reuse in `Program` is memoization on the value itself (`fingerprint`,
/// `stats`, `output_buffer_indices`, `has_indirect_dispatch`) rather than
/// lookup in a shared table. That distinction is the whole reason those are
/// safe while `build_cached` is not, and it is worth an explicit test so a
/// future refactor cannot quietly convert a value memo into a keyed cache.
///
/// This also covers the mutable-state hazard: a memo over a value whose inputs
/// can change is stale by construction unless every mutator invalidates it.
/// `entry_mut` is the mutation path, so it must clear all four.
#[test]
fn program_value_memos_are_stable_and_invalidate_on_mutation() {
    let mut program = Program::wrapped(
        vec![out_buf()],
        [64, 1, 1],
        vec![Node::store("out", Expr::u32(0), Expr::u32(1))],
    );

    // Stable across repeated reads.
    let fingerprint = program.fingerprint();
    assert_eq!(program.fingerprint(), fingerprint);
    assert_eq!(program.stats().node_count, 2);
    assert_eq!(program.stats().node_count, 2);
    assert_eq!(program.output_buffer_indices(), &[0]);
    assert_eq!(program.output_buffer_indices(), &[0]);
    assert!(!program.has_indirect_dispatch());

    // A clone carries the memos and must agree exactly.
    let cloned = program.clone();
    assert_eq!(cloned.fingerprint(), fingerprint);
    assert_eq!(cloned.stats().node_count, 2);

    // Mutating through the sanctioned path must invalidate, not go stale.
    program
        .entry_mut()
        .push(Node::store("out", Expr::u32(1), Expr::u32(2)));
    assert_ne!(
        program.fingerprint(),
        fingerprint,
        "Fix: entry_mut must invalidate the fingerprint memo. A memo over mutable state that is \
         not invalidated is stale by construction."
    );
    assert_eq!(
        program.stats().node_count,
        3,
        "Fix: entry_mut must invalidate the stats memo."
    );
    assert!(
        !program.is_structurally_validated(),
        "Fix: entry_mut must clear structural validation state."
    );
}

// ---------------------------------------------------------------------------
// The invariant that keeps the `build_cached` defect LATENT, made enforceable
// ---------------------------------------------------------------------------
//
// `program_facts_cache_serves_wrong_node_count_and_indices` proves the cache
// hands out a fact table describing a DIFFERENT tree, and
// `optimize_output_is_stable_under_a_poisoned_fact_cache` proves that does not
// reach generated code today. The reason it does not is a single unwritten
// invariant:
//
//     NO CONSUMER OF A FINGERPRINT-KEYED FACT CACHE MAY MIX A CACHED
//     `NodeIndex` WITH THE LIVE PROGRAM TREE, OR BRANCH ON A PROJECTION THAT
//     CANONICALIZATION CAN CHANGE.
//
// Nothing enforced that. The three tests below do, so the invariant fails
// loudly at its boundary instead of silently at a buffer recycle.

/// Source files that read a fingerprint-keyed fact cache, and the exact number
/// of call sites in each. Production only: `#[cfg(test)]` consumers are
/// excluded by skipping `tests` paths.
const CACHED_FACT_CONSUMERS: &[(&str, usize)] = &[
    ("optimizer/megakernel/scratch_reuse.rs", 1),
    (
        "optimizer/passes/algebraic/const_fold/reaching_def_propagate.rs",
        2,
    ),
    ("optimizer/passes/fusion_cse/fusion.rs", 2),
    ("optimizer/passes/loops/loop_software_pipeline.rs", 2),
    ("optimizer/passes/loops/loop_var_range_fold.rs", 2),
    ("optimizer/passes/memory/dead_buffer_elim.rs", 1),
    ("optimizer/passes/memory/vectorization.rs", 1),
    ("optimizer/passes/specialization/autotune.rs", 2),
    ("transform/visit.rs", 1),
    ("validate/linear_type.rs", 1),
];

/// Every constructor that returns facts served from a fingerprint-keyed cache.
const CACHED_FACT_CONSTRUCTORS: &[&str] = &[
    "build_cached(",
    "derive_cached(",
    "derive_shape_and_use_cached(",
    "derive_use_only_cached(",
];

/// `ProgramFacts` projections that canonicalization CAN change, so reading one
/// off a cached table is reading a property of a different program.
///
/// `node_count()` differs (proven: 6 reported for a 5-node program) and
/// `kinds_present()` is a bitset over the raw walk, so an erased `Block` clears
/// a bit that the live tree still sets.
const CANONICALIZATION_VARIANT_PROJECTIONS: &[&str] = &["node_count()", "kinds_present()"];

/// Collect every `.rs` file under `dir`, skipping test modules.
fn collect_production_sources(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
    let entries = std::fs::read_dir(dir)
        .unwrap_or_else(|error| panic!("Fix: cannot read {}: {error}", dir.display()));
    for entry in entries {
        let path = entry
            .unwrap_or_else(|error| panic!("Fix: cannot read entry in {}: {error}", dir.display()))
            .path();
        if path.is_dir() {
            if path.file_name().is_some_and(|name| name == "tests") {
                continue;
            }
            collect_production_sources(&path, out);
        } else if path.extension().is_some_and(|ext| ext == "rs")
            && !path
                .file_name()
                .is_some_and(|name| name.to_string_lossy().ends_with("tests.rs"))
        {
            out.push(path);
        }
    }
}

/// Strip the CONTENTS of double-quoted string literals from one line.
///
/// Why this exists: the detectors below search for text like `build_cached(` or
/// `node_count()`. A needle inside a STRING, typically a panic or `unreachable!`
/// message naming the very contract being guarded, is not a call and not a read.
/// Counting it would make a detector look sensitive while being merely noisy,
/// and it is the likeliest near-miss in THIS codebase precisely because those
/// messages quote the API they protect.
///
/// Boundary: single-line literals only. A raw string spanning lines is not
/// handled, and no consumer file contains one today. A `'"'` char literal would
/// also toggle the state wrongly; none appears in the scanned set.
fn code_outside_string_literals(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut in_string = false;
    let mut escaped = false;
    for ch in line.chars() {
        if escaped {
            escaped = false;
            continue;
        }
        match ch {
            '\\' if in_string => escaped = true,
            '"' => in_string = !in_string,
            _ if !in_string => out.push(ch),
            _ => {}
        }
    }
    out
}

/// True when `line` is a comment or a function DEFINITION rather than a call.
///
/// `fn ` rather than `pub fn `, because a `pub(crate) fn build_cached(` or a
/// private `fn derive_cached(` is still a definition, and counting a definition
/// as a call site inflates the closed-set counts with a site that calls nothing.
fn is_comment_or_definition(line: &str) -> bool {
    let trimmed = line.trim_start();
    trimmed.starts_with("//") || trimmed.contains("fn ")
}

/// Count cache-constructor CALL sites, excluding definitions, comments and
/// needles that appear only inside string literals.
fn count_cached_fact_calls(source: &str) -> usize {
    source
        .lines()
        .filter(|line| !is_comment_or_definition(line))
        .map(|line| {
            let code = code_outside_string_literals(line);
            CACHED_FACT_CONSTRUCTORS
                .iter()
                .map(|needle| code.matches(needle).count())
                .sum::<usize>()
        })
        .sum()
}

/// Which canonicalization-variant projections `source` actually READS.
///
/// Comments and string literals are excluded for the same reason as above. Note
/// this does NOT skip `fn ` lines: a read can legitimately sit on a line that
/// also opens a closure, and excluding those would blind the detector.
fn variant_projections_read_by(source: &str) -> Vec<&'static str> {
    let code: String = source
        .lines()
        .filter(|line| !line.trim_start().starts_with("//"))
        .map(code_outside_string_literals)
        .collect::<Vec<_>>()
        .join("\n");
    CANONICALIZATION_VARIANT_PROJECTIONS
        .iter()
        .copied()
        .filter(|projection| code.contains(projection))
        .collect()
}

/// True when `collect_buffer_uses` keeps its live-tree parameter UNUSED, which
/// the leading underscore is the only in-source evidence of.
fn live_tree_parameter_is_unused(source: &str) -> bool {
    source.contains("_entry: &[Node]")
}

fn production_sources() -> Vec<(String, String)> {
    let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    collect_production_sources(&src, &mut files);
    files
        .into_iter()
        .map(|path| {
            let relative = path
                .strip_prefix(&src)
                .expect("Fix: collected path must live under src/")
                .to_string_lossy()
                .replace('\\', "/");
            let source = std::fs::read_to_string(&path)
                .unwrap_or_else(|error| panic!("Fix: cannot read {}: {error}", path.display()));
            (relative, source)
        })
        .collect()
}

/// The set of consumers of the fingerprint-keyed fact caches is CLOSED, and a
/// new one FAILS THIS TEST rather than silently inheriting the hazard.
///
/// Why this exists: `build_cached` can serve a fact table built from a
/// different raw tree (proven above with exact values). That is harmless only
/// because of what today's ten consumers happen to read. A new consumer is not
/// covered by that reasoning, and the failure mode if it reads the wrong thing
/// is a wrong `NodeIndex` used against the live tree, which surfaces as wrong
/// output or a bad buffer recycle rather than as an error. So a new consumer
/// must be a DECISION, not a default.
///
/// Asserts BOTH directions with exact counts, because either drift matters: a
/// new file means an unreviewed consumer, a vanished file means this list has
/// gone stale and is no longer protecting anything, and a changed count in a
/// known file means a new call site inside already-reviewed code.
///
/// What breaks if this regresses: when this fails, read
/// `program_facts_cache_serves_wrong_node_count_and_indices` first, then decide
/// whether the new call site reads canonicalization-invariant projections
/// (buffer names, use counts, `Let` names, Region generators) or whether it
/// needs the uncached `ProgramFacts::build`. Then update this list.
#[test]
fn cached_fact_consumers_are_a_closed_set_with_exact_call_counts() {
    let mut found: Vec<(String, usize)> = production_sources()
        .into_iter()
        .filter_map(|(relative, source)| {
            let count = count_cached_fact_calls(&source);
            (count > 0).then_some((relative, count))
        })
        .collect();
    found.sort();

    let expected: Vec<(String, usize)> = CACHED_FACT_CONSUMERS
        .iter()
        .map(|(path, count)| ((*path).to_string(), *count))
        .collect();

    assert_eq!(
        found, expected,
        "Fix: the set of consumers of the fingerprint-keyed fact caches changed. A cached fact \
         table can describe a DIFFERENT raw tree than the program in hand, so every consumer must \
         be checked to read only canonicalization-invariant projections. Review the new or changed \
         call site, then update CACHED_FACT_CONSUMERS."
    );
}

/// NO consumer of a cached fact table branches on a projection that
/// canonicalization can change.
///
/// Why this exists: this is the invariant itself, asserted directly. Reading
/// `node_count()` or `kinds_present()` off a cached table is reading a property
/// of whichever program warmed the cache. `node_count()` is proven to differ (6
/// reported for a 5-node program) and `kinds_present()` is a bitset over the raw
/// walk, so an erased `Block` clears a bit the live tree still sets.
///
/// What breaks if this regresses: a pass that skips work because
/// `kinds_present()` says there are no `Loop` nodes, on a program that has one,
/// silently drops an optimization or applies one that is unsound. Nothing
/// errors. If you need either projection, call the uncached
/// `ProgramFacts::build` and take the full walk.
#[test]
fn no_cached_fact_consumer_reads_a_canonicalization_variant_projection() {
    let consumers: Vec<&str> = CACHED_FACT_CONSUMERS
        .iter()
        .map(|(path, _)| *path)
        .collect();
    let mut violations: Vec<String> = Vec::new();

    for (relative, source) in production_sources() {
        if !consumers.contains(&relative.as_str()) {
            continue;
        }
        for projection in variant_projections_read_by(&source) {
            violations.push(format!("{relative} reads {projection}"));
        }
    }

    assert_eq!(
        violations,
        Vec::<String>::new(),
        "Fix: a consumer of a fingerprint-keyed fact cache now branches on a projection that \
         canonicalization can change, so it may be reading a property of a DIFFERENT program. Use \
         the uncached ProgramFacts::build for these projections."
    );
}

/// The NEAREST consumer to memory corruption keeps the live tree UNUSED.
///
/// Why this exists, and why it is the sharpest guard in this file:
/// `optimizer/megakernel/scratch_reuse.rs` decides which buffers a megakernel
/// arm may RECYCLE. It walks `facts.buffer_refs()` and calls
/// `facts.is_descendant_of(*node, region_node)`, where both indices come from
/// the served table, so they agree with each other even when the table
/// describes a different tree. That self-consistency is the only thing making
/// it safe.
///
/// Its helper `collect_buffer_uses` still ACCEPTS the live entry tree, as
/// `_entry`, deliberately unused. That underscore is load-bearing: the moment
/// someone uses that parameter and indexes it with a cached `NodeIndex`, the
/// descendant test answers for the wrong tree, a buffer that is actually live
/// is judged recyclable, and the megakernel writes over data still in use.
/// That is MEMORY CORRUPTION, not a wrong token id, and it would not raise an
/// error anywhere.
///
/// What breaks if this regresses: if the parameter loses its underscore, do not
/// "fix" this test. Either drop the parameter entirely, or switch that call
/// site to the uncached `ProgramFacts::build` so the indices and the tree are
/// guaranteed to describe the same program.
#[test]
fn scratch_reuse_never_indexes_the_live_tree_with_cached_indices() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src/optimizer/megakernel/scratch_reuse.rs");
    let source = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("Fix: cannot read {}: {error}", path.display()));

    assert!(
        source.contains("facts.is_descendant_of("),
        "Fix: scratch_reuse no longer uses is_descendant_of, so this guard has stopped protecting \
         the buffer-recycling decision it was written for. Re-derive which indices that decision \
         now depends on before deleting this test."
    );
    assert!(
        live_tree_parameter_is_unused(&source),
        "Fix: collect_buffer_uses must keep its live-tree parameter UNUSED. Indexing the live tree \
         with a NodeIndex from build_cached lets a buffer that is still live be judged recyclable, \
         which corrupts memory silently. Drop the parameter or use the uncached ProgramFacts::build."
    );
}

/// THE THREE SOURCE-SCANNING GUARDS IN THIS FILE ARE PROVEN TO DISCRIMINATE,
/// and this is the test that proves it. Without it they were DETECTORS, not
/// gates.
///
/// Why this exists, and it is the honest correction to a green I already
/// reported: the other thirteen tests here earned their green by being observed
/// RED on the real defect (five flipped the instant the wire encoder changed;
/// the equality pair flipped on the real `buffer_decl_canonical_key` omission).
/// These three never were, because they scan production source as TEXT and I own
/// none of the files they scan, so I cannot inject a defect to make them fail.
///
/// The failure mode that makes this necessary is specific and quiet: a
/// source-scanning guard breaks by SILENTLY CEASING TO MATCH. Rename the
/// function, reformat the signature across two lines, or change a call to a
/// method chain, and the needle stops appearing. The guard then reports GREEN
/// forever while protecting nothing, which is worse than deleting it, because a
/// deleted guard is visibly absent and a no-op guard looks like coverage.
///
/// Method, which is why this works without touching production code: each
/// detector is a PURE function over source text, so it can be fed SYNTHETIC
/// input here. Every case below is a real behavioural claim, and each near-miss
/// is a case that would otherwise make a detector look sensitive while being
/// merely noisy. A detector never shown to REJECT a near-miss has not been shown
/// to discriminate; it has only been shown to match something.
///
/// What breaks if this regresses: if a case here fails, one of the three guards
/// has stopped meaning what its name says, and the closed-set counts or the
/// projection ban are no longer enforcing anything. Fix the detector, never the
/// expectation.
#[test]
fn the_source_scanning_detectors_fire_on_the_defect_and_reject_near_misses() {
    // Detector 1, POSITIVE: a real call site counts.
    assert_eq!(
        count_cached_fact_calls("    let facts = ProgramFacts::build_cached(program);"),
        1,
        "Fix: the call-site detector no longer sees a plain build_cached call, so every count in \
         CACHED_FACT_CONSUMERS is now vacuously satisfied and a new consumer would slip in."
    );
    // Two DIFFERENT constructors on one line count as two, so a line cannot
    // hide a second call behind the first.
    assert_eq!(
        count_cached_fact_calls(
            "    let a = ProgramFacts::build_cached(p); let b = Shape::derive_cached(p);"
        ),
        2,
        "Fix: the detector counts at most one call per line, so a consumer could add a second \
         cached-fact call without moving its committed count."
    );
    // Detector 1, NEAR-MISSES that must NOT count.
    assert_eq!(
        count_cached_fact_calls("    pub fn build_cached(program: &Program) -> Self {"),
        0,
        "Fix: a public DEFINITION is being counted as a call site."
    );
    assert_eq!(
        count_cached_fact_calls("    pub(crate) fn derive_cached(&self) -> Self {"),
        0,
        "Fix: a pub(crate) DEFINITION is being counted as a call site. This is the case the old \
         `pub fn` filter missed, which would inflate a consumer's count with a site that calls \
         nothing."
    );
    assert_eq!(
        count_cached_fact_calls("    // build_cached(program) is deliberately avoided here"),
        0,
        "Fix: a COMMENT mentioning the constructor is being counted as a call."
    );
    assert_eq!(
        count_cached_fact_calls(
            "        unreachable!(\"derive_use_only_cached( is not permitted here\");"
        ),
        0,
        "Fix: a needle inside a STRING LITERAL is being counted as a call. These messages quote \
         the API they protect, so this is the most likely false positive in this codebase."
    );
    assert_eq!(
        count_cached_fact_calls("    let facts = ProgramFacts::build(program);"),
        0,
        "Fix: the UNCACHED constructor is being counted. It is the safe call and the recommended \
         remedy, so counting it would penalize the fix."
    );

    // Detector 2, POSITIVE: each banned projection is seen.
    assert_eq!(
        variant_projections_read_by("    let n = facts.node_count();"),
        vec!["node_count()"],
        "Fix: the projection detector no longer sees a node_count() read, so the ban on reading \
         canonicalization-variant projections off a cached table is unenforced."
    );
    assert_eq!(
        variant_projections_read_by("    if facts.kinds_present().contains(Loop) {"),
        vec!["kinds_present()"],
        "Fix: the projection detector no longer sees a kinds_present() read."
    );
    // Detector 2, THE NEAR-MISS THAT MATTERS: a field access with no parens is
    // a DIFFERENT property, read off Program::stats() rather than cached facts,
    // and flagging it would make the guard noisy rather than sensitive.
    assert_eq!(
        variant_projections_read_by("    let n = program.stats().node_count;"),
        Vec::<&str>::new(),
        "Fix: the detector now flags `stats().node_count`, a FIELD access on the program's own \
         stats. That is not a cached-facts read, and flagging it makes the guard fire on correct \
         code, which is how a guard gets deleted as noise."
    );
    assert_eq!(
        variant_projections_read_by("    // facts.node_count() would be wrong here"),
        Vec::<&str>::new(),
        "Fix: a COMMENT explaining why the projection is avoided is being reported as a read, so \
         documenting the hazard would trip the guard."
    );
    assert_eq!(
        variant_projections_read_by(
            "        panic!(\"never read node_count() off cached facts\");"
        ),
        Vec::<&str>::new(),
        "Fix: a needle inside a STRING is being reported as a read."
    );

    // Detector 3: the underscore IS the evidence, so it must be load bearing.
    assert!(
        live_tree_parameter_is_unused(
            "fn collect_buffer_uses(_entry: &[Node], facts: &ProgramFacts) {"
        ),
        "Fix: the unused-parameter detector no longer recognizes the underscored parameter, so the \
         sharpest guard in this file passes vacuously."
    );
    assert!(
        !live_tree_parameter_is_unused(
            "fn collect_buffer_uses(entry: &[Node], facts: &ProgramFacts) {"
        ),
        "Fix: the detector accepts the parameter WITHOUT its underscore, which is the exact defect \
         it exists to catch: indexing the live tree with a NodeIndex from build_cached lets a \
         still-live buffer be judged recyclable, corrupting memory with no error."
    );
}

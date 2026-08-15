//! The C-AST parity matrix: one owner for the case list and the scaffold both
//! arms of a C-AST family run.
//!
//! A C-AST family is one fixture file plus two arms. The CPU arm lives in
//! `vyre-libs/tests` and asserts what each construct must classify as. The
//! backend arm lives in a driver's `tests` and asserts that dispatching the
//! same kernels reproduces the CPU oracle byte for byte. Both arms build the
//! same fixtures, run the same four stages, and compare the same packed rows.
//!
//! # What was wrong with a scaffold per directory
//!
//! Four directories restated the stage sequence
//! (`reference_c11_build_vast_nodes` to `reference_c11_annotate_typedef_names`
//! to `reference_c11_classify_vast_node_kinds` to `reference_ast_to_pg_nodes`)
//! once per test function, and each arm's case list was whichever fixtures its
//! own functions happened to name. Nothing compared the two lists, so they
//! drifted: `gnu_restrict_qualifier` was named by the CPU arm and by no
//! backend arm, which left GNU `__restrict` normalization CPU-proven and
//! backend-unproven, and six declarator-matrix cases reached the classifier on
//! a backend but never the property-graph lowerer.
//!
//! Here the case list is a [`ParityCase`] table beside its fixtures, both arms
//! iterate that one table, and `c_ast_parity_case_matrix_gate` fails when a
//! fixture function is missing from it. A construct cannot be proven on one
//! side and unproven on the other, because neither side has a list of its own
//! to forget.
//!
//! # The stages
//!
//! [`cpu_stages`] is the oracle chain. [`ParityArm`] is one dispatch of one
//! [`Program`], which is all a backend arm has to supply: this module owns
//! every program the four stages need, so the buffer names and the argument
//! order have one definition rather than one per driver crate. The typedef
//! stage is three dispatches (prehash identifiers, precompute brace scopes,
//! annotate against those scopes) and it uses the SCOPE-AWARE annotator,
//! because that is the only one that can reproduce
//! `reference_c11_annotate_typedef_names` under shadowing.

use vyre::ir::{Expr, Program};
use vyre_libs::parsing::c::lower::{c_lower_ast_to_pg_nodes, reference_ast_to_pg_nodes};
use vyre_libs::parsing::c::parse::vast::{
    c11_annotate_typedef_names, c11_annotate_typedef_names_precomputed_scope,
    c11_build_vast_nodes, c11_classify_vast_node_kinds, c11_precompute_vast_scopes,
    c11_prehash_vast_identifiers, reference_c11_annotate_typedef_names,
    reference_c11_build_vast_nodes, reference_c11_classify_vast_node_kinds,
};

use super::rows::{
    assert_words_eq, bytes, haystack_words, node_count_from_vast, pg_word_at, word_at,
    VAST_STRIDE_BYTES, VAST_STRIDE_U32,
};
use super::token_fixture::Fixture;

/// One construct both arms of a family evaluate.
///
/// `name` is the fixture function's name without its `fixture_` prefix, which
/// is what lets the case-matrix gate compare a table against the fixture file
/// it sits in.
pub(crate) struct ParityCase {
    pub(crate) name: &'static str,
    pub(crate) build: fn() -> Fixture,
}

impl ParityCase {
    pub(crate) const fn new(name: &'static str, build: fn() -> Fixture) -> Self {
        Self { name, build }
    }
}

/// Every packed buffer the CPU oracle chain produces for one fixture.
pub(crate) struct Stages {
    /// VAST rows straight from the builder, before typedef annotation.
    pub(crate) raw: Vec<u8>,
    /// VAST rows with typedef visibility and declaration flags resolved.
    pub(crate) annotated: Vec<u8>,
    /// VAST rows with AST kinds classified.
    pub(crate) typed: Vec<u8>,
    /// Property-graph rows lowered from the typed VAST.
    pub(crate) pg: Vec<u8>,
}

/// Run one fixture through the four CPU oracles.
pub(crate) fn cpu_stages(fix: &Fixture) -> Stages {
    let raw = reference_c11_build_vast_nodes(&fix.tok_types, &fix.tok_starts, &fix.tok_lens);
    let annotated = reference_c11_annotate_typedef_names(&raw, fix.source.as_bytes());
    let typed = reference_c11_classify_vast_node_kinds(&annotated);
    let pg = reference_ast_to_pg_nodes(&typed);
    Stages {
        raw,
        annotated,
        typed,
        pg,
    }
}

/// Assert the property graph mirrors the typed VAST at EVERY row.
///
/// The lowerer emits one property-graph row per VAST row, copying the kind and
/// the three tree links and turning the token's offset and length into a span,
/// so "row `i` of the graph mirrors row `i` of the VAST" is the whole contract
/// and it holds for every row rather than for a chosen few. Each family used to
/// pin a hand-written index list per case; a list is a member set that goes
/// stale in silence, and the rows it left out were exactly the ones no test
/// covered. Iterating the rows the fixture actually produced needs no list.
pub(crate) fn assert_pg_mirrors_every_vast_row(fix: &Fixture, stages: &Stages, label: &str) {
    let rows = stages.typed.len() / VAST_STRIDE_BYTES;
    assert_eq!(
        rows,
        fix.tok_starts.len(),
        "{label}: the VAST carries one row per token, so a row count that \
         disagrees with the token count means the span columns below are \
         comparing unrelated rows"
    );
    for idx in 0..rows {
        let vast = idx * VAST_STRIDE_U32;
        assert_eq!(
            pg_word_at(&stages.pg, idx, 0),
            word_at(&stages.typed, vast),
            "{label}: PG kind mismatch at row {idx}"
        );
        assert_eq!(
            pg_word_at(&stages.pg, idx, 1),
            fix.tok_starts[idx],
            "{label}: PG span_start mismatch at row {idx}"
        );
        assert_eq!(
            pg_word_at(&stages.pg, idx, 2),
            fix.tok_starts[idx] + fix.tok_lens[idx],
            "{label}: PG span_end mismatch at row {idx}"
        );
        assert_eq!(
            pg_word_at(&stages.pg, idx, 3),
            word_at(&stages.typed, vast + 1),
            "{label}: PG parent mismatch at row {idx}"
        );
        assert_eq!(
            pg_word_at(&stages.pg, idx, 4),
            word_at(&stages.typed, vast + 2),
            "{label}: PG first_child mismatch at row {idx}"
        );
        assert_eq!(
            pg_word_at(&stages.pg, idx, 5),
            word_at(&stages.typed, vast + 3),
            "{label}: PG next_sibling mismatch at row {idx}"
        );
    }
}

// ---------------------------------------------------------------------------
// Programs
// ---------------------------------------------------------------------------

/// The [`Program`] each parity stage dispatches.
///
/// Buffer names and argument order are an ABI between the program and the
/// inputs a caller passes positionally, so they have one definition here rather
/// than one per driver crate.
pub(crate) mod program {
    use super::*;

    /// Build raw VAST rows from a token stream. Outputs `[vast_nodes, count]`.
    pub(crate) fn build_vast(token_count: u32) -> Program {
        c11_build_vast_nodes(
            "tok_types",
            "tok_starts",
            "tok_lens",
            Expr::u32(token_count),
            "out_vast_nodes",
            "out_count",
        )
    }

    /// Hash every identifier row's spelling into the symbol field.
    pub(crate) fn prehash_identifiers(source_len: u32, node_count: u32) -> Program {
        c11_prehash_vast_identifiers(
            "vast_nodes",
            "haystack",
            Expr::u32(source_len),
            Expr::u32(node_count),
            "hashed_vast",
        )
    }

    /// Resolve each row's enclosing brace scope into the scope field.
    pub(crate) fn precompute_scopes(node_count: u32) -> Program {
        c11_precompute_vast_scopes("hashed_vast", Expr::u32(node_count), "scoped_vast")
    }

    /// Annotate typedef flags using the scopes [`precompute_scopes`] resolved.
    pub(crate) fn annotate_scoped_typedefs(source_len: u32, node_count: u32) -> Program {
        c11_annotate_typedef_names_precomputed_scope(
            "vast_nodes",
            "haystack",
            Expr::u32(source_len),
            Expr::u32(node_count),
            "annotated_vast",
        )
    }

    /// Annotate typedef flags in one pass, rescanning source per row.
    pub(crate) fn annotate_typedefs(source_len: u32, node_count: u32) -> Program {
        c11_annotate_typedef_names(
            "vast_nodes",
            "haystack",
            Expr::u32(source_len),
            Expr::u32(node_count),
            "annotated_vast",
        )
    }

    /// Classify each annotated VAST row's AST kind.
    pub(crate) fn classify(node_count: u32) -> Program {
        c11_classify_vast_node_kinds("vast_nodes", Expr::u32(node_count), "typed_vast_nodes")
    }

    /// Lower typed VAST rows to property-graph rows.
    pub(crate) fn lower_pg(node_count: u32) -> Program {
        c_lower_ast_to_pg_nodes("vast_nodes", Expr::u32(node_count), "out_pg_nodes")
    }
}

// ---------------------------------------------------------------------------
// Backend arms
// ---------------------------------------------------------------------------

/// One dispatch of one [`Program`], however the owning crate reaches a backend.
///
/// The driver crate's arm serializes GPU dispatch behind a lock and a watchdog;
/// the CPU-reference arm calls the interpreter directly. Everything above this
/// method is identical, so the stage sequence and the parity assertions live
/// here and only the dispatch differs.
pub(crate) trait ParityArm {
    /// Dispatch `program` with `inputs` in `Program::buffers` order, panicking
    /// with `context` on failure.
    fn dispatch(
        &self,
        context: &'static str,
        program: Program,
        inputs: Vec<Vec<u8>>,
    ) -> Vec<Vec<u8>>;
}

/// The primary output, requiring any trailing output to be zero-byte scratch.
///
/// A megakernel declares scratch buffers a backend must allocate but whose
/// contents are meaningless after the dispatch. Accepting a non-empty trailing
/// output would let a stage silently read the wrong buffer.
pub(crate) fn primary_output(outputs: Vec<Vec<u8>>, context: &str) -> Vec<u8> {
    assert!(
        !outputs.is_empty(),
        "{context}: expected at least one primary output"
    );
    assert!(
        outputs.iter().skip(1).all(Vec::is_empty),
        "{context}: only zero-byte scratch outputs may follow the primary output"
    );
    outputs[0].clone()
}

/// Stage 1: raw VAST rows for a token stream.
pub(crate) fn arm_raw_vast(
    arm: &impl ParityArm,
    tok_types: &[u32],
    tok_starts: &[u32],
    tok_lens: &[u32],
) -> Vec<u8> {
    let outputs = arm.dispatch(
        "C VAST builder",
        program::build_vast(tok_types.len() as u32),
        vec![bytes(tok_types), bytes(tok_starts), bytes(tok_lens)],
    );
    assert_eq!(
        outputs.len(),
        2,
        "C VAST builder: expected [vast_nodes, count]"
    );
    outputs[0].clone()
}

/// Stage 2: typedef flags, resolved against precomputed brace scopes.
///
/// Three dispatches, because the scope-aware annotator reads a scope field an
/// earlier pass has to fill. The global-fast annotator would be one dispatch
/// and cannot be substituted here: it tests an identifier's hash against the
/// set of file-scope typedef names and has no scope model, so it disagrees with
/// `reference_c11_annotate_typedef_names` in both directions under shadowing
/// (an inner `float T` keeps removing the name after its block closes, and a
/// shadowed file-scope name keeps resolving as a type). `c_global_typedef_annotate_parity`
/// in `vyre-libs/tests` pins what that faster path does promise.
pub(crate) fn arm_annotated_vast(arm: &impl ParityArm, source: &[u8], raw_vast: &[u8]) -> Vec<u8> {
    let source_len = source.len() as u32;
    let node_count = node_count_from_vast(raw_vast);
    let haystack = haystack_words(source);

    let hashed = arm.dispatch(
        "typedef identifier prehash",
        program::prehash_identifiers(source_len, node_count),
        vec![raw_vast.to_vec(), haystack.clone(), raw_vast.to_vec()],
    );
    let hashed_vast = primary_output(hashed, "typedef identifier prehash");

    let scope_stack = vec![0u8; node_count.max(1) as usize * core::mem::size_of::<u32>()];
    let scoped = arm.dispatch(
        "typedef scope precompute",
        program::precompute_scopes(node_count),
        vec![hashed_vast.clone(), hashed_vast, scope_stack],
    );
    let scoped_vast = primary_output(scoped, "typedef scope precompute");

    let annotated = arm.dispatch(
        "scoped typedef annotation",
        program::annotate_scoped_typedefs(source_len, node_count),
        vec![scoped_vast, haystack],
    );
    primary_output(annotated, "scoped typedef annotation")
}

/// Stage 3: AST kinds for annotated VAST rows.
pub(crate) fn arm_typed_vast(arm: &impl ParityArm, annotated_vast: &[u8]) -> Vec<u8> {
    let outputs = arm.dispatch(
        "VAST classifier",
        program::classify(node_count_from_vast(annotated_vast)),
        vec![annotated_vast.to_vec()],
    );
    primary_output(outputs, "VAST classifier")
}

/// Stage 4: property-graph rows for typed VAST rows.
pub(crate) fn arm_pg_nodes(arm: &impl ParityArm, typed_vast: &[u8]) -> Vec<u8> {
    let outputs = arm.dispatch(
        "AST-to-PG lower",
        program::lower_pg(node_count_from_vast(typed_vast)),
        vec![typed_vast.to_vec()],
    );
    primary_output(outputs, "AST-to-PG lower")
}

/// Assert an arm reproduces the CPU oracle through the classifier.
///
/// Feeds each stage the arm's OWN previous output rather than the oracle's, so
/// a divergence is attributed to the stage that introduced it instead of being
/// masked by a corrected input.
pub(crate) fn assert_arm_parity_through_classify(
    arm: &impl ParityArm,
    fix: &Fixture,
    label: &str,
) -> Stages {
    let cpu = cpu_stages(fix);

    let raw = arm_raw_vast(arm, &fix.tok_types, &fix.tok_starts, &fix.tok_lens);
    assert_words_eq(&raw, &cpu.raw, &format!("{label}: raw VAST arm/CPU parity"));

    let annotated = arm_annotated_vast(arm, fix.source.as_bytes(), &raw);
    assert_words_eq(
        &annotated,
        &cpu.annotated,
        &format!("{label}: annotated VAST arm/CPU parity"),
    );

    let typed = arm_typed_vast(arm, &annotated);
    assert_words_eq(
        &typed,
        &cpu.typed,
        &format!("{label}: typed VAST arm/CPU parity"),
    );

    cpu
}

/// Assert an arm reproduces the CPU oracle through every stage, and that the
/// oracle's own property graph mirrors its typed VAST row for row.
pub(crate) fn assert_case_parity(arm: &impl ParityArm, case: &ParityCase) {
    let fix = (case.build)();
    let cpu = assert_arm_parity_through_classify(arm, &fix, case.name);

    let pg = arm_pg_nodes(arm, &cpu.typed);
    assert_words_eq(
        &pg,
        &cpu.pg,
        &format!("{}: PG lowering arm/CPU parity", case.name),
    );

    assert_pg_mirrors_every_vast_row(&fix, &cpu, case.name);
}

/// Run every case of a family through [`assert_case_parity`].
pub(crate) fn assert_family_parity(arm: &impl ParityArm, cases: &[ParityCase]) {
    assert!(
        !cases.is_empty(),
        "Fix: a parity family with no cases proves nothing; populate its CASES table"
    );
    for case in cases {
        assert_case_parity(arm, case);
    }
}

// ---------------------------------------------------------------------------
// Case-matrix gate
// ---------------------------------------------------------------------------

/// Fixture builder names a fixture source file declares, without the
/// `fixture_` prefix, in declaration order.
///
/// Read from the file at run time so that adding a fixture changes the member
/// set the gate compares against without anyone maintaining a second list.
pub(crate) fn fixture_names_in_source(source: &str) -> Vec<String> {
    const PREFIX: &str = "pub(crate) fn fixture_";
    source
        .lines()
        .filter_map(|line| {
            let rest = line.trim_start().strip_prefix(PREFIX)?;
            let name = rest.split('(').next()?;
            (!name.is_empty()).then(|| name.to_string())
        })
        .collect()
}

/// Assert a family's case table names every fixture its fixture file declares.
///
/// This is what keeps the two arms on one case list. Both arms iterate `cases`,
/// so a fixture reaching the table reaches both arms at once, and a fixture
/// that never reaches the table is a construct nobody proves. The member set is
/// the fixture file on disk, so adding a builder turns this red until the table
/// records a decision about it.
pub(crate) fn assert_case_table_covers_fixture_file(
    family: &str,
    fixture_source: &str,
    cases: &[ParityCase],
) {
    let declared = fixture_names_in_source(fixture_source);
    assert!(
        !declared.is_empty(),
        "Fix: found no `pub(crate) fn fixture_*` builders in the {family} fixture source; \
         the gate cannot derive a member set, so point it at the right file"
    );

    let tabled: Vec<&str> = cases.iter().map(|case| case.name).collect();
    let missing: Vec<&String> = declared
        .iter()
        .filter(|name| !tabled.contains(&name.as_str()))
        .collect();
    assert!(
        missing.is_empty(),
        "Fix: {family} declares fixture builder(s) {missing:?} that its CASES table does not \
         name, so both arms skip them and the construct is proven on neither backend; add each \
         to CASES in the fixture file"
    );

    let unknown: Vec<&&str> = tabled
        .iter()
        .filter(|name| !declared.iter().any(|declared| declared == *name))
        .collect();
    assert!(
        unknown.is_empty(),
        "Fix: {family} CASES names {unknown:?} with no matching `fixture_*` builder in the \
         fixture file, so the table has drifted from the fixtures it indexes"
    );

    let mut sorted = tabled.clone();
    sorted.sort_unstable();
    let before = sorted.len();
    sorted.dedup();
    assert_eq!(
        before,
        sorted.len(),
        "Fix: {family} CASES names the same fixture twice, which double-runs one case and \
         hides a missing one behind an equal case count"
    );
}

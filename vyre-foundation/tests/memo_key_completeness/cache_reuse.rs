use super::*;

/// FIXED, AND THIS IS THE WITNESS THAT IT STAYS FIXED.
/// `ProgramShapeFacts::derive_cached` no longer serves another program's facts.
///
/// Why this exists: this was a live-shaped wrong-reuse defect of exactly the
/// PTX-digest kind, key equal while artifact differs. The artifact reads
/// `decl.shape_predicate()` to compute `min_count`, `max_count`,
/// `is_fixed_count` and `byte_alignment`, and before wire rev 6 the key could
/// not see `shape_predicate` at all, so priming the cache with an unrefined
/// program made it serve `min_count` 4 for a program whose true `min_count` is
/// 64.
///
/// The consumers are production optimizer passes, not test scaffolding:
/// `passes/loops/loop_var_range_fold.rs` folds loop bounds off `min_count` and
/// `max_count`, and `passes/specialization/autotune.rs` picks a workgroup size
/// off the same facts. A wrong loop bound is wrong output with no error raised
/// anywhere.
///
/// It was latent rather than live only because no production code writes any of
/// the three formerly-invisible fields, so the optimizer never varied them
/// within one cache lifetime. That is a property of today's callers, not of the
/// cache, and it would have been silently untrue the first time a frontend
/// emitted shape refinements per specialization. The wire fix removes the
/// dependence on that accident.
///
/// What breaks if this regresses: the optimizer folds a loop bound from a
/// refinement belonging to a DIFFERENT program. Assert the exact served
/// `min_count`, never that the two facts merely differ, because the wrong value
/// (4) is a perfectly plausible fact for a 4-element buffer.
#[test]
fn shape_facts_cache_no_longer_serves_another_programs_facts() {
    let out = Ident::from("out");
    let plain = program_with(vec![out_buf()]);
    let refined = program_with(vec![
        out_buf().with_shape_predicate(ShapePredicate::AtLeast(64))
    ]);

    // The key now separates them. This is the property the whole fix rests on.
    assert_ne!(
        plain.fingerprint(),
        refined.fingerprint(),
        "Fix: the two programs must not share a cache key. shape_predicate is part of the \
         canonical wire bytes as of rev 6; if this fails the encoder has stopped emitting it."
    );

    // The artifact genuinely differs: exact values, computed uncached.
    assert_eq!(
        ProgramShapeFacts::derive(&plain)
            .get(&out)
            .expect("plain fact")
            .min_count,
        4,
        "static count 4, no predicate"
    );
    assert_eq!(
        ProgramShapeFacts::derive(&refined)
            .get(&out)
            .expect("refined fact")
            .min_count,
        64,
        "AtLeast(64) widens the proved lower bound to 64"
    );

    // Prime the cache with `plain`, then ask for `refined`. This is the exact
    // sequence that used to produce the wrong answer.
    let _primed = ProgramShapeFacts::derive_cached(&plain);
    let served = ProgramShapeFacts::derive_cached(&refined);
    assert_eq!(
        served.get(&out).expect("served fact").min_count,
        64,
        "Fix: derive_cached served the WRONG program's facts. It was primed with an unrefined \
         program and asked about a refined one, and returned the primed program's min_count. The \
         cache key must cover shape_predicate."
    );

    // And the reverse order, because a single-slot cache can be wrong in one
    // direction only: priming with the refined program must not leak 64 into
    // the unrefined answer.
    let _reprimed = ProgramShapeFacts::derive_cached(&refined);
    let served_plain = ProgramShapeFacts::derive_cached(&plain);
    assert_eq!(
        served_plain.get(&out).expect("served fact").min_count,
        4,
        "Fix: derive_cached leaked a refined program's min_count into an unrefined program's \
         facts. A proved lower bound that is too HIGH is the dangerous direction: it licenses a \
         loop-bound fold that reads past the real element count."
    );
}

/// FIXED, same gap reached through the second independent cache.
///
/// Why this exists separately from the test above: `FactSubstrate` is a second,
/// independently-keyed cache (three thread-local slots of its own) that EMBEDS
/// `ProgramShapeFacts`. Fixing one cache would not have fixed the other, so
/// both need their own witness, and a future change that re-breaks only one of
/// them must fail loudly. `passes/memory/vectorization.rs` consumes this one and
/// picks vector widths from it, which is a codegen decision: a vector width
/// chosen from another program's proved alignment reads or writes out of bounds.
#[test]
fn fact_substrate_cache_no_longer_serves_another_programs_shape_facts() {
    let out = Ident::from("out");
    let plain = program_with(vec![out_buf()]);
    let refined = program_with(vec![
        out_buf().with_shape_predicate(ShapePredicate::AtLeast(64))
    ]);

    let _primed = FactSubstrate::derive_shape_and_use_cached(&plain);
    let served = FactSubstrate::derive_shape_and_use_cached(&refined);
    let shape = served.shape.as_deref().expect("shape partition populated");

    assert_eq!(
        shape.get(&out).expect("served fact").min_count,
        64,
        "Fix: FactSubstrate served the WRONG program's shape facts."
    );
    // And the uncached derivation on the same program proves the served value
    // is not merely a coincidence of the fixture.
    assert_eq!(
        ProgramShapeFacts::derive(&refined)
            .get(&out)
            .expect("refined fact")
            .min_count,
        64,
        "the truth for the program that was asked about"
    );
    assert_ne!(
        plain.fingerprint(),
        refined.fingerprint(),
        "Fix: the two programs must not share the cache key that selects these facts."
    );
}

/// LATENT CRITICAL, and a SECOND independent route into the same caches that
/// does not involve the three lossy fields at all.
///
/// Why this exists: `ProgramFacts::build_cached` keys on the fingerprint of the
/// CANONICALIZED program but builds its artifact by walking the RAW entry tree.
/// Canonicalization splices `Let`-free `Block`s, so two programs with different
/// raw trees share a key while their fact tables differ in `node_count` and in
/// every `NodeIndex`.
///
/// Proven here with exact values: the served table reports 6 nodes for a
/// 5-node program, and places the `Let` named "x" at `NodeIndex(2)` when it
/// actually sits at `NodeIndex(1)`.
///
/// WHY LATENT AND NOT LIVE: this one is reachable by production passes.
/// `passes/cleanup/empty_block_collapse.rs` removes exactly the `Block(vec![])`
/// that canonicalization also erases, so it produces a rewrite that changes the
/// raw tree while leaving the fingerprint identical, and it reports
/// `changed = true`. What stops it becoming wrong output is that every current
/// consumer of `build_cached` reads only projections that are either
/// canonicalization-invariant or self-consistent within the served table:
/// buffer NAMES (`transform/visit.rs`, `autotune.rs`), buffer USE COUNTS
/// (`validate/linear_type.rs`), `Let` NAMES (`reaching_def_propagate.rs` uses
/// facts for names only and rescans the real tree for values), and Region
/// generators. None of them mixes a `NodeIndex` from the cache with the live
/// tree.
///
/// EXACTLY WHAT WOULD MAKE IT LIVE: one consumer that uses a `NodeIndex` from
/// `build_cached` to index the current program, or that branches on
/// `node_count()` or `kinds_present()`. `optimizer/megakernel/scratch_reuse.rs`
/// is one line away: it already threads `RegionMeta::node` into
/// `is_descendant_of` to decide which buffers a megakernel arm may recycle, and
/// recycling a buffer that is actually live is memory corruption. It is safe
/// today only because both indices come from the same served table.
#[test]
fn program_facts_cache_serves_wrong_node_count_and_indices() {
    let (target, primer) = (indexed_target(), indexed_primer());
    let x = Ident::from("x");

    assert_eq!(target.fingerprint(), primer.fingerprint(), "keys collide");

    // Uncached truth for each program.
    let truth_target = ProgramFacts::build(&target);
    let truth_primer = ProgramFacts::build(&primer);
    assert_eq!(truth_target.node_count(), 5);
    assert_eq!(truth_primer.node_count(), 6);
    assert_eq!(truth_target.let_sites_of(x.as_str()).len(), 1);
    assert_eq!(truth_primer.let_sites_of(x.as_str()).len(), 1);
    let true_x_target = truth_target.let_sites_of(x.as_str())[0];
    let true_x_primer = truth_primer.let_sites_of(x.as_str())[0];
    assert_ne!(
        true_x_target, true_x_primer,
        "the empty Block shifts every later NodeIndex by one"
    );

    // Prime with the primer, then ask for the target.
    let _primed = ProgramFacts::build_cached(&primer);
    let served = ProgramFacts::build_cached(&target);

    assert_eq!(
        served.node_count(),
        6,
        "PROBED WRONG REUSE: build_cached served a 6-node fact table for a 5-node program. \
         Fix: key the table on something that distinguishes raw trees; this must then read 5."
    );
    assert_eq!(
        served.let_sites_of(x.as_str())[0],
        true_x_primer,
        "PROBED WRONG REUSE: the served Let site is the primer's index, not the target's."
    );
    assert_ne!(
        served.let_sites_of(x.as_str())[0],
        true_x_target,
        "PROBED WRONG REUSE: the served NodeIndex for `x` does not match where `x` actually is."
    );
}

/// FIXED. The validation cache key now SEPARATES a valid program from an
/// invalid one.
///
/// Why this exists: `Program::validate()` memoizes success, and its key is a
/// token taken from `Program::fingerprint()`. Before wire rev 6 both
/// `bytes_extraction` (which gates V013) and `shape_predicate` (which the
/// refinement check reads) changed the VERDICT while leaving the key identical,
/// so the cache could admit an invalid program or reject a valid one. A
/// validator serving a stale verdict is nearly as damaging as a miscompile: it
/// either passes IR the backend cannot compile or rejects IR that is fine, and
/// in both directions the diagnostic points at the wrong program.
///
/// What breaks if this regresses: the two fixtures below differ ONLY in a field
/// that decides their verdict. If their keys collide again, a cached PASS can
/// be served for the program that must FAIL. Assert the verdicts with exact
/// error counts and the V013 code, not `is_err`, so a fixture that starts
/// failing for an unrelated reason cannot masquerade as this test passing.
#[test]
fn validation_cache_key_separates_valid_from_invalid_programs() {
    // Route 1: bytes_extraction gates V013 on a DataType::Bytes store.
    let bytes_entry = vec![Node::store("b", Expr::gid_x(), Expr::u32(1))];
    let rejected = Program::wrapped(
        vec![BufferDecl::storage("b", 0, BufferAccess::ReadWrite, DataType::Bytes).with_count(4)],
        [64, 1, 1],
        bytes_entry.clone(),
    );
    let accepted = Program::wrapped(
        vec![
            BufferDecl::storage("b", 0, BufferAccess::ReadWrite, DataType::Bytes)
                .with_count(4)
                .with_bytes_extraction(true),
        ],
        [64, 1, 1],
        bytes_entry,
    );
    let rejected_errors = vyre_foundation::validate::validate(&rejected);
    assert_eq!(
        rejected_errors.len(),
        1,
        "a Bytes store without the opt-in must raise exactly one error"
    );
    assert!(
        rejected_errors[0].message().contains("V013"),
        "the error must be V013, got: {}",
        rejected_errors[0].message()
    );
    assert_eq!(
        vyre_foundation::validate::validate(&accepted).len(),
        0,
        "the same program with the opt-in must be clean"
    );
    assert_ne!(
        rejected.fingerprint(),
        accepted.fingerprint(),
        "Fix: one program is INVALID and one is VALID, and they must not share the validation \
         cache key. If they do, a memoized PASS can be served for the failing program."
    );

    // Route 2: a shape predicate contradicted by the static count.
    let contradicted = program_with(vec![
        out_buf().with_shape_predicate(ShapePredicate::Exactly(9))
    ]);
    let consistent = program_with(vec![out_buf()]);
    assert_eq!(
        vyre_foundation::validate::validate(&contradicted).len(),
        1,
        "count 4 against Exactly(9) must raise exactly one error"
    );
    assert_eq!(vyre_foundation::validate::validate(&consistent).len(), 0);
    assert_ne!(
        contradicted.fingerprint(),
        consistent.fingerprint(),
        "Fix: a valid and a predicate-violating program must not share the validation cache key."
    );
}

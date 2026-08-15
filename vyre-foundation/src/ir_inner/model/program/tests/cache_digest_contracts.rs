//! Contracts for the normalized compiled-pipeline cache digest.
//!
//! Two distinct failure families are locked out here. First, cost: the digest
//! used to be recomputed from scratch on every dispatch, walking the whole node
//! list, which is why it showed up as host overhead. Second, correctness: the
//! digest is a cache KEY, so a missing input silently serves one program's
//! compiled artifact for another, and an extra input silently misses cache.

use super::*;
use crate::ir_inner::model::program::MemoryKind;
use crate::ir_inner::model::spec_types::BufferAccess;
use std::sync::atomic::Ordering;

/// Uncached digest computations performed on THIS thread.
///
/// The counter behind this is thread-local, which is what makes a delta an
/// exact count instead of an upper bound. An earlier process-global counter
/// reported 6 computations for a window containing exactly 1, because `cargo
/// test` runs test functions on parallel threads and sibling tests in this file
/// were computing digests concurrently. Serializing the tests with a mutex
/// would have hidden that, but only for as long as every future test in the
/// binary remembered to take the lock; a thread-local count cannot be polluted
/// by another thread at all.
fn computations() -> usize {
    DIGEST_COMPUTATIONS.with(std::cell::Cell::get)
}

/// Program with `count` distinct body statements, for cost tests where the
/// walked-node-list size is what matters.
fn body_of(count: u32) -> Vec<Node> {
    let mut body = Vec::with_capacity(count as usize);
    for index in 0..count {
        body.push(Node::store(
            "out",
            Expr::u32(index),
            Expr::u32(index * 3 + 1),
        ));
    }
    body
}

fn program_from_body(buffer_count: u32, body: Vec<Node>) -> Program {
    Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32).with_count(buffer_count.max(1))],
        [64, 1, 1],
        body,
    )
}

fn program_with_body(count: u32) -> Program {
    program_from_body(count, body_of(count))
}

fn digest_of(program: &Program) -> [u8; 32] {
    program
        .try_normalized_cache_digest()
        .expect("Fix: fixture program must produce a normalized cache digest")
}

// ---------------------------------------------------------------------------
// Memoization: the cost fix
// ---------------------------------------------------------------------------

/// The digest must be computed EXACTLY once per `Program` value, no matter how
/// many times it is requested.
///
/// This is the whole point of the change. Before it, every dispatch recomputed
/// the digest by walking the entire node list, so a resident program dispatched
/// in a loop paid a full IR traversal per iteration. A count, not a duration,
/// because the count is load-independent and cannot flake under contention.
/// If this regresses, per-dispatch host cost grows with program size again.
#[test]
fn normalized_cache_digest_computes_exactly_once_per_program_value() {
    let program = program_with_body(32);

    let before = computations();
    let first = digest_of(&program);
    for _ in 0..64 {
        assert_eq!(
            digest_of(&program),
            first,
            "Fix: memoized normalized cache digest must be stable across reads."
        );
    }
    let performed = computations() - before;

    assert_eq!(
        performed, 1,
        "Fix: normalized cache digest must be memoized on the Program value; \
         65 reads performed {performed} computations."
    );
}

/// The memoized value must equal what an uncached recompute produces.
///
/// A memo that is populated once and is WRONG passes every test that only reads
/// the cached path, and would corrupt cache identity permanently for that
/// program value. Locks out a future refactor that writes the memo from the
/// wrong source (for example from `fingerprint`, which keys different inputs).
#[test]
fn memoized_digest_equals_uncached_recompute() {
    let program = program_with_body(9);
    let memoized = digest_of(&program);
    let recomputed = program
        .compute_normalized_cache_digest()
        .expect("Fix: uncached recompute must succeed for a valid fixture");

    assert_eq!(
        memoized, recomputed,
        "Fix: memoized normalized cache digest must equal an uncached recompute."
    );
}

/// A `Program` clone must carry the memo, not recompute it.
///
/// Programs are cloned on plan construction and on resident-handle paths. If
/// `Clone` reset the memo, every clone would pay a fresh IR walk and the fix
/// would silently not apply to the paths that clone, which are exactly the hot
/// ones. Asserts a computation count of zero for the clone, plus digest
/// equality so a memo carried across a clone cannot be carrying a stale value
/// for a program whose contents differ.
#[test]
fn clone_carries_normalized_cache_digest_memo() {
    let program = program_with_body(16);
    let original = digest_of(&program);

    let before = computations();
    let cloned = program.clone();
    let cloned_digest = digest_of(&cloned);
    let performed = computations() - before;

    assert_eq!(
        performed, 0,
        "Fix: Program::clone must propagate the normalized cache digest memo; \
         reading the clone performed {performed} computations."
    );
    assert_eq!(
        cloned_digest, original,
        "Fix: a cloned Program must have the same normalized cache digest."
    );
}

/// A cache-invalidating mutation must clear the memo, and the recomputed value
/// must match a program freshly built in the mutated shape.
///
/// A memo that survives mutation is worse than no memo: it serves a compiled
/// artifact built for the pre-mutation program. Checks both halves, because
/// asserting only that the digest changed would also pass if the memo were
/// cleared but recomputed from partially stale state.
#[test]
fn cache_invalidating_mutation_clears_normalized_cache_digest_memo() {
    let mut program = program_with_body(4);
    let before_mutation = digest_of(&program);

    let count_before = computations();
    program.set_workgroup_size([128, 2, 1]);
    let after_mutation = digest_of(&program);
    let performed = computations() - count_before;

    assert_eq!(
        performed, 1,
        "Fix: set_workgroup_size must clear the normalized cache digest memo; \
         the post-mutation read performed {performed} computations."
    );
    assert_ne!(
        before_mutation, after_mutation,
        "Fix: workgroup_size is baked into generated code and must change the cache digest."
    );

    let mut rebuilt = program_with_body(4);
    rebuilt.set_workgroup_size([128, 2, 1]);
    assert_eq!(
        after_mutation,
        digest_of(&rebuilt),
        "Fix: a mutated Program's digest must equal that of an equivalently built Program."
    );
}

/// Mutating the entry body through `entry_mut` must clear the memo, and a
/// mutate-then-revert round trip must restore the original digest exactly.
///
/// `set_workgroup_size` and `entry_mut` reach the memo through the same
/// `invalidate_caches_for` call, but body mutation is the case where a stale
/// digest is most damaging: the shader text changes completely while the cache
/// key does not, so the wrong compiled kernel runs and produces wrong results
/// with no error anywhere.
///
/// The revert is the load-bearing half. The digest is a pure function of the
/// program's current content, so removing the appended node must return the
/// exact original 32 bytes. That fails if the memo is only partially cleared,
/// if a recompute reads any state left over from the previous computation, or
/// if the thread-local scratch buffer is not reset between computations, which
/// an append-only `assert_ne!` would never notice.
#[test]
fn body_mutation_clears_normalized_cache_digest_memo() {
    let extra = Node::store("out", Expr::u32(0), Expr::u32(99));
    let mut program = program_with_body(3);
    let before = digest_of(&program);

    let count_before = computations();
    program.entry_mut().push(extra.clone());
    let after = digest_of(&program);
    let performed = computations() - count_before;

    assert_eq!(
        performed, 1,
        "Fix: entry_mut must clear the normalized cache digest memo; the post-mutation \
         read performed {performed} computations."
    );
    assert_ne!(
        before, after,
        "Fix: appending a node to the entry body must change the normalized cache digest."
    );

    // Revert and require the digest to come all the way back. Also proves the
    // second invalidation cleared the memo, since a surviving memo would return
    // the post-mutation bytes here.
    program
        .entry_mut()
        .pop()
        .expect("Fix: the appended node must be present to revert");
    let reverted = digest_of(&program);
    let performed_total = computations() - count_before;

    assert_eq!(
        reverted, before,
        "Fix: mutate-then-revert must restore the exact original cache digest; the \
         digest must be a pure function of current program content."
    );
    assert_eq!(
        performed_total, 2,
        "Fix: each body mutation must invalidate the memo exactly once; the two reads \
         around two mutations performed {performed_total} computations."
    );
}

// ---------------------------------------------------------------------------
// Discrimination: inputs that MUST be keyed
// ---------------------------------------------------------------------------

/// Distinct programs must get distinct digests.
///
/// The floor requirement for a cache key. Kept as an explicit all-pairs check
/// over programs differing in one keyed dimension each, so a regression that
/// drops any single input fails here rather than silently serving one program's
/// compiled artifact for another.
#[test]
fn distinct_programs_get_distinct_normalized_cache_digests() {
    let base = program_with_body(2);

    let mut different_workgroup = program_with_body(2);
    different_workgroup.set_workgroup_size([32, 1, 1]);

    let different_body = program_with_body(3);

    let different_element = Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::F32).with_count(2)],
        [64, 1, 1],
        vec![
            Node::store("out", Expr::u32(0), Expr::u32(1)),
            Node::store("out", Expr::u32(1), Expr::u32(4)),
        ],
    );

    let different_name = Program::wrapped(
        vec![BufferDecl::output("result", 0, DataType::U32).with_count(2)],
        [64, 1, 1],
        vec![
            Node::store("result", Expr::u32(0), Expr::u32(1)),
            Node::store("result", Expr::u32(1), Expr::u32(4)),
        ],
    );

    let different_op_id = program_with_body(2).with_entry_op_id("op-7");

    let labelled = [
        ("base", digest_of(&base)),
        ("workgroup", digest_of(&different_workgroup)),
        ("body", digest_of(&different_body)),
        ("element", digest_of(&different_element)),
        ("name", digest_of(&different_name)),
        ("op_id", digest_of(&different_op_id)),
    ];

    for (index, (left_name, left)) in labelled.iter().enumerate() {
        for (right_name, right) in labelled.iter().skip(index + 1) {
            assert_ne!(
                left, right,
                "Fix: programs differing in {left_name} vs {right_name} must not share \
                 a compiled-pipeline cache digest."
            );
        }
    }
}

/// Buffer `binding` must be keyed unconditionally.
///
/// The two programs here are identical except that the two buffers swap binding
/// slots. Descriptor lowering reads `binding` into the descriptor and the
/// generated bind-group layout, so the emitted artifacts genuinely differ.
/// Before v3 the digest omitted `binding` entirely, so a compiled-pipeline
/// cache would serve the shader compiled for the other layout: writes land in
/// the wrong buffer, with no error.
#[test]
fn normalized_cache_digest_separates_buffer_binding_layouts() {
    let body = vec![
        Node::store("a", Expr::u32(0), Expr::u32(1)),
        Node::store("b", Expr::u32(0), Expr::u32(2)),
    ];
    let straight = Program::wrapped(
        vec![
            BufferDecl::output("a", 0, DataType::U32).with_count(1),
            BufferDecl::output("b", 1, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        body.clone(),
    );
    let swapped = Program::wrapped(
        vec![
            BufferDecl::output("a", 1, DataType::U32).with_count(1),
            BufferDecl::output("b", 0, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        body,
    );

    assert_ne!(
        digest_of(&straight),
        digest_of(&swapped),
        "Fix: buffer binding indices reach the generated bind-group layout and must be \
         part of the compiled-pipeline cache digest."
    );
}

/// A static workgroup array length must be keyed.
///
/// `MemoryKind::Shared` is the one class whose `element_count` an emitter bakes
/// into the shader text, as a fixed-length workgroup array. Two programs
/// differing only in N compile to different shaders. Before v3 the digest
/// omitted `count` for every class, so a compiled-pipeline cache returned the
/// shader built for the other N: the kernel then indexes past its workgroup
/// array.
#[test]
fn normalized_cache_digest_separates_workgroup_static_array_lengths() {
    let build = |shared_len: u32| {
        Program::wrapped(
            vec![
                BufferDecl::output("out", 0, DataType::U32).with_count(1),
                BufferDecl::workgroup("tile", shared_len, DataType::U32),
            ],
            [64, 1, 1],
            vec![Node::store("out", Expr::u32(0), Expr::u32(1))],
        )
    };

    assert_ne!(
        digest_of(&build(64)),
        digest_of(&build(128)),
        "Fix: a static workgroup array length is baked into shader text and must be \
         part of the compiled-pipeline cache digest."
    );
}

/// An interior NUL in a buffer name must not let one program impersonate
/// another.
///
/// The digest is a flat byte string with NUL-delimited sections. Buffer names
/// are attacker-adjacent in that they come from user IR, so a name containing
/// the delimiter is the classic way to shift the parse and collide two distinct
/// programs. Locks in the length prefix on every variable-length field.
#[test]
fn normalized_cache_digest_resists_delimiter_injection_in_buffer_names() {
    let split = Program::wrapped(
        vec![
            BufferDecl::output("a\0b", 0, DataType::U32).with_count(1),
            BufferDecl::output("c", 1, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        vec![Node::store("c", Expr::u32(0), Expr::u32(1))],
    );
    let joined = Program::wrapped(
        vec![
            BufferDecl::output("a", 0, DataType::U32).with_count(1),
            BufferDecl::output("b\0c", 1, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        vec![Node::store("a", Expr::u32(0), Expr::u32(1))],
    );

    assert_ne!(
        digest_of(&split),
        digest_of(&joined),
        "Fix: variable-length cache-digest fields must be length-prefixed so a name \
         containing the field delimiter cannot impersonate a different program."
    );
}

/// An interior NUL in the entry op id must not alias either.
///
/// Same injection shape as the buffer-name case but on a different field, and
/// worth its own test because the op id has a distinct encoding path (an
/// `Option` whose `None` arm writes four zero bytes). A four-byte length prefix
/// and a four-byte `None` marker are only unambiguous together.
#[test]
fn normalized_cache_digest_resists_delimiter_injection_in_entry_op_id() {
    let injected = program_with_body(1).with_entry_op_id("x\0bufs\0");
    let plain = program_with_body(1).with_entry_op_id("x");
    let absent = program_with_body(1);

    let digests = [digest_of(&injected), digest_of(&plain), digest_of(&absent)];
    assert_ne!(
        digests[0], digests[1],
        "Fix: entry op ids differing after an interior NUL must produce distinct digests."
    );
    assert_ne!(
        digests[0], digests[2],
        "Fix: an entry op id must never collide with the absent-op-id encoding."
    );
    assert_ne!(
        digests[1], digests[2],
        "Fix: Some(op) and None entry op ids must produce distinct digests."
    );
}

// ---------------------------------------------------------------------------
// Erasure: inputs that MUST NOT be keyed
// ---------------------------------------------------------------------------

/// A runtime storage buffer's `element_count` must NOT be keyed.
///
/// Storage and uniform buffer lengths are erased in the shader, so resizing an
/// input must reuse the compiled shader. If this regresses, a compiled-pipeline
/// cache misses on every new input size and each miss is a full shader compile,
/// which is orders of magnitude more expensive than the dispatch it precedes.
/// This is the invariant that makes the count field conditional rather than
/// unconditional.
#[test]
fn normalized_cache_digest_erases_runtime_storage_lengths() {
    let build = |count: u32| {
        Program::wrapped(
            vec![BufferDecl::output("out", 0, DataType::U32).with_count(count)],
            [64, 1, 1],
            vec![Node::store("out", Expr::u32(0), Expr::u32(1))],
        )
    };

    assert_eq!(
        digest_of(&build(1024)),
        digest_of(&build(1_048_576)),
        "Fix: runtime storage buffer lengths are erased in generated shader text and must \
         stay out of the cache digest, or every resize forces a recompile."
    );
}

/// `has_static_element_count` must agree with descriptor lowering's memory
/// classification across the entire `MemoryKind` x `BufferAccess` product.
///
/// The predicate is a hand-mirror of `vyre-lower`'s `memory_class` arms, and a
/// hand-mirror drifts. Enumerating the full product means adding a
/// `MemoryKind` variant fails this test instead of silently defaulting the new
/// variant to erased, which would be a cache collision, or to keyed, which
/// would be a recompile storm. `Persistent` is excluded from lowering by an
/// earlier error, so it must report false rather than guess.
#[test]
fn static_element_count_predicate_matches_lowering_memory_classes() {
    const KINDS: [MemoryKind; 7] = [
        MemoryKind::Global,
        MemoryKind::Shared,
        MemoryKind::Uniform,
        MemoryKind::Local,
        MemoryKind::Readonly,
        MemoryKind::Persistent,
        MemoryKind::Push,
    ];
    // Expected values are a hand-written FIXTURE, not a re-derivation of the
    // predicate, so this test disagrees with the implementation if either side
    // changes. Columns follow ACCESS_ORDER.
    const ACCESS_ORDER: [BufferAccess; 5] = [
        BufferAccess::ReadOnly,
        BufferAccess::ReadWrite,
        BufferAccess::WriteOnly,
        BufferAccess::Uniform,
        BufferAccess::Workgroup,
    ];
    const EXPECTED: [(MemoryKind, [bool; 5]); 7] = [
        // Host-visible classes: only an explicit Workgroup access makes the
        // count static, because that is what lowering calls Scratch.
        (MemoryKind::Global, [false, false, false, false, true]),
        (MemoryKind::Uniform, [false, false, false, false, true]),
        (MemoryKind::Readonly, [false, false, false, false, true]),
        (MemoryKind::Push, [false, false, false, false, true]),
        // Shared and Local ARE the static-array classes regardless of access.
        (MemoryKind::Shared, [true, true, true, true, true]),
        (MemoryKind::Local, [true, true, true, true, true]),
        // Precedence boundary: Persistent must win over the Workgroup access
        // arm. Lowering rejects Persistent before it ever classifies memory, so
        // a Persistent buffer has no emitted array whose length could be baked,
        // and the Workgroup column here must stay false. Reordering the arms in
        // the predicate flips exactly this one cell.
        (MemoryKind::Persistent, [false, false, false, false, false]),
    ];

    // The fixture must cover every MemoryKind exactly once. Without this, adding
    // a variant and forgetting a row would silently shrink the product under
    // test rather than fail, which is the drift this test exists to catch.
    for kind in KINDS {
        assert_eq!(
            EXPECTED.iter().filter(|(row, _)| *row == kind).count(),
            1,
            "Fix: the has_static_element_count fixture must cover MemoryKind {} exactly \
             once; an uncovered kind is an untested compiled-pipeline cache-key decision.",
            memory_kind_label(kind)
        );
    }

    for (kind, expected_row) in EXPECTED {
        for (access, expected) in ACCESS_ORDER.into_iter().zip(expected_row) {
            let buffer = BufferDecl::storage("b", 0, access.clone(), DataType::U32)
                .with_kind(kind)
                .with_count(8);

            assert_eq!(
                buffer.has_static_element_count(),
                expected,
                "Fix: has_static_element_count disagrees with lowering's memory class for \
                 kind {} access {}; the cache digest keys count from this predicate, so \
                 drift is a cache collision on one side or a recompile storm on the other.",
                memory_kind_label(kind),
                access_label(&access)
            );
        }
    }
}

/// Computing the digest must not read structural validation state.
///
/// Verified by side effect rather than by value, which makes the test immune to
/// a future digest that happens to hash the flag to the same bytes.
/// `is_structurally_validated()` re-derives a canonical wire encoding to check
/// its recorded token and CLEARS the flag on mismatch. So: validate, then write
/// a public IR field directly to desynchronize the token, then compute the
/// digest. If the digest consulted the flag, the flag would now be false.
///
/// Two things regress if this fails. Cost: the v2 digest paid a full canonical
/// re-encode on every validated program, which is far more expensive than the
/// digest itself. Correctness: validation state is mutable atomic state that
/// provably does not change generated code, so keying it splits the cache into
/// validated and unvalidated halves for identical artifacts.
#[test]
fn normalized_cache_digest_does_not_read_structural_validation_state() {
    let mut program = program_with_body(2);
    program
        .validate()
        .expect("Fix: fixture must pass structural validation");
    assert!(
        program.structural_validated.load(Ordering::Acquire),
        "Fix: precondition, the fixture must be marked structurally validated"
    );

    // Desynchronize the recorded token WITHOUT going through a mutator, so the
    // flag is still set but would be cleared by any is_structurally_validated
    // call.
    program.workgroup_size = [8, 1, 1];

    let _ = program.compute_normalized_cache_digest();

    assert!(
        program.structural_validated.load(Ordering::Acquire),
        "Fix: computing the normalized cache digest must not call \
         is_structurally_validated; doing so pays a canonical wire re-encode per \
         computation and keys mutable state that does not affect generated code."
    );
}

/// Validated and unvalidated programs with identical IR must share a digest.
///
/// The value-level twin of the side-effect test above, and the one that states
/// the intent directly: validation is a host-side assertion about a program, not
/// a property of the artifact compiled from it, so it must not partition the
/// compiled-pipeline cache.
#[test]
fn normalized_cache_digest_ignores_whether_program_was_validated() {
    let unvalidated = program_with_body(5);
    let validated = program_with_body(5);
    validated
        .validate()
        .expect("Fix: fixture must pass structural validation");

    assert_eq!(
        digest_of(&unvalidated),
        digest_of(&validated),
        "Fix: structural validation state must not change the compiled-pipeline cache digest."
    );
}

/// Pin the digest's exact byte framing, including the version label.
///
/// This replaces a test that LEAKED. The previous version asserted the digest
/// differed from `blake3("\0wg\0" + workgroup bytes)`, which is a hash of a
/// PREFIX of the real stream, so it stayed green with the version label removed
/// from production entirely. A negative control proved that: injecting the
/// defect left the old test passing. It was a check with no information behind
/// it, which is the exact failure it was supposed to guard against.
///
/// The structure here is deliberate and has two halves that must stay together:
///
/// 1. EQUIVALENCE. `expected` is a mirror of `compute_normalized_cache_digest`,
///    rebuilt field by field, and the first assertion requires it to reproduce
///    the shipped digest byte for byte. A mirror that has drifted from
///    production fails here, so the second half cannot be proving something
///    about a copy that no longer resembles the real encoder.
/// 2. PERTURBATION. The second assertion strips exactly the version prefix from
///    that proven-equivalent stream and requires the shipped digest to differ.
///    A label that is declared but never hashed makes this fail, which is the
///    defect that makes every future version bump a silent no-op.
///
/// Because the mirror is byte-exact, this also pins the section tags
/// (`\0wg\0`, `\0op\0`, `\0bufs\0`, `\0body\0`), their ORDER, the little-endian
/// axis encoding, and the four-zero-byte encoding of an absent `entry_op_id`.
/// Any of those changing without a version bump would silently alias cache
/// entries written under the old framing.
#[test]
fn normalized_cache_digest_pins_exact_byte_framing_and_hashes_the_version_label() {
    let version = crate::ir_inner::model::program::NORMALIZED_PROGRAM_CACHE_DIGEST_VERSION;
    assert_eq!(
        version, "vyre-pipeline-cache-norm-v3",
        "Fix: the normalized cache digest keyed input set changed in v3; the version label \
         must be bumped with it so pre-v3 cache entries cannot be served."
    );

    // No buffers, so the buffer lane is empty and the framing is fully visible.
    let program = Program::wrapped(vec![], [64, 1, 1], vec![]);

    let mut expected = Vec::new();
    expected.extend_from_slice(version.as_bytes());
    expected.extend_from_slice(b"\0wg\0");
    for axis in [64u32, 1, 1] {
        expected.extend_from_slice(&axis.to_le_bytes());
    }
    expected.extend_from_slice(b"\0op\0");
    expected.extend_from_slice(&[0u8; 4]);
    expected.extend_from_slice(b"\0bufs\0");
    expected.extend_from_slice(b"\0body\0");
    crate::serial::wire::append_node_list_fingerprint(&mut expected, program.entry())
        .expect("Fix: fixture entry body must fingerprint");

    assert_eq!(
        digest_of(&program),
        *blake3::hash(&expected).as_bytes(),
        "Fix: the normalized cache digest byte framing changed. If that was deliberate, \
         bump NORMALIZED_PROGRAM_CACHE_DIGEST_VERSION in the same patch so entries written \
         under the old framing cannot be served, then update this expected stream."
    );

    let without_version = expected[version.len()..].to_vec();
    assert_ne!(
        digest_of(&program),
        *blake3::hash(&without_version).as_bytes(),
        "Fix: the version label must be HASHED, not merely declared; otherwise every future \
         version bump is a silent no-op and pre-bump cache entries stay servable."
    );
}

/// Wildcard-free `MemoryKind` name, so adding a variant fails to COMPILE here.
///
/// A `{:?}` format would keep compiling and let a new variant slip into the
/// digest with whatever `has_static_element_count` happens to return for it.
/// That default is a coin flip between a cache collision and a recompile storm,
/// so the decision must be made deliberately, and this match is what forces it.
fn memory_kind_label(kind: MemoryKind) -> &'static str {
    match kind {
        MemoryKind::Global => "Global",
        MemoryKind::Shared => "Shared",
        MemoryKind::Uniform => "Uniform",
        MemoryKind::Local => "Local",
        MemoryKind::Readonly => "Readonly",
        MemoryKind::Persistent => "Persistent",
        MemoryKind::Push => "Push",
    }
}

/// `BufferAccess` name for assertion messages.
///
/// Unlike [`memory_kind_label`] this CANNOT force a compile error on a new
/// variant: `vyre_spec::BufferAccess` is `#[non_exhaustive]`, so the wildcard
/// arm is mandatory and a variant added upstream would silently fall through
/// it. That is why the arm returns a loud marker and
/// `access_order_covers_every_known_buffer_access` asserts no tested access
/// reaches it. A new upstream access mode must be added to `ACCESS_ORDER` and
/// to every `EXPECTED` row by hand, because it changes which buffers lowering
/// classifies as Scratch and therefore changes the cache key.
fn access_label(access: &BufferAccess) -> &'static str {
    match access {
        BufferAccess::ReadOnly => "ReadOnly",
        BufferAccess::ReadWrite => "ReadWrite",
        BufferAccess::WriteOnly => "WriteOnly",
        BufferAccess::Uniform => "Uniform",
        BufferAccess::Workgroup => "Workgroup",
        _ => UNKNOWN_ACCESS,
    }
}

/// Marker returned for a `BufferAccess` variant this file does not know about.
const UNKNOWN_ACCESS: &str = "UNKNOWN-ACCESS";

/// Every `BufferAccess` variant the drift fixture tests must be one this file
/// recognizes, and the five must be distinct.
///
/// The compiler cannot enforce coverage of a `#[non_exhaustive]` enum, so this
/// is the runtime substitute. It fails if someone adds a variant to
/// `ACCESS_ORDER` without adding it to `access_label`, which would otherwise
/// produce a drift-test failure message naming "UNKNOWN-ACCESS" and send the
/// next reader hunting the wrong bug.
#[test]
fn access_order_covers_every_known_buffer_access() {
    let labels: Vec<&str> = [
        BufferAccess::ReadOnly,
        BufferAccess::ReadWrite,
        BufferAccess::WriteOnly,
        BufferAccess::Uniform,
        BufferAccess::Workgroup,
    ]
    .iter()
    .map(access_label)
    .collect();

    assert_eq!(
        labels,
        vec!["ReadOnly", "ReadWrite", "WriteOnly", "Uniform", "Workgroup"],
        "Fix: access_label must name every BufferAccess variant the cache-digest drift \
         fixture exercises."
    );
    assert!(
        !labels.contains(&UNKNOWN_ACCESS),
        "Fix: a BufferAccess variant fell through to the wildcard arm; add it to \
         access_label, to ACCESS_ORDER, and to every EXPECTED row in the \
         has_static_element_count drift fixture."
    );
}

/// Warm every memoized field on `program`, in the order that makes each
/// `OnceLock` observable afterwards.
///
/// `fingerprint()` deliberately warms TWO cells: meta.rs sets `hash` from the
/// same wire hash it fingerprints, so a caller cannot warm one without the
/// other. That coupling is why the propagation gate below checks both.
fn warm_every_memo(program: &Program) {
    let _ = program.fingerprint();
    let _ = digest_of(program);
    let _ = program.output_buffer_indices();
    let _ = program.has_indirect_dispatch();
    let _ = program.stats();
}

/// Names of the six memo cells, paired with a probe reporting whether each is
/// warm on a given program. Keeping them in one table means a newly added
/// `OnceLock` on `Program` is a one-line change here rather than a silently
/// uncovered field.
fn memo_warmth(program: &Program) -> Vec<(&'static str, bool)> {
    vec![
        ("hash", program.hash.get().is_some()),
        ("fingerprint", program.fingerprint.get().is_some()),
        (
            "normalized_cache_digest",
            program.normalized_cache_digest.get().is_some(),
        ),
        (
            "output_buffer_index",
            program.output_buffer_index.get().is_some(),
        ),
        (
            "has_indirect_dispatch",
            program.has_indirect_dispatch.get().is_some(),
        ),
        ("stats", program.stats.get().is_some()),
    ]
}

/// `Program::clone` must carry EVERY warm memo to the clone.
///
/// Why this exists: `impl Clone for Program` initialises all six `OnceLock`
/// fields with `OnceLock::new()` in its struct literal, because a populated
/// `OnceLock` cannot be constructed inline, and only then copies each warm
/// value across with `if let Some(..)`. Reading only the literal makes the
/// clone look like it DROPS every memo. Three separate readers reached exactly
/// that conclusion in one session and one of them nearly landed a
/// "propagate the memo" change that was already there.
///
/// The exact bug this locks out: deleting or short-circuiting any of those six
/// copies. That is pure cost, never a wrong answer, so it is invisible to every
/// value-based test in the tree: a dropped memo is recomputed and the recomputed
/// value is identical. `fingerprint_matches_across_clone_when_canonical_wire_encode_rejects_workgroup`
/// in call_collection_contracts.rs asserts fingerprint EQUALITY across a clone
/// and passes with the propagation fully deleted, which is why equality is not
/// enough and warmth has to be observed directly.
///
/// If this regresses, every clone of a Program re-hashes the whole node list on
/// its next cache lookup. `Program::clone` is on the dispatch path (a driver's
/// capability lowering clones the program per dispatch), so the cost returns
/// per dispatch with all gates green.
#[test]
fn clone_carries_every_warm_memo() {
    let program = program_with_body(16);
    warm_every_memo(&program);

    let cold: Vec<&str> = memo_warmth(&program)
        .into_iter()
        .filter(|(_, warm)| !warm)
        .map(|(name, _)| name)
        .collect();
    assert!(
        cold.is_empty(),
        "Fix: warm_every_memo must warm all six cells before the propagation \
         assertion can mean anything; these stayed cold: {cold:?}"
    );

    let clone = program.clone();

    let dropped: Vec<&str> = memo_warmth(&clone)
        .into_iter()
        .filter(|(_, warm)| !warm)
        .map(|(name, _)| name)
        .collect();
    assert_eq!(
        dropped,
        Vec::<&str>::new(),
        "Fix: Program::clone must copy every warm memo to the clone; these were \
         dropped and will be recomputed on the clone's next cache lookup."
    );
}

/// A clone must propagate memo VALUES field-for-field, never a crossed pair.
///
/// Why this exists: `fingerprint` and `normalized_cache_digest` are both
/// `OnceLock<[u8; 32]>`, so `cloned.fingerprint.set(*normalized_cache_digest)`
/// compiles cleanly. The six copies in `impl Clone` are near-identical
/// copy-paste blocks, which is exactly where a crossed field survives review,
/// and the type checker cannot catch this pair.
///
/// The exact bug this locks out: swapping those two values during a clone. The
/// result is catastrophic and silent, because the fingerprint keys the
/// validation cache and four optimizer passes while the digest keys compiled
/// pipelines, so a crossed clone serves one cache's answer under the other's
/// key. Asserting the two differ first is what gives the crossing something to
/// be caught by; if a future fixture made them equal, the assertion below would
/// pass vacuously and this comment is the warning.
#[test]
fn clone_propagates_each_memo_value_without_crossing_fields() {
    let program = program_with_body(12);
    let fingerprint = program.fingerprint();
    let digest = digest_of(&program);
    assert_ne!(
        fingerprint, digest,
        "Fix: this control needs the fingerprint and the cache digest to differ, \
         otherwise a crossed-field clone is undetectable. Both are [u8; 32] over \
         different inputs, so equality means the fixture stopped discriminating."
    );

    let clone = program.clone();

    assert_eq!(
        clone.fingerprint.get().copied(),
        Some(fingerprint),
        "Fix: the clone's fingerprint memo must hold the original's fingerprint, \
         not another cell's 32 bytes."
    );
    assert_eq!(
        clone.normalized_cache_digest.get().copied(),
        Some(digest),
        "Fix: the clone's cache-digest memo must hold the original's digest, not \
         another cell's 32 bytes."
    );
    assert_eq!(
        clone.hash.get().map(|hash| *hash.as_bytes()),
        Some(fingerprint),
        "Fix: hash and fingerprint are derived from the same wire hash, so a \
         clone must carry them consistently."
    );
}

/// Cloning a cold program must not warm anything.
///
/// Why this exists: it is the boundary case of the propagation gate above, and
/// it fails on the obvious over-correction. Someone told "the clone loses the
/// memo, propagate it" can satisfy that by computing the digest inside
/// `Program::clone`, which makes the propagation gate pass while adding a full
/// program walk to every clone, including the many throwaway clones that
/// fixpoint iteration performs and never queries. `PassResult::from_programs`
/// clones a Program purely to compare it against itself.
///
/// The exact bug this locks out: eager computation in `Clone`. If it regresses,
/// clone cost becomes proportional to program size for every clone rather than
/// only the queried ones.
#[test]
fn cloning_a_cold_program_warms_no_memo() {
    let program = program_with_body(8);
    let before = computations();

    let clone = program.clone();

    assert_eq!(
        computations(),
        before,
        "Fix: Program::clone must not compute the normalized cache digest; a \
         cold program's clone stays cold until something asks for the digest."
    );
    let warm: Vec<&str> = memo_warmth(&clone)
        .into_iter()
        .filter(|(_, warm)| *warm)
        .map(|(name, _)| name)
        .collect();
    assert_eq!(
        warm,
        Vec::<&str>::new(),
        "Fix: cloning a cold Program must leave every memo cold; eager \
         computation in Clone charges every throwaway clone for a walk nobody \
         asked for."
    );
}

/// Every supported in-place mutation must clear ALL six memos, not just the
/// digest.
///
/// Why this exists: propagating memos across a clone is safe ONLY IF every
/// mutation path invalidates them. The two invariants are a pair, and testing
/// propagation without testing invalidation converts a wasted-work bug into a
/// wrong-reuse bug. `hash` and `fingerprint` matter most here: the fingerprint
/// keys `ProgramShapeFacts::derive_cached`, `FactCache`, the validation
/// cache, and the `fingerprint_program(&optimized) != before` comparison in
/// four optimizer passes, so a stale fingerprint makes a mutated program claim
/// it did not change.
///
/// The exact bug this locks out: adding a mutating method, or reordering an
/// existing one, so that it skips `invalidate_caches_for`. Note that
/// `entry_mut` invalidates BEFORE `Arc::make_mut`; either order clears the
/// cells, so this gate pins the observable postcondition rather than the
/// sequence.
///
/// Not covered on purpose: `Program` exposes `buffers`, `workgroup_size`,
/// `entry`, `entry_op_id` and `non_composable_with_self` as PUBLIC fields, so a
/// holder of `&mut Program` can assign them directly and leave all three
/// digests stale. No production path in the workspace does this (every direct
/// assignment found is in tests, which exploit it deliberately, as
/// validation_cache_contracts.rs does to prove the validation cache compares
/// against current bytes instead of trusting a flag). Asserting the stale
/// result here would enshrine the hazard as intended behaviour, so this gate
/// covers the supported API and the hazard is recorded on the fields.
#[test]
fn every_supported_mutation_clears_every_memo() {
    let mutations: Vec<(&str, fn(&mut Program))> = vec![
        ("set_workgroup_size", |program| {
            program.set_workgroup_size([8, 1, 1]);
        }),
        ("set_parallel_region_size", |program| {
            program.set_parallel_region_size([16, 1, 1]);
        }),
        ("entry_mut", |program| {
            program
                .entry_mut()
                .push(Node::store("out", Expr::u32(0), Expr::u32(7)));
        }),
        ("mark_unknown_mutation_provenance", |program| {
            program.mark_unknown_mutation_provenance();
        }),
    ];

    for (name, mutate) in mutations {
        let mut program = program_with_body(6);
        warm_every_memo(&program);

        mutate(&mut program);

        let stale: Vec<&str> = memo_warmth(&program)
            .into_iter()
            .filter(|(_, warm)| *warm)
            .map(|(cell, _)| cell)
            .collect();
        assert_eq!(
            stale,
            Vec::<&str>::new(),
            "Fix: `{name}` must route through invalidate_caches_for so every \
             memo is dropped; these cells survived the mutation and now describe \
             the pre-mutation program."
        );
    }
}

/// A mutation followed by an exact revert must restore the original digest AND
/// the original fingerprint.
///
/// Why this exists: `every_supported_mutation_clears_every_memo` proves the
/// cells go cold, which a method could satisfy while corrupting the value
/// recomputed afterwards. This closes that gap by round-tripping to a known
/// state and demanding the original 32 bytes back, for both digests, so
/// invalidation is proven to restore rather than merely to clear.
///
/// The exact bug this locks out: an invalidation path that clears the cells but
/// leaves the program in a state whose recomputed identity differs from the
/// original, which would make cache keys depend on mutation HISTORY rather than
/// on the current program value.
#[test]
fn mutate_then_revert_restores_both_identities() {
    let mut program = program_with_body(5);
    let fingerprint = program.fingerprint();
    let digest = digest_of(&program);

    program.set_workgroup_size([8, 1, 1]);
    let mutated_digest = digest_of(&program);
    assert_ne!(
        mutated_digest, digest,
        "Fix: this control needs the mutation to actually change the digest, \
         otherwise the revert below proves nothing."
    );

    program.set_workgroup_size([64, 1, 1]);

    assert_eq!(
        program.fingerprint(),
        fingerprint,
        "Fix: reverting a mutation must restore the exact original fingerprint; \
         program identity must be a function of the current value, not of \
         mutation history."
    );
    assert_eq!(
        digest_of(&program),
        digest,
        "Fix: reverting a mutation must restore the exact original cache digest."
    );
}

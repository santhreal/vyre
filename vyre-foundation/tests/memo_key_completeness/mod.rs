//! Completeness probe for every `(key, artifact)` reuse pair keyed on
//! `Program::fingerprint()`, plus the wire-layer defect that causes the gaps.
//!
//! # Why this file exists
//!
//! A peer found a silent miscompile in the backend PTX cache by tabulating,
//! for six program pairs, whether the generated artifact differed against
//! whether the cache key differed. Two rows came back "artifact differs, key
//! identical", which is the shape that serves code compiled for a different
//! program. Every pre-existing test asked "does this program compile
//! correctly" and none asked "do these two programs share a key they must
//! not", so the defect survived for a long time.
//!
//! This file applies that tabulation to `Program::fingerprint()`, which is the
//! cache key for six other memoized artifacts in this crate. Each test states
//! whether it demonstrates a LIVE wrong reuse (production reaches it today),
//! a LATENT one (the cache provably returns another program's artifact, but no
//! current consumer turns that into a wrong result), or a PROVEN-ABSENT result.
//! That live-versus-latent distinction is load-bearing: a probe can be correct
//! about a key gap and still be wrong about severity, and severity is what
//! decides whether something blocks a release.
//!
//! # Root cause, established by probe not by reading
//!
//! `Program::fingerprint()` is BLAKE3 over the canonical VIR0 wire bytes, so
//! it inherits every gap in the wire encoding. The wire encoding is LOSSY:
//! `to_wire` then `from_wire` silently drops three `BufferDecl` fields.
//! Everything else in this file is a consequence of that one defect, which is
//! why the fix belongs at the wire layer and not in six cache keys.

use std::ops::Range;
use std::sync::Arc;

use vyre_foundation::ir::{
    BufferAccess, BufferDecl, CacheLocality, DataType, Expr, Ident, LinearType, MemoryHints,
    MemoryKind, Node, Program, ShapePredicate,
};
use vyre_foundation::optimizer::fact_cache::FactCache;
use vyre_foundation::optimizer::program_shape_facts::ProgramShapeFacts;
use vyre_foundation::optimizer::program_soa::ProgramFacts;

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

fn out_buf() -> BufferDecl {
    BufferDecl::storage("out", 0, BufferAccess::ReadWrite, DataType::U32).with_count(4)
}

fn one_store() -> Vec<Node> {
    vec![Node::store("out", Expr::gid_x(), Expr::u32(1))]
}

fn program_with(buffers: Vec<BufferDecl>) -> Program {
    Program::wrapped(buffers, [64, 1, 1], one_store())
}

/// Two Lets and two Stores. Reused as the "target" of a cache-poisoning probe.
fn indexed_target() -> Program {
    Program::wrapped(
        vec![out_buf()],
        [64, 1, 1],
        vec![
            Node::let_bind("x", Expr::u32(7)),
            Node::store("out", Expr::u32(0), Expr::var("x")),
            Node::let_bind("y", Expr::u32(9)),
            Node::store("out", Expr::u32(1), Expr::var("y")),
        ],
    )
}

/// [`indexed_target`] plus a leading `Node::Block(vec![])`.
///
/// Canonicalization splices any `Block` that owns no `Let` bindings, and an
/// empty `Block` trivially owns none, so this node is erased before hashing.
/// The two programs therefore share a fingerprint while every `NodeIndex`
/// after position zero differs by exactly one.
fn indexed_primer() -> Program {
    Program::wrapped(
        vec![out_buf()],
        [64, 1, 1],
        vec![
            Node::Block(Vec::new()),
            Node::let_bind("x", Expr::u32(7)),
            Node::store("out", Expr::u32(0), Expr::var("x")),
            Node::let_bind("y", Expr::u32(9)),
            Node::store("out", Expr::u32(1), Expr::var("y")),
        ],
    )
}

// ---------------------------------------------------------------------------
// 1. Compile-time completeness guard
// ---------------------------------------------------------------------------

/// Exhaustive `BufferDecl` struct literal, so a NEW FIELD BREAKS COMPILATION.
///
/// Why this exists: the tabulation below partitions `BufferDecl`'s fields into
/// "the fingerprint can see it" and "the fingerprint is blind to it". A
/// partition is only trustworthy if it is exhaustive, and a hand-written list
/// of field names silently stops being exhaustive the day someone adds a
/// field. A struct literal must name every field, so this function stops
/// compiling instead, forcing the author to decide which side of the partition
/// their new field belongs on and to extend
/// [`wire_round_trip_drops_exactly_three_bufferdecl_fields`].
///
/// What breaks if this regresses: a field added without a decision inherits
/// "the fingerprint cannot see it" by default, which is the miscompile side,
/// and no test notices. That is precisely how the PTX digest lost `binding`.
fn exhaustive_buffer_decl() -> BufferDecl {
    BufferDecl {
        name: Arc::from("out"),
        binding: 0,
        access: BufferAccess::ReadWrite,
        kind: MemoryKind::Global,
        element: DataType::U32,
        count: 4,
        is_output: false,
        pipeline_live_out: false,
        output_byte_range: None,
        hints: MemoryHints {
            coalesce_axis: None,
            preferred_alignment: 0,
            cache_locality: CacheLocality::Temporal,
        },
        bytes_extraction: false,
        linear_type: LinearType::Unrestricted,
        shape_predicate: None,
    }
}

mod cache_reuse;
mod proven_absent;
mod wire_contracts;

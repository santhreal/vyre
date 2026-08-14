//! The ONE packed-bitset addressing skeleton every graph kernel body uses.
//!
//! `bitset::bitset_words` owns the HOST-side word count. This module owns its
//! device-side twin: given a bit index as an [`Expr`], the word that holds it and
//! the single-bit mask that selects it, plus the two things a graph kernel ever
//! does with that pair. Probe the bit and run a body when it is set; or set the
//! bit with an atomic OR and run a body only when this lane flipped it 0 to 1.
//!
//! It lives at `graph/` level, a peer of [`crate::graph::edge_scan`], for the
//! same reason that module gives: every consumer subsystem is a sibling, so
//! parking the skeleton inside one of them forces the others to reach across.
//!
//! The skeleton was hand-written more than twenty times across `graph/`, in
//! bodies whose surrounding logic genuinely differs (forward step, backward step,
//! dominance frontier, SCC stamping, queue compaction, tensor lanes, adaptive
//! dense/sparse steps). Every copy was byte-identical in the part that addresses
//! the bitset, and near-duplicate bodies are where the drift in this directory
//! has hidden: a missing `.max(1)`, an unfloored word count, a binding order that
//! disagreed with its own module's documented order.
//!
//! # Names are part of the ABI
//!
//! The emitted IR is compared byte for byte by the graph oracle and fixpoint
//! parity matrices, and `vyre-foundation`'s validator forbids a reused binding
//! name inside one kernel. So the local binding names are caller-supplied
//! ([`BitAccess`]) rather than fixed here: a caller keeps the names it already
//! emitted, and a caller that inlines the skeleton twice passes distinct ones.

use vyre_foundation::ir::{Expr, Node};

/// The local bindings one packed-bitset access introduces.
///
/// `word` holds the word index, `mask` the single-bit selector, and `value` the
/// word the access reads: the loaded word for a probe, the pre-OR word for a set.
#[derive(Clone, Copy)]
pub(in crate::graph) struct BitAccess<'a> {
    /// Binding name for the word index of the addressed bit.
    pub word: &'a str,
    /// Binding name for the single-bit mask selecting the bit within its word.
    pub mask: &'a str,
    /// Binding name for the word value the access reads.
    pub value: &'a str,
}

/// Bind the word index and single-bit mask addressing `bit` in a packed bitset.
///
/// `word_index` maps a bitset WORD index to its position in the target buffer:
/// identity for a standalone bitset, `base + word` for one query's slice of a
/// flat per-query batch.
pub(in crate::graph) fn bind_bit_address(
    bit: &Expr,
    word: &str,
    mask: &str,
    word_index: impl FnOnce(Expr) -> Expr,
) -> [Node; 2] {
    [
        Node::let_bind(word, word_index(Expr::shr(bit.clone(), Expr::u32(5)))),
        Node::let_bind(
            mask,
            Expr::shl(Expr::u32(1), Expr::bitand(bit.clone(), Expr::u32(31))),
        ),
    ]
}

/// Load the word holding the addressed bit from `buffer` into `names.value`.
pub(in crate::graph) fn bind_word(buffer: &str, names: BitAccess<'_>) -> Node {
    Node::let_bind(names.value, Expr::load(buffer, Expr::var(names.word)))
}

/// `(value & mask) != 0`: the addressed bit is set in the word already bound.
pub(in crate::graph) fn bit_is_set(names: BitAccess<'_>) -> Expr {
    Expr::ne(
        Expr::bitand(Expr::var(names.value), Expr::var(names.mask)),
        Expr::u32(0),
    )
}

/// `(value & mask) == 0`: the addressed bit is clear in the word already bound.
pub(in crate::graph) fn bit_is_clear(value: &str, mask: &str) -> Expr {
    Expr::eq(
        Expr::bitand(Expr::var(value), Expr::var(mask)),
        Expr::u32(0),
    )
}

/// Run `body` only when bit `bit` of packed bitset `buffer` is set.
///
/// Emits the address bindings, the word load, and the guarded body: the "this
/// lane's bit is live" preamble of every frontier-driven kernel.
///
/// `word` names the binding that holds the word index, or is `None` to inline
/// that index into the load. Both shapes are live because both are ABI: a kernel
/// that reads the index again keeps it bound, and a kernel whose only reader is
/// the load does not introduce a binding it never reads a second time.
pub(in crate::graph) fn when_bit_set(
    buffer: &str,
    bit: &Expr,
    word: Option<&str>,
    value: &str,
    mask: &str,
    word_index: impl FnOnce(Expr) -> Expr,
    body: Vec<Node>,
) -> Vec<Node> {
    let index = word_index(Expr::shr(bit.clone(), Expr::u32(5)));
    let bit_mask = Node::let_bind(
        mask,
        Expr::shl(Expr::u32(1), Expr::bitand(bit.clone(), Expr::u32(31))),
    );
    let guard = Node::if_then(
        Expr::ne(
            Expr::bitand(Expr::var(value), Expr::var(mask)),
            Expr::u32(0),
        ),
        body,
    );
    match word {
        Some(word) => vec![
            Node::let_bind(word, index),
            bit_mask,
            Node::let_bind(value, Expr::load(buffer, Expr::var(word))),
            guard,
        ],
        None => vec![
            Node::let_bind(value, Expr::load(buffer, index)),
            bit_mask,
            guard,
        ],
    }
}

/// Set bit `bit` of packed bitset `buffer` with an atomic OR, running
/// `on_new_bit` only when this lane flipped the bit from 0 to 1.
///
/// An empty `on_new_bit` emits the bare OR and no flip guard, so a set-only
/// caller keeps its minimal IR: no unused binding and no empty conditional. The
/// atomic is what makes the 0-to-1 observation exclusive, so exactly one lane per
/// newly reached node runs `on_new_bit` however many lanes race for it.
pub(in crate::graph) fn set_bit(
    buffer: &str,
    bit: &Expr,
    names: BitAccess<'_>,
    word_index: impl FnOnce(Expr) -> Expr,
    on_new_bit: Vec<Node>,
) -> Vec<Node> {
    let [word, mask] = bind_bit_address(bit, names.word, names.mask, word_index);
    let or = Node::let_bind(
        names.value,
        Expr::atomic_or(buffer, Expr::var(names.word), Expr::var(names.mask)),
    );
    if on_new_bit.is_empty() {
        return vec![word, mask, or];
    }
    vec![
        word,
        mask,
        or,
        Node::if_then(bit_is_clear(names.value, names.mask), on_new_bit),
    ]
}

/// Guard `active_body` on lane `source` naming an in-bounds node whose bit is set
/// in `frontier_in` and, when `excluded_sources` is given, clear in that bitset.
///
/// This is the preamble of every source-lane CSR kernel. The bounds guard is not
/// optional and the backend cannot supply it: a dispatch launches whole groups of
/// lanes, so the tail group always over-runs `node_count`, and without the guard
/// those lanes index `edge_offsets` and the frontier past their declared counts.
///
/// `src`, `word_idx`, `bit_mask`, `src_word` and `excluded_word` are the canonical
/// unprefixed names a standalone source-lane program emits.
pub(in crate::graph) fn active_source_lane(
    node_count: u32,
    frontier_in: &str,
    excluded_sources: Option<&str>,
    source: Expr,
    active_body: Vec<Node>,
) -> Node {
    let names = BitAccess {
        word: "word_idx",
        mask: "bit_mask",
        value: "src_word",
    };
    let [word, mask] = bind_bit_address(&Expr::var("src"), names.word, names.mask, |word| word);
    let mut lane = vec![
        Node::let_bind("src", source.clone()),
        word,
        mask,
        bind_word(frontier_in, names),
    ];
    let live = match excluded_sources {
        Some(excluded_sources) => {
            lane.push(Node::let_bind(
                "excluded_word",
                Expr::load(excluded_sources, Expr::var(names.word)),
            ));
            Expr::and(bit_is_set(names), bit_is_clear("excluded_word", names.mask))
        }
        None => bit_is_set(names),
    };
    lane.push(Node::if_then(live, active_body));
    Node::if_then(Expr::lt(source, Expr::u32(node_count)), lane)
}

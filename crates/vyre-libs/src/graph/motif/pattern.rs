//! What a motif is: directed edges over pattern-local node numbers.

/// One directed motif edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MotifEdge {
    /// Source node id.
    pub from: u32,
    /// Edge-kind mask that must match.
    pub kind_mask: u32,
    /// Destination node id.
    pub to: u32,
}

/// The two-edge directed path `0 -> 1 -> 2`, both edges on kind mask 1.
///
/// The smallest motif that is not a single edge, so it is the pattern every
/// parity suite reaches for: it distinguishes a matcher that walks the whole
/// pattern from one that stops after the first edge. Four crates had each
/// written the two `MotifEdge` literals out by hand, and a change to what
/// `kind_mask` means, or to the node numbering a witness is indexed by, would
/// have had to be applied to every copy separately.
pub const TWO_EDGE_PATH_MOTIF: [MotifEdge; 2] = [
    MotifEdge {
        from: 0,
        kind_mask: 1,
        to: 1,
    },
    MotifEdge {
        from: 1,
        kind_mask: 1,
        to: 2,
    },
];

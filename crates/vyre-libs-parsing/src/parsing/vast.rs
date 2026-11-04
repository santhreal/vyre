//! Packed AST (VAST). The layout and every host-side operation over it belong
//! to [`vyre_foundation::vast`]; this module re-exports the surface parser
//! dialects use so a caller names one path.

pub use vyre_foundation::vast::{
    pack_spine_vast, validate_vast, walk_postorder_indices, walk_preorder_indices, VastError,
    VastHeader, VastNode, HEADER_LEN, NODE_STRIDE_U32, SENTINEL, VAST_MAGIC, VAST_VERSION,
};

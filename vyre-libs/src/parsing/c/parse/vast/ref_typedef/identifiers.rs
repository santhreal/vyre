use super::*;
pub(super) use vyre_primitives::hash::fnv1a::fnv1a32;

pub(super) fn identifier_lexeme<'a>(
    vast_nodes: &[u32],
    node_idx: usize,
    haystack: &'a [u8],
) -> Option<&'a [u8]> {
    if kind_at(vast_nodes, node_idx) != TOK_IDENTIFIER {
        return None;
    }
    let start = vast_field_at(vast_nodes, node_idx, 5) as usize;
    let len = vast_field_at(vast_nodes, node_idx, 6) as usize;
    haystack.get(start..start.saturating_add(len))
}

pub(super) fn is_gnu_typeof_hash_raw(hash: u32) -> bool {
    C_GNU_TYPEOF_HASHES.contains(&hash)
}

pub(super) fn is_gnu_auto_type_hash_raw(hash: u32) -> bool {
    hash == C_GNU_AUTO_TYPE_HASH
}

pub(super) fn symbol_hash_at(vast_nodes: &[u32], node_idx: usize) -> u32 {
    vast_field_at(vast_nodes, node_idx, VAST_TYPEDEF_SYMBOL_FIELD as usize)
}

/// The hash the identifier-hash phase leaves in the symbol field: the stored
/// hash when the row already carries one, and FNV-1a over the row's source
/// bytes otherwise.
///
/// The phase reads bytes only while the span stays inside the haystack, so a
/// row whose span runs past the source hashes its prefix.
pub(in crate::parsing::c::parse::vast) fn identifier_row_hash(
    vast_nodes: &[u32],
    node_idx: usize,
    haystack: &[u8],
) -> u32 {
    let stored = symbol_hash_at(vast_nodes, node_idx);
    if stored != 0 {
        return stored;
    }
    let start = vast_field_at(vast_nodes, node_idx, 5) as usize;
    let len = vast_field_at(vast_nodes, node_idx, 6) as usize;
    let mut hash = 0x811c_9dc5_u32;
    for offset in 0..len {
        let Some(byte) = haystack.get(start.saturating_add(offset)) else {
            continue;
        };
        hash = (hash ^ u32::from(*byte)).wrapping_mul(0x0100_0193);
    }
    hash
}

pub(super) fn is_typeof_operator_raw(kind: u32, symbol_hash: u32) -> bool {
    matches!(kind, TOK_GNU_TYPEOF | TOK_GNU_TYPEOF_UNQUAL)
        || (kind == TOK_IDENTIFIER && is_gnu_typeof_hash_raw(symbol_hash))
}

pub(super) fn is_decl_prefix_at(vast_nodes: &[u32], node_idx: usize) -> bool {
    let kind = kind_at(vast_nodes, node_idx);
    let symbol_hash = symbol_hash_at(vast_nodes, node_idx);
    is_decl_prefix_raw(kind)
        || is_typeof_operator_raw(kind, symbol_hash)
        || (kind == TOK_IDENTIFIER && is_gnu_auto_type_hash_raw(symbol_hash))
}

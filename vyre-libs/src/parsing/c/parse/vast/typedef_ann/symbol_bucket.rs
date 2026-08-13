use super::*;

/// Map an identifier hash onto one of `buckets` typedef symbol-table slots.
///
/// `buckets` is a power of two, so the mask is exact; the xor-fold spreads the
/// high half of the hash into the low bits the mask keeps.
pub(super) fn typedef_symbol_bucket(hash: Expr, buckets: u32) -> Expr {
    let mixed = Expr::bitxor(hash.clone(), Expr::shr(hash, Expr::u32(16)));
    Expr::bitand(mixed, Expr::u32(buckets - 1))
}

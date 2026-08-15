//! Canonical bytes for a buffer declaration, and the order-independent
//! comparison built on them.

pub(crate) fn buffers_equal_ignoring_declaration_order(
    left: &[crate::ir_inner::model::program::BufferDecl],
    right: &[crate::ir_inner::model::program::BufferDecl],
) -> bool {
    if left.len() != right.len() {
        return false;
    }

    // VYRE_IR_HOTSPOTS HIGH (meta.rs:360-379): previous impl allocated
    // two Vec<Vec<u8>> then sorted on every equality call. Fast-path:
    // if the slices compare equal in-place (declaration orders match)
    // we skip the key-materialization entirely. This catches every
    // Program::clone(prog) == prog and every `Arc::clone`-equivalent
    // comparison, which dominate the call distribution.
    if left == right {
        return true;
    }

    let mut left_keys = Vec::with_capacity(left.len());
    left_keys.extend(left.iter().map(buffer_decl_canonical_key));
    let mut right_keys = Vec::with_capacity(right.len());
    right_keys.extend(right.iter().map(buffer_decl_canonical_key));
    left_keys.sort_unstable();
    right_keys.sort_unstable();
    left_keys == right_keys
}

/// Canonical bytes for one buffer declaration, used to compare buffer sets
/// independently of declaration order.
///
/// THIS KEY DECIDES PROGRAM EQUALITY. `Program::eq` delegates to
/// `structural_eq`, which compares buffers through
/// `buffers_equal_ignoring_declaration_order`, which keys each declaration
/// here. So any field this function omits is a field two programs may differ in
/// while comparing EQUAL.
///
/// It previously omitted `linear_type` and `shape_predicate` while including
/// `bytes_extraction`: one of three fields wired in, the other two forgotten.
/// Both omitted fields DECIDE VALIDATION VERDICTS (`validate::linear_type` and
/// `validate::shape_predicate::check_shape_predicates`), so a VALID program and
/// an INVALID one compared equal, and any caller shaped as "a == b, therefore
/// same program, therefore safe to reuse" was wrong for them.
///
/// COMPLETENESS IS ENFORCED BY THE COMPILER, not by review. The destructuring
/// below binds every field by name, so adding a field to `BufferDecl` FAILS TO
/// COMPILE here until somebody decides whether program equality depends on it.
/// That is the only mechanism that catches the next omission at the moment it is
/// introduced rather than years later through a wrong result. Do not replace it
/// with `buffer.field` accesses, and do not silence a new field with `..`.
///
/// `linear_type` and `shape_predicate` are encoded by the SAME functions the
/// wire encoder uses, so program equality and `Program::fingerprint` cannot
/// drift apart on these fields.
pub(crate) fn buffer_decl_canonical_key(buffer: &crate::ir_inner::model::program::BufferDecl) -> Vec<u8> {
    use crate::serial::wire::encode::to_wire::{linear_type_tag, put_shape_predicate};
    use crate::serial::wire::framing::{put_len_u32, put_u32, put_u8};
    use crate::serial::wire::tags::put_data_type;

    // Exhaustive by field name: see the completeness note above. A new
    // `BufferDecl` field breaks this pattern and forces a decision.
    let crate::ir_inner::model::program::BufferDecl {
        name,
        binding,
        access,
        kind,
        element,
        count,
        is_output,
        pipeline_live_out,
        output_byte_range,
        hints,
        bytes_extraction,
        linear_type,
        shape_predicate,
    } = buffer;

    let mut key = Vec::with_capacity(96);
    if let Err(error) = put_len_u32(&mut key, name.len(), "buffer name length") {
        key.extend_from_slice(b"\0name-length-error\0");
        key.extend_from_slice(error.as_bytes());
    }
    key.extend_from_slice(name.as_bytes());
    put_u32(&mut key, *binding);
    match crate::serial::wire::tags::access_tag::access_tag(access) {
        Ok(tag) => put_u8(&mut key, tag),
        Err(error) => {
            put_u8(&mut key, u8::MAX);
            key.extend_from_slice(error.as_bytes());
        }
    }
    put_u8(&mut key, crate::ir_inner::model::program::cache_digest::memory_kind_cache_tag(*kind));
    if let Err(error) = put_data_type(&mut key, element) {
        key.extend_from_slice(b"\0dtype-error\0");
        key.extend_from_slice(error.as_bytes());
    }
    put_u32(&mut key, *count);
    put_u8(&mut key, u8::from(*is_output));
    put_u8(&mut key, u8::from(*pipeline_live_out));
    match output_byte_range {
        Some(range) => {
            put_u8(&mut key, 1);
            match u32::try_from(range.start) {
                Ok(start) => put_u32(&mut key, start),
                Err(error) => {
                    put_u32(&mut key, u32::MAX);
                    key.extend_from_slice(error.to_string().as_bytes());
                }
            }
            match u32::try_from(range.end) {
                Ok(end) => put_u32(&mut key, end),
                Err(error) => {
                    put_u32(&mut key, u32::MAX);
                    key.extend_from_slice(error.to_string().as_bytes());
                }
            }
        }
        None => put_u8(&mut key, 0),
    }
    match hints.coalesce_axis {
        Some(axis) => {
            put_u8(&mut key, 1);
            put_u8(&mut key, axis);
        }
        None => put_u8(&mut key, 0),
    }
    put_u32(&mut key, hints.preferred_alignment);
    put_u8(
        &mut key,
        match hints.cache_locality {
            crate::ir_inner::model::program::CacheLocality::Streaming => 0,
            crate::ir_inner::model::program::CacheLocality::Temporal => 1,
            crate::ir_inner::model::program::CacheLocality::Random => 2,
        },
    );
    put_u8(&mut key, u8::from(*bytes_extraction));
    put_u8(&mut key, linear_type_tag(*linear_type));
    if let Err(error) = put_shape_predicate(&mut key, shape_predicate.as_ref(), 0) {
        // The encoder refuses a predicate nested past the wire depth bound, and
        // it may already have emitted a prefix before refusing. A bare error
        // marker would let two DIFFERENT over-deep predicates share a key,
        // reintroducing the exact collision this field was added to close, so
        // fall back to the Debug form, which stays injective at any depth.
        key.extend_from_slice(b"\0shape-predicate-error\0");
        key.extend_from_slice(error.as_bytes());
        key.extend_from_slice(format!("{shape_predicate:?}").as_bytes());
    }
    key
}

//! Encoders for the buffer table and its memory-region payload.
//!
//! Everything below the buffer table's own record: the discipline tag, the
//! shape refinement, the memory-region block and the quantization scalars
//! inside it. [`linear_type_tag`] and [`put_shape_predicate`] are also the
//! encoders program equality keys on, so the wire fingerprint and
//! `Program::eq` cannot disagree about what a buffer declares.

use crate::ir_inner::model::program::{
    BufferDecl, CacheLocality, LinearType, MemoryKind, ShapePredicate,
};
use crate::ir_inner::model::spec_types::DataType;
use crate::serial::wire::encode::WireEncodeErr;
use crate::serial::wire::framing::{put_u32, put_u8, MAX_SHAPE_PREDICATE_DEPTH};
use crate::serial::wire::tags::access_tag::access_tag;
use crate::serial::wire::tags::data_type_tag::data_type_tag;
use crate::serial::{put_leb_u32, put_leb_u64};

struct RegionPayloadScratch {
    shape: Vec<u8>,
    hints: Vec<u8>,
}

/// Stable wire tag for a [`LinearType`].
///
/// Exhaustive on purpose. `LinearType` is `#[non_exhaustive]` to outside
/// crates, but this match lives in the defining crate, so adding a variant
/// FAILS TO COMPILE here instead of silently encoding as some default. A
/// discipline that encodes as the wrong variant is a validation verdict
/// changing under serialization.
///
/// SHARED BY TWO KEYS ON PURPOSE, and that sharing is the point. This encodes
/// `linear_type` for the wire payload (hence for `Program::fingerprint`) and
/// `buffer_decl_canonical_key` in `ir_inner::model::program::meta` calls the
/// SAME function for program equality. Both keys are derived over the same
/// struct, so giving them one encoder means a new `LinearType` variant breaks
/// one build site and fixes both keys at once. The alternative, a second
/// private copy of this match, is exactly how `binding` went missing from a
/// target digest while remaining present in the code generator.
pub(crate) const fn linear_type_tag(value: LinearType) -> u8 {
    match value {
        LinearType::Linear => 0,
        LinearType::Affine => 1,
        LinearType::Relevant => 2,
        LinearType::Unrestricted => 3,
    }
}

/// Encode an optional [`ShapePredicate`], tag `0` for `None`.
///
/// `And`, `Or` and `Not` are recursive, so `depth` is checked against
/// [`MAX_SHAPE_PREDICATE_DEPTH`] before descending. The decoder enforces the
/// same bound from the same constant, so a predicate this encoder accepts is
/// one that decoder can read back.
///
/// Exhaustive for the same reason as [`linear_type_tag`]: a new predicate
/// variant must not silently encode as something else.
///
/// Shared with `buffer_decl_canonical_key` for the same reason as
/// [`linear_type_tag`]. Program equality and the wire fingerprint MUST agree on
/// what a shape refinement is, because both are used to decide whether two
/// programs are interchangeable, and `shape_predicate` decides a validation
/// verdict through `validate::shape_predicate::check_shape_predicates`.
pub(crate) fn put_shape_predicate(
    out: &mut Vec<u8>,
    predicate: Option<&ShapePredicate>,
    depth: usize,
) -> Result<(), WireEncodeErr> {
    let Some(predicate) = predicate else {
        put_u8(out, 0);
        return Ok(());
    };
    if depth >= MAX_SHAPE_PREDICATE_DEPTH {
        return Err(WireEncodeErr::static_msg(
            "Fix: shape predicate nests deeper than the wire limit; flatten the And/Or/Not tree.",
        ));
    }
    match predicate {
        ShapePredicate::AtLeast(count) => {
            put_u8(out, 1);
            put_u32(out, *count);
        }
        ShapePredicate::AtMost(count) => {
            put_u8(out, 2);
            put_u32(out, *count);
        }
        ShapePredicate::Exactly(count) => {
            put_u8(out, 3);
            put_u32(out, *count);
        }
        ShapePredicate::MultipleOf(count) => {
            put_u8(out, 4);
            put_u32(out, *count);
        }
        ShapePredicate::ModEquals { modulus, remainder } => {
            put_u8(out, 5);
            put_u32(out, *modulus);
            put_u32(out, *remainder);
        }
        ShapePredicate::AffineRange {
            scale,
            offset,
            min,
            max,
        } => {
            put_u8(out, 6);
            // Two's-complement bit pattern via to_le_bytes, so the round trip
            // is exact for negative coefficients without a sign-loss cast.
            for value in [*scale, *offset, *min, *max] {
                put_leb_u64(out, u64::from_le_bytes(value.to_le_bytes()));
            }
        }
        ShapePredicate::And(left, right) => {
            put_u8(out, 7);
            put_shape_predicate(out, Some(left), depth + 1)?;
            put_shape_predicate(out, Some(right), depth + 1)?;
        }
        ShapePredicate::Or(left, right) => {
            put_u8(out, 8);
            put_shape_predicate(out, Some(left), depth + 1)?;
            put_shape_predicate(out, Some(right), depth + 1)?;
        }
        ShapePredicate::Not(inner) => {
            put_u8(out, 9);
            put_shape_predicate(out, Some(inner), depth + 1)?;
        }
    }
    Ok(())
}

pub(super) fn put_memory_regions(out: &mut Vec<u8>, buffers: &[BufferDecl]) -> Result<(), WireEncodeErr> {
    let mut scratch = RegionPayloadScratch {
        shape: Vec::with_capacity(16),
        hints: Vec::with_capacity(16),
    };
    put_memory_regions_with_scratch(out, buffers, &mut scratch.shape, &mut scratch.hints)
}

pub(super) fn put_memory_regions_with_scratch(
    out: &mut Vec<u8>,
    buffers: &[BufferDecl],
    shape: &mut Vec<u8>,
    hints: &mut Vec<u8>,
) -> Result<(), WireEncodeErr> {
    put_leb_u64(
        out,
        u64::try_from(buffers.len()).map_err(|_| {
            WireEncodeErr::static_msg("Fix: memory-region count cannot fit u64; split the Program.")
        })?,
    );
    for (index, buffer) in buffers.iter().enumerate() {
        put_leb_u32(
            out,
            u32::try_from(index).map_err(|_| {
                WireEncodeErr::fmt_usize(
                    "Fix: memory-region id ",
                    index,
                    " cannot fit u32; split the Program.",
                )
            })?,
        );
        put_u8(out, memory_kind_tag(buffer.kind()));
        put_u8(
            out,
            access_tag(&buffer.access()).map_err(WireEncodeErr::from)?,
        );
        put_u8(out, dense_element_tag(&buffer.element())?);
        put_u8(out, 0);
        shape.clear();
        put_leb_u64(shape, u64::from(buffer.count()));
        if let DataType::Array { element_size } = buffer.element() {
            put_leb_u64(
                shape,
                u64::try_from(element_size).map_err(|_| {
                    WireEncodeErr::static_msg(
                        "Fix: array element size cannot fit u64; cap the element size.",
                    )
                })?,
            );
        }
        if let DataType::Handle(id) = buffer.element() {
            put_leb_u64(shape, u64::from(id.as_u32()));
        }
        if let DataType::Opaque(id) = buffer.element() {
            // Opaque payload = u32 extension id (LEB-encoded as u64 to match
            // the surrounding wire convention; decoder caps at u32::MAX).
            put_leb_u64(shape, u64::from(id.as_u32()));
        }
        if let DataType::Quantized {
            storage,
            scale,
            zero_point,
        } = buffer.element()
        {
            if !storage.is_quantized_storage() {
                return Err(WireEncodeErr::static_msg(
                    "Fix: quantized memory-region storage must be I4/I8/I16/U8/U16/F8E4M3/F8E5M2/FP4/NF4.",
                ));
            }
            // `buffer.element()` returns DataType by value, so the
            // `Quantized { storage, scale, zero_point }` pattern binds
            // by value: storage is Box<DataType>, scale and zero_point
            // are owned. The helpers want references - &*storage drops
            // the Box to get &DataType.
            put_leb_u64(shape, u64::from(dense_element_tag(&storage)?));
            put_dense_quantization_scale(shape, &scale)?;
            put_dense_quantization_zero_point(shape, &zero_point)?;
        }
        put_leb_u64(
            out,
            u64::try_from(shape.len()).map_err(|_| {
                WireEncodeErr::static_msg(
                    "Fix: shape payload length cannot fit u64; split the Program.",
                )
            })?,
        );
        out.extend_from_slice(shape);
        hints.clear();
        put_hints_payload(hints, buffer.hints());
        put_leb_u64(
            out,
            u64::try_from(hints.len()).map_err(|_| {
                WireEncodeErr::static_msg(
                    "Fix: hints payload length cannot fit u64; split the Program.",
                )
            })?,
        );
        out.extend_from_slice(hints);
    }
    Ok(())
}

pub(super) fn put_hints_payload(out: &mut Vec<u8>, hints: crate::ir::MemoryHints) {
    match hints.coalesce_axis {
        Some(axis) => {
            put_u8(out, 1);
            put_u8(out, axis);
        }
        None => put_u8(out, 0),
    }
    put_u32(out, hints.preferred_alignment);
    put_u8(
        out,
        match hints.cache_locality {
            CacheLocality::Streaming => 0,
            CacheLocality::Temporal => 1,
            CacheLocality::Random => 2,
        },
    );
}

pub(super) fn memory_kind_tag(kind: MemoryKind) -> u8 {
    match kind {
        MemoryKind::Global => 0,
        MemoryKind::Shared => 1,
        MemoryKind::Uniform => 2,
        MemoryKind::Local => 3,
        MemoryKind::Readonly => 4,
        MemoryKind::Push => 5,
        MemoryKind::Persistent => 6,
    }
}

/// The wire tag for a DENSE memory-region element type.
///
/// The tag itself comes from `tags::data_type_tag`, the one owner of that
/// mapping. What is local here is the narrower domain: a sparse or
/// device-mesh type is a legal `DataType` and has a wire tag, but it is not a
/// legal element of a dense memory region, so this encoder refuses it rather
/// than emitting a blob whose region stride is meaningless. Restating the tag
/// table to express that restriction is what let the two copies drift.
pub(super) fn dense_element_tag(value: &DataType) -> Result<u8, WireEncodeErr> {
    if matches!(
        value,
        DataType::SparseCsr { .. }
            | DataType::SparseCoo { .. }
            | DataType::SparseBsr { .. }
            | DataType::DeviceMesh { .. }
    ) {
        return Err(dense_element_rejection());
    }
    data_type_tag(value).map_err(|_| dense_element_rejection())
}

fn dense_element_rejection() -> WireEncodeErr {
    WireEncodeErr::static_msg(
        "Fix: unknown DataType variant cannot be serialized into VYRE wire format. \
         Sparse/Vec/TensorShaped/DeviceMesh types are not valid buffer elements in the \
         dense memory-region encoder; lower to a supported scalar/array/handle/opaque \
         first.",
    )
}

fn put_dense_quantization_scale(
    out: &mut Vec<u8>,
    scale: &vyre_spec::QuantizationScale,
) -> Result<(), WireEncodeErr> {
    match scale {
        vyre_spec::QuantizationScale::PerTensor => {
            put_leb_u64(out, 0);
            put_leb_u64(out, 0);
        }
        vyre_spec::QuantizationScale::PerChannel { axis } => {
            put_leb_u64(out, 1);
            put_leb_u64(out, u64::from(*axis));
        }
        vyre_spec::QuantizationScale::PerGroup { group_size } => {
            if *group_size == 0 {
                return Err(WireEncodeErr::static_msg(
                    "Fix: quantized PerGroup scale requires group_size > 0.",
                ));
            }
            put_leb_u64(out, 2);
            put_leb_u64(out, u64::from(*group_size));
        }
    }
    Ok(())
}

fn put_dense_quantization_zero_point(
    out: &mut Vec<u8>,
    zero_point: &vyre_spec::QuantizationZeroPoint,
) -> Result<(), WireEncodeErr> {
    match zero_point {
        vyre_spec::QuantizationZeroPoint::Absent => {
            put_leb_u64(out, 0);
            put_leb_u64(out, 0);
        }
        vyre_spec::QuantizationZeroPoint::PerTensor => {
            put_leb_u64(out, 1);
            put_leb_u64(out, 0);
        }
        vyre_spec::QuantizationZeroPoint::PerChannel { axis } => {
            put_leb_u64(out, 2);
            put_leb_u64(out, u64::from(*axis));
        }
        vyre_spec::QuantizationZeroPoint::PerGroup { group_size } => {
            if *group_size == 0 {
                return Err(WireEncodeErr::static_msg(
                    "Fix: quantized PerGroup zero-point requires group_size > 0.",
                ));
            }
            put_leb_u64(out, 3);
            put_leb_u64(out, u64::from(*group_size));
        }
    }
    Ok(())
}

pub(super) fn put_leb_str(out: &mut Vec<u8>, value: &str) -> Result<(), WireEncodeErr> {
    put_leb_u64(
        out,
        u64::try_from(value.len()).map_err(|_| {
            WireEncodeErr::static_msg("Fix: string length cannot fit u64; shorten the identifier.")
        })?,
    );
    out.extend_from_slice(value.as_bytes());
    Ok(())
}

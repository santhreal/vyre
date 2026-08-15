//! Decoders for the buffer table and its memory-region payload.
//!
//! The mirror of `encode::to_wire::buffer_table`, and it runs on untrusted
//! bytes: an unknown discipline tag is rejected rather than defaulted, and the
//! recursive shape predicate is bounded by the same constant the encoder
//! checks, so a hostile blob cannot nest connectives until the stack ends.

use super::payload::{
    data_type_from_tag, memory_kind_from_tag, read_dense_quantization_scale,
    read_dense_quantization_zero_point, read_hints,
};
use super::{DecodedMetadata, LebReader};
use crate::ir_inner::model::program::{LinearType, ShapePredicate};
use crate::ir_inner::model::spec_types::{BufferAccess, DataType};
use crate::serial::wire::decode::{invariants, reject_reserved_extension_id};
use crate::serial::wire::framing::{MAX_SHAPE_PREDICATE_DEPTH, WIRE_FORMAT_VERSION};
use crate::serial::wire::tags::access_from_tag::access_from_tag;
use crate::serial::wire::{Reader, MAX_BUFFERS};

/// Decode a [`LinearType`] wire tag.
///
/// Unknown tags are REJECTED rather than defaulted. Defaulting here would turn
/// a payload this decoder does not understand into a program with a weaker
/// linear discipline than its author declared, which is a validation verdict
/// changing silently under deserialization.
pub(super) fn linear_type_from_tag(tag: u8) -> Result<LinearType, String> {
    match tag {
        0 => Ok(LinearType::Linear),
        1 => Ok(LinearType::Affine),
        2 => Ok(LinearType::Relevant),
        3 => Ok(LinearType::Unrestricted),
        value => Err(format!(
            "InvalidDiscriminant: field linear_type has tag {value}. Fix: reserialize with Program::to_wire()."
        )),
    }
}

/// Decode an optional [`ShapePredicate`]; tag `0` means `None`.
///
/// `depth` is bounded by [`MAX_SHAPE_PREDICATE_DEPTH`], the same constant the
/// encoder checks, because `And` / `Or` / `Not` are recursive and this runs on
/// untrusted bytes. Without the bound a hostile blob nests them until the
/// decoder stack overflows.
pub(super) fn read_shape_predicate(
    reader: &mut Reader<'_>,
    depth: usize,
) -> Result<Option<ShapePredicate>, String> {
    let tag = reader.u8()?;
    if tag == 0 {
        return Ok(None);
    }
    if depth >= MAX_SHAPE_PREDICATE_DEPTH {
        return Err(format!(
            "Fix: shape predicate exceeds maximum nesting depth {MAX_SHAPE_PREDICATE_DEPTH}; reject this untrusted blob or flatten the And/Or/Not tree."
        ));
    }
    let predicate = match tag {
        1 => ShapePredicate::AtLeast(reader.u32()?),
        2 => ShapePredicate::AtMost(reader.u32()?),
        3 => ShapePredicate::Exactly(reader.u32()?),
        4 => ShapePredicate::MultipleOf(reader.u32()?),
        5 => {
            let modulus = reader.u32()?;
            let remainder = reader.u32()?;
            ShapePredicate::ModEquals { modulus, remainder }
        }
        6 => {
            let scale = read_i64(reader)?;
            let offset = read_i64(reader)?;
            let min = read_i64(reader)?;
            let max = read_i64(reader)?;
            ShapePredicate::AffineRange {
                scale,
                offset,
                min,
                max,
            }
        }
        7 => {
            let left = read_nested_shape_predicate(reader, depth)?;
            let right = read_nested_shape_predicate(reader, depth)?;
            ShapePredicate::And(Box::new(left), Box::new(right))
        }
        8 => {
            let left = read_nested_shape_predicate(reader, depth)?;
            let right = read_nested_shape_predicate(reader, depth)?;
            ShapePredicate::Or(Box::new(left), Box::new(right))
        }
        9 => ShapePredicate::Not(Box::new(read_nested_shape_predicate(reader, depth)?)),
        value => {
            return Err(format!(
                "InvalidDiscriminant: field shape_predicate has tag {value}. Fix: reserialize with Program::to_wire()."
            ));
        }
    };
    Ok(Some(predicate))
}

/// Read an operand of `And` / `Or` / `Not`, which must be present.
///
/// A nested `None` tag would decode into a connective with a missing operand,
/// so it is rejected rather than silently dropping the connective.
pub(super) fn read_nested_shape_predicate(
    reader: &mut Reader<'_>,
    depth: usize,
) -> Result<ShapePredicate, String> {
    read_shape_predicate(reader, depth + 1)?.ok_or_else(|| {
        "InvalidDiscriminant: shape predicate connective has an absent operand. Fix: reserialize with Program::to_wire().".to_string()
    })
}

/// Read an `i64` written as its little-endian two's-complement bit pattern.
pub(super) fn read_i64(reader: &mut Reader<'_>) -> Result<i64, String> {
    Ok(i64::from_le_bytes(reader.leb_u64()?.to_le_bytes()))
}

pub(super) fn read_memory_regions(
    reader: &mut Reader<'_>,
    metadata: &mut DecodedMetadata,
) -> Result<(), String> {
    let count = reader.leb_len(MAX_BUFFERS, "memory-region count")?;
    if count != metadata.buffers.len() {
        return Err(format!(
            "InvalidDiscriminant: memory-region count {count} does not match metadata buffer count {}. Fix: reject tampered Program bytes.",
            metadata.buffers.len()
        ));
    }
    for index in 0..count {
        let id = reader.leb_u32()?;
        if usize::try_from(id).ok() != Some(index) {
            return Err(format!(
                "InvalidDiscriminant: memory-region id {id} is out of canonical order at index {index}. Fix: reserialize with Program::to_wire()."
            ));
        }
        let kind = memory_kind_from_tag(reader.u8()?)?;
        let access = access_from_tag(reader.u8()?)?;
        let element_tag = reader.u8()?;
        let shape_tag = reader.u8()?;
        if shape_tag != 0 {
            return Err(format!(
                "InvalidDiscriminant: field shape_tag has value {shape_tag}. Fix: this decoder supports Dense regions only in schema {WIRE_FORMAT_VERSION}."
            ));
        }
        // VYRE_IR_HOTSPOTS CRIT (from_wire.rs:355): `.to_vec()` on
        // every buffer's shape payload cost one heap alloc per
        // buffer. Using the raw sub-slice from the parent reader
        // keeps decoding zero-copy.
        let shape_len = reader.leb_len(64, "shape payload length")?;
        let shape_payload = reader.take(shape_len)?;
        let mut shape_reader = Reader {
            bytes: shape_payload,
            pos: 0,
            depth: 0,
        };
        let count_value = u32::try_from(shape_reader.leb_u64()?).map_err(|err| {
            format!("TruncatedPayload: dense shape count cannot fit u32 ({err}). Fix: split the memory region.")
        })?;
        let element = if element_tag == 0x08 {
            let element_size = usize::try_from(shape_reader.leb_u64()?).map_err(|err| {
                format!("TruncatedPayload: array element size cannot fit usize ({err}). Fix: reject this payload on this target.")
            })?;
            DataType::Array { element_size }
        } else if element_tag == 0x13 {
            let id_value = u32::try_from(shape_reader.leb_u64()?).map_err(|err| {
                format!("TruncatedPayload: handle DataType id cannot fit u32 ({err}). Fix: reject this payload.")
            })?;
            DataType::Handle(vyre_spec::data_type::TypeId(id_value))
        } else if element_tag == 0x80 {
            let id_value = u32::try_from(shape_reader.leb_u64()?).map_err(|err| {
                format!("TruncatedPayload: opaque DataType id cannot fit u32 ({err}). Fix: reject this payload.")
            })?;
            let id_value = reject_reserved_extension_id(id_value, "DataType")?;
            DataType::Opaque(vyre_spec::extension::ExtensionDataTypeId(id_value))
        } else if element_tag == 0x1F {
            let storage_tag = u8::try_from(shape_reader.leb_u64()?).map_err(|err| {
                format!("TruncatedPayload: quantized storage DataType tag cannot fit u8 ({err}). Fix: reject this payload.")
            })?;
            let storage = data_type_from_tag(storage_tag)?;
            if !storage.is_quantized_storage() {
                return Err(format!(
                    "InvalidDiscriminant: quantized storage tag {storage_tag} decodes to `{storage}`, which is not a valid quantized storage type. Fix: reserialize with I4/I8/I16/U8/U16/F8/FP4/NF4 storage."
                ));
            }
            let scale = read_dense_quantization_scale(&mut shape_reader)?;
            let zero_point = read_dense_quantization_zero_point(&mut shape_reader)?;
            DataType::Quantized {
                storage: Box::new(storage),
                scale,
                zero_point,
            }
        } else {
            data_type_from_tag(element_tag)?
        };
        if shape_reader.pos != shape_reader.bytes.len() {
            return Err("TruncatedPayload: shape payload has trailing bytes. Fix: reject non-canonical Program bytes.".to_string());
        }
        // VYRE_IR_HOTSPOTS CRIT (from_wire.rs:387): same zero-copy
        // sub-slice pattern as the shape payload above.
        let hints_len = reader.leb_len(64, "hints payload length")?;
        let hints_payload = reader.take(hints_len)?;
        let mut hints_reader = Reader {
            bytes: hints_payload,
            pos: 0,
            depth: 0,
        };
        let hints = read_hints(&mut hints_reader)?;
        if hints_reader.pos != hints_reader.bytes.len() {
            return Err("TruncatedPayload: hints payload has trailing bytes. Fix: reject non-canonical Program bytes.".to_string());
        }
        let metadata_buffer = &metadata.buffers[index];
        if metadata_buffer.count != count_value {
            return Err(format!(
                "InvalidDiscriminant: memory-region count {count_value} does not match metadata count {} for buffer `{}`. Fix: reserialize with Program::to_wire().",
                metadata_buffer.count, metadata_buffer.name
            ));
        }
        if count_value == 0 && access == BufferAccess::Workgroup {
            return Err(format!(
                "InvalidDiscriminant: workgroup buffer `{}` has count 0. Fix: workgroup memory requires a concrete positive element count.",
                metadata_buffer.name
            ));
        }
        if count_value == 0 && metadata_buffer.is_output {
            return Err(format!(
                "InvalidDiscriminant: output buffer `{}` has count 0. Fix: output buffers need a concrete positive element count before serialization.",
                metadata_buffer.name
            ));
        }
        if count_value == 0 && metadata_buffer.pipeline_live_out {
            return Err(format!(
                "InvalidDiscriminant: live-out buffer `{}` has count 0. Fix: externally-visible buffers need a concrete positive element count before serialization.",
                metadata_buffer.name
            ));
        }
        invariants::validate_output_range_fits(metadata_buffer, &element, count_value)?;
        let buffer = &mut metadata.buffers[index];
        buffer.kind = kind;
        buffer.access = access;
        buffer.element = element;
        buffer.count = count_value;
        buffer.hints = hints;
    }
    Ok(())
}

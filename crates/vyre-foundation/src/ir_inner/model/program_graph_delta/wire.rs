//! Length-prefixed reads and writes for the delta wire format.
//!
//! Every helper bounds what it accepts before it allocates: a length field
//! read off the wire is attacker-controlled, so a decode that trusts it turns
//! a malformed byte into an allocation the size of the field.

use super::*;

pub(super) fn put_string(bytes: &mut Vec<u8>, s: &str) -> Result<(), GraphDeltaError> {
    let len =
        u32::try_from(s.len()).map_err(|_| GraphDeltaError::Wire("string too long".into()))?;
    bytes.extend_from_slice(&len.to_le_bytes());
    bytes.extend_from_slice(s.as_bytes());
    Ok(())
}

pub(super) fn put_bytes(bytes: &mut Vec<u8>, data: &[u8]) -> Result<(), GraphDeltaError> {
    let len = u32::try_from(data.len())
        .map_err(|_| GraphDeltaError::Wire("bytes slice too long".into()))?;
    bytes.extend_from_slice(&len.to_le_bytes());
    bytes.extend_from_slice(data);
    Ok(())
}

pub(super) fn put_contract(
    bytes: &mut Vec<u8>,
    contract: &ValueContract,
) -> Result<(), GraphDeltaError> {
    // The wire tag for a `DataType` is owned by `serial::wire::tags`. A second
    // mapping here would go stale the next time a data type is added, and the
    // one it replaced listed twelve of the thirty-four variants.
    let dtype_code = crate::serial::wire::tags::data_type_tag(&contract.dtype)
        .map_err(|error| GraphDeltaError::Wire(error.to_string()))?;
    bytes.push(dtype_code);
    // `BufferAccess` is non-exhaustive, so a match here is a compile error the
    // moment an access is added, and the tag is owned by the same module the
    // data-type tag is.
    let access_code = crate::serial::wire::tags::access_tag::access_tag(&contract.access)
        .map_err(GraphDeltaError::Wire)?;
    bytes.push(access_code);
    let lifetime_code: u8 = match contract.lifetime {
        ValueLifetime::Constant => 1,
        ValueLifetime::Invocation => 2,
        ValueLifetime::Retained => 3,
        ValueLifetime::Output => 4,
        ValueLifetime::Stream => 5,
    };
    bytes.push(lifetime_code);
    let rank = u32::try_from(contract.shape.len())
        .map_err(|_| GraphDeltaError::Wire("rank too large".into()))?;
    bytes.extend_from_slice(&rank.to_le_bytes());
    for dim in &contract.shape {
        match dim {
            ShapeDim::Known(extent) => {
                bytes.push(1);
                bytes.extend_from_slice(&extent.to_le_bytes());
            }
            ShapeDim::Symbol(sym) => {
                bytes.push(2);
                put_string(bytes, sym)?;
            }
            ShapeDim::Unresolved => {
                bytes.push(3);
            }
            ShapeDim::Expr(expr_id) => {
                bytes.push(4);
                bytes.extend_from_slice(&expr_id.0.to_le_bytes());
            }
        }
    }
    Ok(())
}

pub(super) fn read_u32(bytes: &[u8], cursor: &mut usize) -> Result<u32, GraphDeltaError> {
    if *cursor + 4 > bytes.len() {
        return Err(GraphDeltaError::Wire("EOF reading u32".into()));
    }
    let val = u32::from_le_bytes([
        bytes[*cursor],
        bytes[*cursor + 1],
        bytes[*cursor + 2],
        bytes[*cursor + 3],
    ]);
    *cursor += 4;
    Ok(val)
}

pub(super) fn read_u64(bytes: &[u8], cursor: &mut usize) -> Result<u64, GraphDeltaError> {
    if *cursor + 8 > bytes.len() {
        return Err(GraphDeltaError::Wire("EOF reading u64".into()));
    }
    let mut b = [0u8; 8];
    b.copy_from_slice(&bytes[*cursor..*cursor + 8]);
    *cursor += 8;
    Ok(u64::from_le_bytes(b))
}

pub(super) fn read_string(bytes: &[u8], cursor: &mut usize) -> Result<String, GraphDeltaError> {
    let len = read_u32(bytes, cursor)? as usize;
    if len > MAX_NAME_BYTES {
        return Err(GraphDeltaError::Wire(format!(
            "string length {len} exceeds limit {MAX_NAME_BYTES}"
        )));
    }
    if *cursor + len > bytes.len() {
        return Err(GraphDeltaError::Wire("EOF reading string".into()));
    }
    let s = std::str::from_utf8(&bytes[*cursor..*cursor + len])
        .map_err(|e| GraphDeltaError::Wire(format!("utf-8 error: {e}")))?
        .to_string();
    *cursor += len;
    Ok(s)
}

pub(super) fn read_bytes(bytes: &[u8], cursor: &mut usize) -> Result<Vec<u8>, GraphDeltaError> {
    let len = read_u32(bytes, cursor)? as usize;
    if len > MAX_DELTA_WIRE_BYTES {
        return Err(GraphDeltaError::Wire(format!(
            "byte buffer length {len} exceeds limit {MAX_DELTA_WIRE_BYTES}"
        )));
    }
    if *cursor + len > bytes.len() {
        return Err(GraphDeltaError::Wire("EOF reading byte buffer".into()));
    }
    let data = bytes[*cursor..*cursor + len].to_vec();
    *cursor += len;
    Ok(data)
}

pub(super) fn read_contract(
    bytes: &[u8],
    cursor: &mut usize,
) -> Result<ValueContract, GraphDeltaError> {
    if *cursor + 3 > bytes.len() {
        return Err(GraphDeltaError::Wire("EOF reading contract header".into()));
    }
    // Each tag is decoded by the authority that wrote it, and an unknown tag is
    // an error. The three matches these replaced ended in a catch-all that
    // returned `F32`, `ReadOnly` and `Invocation`, so a delta written by a newer
    // encoder, or a corrupted one, decoded into a contract that named a
    // different type than the bytes did and was served as if it were the
    // caller's.
    let dtype = crate::serial::wire::tags::data_type_from_tag::data_type_from_tag(bytes[*cursor])
        .map_err(GraphDeltaError::Wire)?;
    *cursor += 1;
    let access = crate::serial::wire::tags::access_from_tag::access_from_tag(bytes[*cursor])
        .map_err(GraphDeltaError::Wire)?;
    *cursor += 1;
    let lifetime = match bytes[*cursor] {
        1 => ValueLifetime::Constant,
        2 => ValueLifetime::Invocation,
        3 => ValueLifetime::Retained,
        4 => ValueLifetime::Output,
        5 => ValueLifetime::Stream,
        unknown => {
            return Err(GraphDeltaError::Wire(format!(
                "value lifetime tag {unknown} is not one of the valid `ValueLifetime` tags"
            )));
        }
    };
    *cursor += 1;
    let rank = read_u32(bytes, cursor)? as usize;
    if rank > MAX_RANK {
        return Err(GraphDeltaError::Wire(format!(
            "tensor rank {rank} exceeds limit {MAX_RANK}"
        )));
    }
    let mut shape = Vec::with_capacity(rank.min(bytes.len() - *cursor));
    for _ in 0..rank {
        if *cursor >= bytes.len() {
            return Err(GraphDeltaError::Wire("EOF reading shape dim".into()));
        }
        let tag = bytes[*cursor];
        *cursor += 1;
        match tag {
            1 => {
                let ext = read_u64(bytes, cursor)?;
                shape.push(ShapeDim::Known(ext));
            }
            2 => {
                let sym = read_string(bytes, cursor)?;
                shape.push(ShapeDim::Symbol(sym));
            }
            3 => {
                shape.push(ShapeDim::Unresolved);
            }
            4 => {
                let id = read_u32(bytes, cursor)?;
                shape.push(ShapeDim::Expr(crate::types::ShapeExprId(id)));
            }
            unknown => {
                return Err(GraphDeltaError::Wire(format!(
                    "shape dim tag {unknown} is not a valid `ShapeDim` tag"
                )));
            }
        }
    }
    Ok(ValueContract {
        dtype,
        shape,
        access,
        lifetime,
    })
}

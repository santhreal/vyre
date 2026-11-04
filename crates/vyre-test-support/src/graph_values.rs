//! Value contracts and output bindings for graph fixtures.
//!
//! A graph output states four fields: the program buffer it drains, the graph
//! value name it publishes, the value contract, and the retained-successor slot
//! that only a resident fixture fills. Every fixture that declares one repeats
//! those four fields with the successor slot empty, and every scalar contract
//! repeats the same `u32` element type. Both are stated once here so a fixture
//! names what differs.

use vyre_foundation::ir::{
    BufferAccess, DataType, GraphOutput, ShapeDim, ValueContract, ValueLifetime,
};

/// A single-element `u32` contract under `access` and `lifetime`.
#[must_use]
pub fn u32_scalar(access: BufferAccess, lifetime: ValueLifetime) -> ValueContract {
    ValueContract {
        dtype: DataType::U32,
        shape: vec![ShapeDim::Known(1)],
        access,
        lifetime,
    }
}

/// A contract of `count` elements of `dtype` under `access` and `lifetime`.
#[must_use]
pub fn typed_vector(
    count: u32,
    dtype: DataType,
    access: BufferAccess,
    lifetime: ValueLifetime,
) -> ValueContract {
    ValueContract {
        dtype,
        shape: vec![ShapeDim::Known(u64::from(count))],
        access,
        lifetime,
    }
}

/// A `u32` contract of `count` elements under `access` and `lifetime`.
#[must_use]
pub fn u32_vector(count: u32, access: BufferAccess, lifetime: ValueLifetime) -> ValueContract {
    typed_vector(count, DataType::U32, access, lifetime)
}

/// A `u32` contract whose extent is the symbol `items`, under `access` and `lifetime`.
#[must_use]
pub fn u32_symbolic(access: BufferAccess, lifetime: ValueLifetime) -> ValueContract {
    ValueContract {
        dtype: DataType::U32,
        shape: vec![ShapeDim::Symbol("items".into())],
        access,
        lifetime,
    }
}

/// A graph output draining `buffer` into a value of the same name under `contract`.
///
/// The retained-successor slot is empty. A resident fixture that pins a
/// successor states the whole binding, because the successor identity is the
/// only reason that fixture exists.
#[must_use]
pub fn graph_output(buffer: &str, contract: ValueContract) -> GraphOutput {
    GraphOutput {
        buffer: buffer.into(),
        name: buffer.into(),
        contract,
        retained_successor_of: None,
    }
}

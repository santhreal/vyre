//! `ProgramDescriptor`  -  lightweight structural introspection of a
//! Cat-A composition's Program without running the full IR builder.
//!
//! The descriptor answers questions like "how many buffers does this
//! op declare?" and "what's its canonical workgroup size?" without
//! forcing the caller to pay the full Program construction cost.
//! P2.9 ships descriptors derived from a built Program; a future
//! optimization may lazy-construct descriptors without materializing
//! the full IR, but the surface here is the contract external
//! tooling pins against.

use vyre_foundation::ir::{BufferAccess, DataType, Program};

/// Structural description of a Cat-A Program.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ProgramDescriptor {
    /// Number of declared buffers.
    pub buffer_count: usize,
    /// Canonical workgroup dispatch size.
    pub workgroup_size: [u32; 3],
    /// Buffer summaries, one per declared buffer.
    pub buffers: Vec<BufferDescriptor>,
    /// Total element-bytes declared across ReadWrite buffers. Useful
    /// for rough memory-footprint estimates. Missing counts (runtime-
    /// determined buffer sizes) contribute zero.
    pub rw_bytes_lower_bound: usize,
    /// Number of top-level nodes in the entry body.
    pub entry_node_count: usize,
}

/// One buffer summary inside a [`ProgramDescriptor`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct BufferDescriptor {
    /// Declared name (matches `TensorRef::name`).
    pub name: String,
    /// Storage-class access mode.
    pub access: BufferAccess,
    /// Element dtype.
    pub dtype: DataType,
    /// Element count, or 0 when the size is runtime-determined.
    pub count: u32,
}

impl BufferDescriptor {
    /// Construct a `BufferDescriptor` from explicit fields. External
    /// tooling that synthesizes buffer summaries uses this constructor
    /// (V7-EXT-022).
    #[must_use]
    pub fn new(name: String, access: BufferAccess, dtype: DataType, count: u32) -> Self {
        Self {
            name,
            access,
            dtype,
            count,
        }
    }
}

impl ProgramDescriptor {
    /// Construct a `ProgramDescriptor` directly from explicit fields.
    /// External tooling that synthesizes descriptors without going
    /// through `from_program` uses this constructor (V7-EXT-023).
    #[must_use]
    pub fn new(
        buffer_count: usize,
        workgroup_size: [u32; 3],
        buffers: Vec<BufferDescriptor>,
        rw_bytes_lower_bound: usize,
        entry_node_count: usize,
    ) -> Self {
        Self {
            buffer_count,
            workgroup_size,
            buffers,
            rw_bytes_lower_bound,
            entry_node_count,
        }
    }

    /// Derive a descriptor from an already-built Program. Zero-allocation
    /// aside from the owned buffer-name strings (one per declared
    /// buffer); consumers that need every dispatch to stay cheap
    /// should cache the descriptor once and reuse it.
    #[must_use]
    pub fn from_program(program: &Program) -> Self {
        let buffers: Vec<BufferDescriptor> = program
            .buffers()
            .iter()
            .map(|b| BufferDescriptor {
                name: b.name.to_string(),
                access: b.access.clone(),
                dtype: b.element.clone(),
                count: b.count,
            })
            .collect();

        let rw_bytes_lower_bound: usize = buffers
            .iter()
            .filter(|b| matches!(b.access, BufferAccess::ReadWrite))
            .map(|b| {
                let elem_bytes = b.dtype.size_bytes().unwrap_or(0);
                (b.count as usize).saturating_mul(elem_bytes)
            })
            .sum();

        Self {
            buffer_count: buffers.len(),
            workgroup_size: program.workgroup_size(),
            rw_bytes_lower_bound,
            entry_node_count: program.entry().len(),
            buffers,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vyre_foundation::ir::{BufferDecl, Node};

    #[test]
    fn descriptor_summarizes_program() {
        let program = Program::wrapped(
            vec![
                BufferDecl::storage("in", 0, BufferAccess::ReadWrite, DataType::F32).with_count(64),
                BufferDecl::output("out", 1, DataType::F32).with_count(64),
                BufferDecl::storage("param", 2, BufferAccess::ReadOnly, DataType::F32)
                    .with_count(64),
            ],
            [64, 1, 1],
            Vec::new(),
        );

        let desc = ProgramDescriptor::from_program(&program);

        assert_eq!(desc.buffer_count, 3);
        assert_eq!(desc.workgroup_size, [64, 1, 1]);
        // `Program::wrapped` installs a root region around the entry, so an
        // empty entry still reports one top-level node.
        assert_eq!(desc.entry_node_count, 1);
        assert!(
            matches!(program.entry()[0], Node::Region { .. }),
            "Fix: the single top-level node must be the root region wrap."
        );
        assert_eq!(desc.buffers[0].name, "in");
        assert_eq!(desc.buffers[1].name, "out");
        assert_eq!(desc.buffers[2].name, "param");
        // `BufferDecl::output` is read-write storage, so "in" and "out" both
        // count at 64 F32 elements times 4 bytes. The read-only "param" is
        // excluded, which is what makes the access filter observable here.
        assert_eq!(desc.rw_bytes_lower_bound, 512);
    }
}

//! Canonical target-only I/O and memory operation registrations.

use vyre_foundation::dialect_lookup::{Signature, TypedParam};
use vyre_foundation::operation::{OperationRegistration, OperationTier};

const OP_DMA_FROM_NVME: &str = "io.dma_from_nvme";
const OP_WRITE_BACK_TO_NVME: &str = "io.write_back_to_nvme";
const OP_ZEROCOPY_MAP: &str = "mem.zerocopy_map";
const OP_UNMAP: &str = "mem.unmap";

const SIG_DMA_FROM_NVME: Signature = Signature {
    inputs: &[
        TypedParam {
            name: "fd",
            ty: "i32",
        },
        TypedParam {
            name: "offset",
            ty: "u64",
        },
        TypedParam {
            name: "length",
            ty: "u64",
        },
    ],
    outputs: &[TypedParam {
        name: "handle",
        ty: "GpuBufferHandle",
    }],
    attrs: &[],
    bytes_extraction: false,
};

const SIG_WRITE_BACK_TO_NVME: Signature = Signature {
    inputs: &[
        TypedParam {
            name: "handle",
            ty: "GpuBufferHandle",
        },
        TypedParam {
            name: "fd",
            ty: "i32",
        },
        TypedParam {
            name: "offset",
            ty: "u64",
        },
    ],
    outputs: &[],
    attrs: &[],
    bytes_extraction: false,
};

const SIG_ZEROCOPY_MAP: Signature = Signature {
    inputs: &[TypedParam {
        name: "fd",
        ty: "i32",
    }],
    outputs: &[TypedParam {
        name: "handle",
        ty: "GpuBufferHandle",
    }],
    attrs: &[],
    bytes_extraction: false,
};

const SIG_UNMAP: Signature = Signature {
    inputs: &[TypedParam {
        name: "handle",
        ty: "GpuBufferHandle",
    }],
    outputs: &[],
    attrs: &[],
    bytes_extraction: false,
};

inventory::submit! {
    OperationRegistration::new(OP_DMA_FROM_NVME, OperationTier::Runtime, None, None, None)
        .with_signature(SIG_DMA_FROM_NVME)
        .with_category("io")
}
inventory::submit! {
    OperationRegistration::new(OP_WRITE_BACK_TO_NVME, OperationTier::Runtime, None, None, None)
        .with_signature(SIG_WRITE_BACK_TO_NVME)
        .with_category("io")
}
inventory::submit! {
    OperationRegistration::new(OP_ZEROCOPY_MAP, OperationTier::Runtime, None, None, None)
        .with_signature(SIG_ZEROCOPY_MAP)
        .with_category("mem")
}
inventory::submit! {
    OperationRegistration::new(OP_UNMAP, OperationTier::Runtime, None, None, None)
        .with_signature(SIG_UNMAP)
        .with_category("mem")
}

#[cfg(test)]
mod tests {
    use super::*;
    use vyre_foundation::operation::OperationRegistry;

    #[test]
    fn io_operations_are_signature_only_semantic_records() {
        for (id, category) in [
            (OP_DMA_FROM_NVME, "io"),
            (OP_WRITE_BACK_TO_NVME, "io"),
            (OP_ZEROCOPY_MAP, "mem"),
            (OP_UNMAP, "mem"),
        ] {
            let operation = OperationRegistry::global()
                .get(id)
                .expect("canonical I/O operation");
            assert_eq!(operation.category, Some(category));
            assert!(operation.signature.is_some());
            assert!(operation.program().is_none());
        }
    }
}

//! One operation signature, taken from a built Program or a declaration.

use vyre::ir::Program;

use super::schema::{BufferSignature, OperationSignature, TypedParameter};

pub(super) fn signature_from_program(program: &Program) -> OperationSignature {
    OperationSignature {
        kind: "program_buffers".to_string(),
        buffers: program
            .buffers()
            .iter()
            .map(|buffer| BufferSignature {
                binding: buffer.binding,
                name: buffer.name.to_string(),
                access: format!("{:?}", buffer.access),
                memory: format!("{:?}", buffer.kind),
                element: format!("{:?}", buffer.element),
                count: buffer.count,
                pipeline_live_out: buffer.pipeline_live_out,
            })
            .collect(),
        inputs: Vec::new(),
        outputs: Vec::new(),
        attributes: Vec::new(),
        bytes_extraction: program
            .buffers()
            .iter()
            .any(|buffer| buffer.bytes_extraction),
    }
}

pub(super) fn signature_from_declaration(
    signature: &vyre_foundation::dialect_lookup::Signature,
) -> OperationSignature {
    OperationSignature {
        kind: "dialect_parameters".to_string(),
        buffers: Vec::new(),
        inputs: signature
            .inputs
            .iter()
            .map(|parameter| TypedParameter {
                name: parameter.name.to_string(),
                data_type: parameter.ty.to_string(),
            })
            .collect(),
        outputs: signature
            .outputs
            .iter()
            .map(|parameter| TypedParameter {
                name: parameter.name.to_string(),
                data_type: parameter.ty.to_string(),
            })
            .collect(),
        attributes: signature
            .attrs
            .iter()
            .map(|attribute| format!("{attribute:?}"))
            .collect(),
        bytes_extraction: signature.bytes_extraction,
    }
}

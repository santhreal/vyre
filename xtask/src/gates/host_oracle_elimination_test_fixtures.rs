//! Source fixtures the host oracle elimination tests share.
//!
//! Every case in this family needs a function body the gate must judge: a host
//! loop that derives bytes from input words. A case varies the item that wraps
//! that body, the name it gives it, and whether the item is test scoped. The
//! body itself is never what a case varies, so it is written here and each case
//! supplies the rest. Indentation is cosmetic to the parser, so one form serves
//! a free function, a method and a trait default alike.

/// A host oracle body that folds each input word into output bytes.
///
/// `op` is the arithmetic applied to each word, so a case that must be
/// distinguishable from another case states its own operation.
pub(super) fn oracle_body(op: &str) -> String {
    format!(
        "    let mut out = Vec::new();
    for &x in input {{
        out.extend_from_slice(&x.{op}.to_le_bytes());
    }}
    out"
    )
}

/// A host oracle body that adds one to each input word.
pub(super) fn incrementing_oracle_body() -> String {
    oracle_body("wrapping_add(1)")
}

/// A staging module that packs graph input bytes into a named carrier type.
///
/// The semantic seam binds byte payloads to graph values, so staging produces
/// bytes and nothing else: no device handle, no launch geometry. `field_vis`
/// varies because a carrier with public fields can be built without its
/// producer. Each `(name, mask)` pair adds one producer of the same nominal
/// type, so a case presents either a unique producer or a second unused one.
pub(super) fn resident_staging_source(
    type_name: &str,
    field_vis: &str,
    producers: &[(&str, &str)],
) -> String {
    let mut source = format!(
        "use vyre_megakernel::SemanticExecutionError;

pub struct {type_name} {{
    {field_vis}packed: Vec<u8>,
}}
"
    );
    for (name, mask) in producers {
        source.push_str(&format!(
            "
pub fn {name}(
    node_count: u32,
    edges: &[u32],
) -> Result<{type_name}, SemanticExecutionError> {{
    let mut packed = Vec::with_capacity(edges.len() * 4);
    for &edge in edges {{
        let encoded = (edge ^ {mask}).wrapping_mul(node_count | 1);
        packed.extend_from_slice(&encoded.to_le_bytes());
    }}
    Ok({type_name} {{ packed }})
}}
"
        ));
    }
    source
}

/// The imports a dispatch fixture needs to build and submit a semantic request.
pub(super) const CANONICAL_DISPATCH_IMPORTS: &str = "use std::collections::BTreeMap;
use vyre_foundation::ir::GraphValueId;
use vyre_foundation::logical::LogicalProgramGraph;
use vyre_megakernel::{
    SemanticExecutionError, SemanticExecutionPolicy, SemanticExecutionRequest, SemanticExecutor,
};
";

/// The request construction every dispatch fixture reaches the seam through.
///
/// Stated once because a case varies what surrounds the submission, never the
/// arguments the seam takes. A second copy of these arguments in a fixture is
/// what `dup-scan` counts, and a case that drifts from this one proves the
/// scanner against a request shape no production caller writes.
pub(super) const CANONICAL_REQUEST_ARGUMENTS: &str = "        logical,
        inputs,
        policy.clone(),";

/// A function that binds staged bytes into a request and submits it.
///
/// `vis` varies because a case distinguishes a public entry point from a private
/// helper reached through one. `extra_params` and `extra_body` carry whatever
/// else the case places around the submission.
pub(super) fn canonical_dispatch_fn(
    vis: &str,
    name: &str,
    carrier: &str,
    extra_params: &str,
    extra_body: &str,
) -> String {
    format!(
        "
{vis}fn {name}(
    dispatcher: &impl SemanticExecutor,
    logical: &LogicalProgramGraph<'_>,
    policy: &SemanticExecutionPolicy,
    graph: &{carrier},{extra_params}
) -> Result<(), SemanticExecutionError> {{{extra_body}
    let mut inputs = BTreeMap::new();
    inputs.insert(GraphValueId(0), graph.packed.as_slice());
    let request = SemanticExecutionRequest::new(
{CANONICAL_REQUEST_ARGUMENTS}
    )?;
    dispatcher.execute(&request)?;
    Ok(())
}}
"
    )
}

/// A staging function that packs a payload, optionally binding it itself.
///
/// `binds` varies because the case that must be convicted is this exact host
/// loop with the binding call removed: the caller, the arithmetic and the name
/// stay identical, so the only difference between the permitted and the
/// convicted shape is whether the producer terminates in canonical binding.
pub(super) fn self_binding_staging_source(binds: bool) -> String {
    let host_loop = "    for &edge in edges {
        let encoded = (edge ^ 0x5A5A_5A5A).wrapping_mul(node_count | 1);
        packed.extend_from_slice(&encoded.to_le_bytes());
    }";
    if binds {
        format!(
            "{CANONICAL_DISPATCH_IMPORTS}
pub fn stage_demo_request<'a>(
    logical: &'a LogicalProgramGraph<'a>,
    policy: &SemanticExecutionPolicy,
    packed: &'a mut Vec<u8>,
    node_count: u32,
    edges: &[u32],
) -> Result<SemanticExecutionRequest<'a>, SemanticExecutionError> {{
{host_loop}
    let mut inputs = BTreeMap::new();
    inputs.insert(GraphValueId(0), &packed[..]);
    SemanticExecutionRequest::new(
{CANONICAL_REQUEST_ARGUMENTS}
    )
}}
"
        )
    } else {
        format!(
            "{CANONICAL_DISPATCH_IMPORTS}
pub fn stage_demo_request(
    node_count: u32,
    edges: &[u32],
) -> Result<Vec<u8>, SemanticExecutionError> {{
    let mut packed = Vec::new();
{host_loop}
    Ok(packed)
}}
"
        )
    }
}

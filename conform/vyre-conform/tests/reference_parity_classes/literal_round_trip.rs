//! WHY: closes the class "a lowered literal does not survive the target-module
//! bundle", which took `(cuda, vyre-libs::nn::top_k)`, `(wgpu, nn::top_k)` and
//! `(wgpu, nn::softmax_top_k)` out of the conformance certificate with `target
//! module bundle failed: invalid type: null, expected f32`. The bundle is JSON,
//! JSON has no non-finite number, and an op that seeds a running maximum with
//! negative infinity therefore wrote a literal the decoder refused.
//!
//! The roster is the operation registry: every op the registry carries a builder
//! for is lowered and its descriptor round-tripped through the real bundle
//! encoder and decoder. A new op whose literal pool holds a value the encoding
//! cannot carry turns this red without anyone editing a list.
//!
//! What it does not catch: a value that round-trips but is wrong, and any
//! encoding loss outside the descriptor. `bytes` is the dialect's own module
//! image and is opaque here, so a dialect that loses a literal inside its own
//! text is that dialect's golden corpus to catch, not this.

use std::collections::BTreeMap;

use vyre_foundation::ir::{Program, ProgramGraph};
use vyre_foundation::logical::LogicalProgramGraph;
use vyre_foundation::schedule::SelectedSchedule;
use vyre_lower::LiteralValue;
use vyre_megakernel::{
    ArtifactNodeId, FusionGroupId, TargetModuleBundle, TargetModuleImage,
    TARGET_MODULE_BUNDLE_SCHEMA_VERSION,
};
use vyre_registry_link::operation::live_operation_registry;

fn image_for(operation_id: &str, program: &Program) -> Result<TargetModuleImage, String> {
    let graph = ProgramGraph::from_program(operation_id, program.clone())
        .map_err(|error| format!("invalid program graph: {error}"))?;
    let logical = LogicalProgramGraph::validate(&graph, &BTreeMap::new())
        .map_err(|error| format!("invalid logical domain: {error}"))?;
    let schedule = vyre_megakernel::baseline_schedule(&logical);
    let phase = schedule
        .phases
        .first()
        .map(|phase| phase.id)
        .ok_or_else(|| "selected schedule has no phase".to_string())?;
    let lowered = vyre_lower::lower_scheduled(program, &schedule, phase)
        .map_err(|error| format!("selected-schedule lowering failed: {error}"))?;
    let program = lowered
        .program
        .to_wire()
        .map_err(|error| format!("physical program encoding failed: {error}"))?;
    let descriptor = lowered.into_descriptor();
    Ok(TargetModuleImage {
        group: FusionGroupId(0),
        stage: 0,
        nodes: vec![ArtifactNodeId(0)],
        program,
        descriptor,
        entry_point: "main".to_string(),
        bytes: Vec::new(),
    })
}

/// Re-seal a bundle body so a hand-edited copy passes the digest check and is
/// judged on its content, which is what a stored payload gets judged on.
fn sealed(body: &[u8]) -> Vec<u8> {
    let digest = blake3::hash(body);
    let mut bytes = Vec::with_capacity(32 + body.len());
    bytes.extend_from_slice(digest.as_bytes());
    bytes.extend_from_slice(body);
    bytes
}

#[test]
fn every_registered_op_descriptor_survives_the_target_module_bundle() {
    let mut lost = Vec::new();
    let mut checked = 0usize;
    for operation in live_operation_registry().iter() {
        let Some(program) = operation.program() else {
            continue;
        };
        let image = match image_for(operation.id, &program) {
            Ok(image) => image,
            Err(error) => {
                lost.push(format!("{}: {error}", operation.id));
                continue;
            }
        };
        let expected = image.descriptor.clone();
        let bundle = TargetModuleBundle::new(vec![image]);
        let bytes = match bundle.to_bytes() {
            Ok(bytes) => bytes,
            Err(error) => {
                lost.push(format!("{}: encode failed: {error}", operation.id));
                continue;
            }
        };
        checked += 1;
        match TargetModuleBundle::from_bytes(&bytes) {
            Ok(decoded) => {
                let descriptor = &decoded.modules[0].descriptor;
                if *descriptor != expected {
                    lost.push(format!(
                        "{}: descriptor changed across the bundle round trip",
                        operation.id
                    ));
                }
            }
            Err(error) => lost.push(format!("{}: decode failed: {error}", operation.id)),
        }
    }
    assert!(
        checked > 0,
        "Fix: no registered op lowered, so this test judged nothing."
    );
    assert!(
        lost.is_empty(),
        "Fix: a lowered descriptor does not survive the target-module bundle, so no backend can materialize the payload. Make the encoding carry the value rather than dropping the op.\n{}",
        lost.join("\n")
    );
}

#[test]
fn a_non_finite_f32_literal_survives_the_descriptor_encoding() {
    for value in [
        f32::NEG_INFINITY,
        f32::INFINITY,
        f32::NAN,
        -0.0,
        0.0,
        f32::MIN_POSITIVE / 2.0,
        1.0,
    ] {
        let literal = LiteralValue::F32(value);
        let encoded = serde_json::to_string(&literal)
            .expect("Fix: an f32 literal must encode in the bundle's own format.");
        let decoded: LiteralValue = serde_json::from_str(&encoded).unwrap_or_else(|error| {
            panic!("Fix: `{encoded}` must decode back to an f32 literal: {error}")
        });
        let LiteralValue::F32(returned) = decoded else {
            panic!("Fix: an f32 literal must decode as an f32 literal, got {decoded:?}");
        };
        assert_eq!(
            returned.to_bits(),
            value.to_bits(),
            "Fix: the descriptor encoding must carry every f32 bit pattern exactly; `{value:?}` came back as `{returned:?}`."
        );
    }
}

/// A finite literal keeps the encoding it had before the non-finite escape
/// existed, which is what bounds the change to the values JSON could not carry.
/// Every other surface that serializes a descriptor (a golden corpus, a
/// descriptor cache, tooling reading one as JSON) reads a finite literal, so
/// this is the assertion that says those surfaces did not change shape.
///
/// The expectation is derived from `serde_json`'s own f32 encoding rather than
/// written out, so it stays correct if that shortest-round-trip spelling ever
/// changes.
#[test]
fn a_finite_f32_literal_keeps_the_encoding_every_other_reader_already_parses() {
    for value in [
        0.0_f32,
        -0.0,
        1.0,
        -1.5,
        f32::MIN_POSITIVE,
        f32::MIN_POSITIVE / 2.0,
        f32::MAX,
        f32::MIN,
        1e-7,
        0.1,
    ] {
        let plain = serde_json::to_string(&value)
            .expect("Fix: a bare f32 must encode before it can be the expectation.");
        let encoded = serde_json::to_string(&LiteralValue::F32(value))
            .expect("Fix: a finite f32 literal must encode.");
        assert_eq!(
            encoded,
            format!("{{\"F32\":{plain}}}"),
            "Fix: a finite f32 literal must encode as the plain number, so no reader outside the target-module bundle sees a new shape; escape only the values JSON cannot represent."
        );
    }
}

/// A finite value written in the non-finite escape is refused rather than
/// accepted, because two spellings for one literal break the bundle's
/// canonical-bytes check: `from_bytes` re-encodes and demands byte identity, so
/// an accepted second spelling would fail there with a digest complaint that
/// names nothing.
#[test]
fn a_finite_f32_literal_written_in_the_non_finite_escape_is_refused() {
    let escaped = format!("{{\"F32\":\"0x{:08x}\"}}", 1.0_f32.to_bits());
    let error = serde_json::from_str::<LiteralValue>(&escaped)
        .expect_err("Fix: a finite value in the bit-pattern escape must be refused; one literal must have one spelling.")
        .to_string();
    assert!(
        error.contains("non-finite"),
        "Fix: the refusal must say the escape is for non-finite values only, because that is what tells the writer to emit a number instead; got: {error}"
    );
}

#[test]
fn a_bundle_written_under_an_earlier_schema_is_refused_by_version() {
    let (operation_id, program) = live_operation_registry()
        .iter()
        .find_map(|operation| operation.program().map(|program| (operation.id, program)))
        .expect("Fix: the operation registry must carry at least one buildable program.");
    let image = image_for(operation_id, &program)
        .expect("Fix: a registered program must lower before this contract can be judged.");
    let bytes = TargetModuleBundle::new(vec![image])
        .to_bytes()
        .expect("Fix: a lowered bundle must encode.");
    let mut body: serde_json::Value = serde_json::from_slice(&bytes[32..])
        .expect("Fix: the bundle body must be readable as the format it is written in.");
    let stale = TARGET_MODULE_BUNDLE_SCHEMA_VERSION - 1;
    body["schema_version"] = serde_json::json!(stale);
    let stale_bytes =
        sealed(&serde_json::to_vec(&body).expect("Fix: the edited bundle body must re-encode."));

    let error = TargetModuleBundle::from_bytes(&stale_bytes)
        .expect_err("Fix: a bundle from an earlier schema must be refused, not reinterpreted: its fields were written under a different encoding.")
        .to_string();
    assert!(
        error.contains(&format!("schema {stale} is unsupported")),
        "Fix: the refusal must name the schema it read and the one it expects, because that is what tells an operator to rebuild the payload; got: {error}"
    );
}

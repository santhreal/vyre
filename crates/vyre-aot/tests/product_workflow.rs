//! The checkpoint-to-package-to-inspection workflow across three crates.
//!
//! `vyre-safetensors` verifies checkpoint bytes, `vyre-aot` compiles a program
//! and packages those bytes beside the artifact, and `vyre-debug` projects the
//! installed envelope for inspection. Nothing in the workspace depends on the
//! three together, so the seams between them are only exercised here.

use crate::fixture_target;

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use vyre_aot::{
    compile, install_package, load_installed_package, package_artifact, TargetId,
    ValidatedCompileRequest,
};
use vyre_debug::ArtifactReport;
use vyre_foundation::ir::{BufferDecl, DataType, Expr, Node, Program, ProgramGraph};
use vyre_foundation::validate::BackendCapabilities;
use vyre_megakernel::{
    ArtifactEnvelope, CompileObjective, CompileRequest, DeviceFacts, Digest, ExternalFacts,
    ObjectiveMetric, SearchBudget,
};
use vyre_safetensors::{ExpectedShardDigest, SafetensorError, ShardedSafetensorIndex};

/// Elements in the `params` resource the checkpoint supplies.
const PARAM_ELEMENTS: usize = 64;
/// Byte width of one `params` element. `DataType::U32` and safetensors `U32`
/// must agree, or the checkpoint cannot fill the resource it is packaged for.
const PARAM_ELEMENT_BYTES: usize = 4;
const SHARD_FILE: &str = "params.safetensors";

/// A program whose first bound resource is the one the checkpoint fills.
fn params_copy_program() -> Program {
    Program::wrapped(
        vec![
            BufferDecl::read("params", 0, DataType::U32).with_count(PARAM_ELEMENTS as u32),
            BufferDecl::read_write("out", 1, DataType::U32).with_count(PARAM_ELEMENTS as u32),
        ],
        [1, 1, 1],
        vec![
            Node::let_bind("idx", Expr::u32(0)),
            Node::store(
                "out",
                Expr::var("idx"),
                Expr::load("params", Expr::var("idx")),
            ),
        ],
    )
}

fn validated_request(external_facts: u8) -> ValidatedCompileRequest {
    let graph = ProgramGraph::from_program("main", params_copy_program())
        .expect("the fixture program must resolve into a graph");
    CompileRequest::new(
        graph,
        ExternalFacts::new(Digest([external_facts; 32]), BTreeMap::new()),
        DeviceFacts::new(BackendCapabilities::default(), 1024),
        SearchBudget::new(8, 1_000, 2, 0, 10_000_000),
        CompileObjective::minimize_latency()
            .with_bound(ObjectiveMetric::ArtifactBytes, 64 * 1024 * 1024),
    )
    .validate()
    .expect("the fixture request must validate")
}

/// Write a single-shard safetensors checkpoint holding `params` and its index.
fn write_checkpoint(root: &Path, fill: u8) -> [u8; 32] {
    let payload: Vec<u8> = (0..PARAM_ELEMENTS * PARAM_ELEMENT_BYTES)
        .map(|byte| fill.wrapping_add(byte as u8))
        .collect();
    let header = format!(
        r#"{{"params":{{"dtype":"U32","shape":[{PARAM_ELEMENTS}],"data_offsets":[0,{}]}}}}"#,
        payload.len()
    );
    let mut bytes = Vec::with_capacity(8 + header.len() + payload.len());
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(&payload);
    fs::write(root.join(SHARD_FILE), &bytes).expect("the fixture shard must be writable");
    fs::write(
        root.join("model.safetensors.index.json"),
        format!(r#"{{"weight_map":{{"params":"{SHARD_FILE}"}}}}"#),
    )
    .expect("the fixture index must be writable");
    *blake3::hash(&bytes).as_bytes()
}

/// Verify the checkpoint under `root` and return the `params` bytes.
fn verified_params(root: &Path, shard_digest: [u8; 32]) -> Vec<u8> {
    let index = ShardedSafetensorIndex::open(root, &root.join("model.safetensors.index.json"))
        .expect("the fixture checkpoint index must open");
    let checkpoint = index
        .verify_transactional([ExpectedShardDigest {
            shard: Path::new(SHARD_FILE),
            blake3: shard_digest,
        }])
        .expect("the fixture shard must verify against its own digest");
    checkpoint
        .tensor_reader("params")
        .expect("the verified checkpoint must expose `params`")
        .read_bytes()
        .expect("verified tensor bytes must be readable")
}

/// Package `weights` beside a compile of `request`, install it, and load it back.
fn package_install_and_load(
    root: &Path,
    request: &ValidatedCompileRequest,
    weights: &[u8],
    artifact_name: &str,
) -> (
    vyre_aot::Manifest,
    ArtifactEnvelope,
    Vec<u8>,
    ArtifactEnvelope,
) {
    let target: TargetId = fixture_target::fixture_target();
    let compiled = compile(request, target.clone()).expect("the fixture target must compile");
    let archive = root.join(format!("{artifact_name}-archive"));
    let install = root.join(format!("{artifact_name}-install"));
    package_artifact(
        &archive,
        &compiled,
        target,
        weights,
        artifact_name,
        "checkpoint weights verified by vyre-safetensors",
    )
    .expect("the compiled artifact must package");
    install_package(&archive, &install).expect("the package must install");
    let (manifest, envelope, loaded_weights) =
        load_installed_package(&install).expect("the installed package must load");
    (manifest, envelope, loaded_weights, compiled)
}

/// WHY: this is the only place the three crates meet. The bytes a checkpoint
/// verifies must be the bytes an installed package hands back, unchanged by
/// compression, by the manifest digest round trip, or by the install copy.
/// A plausible bug at any one of those seams returns different bytes while
/// every individual crate's own tests stay green.
#[test]
fn verified_checkpoint_bytes_survive_packaging_and_installation_unchanged() {
    let temp = tempfile::tempdir().expect("tempdir");
    let checkpoint_root = temp.path().join("checkpoint");
    fs::create_dir_all(&checkpoint_root).expect("checkpoint directory must be creatable");
    let shard_digest = write_checkpoint(&checkpoint_root, 0x10);
    let params = verified_params(&checkpoint_root, shard_digest);
    assert_eq!(params.len(), PARAM_ELEMENTS * PARAM_ELEMENT_BYTES);

    let (manifest, _, loaded_weights, _) = package_install_and_load(
        temp.path(),
        &validated_request(0x55),
        &params,
        "checkpoint-package",
    );

    assert_eq!(
        loaded_weights, params,
        "Fix: the installed package must return the verified checkpoint bytes byte for byte"
    );
    assert_eq!(
        manifest.weights_sha256_hex,
        sha256_hex(&params),
        "Fix: the manifest must record the digest of the uncompressed weights"
    );
}

/// WHY: artifact identity is over the program and the compile request, and the
/// checkpoint is data the artifact is packaged with. Two checkpoints compiled
/// from the same request must produce the same artifact digest, the same
/// request digest and the same selected plan, and must differ only in the
/// weights the manifest records. Folding weight bytes into artifact identity
/// would defeat every artifact cache; reporting one constant identity for both
/// would defeat the inspection surface. This case fails on either.
#[test]
fn the_debug_report_separates_artifact_identity_from_the_packaged_checkpoint() {
    let temp = tempfile::tempdir().expect("tempdir");
    let request = validated_request(0x55);

    let first_root = temp.path().join("checkpoint-a");
    let second_root = temp.path().join("checkpoint-b");
    fs::create_dir_all(&first_root).expect("first checkpoint directory must be creatable");
    fs::create_dir_all(&second_root).expect("second checkpoint directory must be creatable");
    let first_params = verified_params(&first_root, write_checkpoint(&first_root, 0x10));
    let second_params = verified_params(&second_root, write_checkpoint(&second_root, 0x90));
    assert_ne!(
        first_params, second_params,
        "the two fixture checkpoints must differ, or this case proves nothing"
    );

    let (first_manifest, first_envelope, _, first_compiled) =
        package_install_and_load(temp.path(), &request, &first_params, "package-a");
    let (second_manifest, second_envelope, _, _) =
        package_install_and_load(temp.path(), &request, &second_params, "package-b");

    let first = ArtifactReport::from_envelope(&first_envelope);
    let second = ArtifactReport::from_envelope(&second_envelope);

    assert_eq!(
        first, second,
        "Fix: packaged checkpoint bytes must not reach artifact identity or the selected plan"
    );
    assert_ne!(
        first_manifest.weights_sha256_hex, second_manifest.weights_sha256_hex,
        "Fix: the manifest must distinguish two different checkpoints"
    );

    // The report is a projection of the compiled envelope, not of the package.
    assert_eq!(
        first,
        ArtifactReport::from_envelope(&first_compiled),
        "Fix: packaging and installation must not alter what the debug report projects"
    );

    // Changing the compile request must move the request digest, so the report
    // is distinguishing compiles rather than reporting a constant.
    let (_, other_envelope, _, _) = package_install_and_load(
        temp.path(),
        &validated_request(0xAA),
        &first_params,
        "package-c",
    );
    let other = ArtifactReport::from_envelope(&other_envelope);
    assert_ne!(
        first.request, other.request,
        "Fix: the debug report must project the compile request the artifact was built from"
    );
    assert_ne!(
        first.artifact, other.artifact,
        "Fix: a different compile request must yield a different artifact identity"
    );
}

/// WHY: the packaged weights are authenticated against the manifest, so an
/// edit to the installed file has to be refused rather than served. Without
/// this the checkpoint verification `vyre-safetensors` performs is undone the
/// moment the bytes land on disk, and the workflow above proves nothing.
#[test]
fn an_edited_installed_weights_file_is_refused_by_the_loader() {
    let temp = tempfile::tempdir().expect("tempdir");
    let checkpoint_root = temp.path().join("checkpoint");
    fs::create_dir_all(&checkpoint_root).expect("checkpoint directory must be creatable");
    let params = verified_params(&checkpoint_root, write_checkpoint(&checkpoint_root, 0x10));

    let target: TargetId = fixture_target::fixture_target();
    let compiled =
        compile(&validated_request(0x55), target.clone()).expect("the fixture target must compile");
    let archive = temp.path().join("archive");
    let install = temp.path().join("install");
    package_artifact(
        &archive,
        &compiled,
        target,
        &params,
        "tamper-package",
        "checkpoint weights verified by vyre-safetensors",
    )
    .expect("the compiled artifact must package");
    install_package(&archive, &install).expect("the package must install");

    let installed_weights = install.join("active").join("weights.brotli");
    let mut bytes = fs::read(&installed_weights).expect("installed weights must be readable");
    let last = bytes.len() - 1;
    bytes[last] ^= 0xFF;
    fs::write(&installed_weights, &bytes).expect("installed weights must be writable");

    let error = load_installed_package(&install)
        .expect_err("an edited weights file must not load as an installed package");
    let rendered = error.to_string();
    assert!(
        rendered.contains("weights") || rendered.contains("brotli"),
        "Fix: the loader must name the weights as the rejected part, got `{rendered}`"
    );
}

/// WHY: the checkpoint fills a canonical resource, so its byte length must
/// equal what the compiled artifact declares for that resource. A checkpoint
/// whose element width disagrees with the resource packages a short payload,
/// and every later digest check accepts it because the digest is taken over
/// the short payload. The artifact's own resource table is the authority.
#[test]
fn the_checkpoint_payload_exactly_fills_the_resource_the_artifact_declares() {
    let temp = tempfile::tempdir().expect("tempdir");
    let root = temp.path().join("checkpoint");
    fs::create_dir_all(&root).expect("checkpoint directory must be creatable");
    let params = verified_params(&root, write_checkpoint(&root, 0x10));

    let compiled = compile(&validated_request(0x55), fixture_target::fixture_target())
        .expect("the fixture target must compile");
    let resource = compiled
        .neutral()
        .resources()
        .iter()
        .find(|resource| resource.name == "params")
        .expect("the compiled artifact must declare a `params` resource");

    assert_eq!(
        params.len() as u64,
        resource.byte_count,
        "Fix: the checkpoint payload must exactly fill the resource the artifact declares"
    );
    assert_eq!(
        resource.byte_count / resource.element_count,
        PARAM_ELEMENT_BYTES as u64,
        "Fix: the safetensors `U32` width and the artifact resource element width must agree"
    );
}

/// WHY: a checkpoint that names a tensor the program does not declare is a
/// caller error, and the workflow must surface it by name rather than
/// packaging whatever bytes happened to verify.
#[test]
fn a_checkpoint_missing_the_named_tensor_is_refused_by_name() {
    let temp = tempfile::tempdir().expect("tempdir");
    let root = temp.path().join("checkpoint");
    fs::create_dir_all(&root).expect("checkpoint directory must be creatable");
    let shard_digest = write_checkpoint(&root, 0x10);
    let index = ShardedSafetensorIndex::open(&root, &root.join("model.safetensors.index.json"))
        .expect("the fixture checkpoint index must open");
    let checkpoint = index
        .verify_transactional([ExpectedShardDigest {
            shard: Path::new(SHARD_FILE),
            blake3: shard_digest,
        }])
        .expect("the fixture shard must verify");

    let error = checkpoint
        .tensor_reader("bias")
        .expect_err("a tensor the checkpoint does not hold must not produce a reader");
    assert!(
        matches!(&error, SafetensorError::MissingRequiredTensor { name } if name == "bias"),
        "Fix: a missing checkpoint tensor must be refused by name, got {error:?}"
    );
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::Digest as _;
    let mut hasher = sha2::Sha256::new();
    hasher.update(bytes);
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

//! Proves the safetensors checkpoint ingestion contract of [`CheckpointManifest`].
//!
//! Every case writes a real sharded safetensors checkpoint into a scratch
//! directory under this package and ingests it through the verified path.
//! The element-type coverage is closed twice: an exhaustive `match` over
//! [`SafetensorDtype`] with no catch-all arm, and a run-time comparison of the
//! covered variant names against the enum declaration in `vyre-safetensors`.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use vyre::ir::DataType;
use vyre_model_compiler::{
    CheckpointManifest, ExpectedShardDigest, ManifestError, SafetensorDtype, SafetensorError,
};

/// One tensor written into a scratch shard.
struct TensorSpec {
    name: String,
    dtype: SafetensorDtype,
    shape: Vec<u64>,
    payload: Vec<u8>,
}

impl TensorSpec {
    fn new(name: &str, dtype: SafetensorDtype, shape: &[u64]) -> Self {
        let elements: u64 = shape.iter().product();
        let byte_len = elements * dtype_facts(dtype).byte_width;
        let payload = (0..byte_len).map(|index| (index % 251) as u8).collect();
        Self {
            name: name.to_string(),
            dtype,
            shape: shape.to_vec(),
            payload,
        }
    }
}

/// Independent expectation for one safetensors element type.
struct DtypeFacts {
    /// Element type string as it appears in a safetensors header.
    header: &'static str,
    /// Stored width of one element in bytes.
    byte_width: u64,
    /// IR data type the checkpoint ingestion path must produce, if any.
    data_type: Option<DataType>,
}

/// Restate the element-type contract independently of the crate under test.
///
/// The `match` is exhaustive with no catch-all arm, so an element type added to
/// `vyre-safetensors` stops this test from compiling.
fn dtype_facts(dtype: SafetensorDtype) -> DtypeFacts {
    let (header, byte_width, data_type) = match dtype {
        SafetensorDtype::BOOL => ("BOOL", 1, Some(DataType::Bool)),
        SafetensorDtype::U8 => ("U8", 1, Some(DataType::U8)),
        SafetensorDtype::I8 => ("I8", 1, Some(DataType::I8)),
        SafetensorDtype::U16 => ("U16", 2, Some(DataType::U16)),
        SafetensorDtype::I16 => ("I16", 2, Some(DataType::I16)),
        SafetensorDtype::U32 => ("U32", 4, Some(DataType::U32)),
        SafetensorDtype::I32 => ("I32", 4, Some(DataType::I32)),
        SafetensorDtype::U64 => ("U64", 8, Some(DataType::U64)),
        SafetensorDtype::I64 => ("I64", 8, Some(DataType::I64)),
        SafetensorDtype::F16 => ("F16", 2, Some(DataType::F16)),
        SafetensorDtype::BF16 => ("BF16", 2, Some(DataType::BF16)),
        SafetensorDtype::F32 => ("F32", 4, Some(DataType::F32)),
        SafetensorDtype::F64 => ("F64", 8, Some(DataType::F64)),
        SafetensorDtype::F8E4M3 => ("F8_E4M3", 1, None),
        SafetensorDtype::F8E5M2 => ("F8_E5M2", 1, None),
    };
    DtypeFacts {
        header,
        byte_width,
        data_type,
    }
}

/// Rust identifier of one element type, used to compare against the declaration.
///
/// Exhaustive with no catch-all arm for the same reason as [`dtype_facts`].
const fn dtype_variant_name(dtype: SafetensorDtype) -> &'static str {
    match dtype {
        SafetensorDtype::BOOL => "BOOL",
        SafetensorDtype::U8 => "U8",
        SafetensorDtype::I8 => "I8",
        SafetensorDtype::U16 => "U16",
        SafetensorDtype::I16 => "I16",
        SafetensorDtype::U32 => "U32",
        SafetensorDtype::I32 => "I32",
        SafetensorDtype::U64 => "U64",
        SafetensorDtype::I64 => "I64",
        SafetensorDtype::F16 => "F16",
        SafetensorDtype::BF16 => "BF16",
        SafetensorDtype::F32 => "F32",
        SafetensorDtype::F64 => "F64",
        SafetensorDtype::F8E4M3 => "F8E4M3",
        SafetensorDtype::F8E5M2 => "F8E5M2",
    }
}

/// Element types this test drives. Held to the declaration at run time by
/// `covered_dtypes_match_the_safetensors_declaration`.
const COVERED_DTYPES: &[SafetensorDtype] = &[
    SafetensorDtype::BOOL,
    SafetensorDtype::U8,
    SafetensorDtype::I8,
    SafetensorDtype::U16,
    SafetensorDtype::I16,
    SafetensorDtype::U32,
    SafetensorDtype::I32,
    SafetensorDtype::U64,
    SafetensorDtype::I64,
    SafetensorDtype::F16,
    SafetensorDtype::BF16,
    SafetensorDtype::F32,
    SafetensorDtype::F64,
    SafetensorDtype::F8E4M3,
    SafetensorDtype::F8E5M2,
];

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("consumers directory")
        .parent()
        .expect("workspace root")
        .to_path_buf()
}

/// Create an empty scratch directory inside this package's ignored build output.
fn scratch(case: &str) -> PathBuf {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("checkpoint-ingestion")
        .join(case);
    if root.exists() {
        fs::remove_dir_all(&root).expect("clear scratch directory");
    }
    fs::create_dir_all(&root).expect("create scratch directory");
    root
}

/// Write one safetensors shard and return its full-file BLAKE3 digest.
fn write_shard(path: &Path, tensors: &[TensorSpec]) -> [u8; 32] {
    let mut header = String::from("{");
    let mut offset = 0_u64;
    for (position, tensor) in tensors.iter().enumerate() {
        if position > 0 {
            header.push(',');
        }
        let end = offset + tensor.payload.len() as u64;
        let shape = tensor
            .shape
            .iter()
            .map(u64::to_string)
            .collect::<Vec<_>>()
            .join(",");
        header.push_str(&format!(
            "\"{}\":{{\"dtype\":\"{}\",\"shape\":[{shape}],\"data_offsets\":[{offset},{end}]}}",
            tensor.name,
            dtype_facts(tensor.dtype).header
        ));
        offset = end;
    }
    header.push('}');

    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    for tensor in tensors {
        bytes.extend_from_slice(&tensor.payload);
    }
    fs::write(path, &bytes).expect("write shard");
    *blake3::hash(&bytes).as_bytes()
}

/// Write a complete checkpoint and return the index path, the relative shard
/// paths, and the trusted full-file digest of every shard.
fn write_checkpoint(
    root: &Path,
    shards: &[(&str, Vec<TensorSpec>)],
) -> (PathBuf, Vec<PathBuf>, Vec<[u8; 32]>) {
    let mut weight_map = String::from("{");
    let mut shard_paths = Vec::with_capacity(shards.len());
    let mut digests = Vec::with_capacity(shards.len());
    for (shard, tensors) in shards {
        for tensor in tensors {
            if weight_map.len() > 1 {
                weight_map.push(',');
            }
            weight_map.push_str(&format!("\"{}\":\"{shard}\"", tensor.name));
        }
        digests.push(write_shard(&root.join(shard), tensors));
        shard_paths.push(PathBuf::from(shard));
    }
    weight_map.push('}');

    let index_path = root.join("model.safetensors.index.json");
    fs::write(&index_path, format!("{{\"weight_map\":{weight_map}}}")).expect("write shard index");
    (index_path, shard_paths, digests)
}

fn expected_digests<'a>(
    shards: &'a [PathBuf],
    digests: &'a [[u8; 32]],
) -> Vec<ExpectedShardDigest<'a>> {
    shards
        .iter()
        .zip(digests)
        .map(|(shard, blake3)| ExpectedShardDigest {
            shard: shard.as_path(),
            blake3: *blake3,
        })
        .collect()
}

#[test]
fn verified_checkpoint_produces_exact_tensor_descriptors() {
    let root = scratch("well-formed");
    let shards: Vec<(&str, Vec<TensorSpec>)> = vec![
        (
            "model-00001-of-00002.safetensors",
            vec![
                TensorSpec::new("block.0.weight", SafetensorDtype::F32, &[2, 3]),
                TensorSpec::new("block.0.bias", SafetensorDtype::F16, &[3]),
            ],
        ),
        (
            "model-00002-of-00002.safetensors",
            vec![
                TensorSpec::new("block.1.weight", SafetensorDtype::BF16, &[4, 2]),
                TensorSpec::new("block.1.mask", SafetensorDtype::BOOL, &[4]),
            ],
        ),
    ];
    let (index_path, shard_paths, digests) = write_checkpoint(&root, &shards);

    let (manifest, checkpoint) = CheckpointManifest::ingest_verified_checkpoint(
        "scratch-model",
        &root,
        &index_path,
        expected_digests(&shard_paths, &digests),
    )
    .expect("well-formed verified checkpoint ingests");

    assert_eq!(manifest.model_name, "scratch-model");
    assert_eq!(
        manifest.tensors.keys().cloned().collect::<Vec<_>>(),
        vec![
            "block.0.bias".to_string(),
            "block.0.weight".to_string(),
            "block.1.mask".to_string(),
            "block.1.weight".to_string(),
        ],
        "the manifest holds exactly the checkpoint tensor set"
    );

    let expected = [
        ("block.0.weight", vec![2_usize, 3], DataType::F32, 24_usize),
        ("block.0.bias", vec![3], DataType::F16, 6),
        ("block.1.weight", vec![4, 2], DataType::BF16, 16),
        ("block.1.mask", vec![4], DataType::Bool, 4),
    ];
    for (name, shape, dtype, byte_size) in expected {
        let descriptor = manifest.get(name).unwrap_or_else(|| {
            panic!("manifest describes {name}");
        });
        assert_eq!(descriptor.name, name);
        assert_eq!(descriptor.shape, shape, "shape of {name}");
        assert_eq!(descriptor.dtype, dtype, "dtype of {name}");
        assert_eq!(descriptor.byte_size, byte_size, "byte size of {name}");
    }

    // The returned checkpoint reads the verified payload the manifest describes.
    let payload = checkpoint
        .read_tensor("block.0.weight")
        .expect("verified tensor reads");
    let source = shards[0].1[0].payload.clone();
    assert_eq!(payload, source, "verified read returns the written payload");
}

#[test]
fn checkpoint_failing_digest_verification_is_rejected() {
    let root = scratch("digest-mismatch");
    let shards: Vec<(&str, Vec<TensorSpec>)> = vec![(
        "model-00001-of-00001.safetensors",
        vec![TensorSpec::new(
            "block.0.weight",
            SafetensorDtype::F32,
            &[2, 2],
        )],
    )];
    let (index_path, shard_paths, mut digests) = write_checkpoint(&root, &shards);

    // Ingestion under the correct digest succeeds, so the rejection below is
    // caused by the digest alone.
    CheckpointManifest::ingest_verified_checkpoint(
        "scratch-model",
        &root,
        &index_path,
        expected_digests(&shard_paths, &digests),
    )
    .expect("matching digest ingests");

    let truth = digests[0];
    digests[0][0] ^= 0x01;
    let error = CheckpointManifest::ingest_verified_checkpoint(
        "scratch-model",
        &root,
        &index_path,
        expected_digests(&shard_paths, &digests),
    )
    .expect_err("a checkpoint whose content digest does not match must be rejected");

    assert_eq!(
        error,
        ManifestError::Checkpoint(SafetensorError::ShardDigestMismatch {
            shard: shard_paths[0].clone(),
            actual: truth,
            expected: digests[0],
        }),
        "the rejection names the shard and both digests"
    );
}

#[test]
fn checkpoint_content_rewritten_after_indexing_is_rejected() {
    let root = scratch("content-rewritten");
    let shards: Vec<(&str, Vec<TensorSpec>)> = vec![(
        "model-00001-of-00001.safetensors",
        vec![TensorSpec::new("block.0.weight", SafetensorDtype::U8, &[8])],
    )];
    let (index_path, shard_paths, digests) = write_checkpoint(&root, &shards);

    // Rewrite one payload byte in place. Length and header are unchanged, so
    // only full-file content verification can catch it.
    let shard_file = root.join(&shard_paths[0]);
    let mut bytes = fs::read(&shard_file).expect("read shard");
    let last = bytes.len() - 1;
    bytes[last] ^= 0xFF;
    fs::write(&shard_file, &bytes).expect("rewrite shard");

    let error = CheckpointManifest::ingest_verified_checkpoint(
        "scratch-model",
        &root,
        &index_path,
        expected_digests(&shard_paths, &digests),
    )
    .expect_err("a rewritten shard must not ingest");

    assert!(
        matches!(
            &error,
            ManifestError::Checkpoint(SafetensorError::ShardDigestMismatch { .. })
        ),
        "expected a shard digest mismatch, got {error}"
    );
}

#[test]
fn every_safetensor_element_type_ingests_or_reports_the_typed_error() {
    for &dtype in COVERED_DTYPES {
        let facts = dtype_facts(dtype);
        let case = format!("dtype-{}", dtype_variant_name(dtype).to_lowercase());
        let root = scratch(&case);
        let shards: Vec<(&str, Vec<TensorSpec>)> = vec![(
            "model-00001-of-00001.safetensors",
            vec![TensorSpec::new("block.0.weight", dtype, &[2, 3])],
        )];
        let (index_path, shard_paths, digests) = write_checkpoint(&root, &shards);

        let outcome = CheckpointManifest::ingest_verified_checkpoint(
            "scratch-model",
            &root,
            &index_path,
            expected_digests(&shard_paths, &digests),
        );

        match facts.data_type {
            Some(expected) => {
                let (manifest, _) =
                    outcome.unwrap_or_else(|error| panic!("{dtype:?} must ingest, got {error}"));
                let descriptor = manifest
                    .get("block.0.weight")
                    .unwrap_or_else(|| panic!("{dtype:?} manifest describes the tensor"));
                assert_eq!(descriptor.dtype, expected, "data type for {dtype:?}");
                assert_eq!(descriptor.shape, vec![2, 3], "shape for {dtype:?}");
                assert_eq!(
                    descriptor.byte_size as u64,
                    6 * facts.byte_width,
                    "byte size for {dtype:?}"
                );
            }
            None => {
                let error = outcome
                    .err()
                    .unwrap_or_else(|| panic!("{dtype:?} has no IR counterpart and must fail"));
                assert_eq!(
                    error,
                    ManifestError::UnsupportedCheckpointDtype {
                        name: "block.0.weight".to_string(),
                        dtype,
                    },
                    "unsupported element type must produce the typed error"
                );
            }
        }
    }
}

#[test]
fn covered_dtypes_match_the_safetensors_declaration() {
    let source_path = workspace_root().join("vyre-safetensors/src/lib.rs");
    let source = fs::read_to_string(&source_path)
        .unwrap_or_else(|error| panic!("read {}: {error}", source_path.display()));

    let anchor = "pub enum SafetensorDtype {";
    let start = source
        .find(anchor)
        .unwrap_or_else(|| panic!("no `{anchor}` declaration in {}", source_path.display()))
        + anchor.len();
    let body = &source[start..];
    let end = body
        .find("\n}")
        .expect("the SafetensorDtype declaration is closed");

    let mut declared = BTreeSet::new();
    for line in body[..end].lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with("//") || line.starts_with('#') {
            continue;
        }
        let name = line
            .split(|c: char| !c.is_alphanumeric() && c != '_')
            .next()
            .unwrap_or_default();
        assert!(
            !name.is_empty(),
            "unreadable variant line in SafetensorDtype: `{line}`"
        );
        assert!(
            declared.insert(name.to_string()),
            "duplicate variant `{name}` in SafetensorDtype"
        );
    }

    let covered: BTreeSet<String> = COVERED_DTYPES
        .iter()
        .map(|&dtype| dtype_variant_name(dtype).to_string())
        .collect();

    assert_eq!(
        covered,
        declared,
        "ingestion coverage must name every SafetensorDtype variant declared in {}",
        source_path.display()
    );
}

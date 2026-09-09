//! Bounded, metadata-only safetensors checkpoint ingestion contracts.

#![forbid(unsafe_code)]

use std::fs;
use std::io::{Seek as _, SeekFrom, Write as _};
use std::path::Path;

use vyre_safetensors::{
    ExpectedShardDigest, SafetensorDtype, SafetensorError, SafetensorIndex, SafetensorRequirement,
    ShardedSafetensorIndex, VERIFIED_CHECKPOINT_IDENTITY_VERSION,
};

fn write_shard(path: &Path, header: &[u8], payload: &[u8]) {
    let mut bytes = Vec::with_capacity(8 + header.len() + payload.len());
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header);
    bytes.extend_from_slice(payload);
    fs::write(path, bytes).expect("Fix: fixture shard must be writable");
}

fn write_sparse_shard(path: &Path, header: &[u8], payload_len: u64) {
    let mut file = fs::File::create(path).expect("Fix: sparse fixture shard must be creatable");
    file.write_all(&(header.len() as u64).to_le_bytes())
        .expect("Fix: sparse fixture prefix must be writable");
    file.write_all(header)
        .expect("Fix: sparse fixture header must be writable");
    file.set_len(8 + header.len() as u64 + payload_len)
        .expect("Fix: sparse fixture payload length must be writable");
}

fn requirement_fixture() -> (tempfile::TempDir, ShardedSafetensorIndex) {
    let temp = tempfile::tempdir().expect("Fix: fixture directory must be creatable");
    let header = br#"{"embedding":{"dtype":"BF16","shape":[2,2],"data_offsets":[0,8]},"state":{"dtype":"F32","shape":[3],"data_offsets":[8,20]}}"#;
    write_shard(&temp.path().join("one.safetensors"), header, &[0; 20]);
    let index_path = temp.path().join("model.safetensors.index.json");
    fs::write(
        &index_path,
        br#"{"weight_map":{"embedding":"one.safetensors","state":"one.safetensors"}}"#,
    )
    .expect("Fix: requirement fixture index must be writable");
    let index = ShardedSafetensorIndex::open(temp.path(), &index_path)
        .expect("Fix: requirement fixture must index");
    (temp, index)
}

/// Proves metadata ingestion returns exact absolute ranges without reading tensor payloads.
#[test]
fn valid_bf16_tensors_have_exact_ranges_and_identity() {
    let temp = tempfile::tempdir().expect("Fix: fixture directory must be creatable");
    let path = temp.path().join("model.safetensors");
    let header = br#"{"model.embed_tokens.weight":{"dtype":"BF16","shape":[2,2],"data_offsets":[0,8]},"model.norm.weight":{"dtype":"BF16","shape":[2],"data_offsets":[8,12]}}"#;
    write_shard(&path, header, &[0_u8; 12]);

    let index = SafetensorIndex::open(&path).expect("Fix: valid shard must index");
    assert_eq!(index.tensors().len(), 2);
    let embedding = index
        .tensor("model.embed_tokens.weight")
        .expect("Fix: embedding metadata must exist");
    assert_eq!(embedding.dtype, SafetensorDtype::BF16);
    assert_eq!(embedding.shape, [2, 2]);
    assert_eq!(
        embedding.file_range,
        (8 + header.len() as u64)..(16 + header.len() as u64)
    );
    assert_eq!(index.identity().file_len, 8 + header.len() as u64 + 12);
    assert_eq!(
        index.identity().header_digest,
        *blake3::hash(header).as_bytes()
    );
}

/// Prevents a hostile prefix from allocating an unbounded metadata buffer.
#[test]
fn oversized_header_is_rejected_before_allocation() {
    let temp = tempfile::tempdir().expect("Fix: fixture directory must be creatable");
    let path = temp.path().join("oversized.safetensors");
    fs::write(&path, (64_u64 * 1024 * 1024 + 1).to_le_bytes())
        .expect("Fix: hostile prefix must be writable");
    assert_eq!(
        SafetensorIndex::open(&path).expect_err("Fix: oversized header must fail"),
        SafetensorError::HeaderTooLarge {
            header_len: 64 * 1024 * 1024 + 1,
            maximum: 64 * 1024 * 1024,
        }
    );
}

/// Prevents a declared metadata range from extending past a truncated shard.
#[test]
fn truncated_payload_fails_closed() {
    let temp = tempfile::tempdir().expect("Fix: fixture directory must be creatable");
    let path = temp.path().join("truncated.safetensors");
    let header = br#"{"weight":{"dtype":"F32","shape":[4],"data_offsets":[0,16]}}"#;
    write_shard(&path, header, &[0_u8; 12]);
    assert_eq!(
        SafetensorIndex::open(&path).expect_err("Fix: truncated tensor must fail"),
        SafetensorError::RangeOutOfBounds {
            name: "weight".into(),
            start: 0,
            end: 16,
            data_len: 12,
        }
    );
}

/// Prevents two tensor names from aliasing mutable bytes in one shard.
#[test]
fn overlapping_tensor_ranges_fail_closed() {
    let temp = tempfile::tempdir().expect("Fix: fixture directory must be creatable");
    let path = temp.path().join("overlap.safetensors");
    let header = br#"{"a":{"dtype":"U8","shape":[8],"data_offsets":[0,8]},"b":{"dtype":"U8","shape":[8],"data_offsets":[4,12]}}"#;
    write_shard(&path, header, &[0_u8; 12]);
    assert_eq!(
        SafetensorIndex::open(&path).expect_err("Fix: overlap must fail"),
        SafetensorError::Overlap {
            left: "a".into(),
            right: "b".into(),
        }
    );
}

/// Prevents JSON duplicate-key behavior from replacing one tensor descriptor with another.
#[test]
fn duplicate_tensor_names_fail_closed() {
    let temp = tempfile::tempdir().expect("Fix: fixture directory must be creatable");
    let path = temp.path().join("duplicate.safetensors");
    let header = br#"{"weight":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"weight":{"dtype":"U8","shape":[1],"data_offsets":[1,2]}}"#;
    write_shard(&path, header, &[0_u8; 2]);
    let error = SafetensorIndex::open(&path).expect_err("Fix: duplicate names must fail");
    assert!(matches!(error, SafetensorError::Metadata(_)));
    assert!(error.to_string().contains("duplicate tensor name `weight`"));
}

/// Prevents shape metadata from under-reporting or over-reporting tensor bytes.
#[test]
fn dtype_shape_and_byte_range_must_agree() {
    let temp = tempfile::tempdir().expect("Fix: fixture directory must be creatable");
    let path = temp.path().join("wrong-length.safetensors");
    let header = br#"{"weight":{"dtype":"BF16","shape":[2,2],"data_offsets":[0,6]}}"#;
    write_shard(&path, header, &[0_u8; 6]);
    assert_eq!(
        SafetensorIndex::open(&path).expect_err("Fix: byte mismatch must fail"),
        SafetensorError::ByteLength {
            name: "weight".into(),
            actual: 6,
            expected: 8,
        }
    );
}

/// Prevents malformed metadata JSON from falling back to an empty checkpoint.
#[test]
fn malformed_metadata_fails_closed() {
    let temp = tempfile::tempdir().expect("Fix: fixture directory must be creatable");
    let path = temp.path().join("malformed.safetensors");
    write_shard(&path, br#"{"weight": "#, &[]);
    let error = SafetensorIndex::open(&path).expect_err("Fix: malformed JSON must fail");
    assert!(matches!(error, SafetensorError::Metadata(_)));
}

/// Proves a checkpoint index resolves exact tensor ranges across multiple shards.
#[test]
fn sharded_index_resolves_every_weight_once() {
    let temp = tempfile::tempdir().expect("Fix: fixture directory must be creatable");
    let first = br#"{"a":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}}"#;
    let second = br#"{"b":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
    write_shard(
        &temp.path().join("model-00001.safetensors"),
        first,
        &[0_u8; 4],
    );
    write_shard(
        &temp.path().join("model-00002.safetensors"),
        second,
        &[0_u8; 4],
    );
    let index_path = temp.path().join("model.safetensors.index.json");
    fs::write(
        &index_path,
        br#"{"weight_map":{"a":"model-00001.safetensors","b":"model-00002.safetensors"}}"#,
    )
    .expect("Fix: shard index must be writable");

    let index = vyre_safetensors::ShardedSafetensorIndex::open(temp.path(), &index_path)
        .expect("Fix: complete sharded checkpoint must index");
    assert_eq!(index.tensors().len(), 2);
    assert_eq!(index.shards().len(), 2);
    assert_eq!(
        index.tensor("a").expect("Fix: tensor a must resolve").shard,
        std::path::PathBuf::from("model-00001.safetensors")
    );
    assert_eq!(
        index
            .tensor("b")
            .expect("Fix: tensor b must resolve")
            .tensor
            .dtype,
        SafetensorDtype::F32
    );
    assert_ne!(index.manifest_digest(), [0_u8; 32]);
}

/// Prevents a weight map from claiming a tensor that its assigned shard lacks.
#[test]
fn missing_mapped_tensor_fails_closed() {
    let temp = tempfile::tempdir().expect("Fix: fixture directory must be creatable");
    let header = br#"{"actual":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#;
    write_shard(&temp.path().join("one.safetensors"), header, &[0]);
    let index_path = temp.path().join("model.safetensors.index.json");
    fs::write(
        &index_path,
        br#"{"weight_map":{"expected":"one.safetensors"}}"#,
    )
    .expect("Fix: shard index must be writable");
    assert_eq!(
        vyre_safetensors::ShardedSafetensorIndex::open(temp.path(), &index_path)
            .expect_err("Fix: absent mapped tensor must fail"),
        SafetensorError::MissingMappedTensor {
            name: "expected".into(),
            shard: "one.safetensors".into(),
        }
    );
}

/// Prevents checkpoint bytes from bypassing the authoritative weight map.
#[test]
fn unmapped_shard_tensor_fails_closed() {
    let temp = tempfile::tempdir().expect("Fix: fixture directory must be creatable");
    let header = br#"{"mapped":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"hidden":{"dtype":"U8","shape":[1],"data_offsets":[1,2]}}"#;
    write_shard(&temp.path().join("one.safetensors"), header, &[0, 0]);
    let index_path = temp.path().join("model.safetensors.index.json");
    fs::write(
        &index_path,
        br#"{"weight_map":{"mapped":"one.safetensors"}}"#,
    )
    .expect("Fix: shard index must be writable");
    assert_eq!(
        vyre_safetensors::ShardedSafetensorIndex::open(temp.path(), &index_path)
            .expect_err("Fix: unmapped shard bytes must fail"),
        SafetensorError::UnmappedShardTensor {
            name: "hidden".into(),
            shard: "one.safetensors".into(),
        }
    );
}

/// Prevents a hostile shard map from escaping the checkpoint root.
#[test]
fn shard_path_traversal_fails_closed() {
    let temp = tempfile::tempdir().expect("Fix: fixture directory must be creatable");
    let index_path = temp.path().join("model.safetensors.index.json");
    fs::write(
        &index_path,
        br#"{"weight_map":{"weight":"../outside.safetensors"}}"#,
    )
    .expect("Fix: hostile shard index must be writable");
    assert_eq!(
        vyre_safetensors::ShardedSafetensorIndex::open(temp.path(), &index_path)
            .expect_err("Fix: traversal must fail"),
        SafetensorError::UnsafeShardPath {
            path: "../outside.safetensors".into(),
        }
    );
}

/// Proves model requirements bind exact dtype and shape while allowing a separate tower.
#[test]
fn exact_tensor_requirements_bind_without_rejecting_unrequested_weights() {
    let (_temp, index) = requirement_fixture();
    index
        .validate_requirements([SafetensorRequirement {
            name: "embedding",
            dtype: SafetensorDtype::BF16,
            shape: &[2, 2],
        }])
        .expect("Fix: exact required tensor must bind while state remains separate");
}

/// Prevents an absent immutable model weight from leaving a compiled port unbound.
#[test]
fn missing_tensor_requirement_fails_closed() {
    let (_temp, index) = requirement_fixture();
    assert_eq!(
        index
            .validate_requirements([SafetensorRequirement {
                name: "missing",
                dtype: SafetensorDtype::BF16,
                shape: &[2, 2],
            }])
            .expect_err("Fix: absent requirement must fail"),
        SafetensorError::MissingRequiredTensor {
            name: "missing".into()
        }
    );
}

/// Prevents a compiler from interpreting stored BF16 bytes as F32 values.
#[test]
fn requirement_dtype_mismatch_fails_closed() {
    let (_temp, index) = requirement_fixture();
    assert_eq!(
        index
            .validate_requirements([SafetensorRequirement {
                name: "embedding",
                dtype: SafetensorDtype::F32,
                shape: &[2, 2],
            }])
            .expect_err("Fix: dtype mismatch must fail"),
        SafetensorError::RequiredDtype {
            name: "embedding".into(),
            actual: SafetensorDtype::BF16,
            expected: SafetensorDtype::F32,
        }
    );
}

/// Prevents a transposed or truncated checkpoint matrix from binding by name alone.
#[test]
fn requirement_shape_mismatch_fails_closed() {
    let (_temp, index) = requirement_fixture();
    assert_eq!(
        index
            .validate_requirements([SafetensorRequirement {
                name: "embedding",
                dtype: SafetensorDtype::BF16,
                shape: &[4, 1],
            }])
            .expect_err("Fix: shape mismatch must fail"),
        SafetensorError::RequiredShape {
            name: "embedding".into(),
            actual: vec![2, 2],
            expected: vec![4, 1],
        }
    );
}

/// Prevents two graph ports from claiming the same immutable tensor requirement.
#[test]
fn duplicate_tensor_requirement_fails_closed() {
    let (_temp, index) = requirement_fixture();
    let requirement = SafetensorRequirement {
        name: "embedding",
        dtype: SafetensorDtype::BF16,
        shape: &[2, 2],
    };
    assert_eq!(
        index
            .validate_requirements([requirement, requirement])
            .expect_err("Fix: duplicate requirement must fail"),
        SafetensorError::DuplicateRequirement {
            name: "embedding".into()
        }
    );
}

/// Prevents malformed shard-index JSON from falling back to an empty model.
#[test]
fn malformed_shard_index_fails_closed() {
    let temp = tempfile::tempdir().expect("Fix: fixture directory must be creatable");
    let index_path = temp.path().join("model.safetensors.index.json");
    fs::write(&index_path, br#"{"weight_map":"#)
        .expect("Fix: malformed index fixture must be writable");
    let error = ShardedSafetensorIndex::open(temp.path(), &index_path)
        .expect_err("Fix: malformed shard index must fail");
    assert!(matches!(error, SafetensorError::ShardIndex(_)));
}

/// Prevents JSON duplicate-key replacement from redirecting one tensor to another shard.
#[test]
fn duplicate_weight_map_tensor_fails_closed() {
    let temp = tempfile::tempdir().expect("Fix: fixture directory must be creatable");
    let index_path = temp.path().join("model.safetensors.index.json");
    fs::write(
        &index_path,
        br#"{"weight_map":{"weight":"first.safetensors","weight":"second.safetensors"}}"#,
    )
    .expect("Fix: duplicate index fixture must be writable");
    let error = ShardedSafetensorIndex::open(temp.path(), &index_path)
        .expect_err("Fix: duplicate weight-map key must fail");
    assert!(matches!(error, SafetensorError::ShardIndex(_)));
    assert!(error
        .to_string()
        .contains("duplicate weight-map tensor `weight`"));
}

/// Prevents a shard index from allocating more than the bounded 64 MiB policy.
#[test]
fn oversized_shard_index_is_rejected_before_reading() {
    let temp = tempfile::tempdir().expect("Fix: fixture directory must be creatable");
    let index_path = temp.path().join("model.safetensors.index.json");
    let file = fs::File::create(&index_path).expect("Fix: sparse index fixture must be creatable");
    file.set_len(64 * 1024 * 1024 + 1)
        .expect("Fix: sparse oversized index must be writable");
    assert_eq!(
        ShardedSafetensorIndex::open(temp.path(), &index_path)
            .expect_err("Fix: oversized shard index must fail"),
        SafetensorError::ShardIndexTooLarge {
            actual: 64 * 1024 * 1024 + 1,
            maximum: 64 * 1024 * 1024,
        }
    );
}

/// Locks manifest identity to exact index bytes and canonical shard metadata.
#[test]
fn manifest_digest_is_deterministic_and_index_sensitive() {
    let (temp, first) = requirement_fixture();
    let index_path = temp.path().join("model.safetensors.index.json");
    let second = ShardedSafetensorIndex::open(temp.path(), &index_path)
        .expect("Fix: unchanged checkpoint must reopen");
    assert_eq!(first.manifest_digest(), second.manifest_digest());

    fs::write(
        &index_path,
        br#"{ "weight_map": { "embedding": "one.safetensors", "state": "one.safetensors" } }"#,
    )
    .expect("Fix: equivalent index bytes must be writable");
    let reformatted = ShardedSafetensorIndex::open(temp.path(), &index_path)
        .expect("Fix: equivalent weight map must remain valid");
    assert_ne!(first.manifest_digest(), reformatted.manifest_digest());
}

/// Prevents an empty shard name from resolving to the checkpoint directory.
#[test]
fn empty_shard_path_fails_closed() {
    let temp = tempfile::tempdir().expect("Fix: fixture directory must be creatable");
    let index_path = temp.path().join("model.safetensors.index.json");
    fs::write(&index_path, br#"{"weight_map":{"weight":""}}"#)
        .expect("Fix: empty path fixture must be writable");
    assert_eq!(
        ShardedSafetensorIndex::open(temp.path(), &index_path)
            .expect_err("Fix: empty shard path must fail"),
        SafetensorError::UnsafeShardPath { path: "".into() }
    );
}

/// Prevents an in-root symlink from redirecting checkpoint reads outside the root.
#[cfg(unix)]
#[test]
fn shard_symlink_escape_fails_closed() {
    let root = tempfile::tempdir().expect("Fix: checkpoint root must be creatable");
    let outside = tempfile::tempdir().expect("Fix: outside fixture directory must be creatable");
    let outside_shard = outside.path().join("outside.safetensors");
    let header = br#"{"weight":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#;
    write_shard(&outside_shard, header, &[0]);
    std::os::unix::fs::symlink(&outside_shard, root.path().join("linked.safetensors"))
        .expect("Fix: escape symlink fixture must be creatable");
    let index_path = root.path().join("model.safetensors.index.json");
    fs::write(
        &index_path,
        br#"{"weight_map":{"weight":"linked.safetensors"}}"#,
    )
    .expect("Fix: symlink index fixture must be writable");
    assert_eq!(
        ShardedSafetensorIndex::open(root.path(), &index_path)
            .expect_err("Fix: shard symlink escape must fail"),
        SafetensorError::UnsafeShardPath {
            path: "linked.safetensors".into()
        }
    );
}

/// Exercises production-scale Qwen3.5-27B ranges without reading 55.6 GB of payloads.
#[test]
fn official_qwen35_metadata_subset_indexes_without_payload_reads() {
    let temp = tempfile::tempdir().expect("Fix: fixture directory must be creatable");
    let shard1 = br#"{"lm_head.weight":{"dtype":"BF16","shape":[248320,5120],"data_offsets":[0,2542796800]},"model.language_model.embed_tokens.weight":{"dtype":"BF16","shape":[248320,5120],"data_offsets":[2542796800,5085593600]}}"#;
    let shard9 = br#"{"model.language_model.layers.0.linear_attn.in_proj_qkv.weight":{"dtype":"BF16","shape":[10240,5120],"data_offsets":[0,104857600]},"model.language_model.layers.3.self_attn.o_proj.weight":{"dtype":"BF16","shape":[5120,6144],"data_offsets":[1971322880,2034237440]}}"#;
    let shard11 = br#"{"model.language_model.layers.0.linear_attn.A_log":{"dtype":"F32","shape":[48],"data_offsets":[0,192]},"model.language_model.layers.0.linear_attn.conv1d.weight":{"dtype":"BF16","shape":[10240,1,4],"data_offsets":[44032,125952]},"model.language_model.layers.0.linear_attn.in_proj_z.weight":{"dtype":"BF16","shape":[6144,5120],"data_offsets":[1109088,64023648]},"model.language_model.layers.3.self_attn.k_proj.weight":{"dtype":"BF16","shape":[1024,5120],"data_offsets":[564961472,575447232]},"model.language_model.layers.3.self_attn.q_norm.weight":{"dtype":"BF16","shape":[256],"data_offsets":[575447232,575447744]},"model.language_model.layers.3.self_attn.v_proj.weight":{"dtype":"BF16","shape":[1024,5120],"data_offsets":[575447744,585933504]}}"#;
    write_sparse_shard(
        &temp
            .path()
            .join("model.safetensors-00001-of-00011.safetensors"),
        shard1,
        5_085_593_600,
    );
    write_sparse_shard(
        &temp
            .path()
            .join("model.safetensors-00009-of-00011.safetensors"),
        shard9,
        2_034_237_440,
    );
    write_sparse_shard(
        &temp
            .path()
            .join("model.safetensors-00011-of-00011.safetensors"),
        shard11,
        585_933_504,
    );
    let index_path = temp.path().join("model.safetensors.index.json");
    fs::write(
        &index_path,
        br#"{"metadata":{"total_size":55562872800},"weight_map":{"lm_head.weight":"model.safetensors-00001-of-00011.safetensors","model.language_model.embed_tokens.weight":"model.safetensors-00001-of-00011.safetensors","model.language_model.layers.0.linear_attn.A_log":"model.safetensors-00011-of-00011.safetensors","model.language_model.layers.0.linear_attn.conv1d.weight":"model.safetensors-00011-of-00011.safetensors","model.language_model.layers.0.linear_attn.in_proj_qkv.weight":"model.safetensors-00009-of-00011.safetensors","model.language_model.layers.0.linear_attn.in_proj_z.weight":"model.safetensors-00011-of-00011.safetensors","model.language_model.layers.3.self_attn.k_proj.weight":"model.safetensors-00011-of-00011.safetensors","model.language_model.layers.3.self_attn.o_proj.weight":"model.safetensors-00009-of-00011.safetensors","model.language_model.layers.3.self_attn.q_norm.weight":"model.safetensors-00011-of-00011.safetensors","model.language_model.layers.3.self_attn.v_proj.weight":"model.safetensors-00011-of-00011.safetensors"}}"#,
    )
    .expect("Fix: official metadata subset index must be writable");

    let index = ShardedSafetensorIndex::open(temp.path(), &index_path)
        .expect("Fix: official Qwen metadata subset must index");
    assert_eq!(index.tensors().len(), 10);
    assert_eq!(index.shards().len(), 3);
    index
        .validate_requirements([
            SafetensorRequirement {
                name: "lm_head.weight",
                dtype: SafetensorDtype::BF16,
                shape: &[248_320, 5_120],
            },
            SafetensorRequirement {
                name: "model.language_model.layers.0.linear_attn.A_log",
                dtype: SafetensorDtype::F32,
                shape: &[48],
            },
            SafetensorRequirement {
                name: "model.language_model.layers.0.linear_attn.conv1d.weight",
                dtype: SafetensorDtype::BF16,
                shape: &[10_240, 1, 4],
            },
            SafetensorRequirement {
                name: "model.language_model.layers.3.self_attn.o_proj.weight",
                dtype: SafetensorDtype::BF16,
                shape: &[5_120, 6_144],
            },
        ])
        .expect("Fix: official Qwen layouts must bind exactly");
    let head = index
        .tensor("lm_head.weight")
        .expect("Fix: official LM head must resolve");
    assert_eq!(
        head.shard,
        Path::new("model.safetensors-00001-of-00011.safetensors")
    );
    assert_eq!(
        head.tensor.file_range.end - head.tensor.file_range.start,
        2_542_796_800
    );
}

/// Proves trusted full-file digests produce one stable immutable checkpoint identity.
#[test]
fn trusted_shard_digests_verify_complete_file_bytes() {
    let (temp, index) = requirement_fixture();
    let shard = Path::new("one.safetensors");
    let bytes = fs::read(temp.path().join(shard)).expect("Fix: fixture shard must be readable");
    let digest = *blake3::hash(&bytes).as_bytes();
    let identity = index
        .verify_shards([ExpectedShardDigest {
            shard,
            blake3: digest,
        }])
        .expect("Fix: exact full-file digest must verify");
    assert_eq!(
        VERIFIED_CHECKPOINT_IDENTITY_VERSION,
        "vyre-verified-safetensors-blake3-v1"
    );
    assert_eq!(identity.manifest_digest(), index.manifest_digest());
    assert_eq!(
        identity
            .shard_digests()
            .map(|(path, actual)| (path.to_path_buf(), *actual))
            .collect::<Vec<_>>(),
        [(shard.to_path_buf(), digest)]
    );
    assert_ne!(identity.content_digest(), [0; 32]);
}

/// Prevents verification from succeeding when any indexed shard lacks a trusted digest.
#[test]
fn missing_trusted_shard_digest_fails_closed() {
    let (_temp, index) = requirement_fixture();
    assert_eq!(
        index
            .verify_shards([])
            .expect_err("Fix: omitted shard digest must fail"),
        SafetensorError::MissingShardDigest {
            shard: "one.safetensors".into()
        }
    );
}

/// Prevents an unrelated trusted digest from being mistaken for an indexed shard.
#[test]
fn unexpected_trusted_shard_digest_fails_closed() {
    let (_temp, index) = requirement_fixture();
    assert_eq!(
        index
            .verify_shards([ExpectedShardDigest {
                shard: Path::new("other.safetensors"),
                blake3: [7; 32],
            }])
            .expect_err("Fix: unknown trusted shard must fail"),
        SafetensorError::UnexpectedShardDigest {
            shard: "other.safetensors".into()
        }
    );
}

/// Prevents duplicate digest inputs from hiding a conflicting trust record.
#[test]
fn duplicate_trusted_shard_digest_fails_closed() {
    let (_temp, index) = requirement_fixture();
    let expected = ExpectedShardDigest {
        shard: Path::new("one.safetensors"),
        blake3: [3; 32],
    };
    assert_eq!(
        index
            .verify_shards([expected, expected])
            .expect_err("Fix: duplicate trusted shard must fail"),
        SafetensorError::DuplicateShardDigest {
            shard: "one.safetensors".into()
        }
    );
}

/// Detects same-length payload mutation after safe metadata indexing.
#[test]
fn mutated_shard_payload_fails_digest_verification() {
    let (temp, index) = requirement_fixture();
    let shard = Path::new("one.safetensors");
    let path = temp.path().join(shard);
    let expected =
        *blake3::hash(&fs::read(&path).expect("Fix: fixture shard must be readable")).as_bytes();
    let mut file = fs::OpenOptions::new()
        .write(true)
        .open(&path)
        .expect("Fix: fixture shard must reopen for mutation");
    file.seek(SeekFrom::End(-1))
        .expect("Fix: fixture payload byte must be seekable");
    file.write_all(&[1])
        .expect("Fix: fixture payload byte must be mutable");
    let actual =
        *blake3::hash(&fs::read(&path).expect("Fix: mutated shard must be readable")).as_bytes();
    assert_eq!(
        index
            .verify_shards([ExpectedShardDigest {
                shard,
                blake3: expected,
            }])
            .expect_err("Fix: payload mutation must fail"),
        SafetensorError::ShardDigestMismatch {
            shard: shard.into(),
            actual,
            expected,
        }
    );
}

/// Detects file replacement or truncation before reading full shard bytes.
#[test]
fn changed_shard_length_fails_before_digest_verification() {
    let (temp, index) = requirement_fixture();
    let shard = Path::new("one.safetensors");
    let path = temp.path().join(shard);
    let indexed = fs::metadata(&path)
        .expect("Fix: fixture shard metadata must exist")
        .len();
    fs::OpenOptions::new()
        .write(true)
        .open(&path)
        .expect("Fix: fixture shard must reopen")
        .set_len(indexed + 1)
        .expect("Fix: fixture shard length must change");
    assert_eq!(
        index
            .verify_shards([ExpectedShardDigest {
                shard,
                blake3: [0; 32],
            }])
            .expect_err("Fix: changed shard length must fail"),
        SafetensorError::ShardLengthChanged {
            shard: shard.into(),
            indexed,
            actual: indexed + 1,
        }
    );
}

/// Proves transactional checkpoint handles protect against symlink swaps after verification.
#[test]
fn symlink_swap_between_verification_and_binding_reads_original_verified_content() {
    let temp = tempfile::tempdir().expect("tempdir");
    let target_a = temp.path().join("real_shard_a.safetensors");
    let target_b = temp.path().join("real_shard_b.safetensors");
    let symlink_path = temp.path().join("shard.safetensors");

    let header_a = br#"{"weights":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
    let payload_a = [10_u8; 8];
    write_shard(&target_a, header_a, &payload_a);

    let header_b = br#"{"weights":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
    let payload_b = [99_u8; 8];
    write_shard(&target_b, header_b, &payload_b);

    // Create symlink pointing to shard A
    #[cfg(unix)]
    std::os::unix::fs::symlink(&target_a, &symlink_path).expect("symlink");

    #[cfg(unix)]
    {
        let index_path = temp.path().join("model.safetensors.index.json");
        fs::write(
            &index_path,
            br#"{"weight_map":{"weights":"shard.safetensors"}}"#,
        )
        .expect("write index");

        let index = ShardedSafetensorIndex::open(temp.path(), &index_path).expect("open index");
        let shard_rel = Path::new("shard.safetensors");
        let expected_digest = *blake3::hash(&fs::read(&target_a).expect("read a")).as_bytes();

        let checkpoint = index
            .verify_transactional([ExpectedShardDigest {
                shard: shard_rel,
                blake3: expected_digest,
            }])
            .expect("verify transactional");

        // Adversary swaps the symlink to target_b on disk after verification!
        fs::remove_file(&symlink_path).expect("remove symlink");
        std::os::unix::fs::symlink(&target_b, &symlink_path).expect("swap symlink to b");

        // The transactional checkpoint holds the open verified handle from target_a.
        let bytes = checkpoint
            .read_tensor("weights")
            .expect("read from transactional checkpoint");
        assert_eq!(
            bytes, payload_a,
            "Transactional handle must read original verified content, not swapped symlink target"
        );

        // Attempting to re-open index or verify against swapped symlink with expected_digest fails with mismatch by name!
        let new_index =
            ShardedSafetensorIndex::open(temp.path(), &index_path).expect("reopen index");
        let err = new_index
            .verify_shards([ExpectedShardDigest {
                shard: shard_rel,
                blake3: expected_digest,
            }])
            .expect_err("verification on swapped symlink must fail");
        assert!(
            matches!(err, SafetensorError::ShardDigestMismatch { ref shard, .. } if shard == shard_rel)
        );
    }
}

/// Proves that a resource whose content changes between verification and binding is refused by name.
#[test]
fn resource_content_change_between_verification_and_binding_is_refused_by_name() {
    let temp = tempfile::tempdir().expect("tempdir");
    let shard_path = temp.path().join("weights.safetensors");
    let header = br#"{"layer.weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
    let original_payload = [7_u8; 8];
    write_shard(&shard_path, header, &original_payload);

    let index_path = temp.path().join("model.safetensors.index.json");
    fs::write(
        &index_path,
        br#"{"weight_map":{"layer.weight":"weights.safetensors"}}"#,
    )
    .expect("write index");

    let index = ShardedSafetensorIndex::open(temp.path(), &index_path).expect("open index");
    let shard_rel = Path::new("weights.safetensors");
    let expected_digest = *blake3::hash(&fs::read(&shard_path).expect("read")).as_bytes();

    let checkpoint = index
        .verify_transactional([ExpectedShardDigest {
            shard: shard_rel,
            blake3: expected_digest,
        }])
        .expect("verify transactional");

    // Verify initial read succeeds
    let read1 = checkpoint.read_tensor("layer.weight").expect("read tensor");
    assert_eq!(read1, original_payload);

    // Adversary modifies the file on disk (truncates / changes length)
    let current_len = fs::metadata(&shard_path).expect("metadata").len();
    fs::OpenOptions::new()
        .write(true)
        .open(&shard_path)
        .expect("open")
        .set_len(current_len + 16)
        .expect("set_len");

    // Subsequent read on the tensor handle detects the length change and refuses by name!
    let err = checkpoint
        .read_tensor("layer.weight")
        .expect_err("read after file length modification must fail");
    assert!(
        matches!(&err, SafetensorError::ShardLengthChanged { shard, .. } if shard == shard_rel),
        "Fix: resource content modification must be refused by name, got {err:?}"
    );
}

/// Proves transactional reader operations read_bytes, read_into, and tensor lookup.
#[test]
fn transactional_tensor_reader_operations() {
    let (temp, index) = requirement_fixture();
    let shard = Path::new("one.safetensors");
    let path = temp.path().join(shard);
    let expected = *blake3::hash(&fs::read(&path).expect("read")).as_bytes();

    let checkpoint = index
        .verify_transactional([ExpectedShardDigest {
            shard,
            blake3: expected,
        }])
        .expect("verify transactional");

    let embedding_handle = checkpoint.tensor("embedding").expect("embedding handle");
    assert_eq!(embedding_handle.tensor().name, "embedding");
    assert_eq!(embedding_handle.shard(), shard);

    let reader = checkpoint.tensor_reader("embedding").expect("reader");
    let bytes = reader.read_bytes().expect("read bytes");
    assert_eq!(bytes.len(), 8);

    let mut buf = vec![0_u8; 8];
    reader.read_into(&mut buf).expect("read into");
    assert_eq!(buf, bytes);

    // Nonexistent tensor is refused by name
    let missing_err = checkpoint
        .read_tensor("missing.tensor")
        .expect_err("missing tensor must fail");
    assert!(
        matches!(missing_err, SafetensorError::MissingRequiredTensor { name } if name == "missing.tensor")
    );
}

/// Proves that substituting a shard file on disk after verification is rejected atomically.
#[test]
fn shard_substitution_between_verification_and_binding_fails_atomically() {
    let temp = tempfile::tempdir().expect("tempdir");
    let shard1_path = temp.path().join("shard1.safetensors");
    let shard2_path = temp.path().join("shard2.safetensors");
    let rogue_path = temp.path().join("rogue.safetensors");

    let header1 = br#"{"weight1":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
    let payload1 = [11_u8; 8];
    write_shard(&shard1_path, header1, &payload1);

    let header2 = br#"{"weight2":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
    let payload2 = [22_u8; 8];
    write_shard(&shard2_path, header2, &payload2);

    let header_rogue = br#"{"weight2":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
    let payload_rogue = [99_u8; 8];
    write_shard(&rogue_path, header_rogue, &payload_rogue);

    let index_path = temp.path().join("model.safetensors.index.json");
    fs::write(
        &index_path,
        br#"{"weight_map":{"weight1":"shard1.safetensors","weight2":"shard2.safetensors"}}"#,
    )
    .expect("write index");

    let index = ShardedSafetensorIndex::open(temp.path(), &index_path).expect("open index");
    let s1_rel = Path::new("shard1.safetensors");
    let s2_rel = Path::new("shard2.safetensors");
    let d1 = *blake3::hash(&fs::read(&shard1_path).expect("read 1")).as_bytes();
    let d2 = *blake3::hash(&fs::read(&shard2_path).expect("read 2")).as_bytes();

    let checkpoint = index
        .verify_transactional([
            ExpectedShardDigest {
                shard: s1_rel,
                blake3: d1,
            },
            ExpectedShardDigest {
                shard: s2_rel,
                blake3: d2,
            },
        ])
        .expect("verify transactional");

    // Shard 2 is substituted on disk with rogue payload (file replacement on disk)!
    fs::remove_file(&shard2_path).expect("remove shard2");
    fs::copy(&rogue_path, &shard2_path).expect("substitute shard2");

    // 1. Transactional checkpoint still safely reads the verified content from the pinned handle
    let read2 = checkpoint.read_tensor("weight2").expect("read weight2");
    assert_eq!(
        read2, payload2,
        "Transactional handle must read original verified content, not substituted file"
    );

    // 2. Re-verifying or opening index against the substituted shard fails with digest mismatch
    let new_index =
        ShardedSafetensorIndex::open(temp.path(), &index_path).expect("reopen index");
    let err = new_index
        .verify_shards([
            ExpectedShardDigest {
                shard: s1_rel,
                blake3: d1,
            },
            ExpectedShardDigest {
                shard: s2_rel,
                blake3: d2,
            },
        ])
        .expect_err("verification on substituted shard must fail");
    assert!(
        matches!(err, SafetensorError::ShardDigestMismatch { ref shard, .. } if shard == s2_rel)
    );
}

/// Proves that sparse-file modification or truncation after verification is detected and refused.
#[test]
fn sparse_file_change_and_stale_manifest_fail_atomically() {
    let temp = tempfile::tempdir().expect("tempdir");
    let shard_path = temp.path().join("sparse_shard.safetensors");
    let header = br#"{"sparse_tensor":{"dtype":"U8","shape":[16],"data_offsets":[0,16]}}"#;
    write_sparse_shard(&shard_path, header, 16);

    let index_path = temp.path().join("model.safetensors.index.json");
    fs::write(
        &index_path,
        br#"{"weight_map":{"sparse_tensor":"sparse_shard.safetensors"}}"#,
    )
    .expect("write index");

    let index = ShardedSafetensorIndex::open(temp.path(), &index_path).expect("open index");
    let shard_rel = Path::new("sparse_shard.safetensors");
    let digest = *blake3::hash(&fs::read(&shard_path).expect("read sparse")).as_bytes();

    let checkpoint = index
        .verify_transactional([ExpectedShardDigest {
            shard: shard_rel,
            blake3: digest,
        }])
        .expect("verify transactional");

    // 1. Initial read succeeds
    let bytes = checkpoint
        .read_tensor("sparse_tensor")
        .expect("read sparse_tensor");
    assert_eq!(bytes.len(), 16);

    // 2. Modifying the sparse file length on disk is detected on subsequent read
    let cur_len = fs::metadata(&shard_path).expect("metadata").len();
    fs::OpenOptions::new()
        .write(true)
        .open(&shard_path)
        .expect("open")
        .set_len(cur_len + 32)
        .expect("set_len");

    let err = checkpoint
        .read_tensor("sparse_tensor")
        .expect_err("read after sparse modification must fail");
    assert!(
        matches!(&err, SafetensorError::ShardLengthChanged { shard, .. } if shard == shard_rel)
    );

    // 3. Stale manifest pointing to a non-existent or modified shard fails atomically
    let stale_index_path = temp.path().join("stale.safetensors.index.json");
    fs::write(
        &stale_index_path,
        br#"{"weight_map":{"tensor_x":"nonexistent_shard.safetensors"}}"#,
    )
    .expect("write stale index");
    assert!(
        ShardedSafetensorIndex::open(temp.path(), &stale_index_path).is_err(),
        "Stale manifest pointing to nonexistent shard must fail atomically"
    );
}

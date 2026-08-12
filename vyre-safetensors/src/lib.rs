//! Bounded safetensors metadata ingestion and immutable checkpoint identity.
//!
//! This adapter validates safetensors headers and sharded indexes without
//! owning runtime allocation, residency, scheduling, or submission.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::Read;
use std::ops::Range;
use std::path::{Component, Path, PathBuf};

use serde::de::{Error as _, IgnoredAny, MapAccess, Visitor};
use serde::{Deserialize, Deserializer as _};
use thiserror::Error;

const MAX_HEADER_BYTES: u64 = 64 * 1024 * 1024;
const MAX_SHARD_INDEX_BYTES: u64 = 64 * 1024 * 1024;
const MAX_TENSORS: usize = 1_000_000;
const MAX_TENSOR_NAME_BYTES: usize = 4_096;
const MAX_SHARD_PATH_BYTES: usize = 4_096;
const SHARD_VERIFY_BUFFER_BYTES: usize = 1024 * 1024;

/// Stable framing version for verified full-checkpoint identities.
pub const VERIFIED_CHECKPOINT_IDENTITY_VERSION: &str = "vyre-verified-safetensors-blake3-v1";

/// Safetensors element representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize)]
pub enum SafetensorDtype {
    /// Boolean byte.
    BOOL,
    /// Unsigned 8-bit integer.
    U8,
    /// Signed 8-bit integer.
    I8,
    /// Unsigned 16-bit integer.
    U16,
    /// Signed 16-bit integer.
    I16,
    /// Unsigned 32-bit integer.
    U32,
    /// Signed 32-bit integer.
    I32,
    /// Unsigned 64-bit integer.
    U64,
    /// Signed 64-bit integer.
    I64,
    /// IEEE binary16.
    F16,
    /// Brain floating point.
    BF16,
    /// IEEE binary32.
    F32,
    /// IEEE binary64.
    F64,
    /// FP8 E4M3.
    #[serde(rename = "F8_E4M3")]
    F8E4M3,
    /// FP8 E5M2.
    #[serde(rename = "F8_E5M2")]
    F8E5M2,
}

impl SafetensorDtype {
    const fn byte_width(self) -> u64 {
        match self {
            Self::BOOL | Self::U8 | Self::I8 | Self::F8E4M3 | Self::F8E5M2 => 1,
            Self::U16 | Self::I16 | Self::F16 | Self::BF16 => 2,
            Self::U32 | Self::I32 | Self::F32 => 4,
            Self::U64 | Self::I64 | Self::F64 => 8,
        }
    }
}

#[derive(Debug, Deserialize)]
struct RawTensor {
    dtype: SafetensorDtype,
    shape: Vec<u64>,
    data_offsets: [u64; 2],
}

#[derive(Debug, Deserialize)]
struct RawShardIndex {
    #[serde(deserialize_with = "deserialize_unique_weight_map")]
    weight_map: BTreeMap<String, String>,
}

/// Validated tensor metadata and its absolute file range.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SafetensorEntry {
    /// Stable checkpoint tensor name.
    pub name: String,
    /// Element representation.
    pub dtype: SafetensorDtype,
    /// Ordered tensor dimensions.
    pub shape: Vec<u64>,
    /// Absolute byte range in the shard file.
    pub file_range: Range<u64>,
}

/// Metadata-only immutable shard identity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SafetensorShardIdentity {
    /// BLAKE3 of the exact metadata header bytes.
    pub header_digest: [u8; 32],
    /// Complete shard length.
    pub file_len: u64,
    /// Start of tensor payload bytes.
    pub data_start: u64,
}

/// Validated metadata index for one safetensors shard.
#[derive(Debug, Clone)]
pub struct SafetensorIndex {
    path: PathBuf,
    identity: SafetensorShardIdentity,
    tensors: BTreeMap<String, SafetensorEntry>,
}

impl SafetensorIndex {
    /// Read and validate one shard header without allocating or reading tensor payloads.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, SafetensorError> {
        let path = path.as_ref();
        let mut file = File::open(path).map_err(|source| SafetensorError::Io {
            path: path.to_path_buf(),
            detail: source.to_string(),
        })?;
        let file_len = file
            .metadata()
            .map_err(|source| SafetensorError::Io {
                path: path.to_path_buf(),
                detail: source.to_string(),
            })?
            .len();
        if file_len < 8 {
            return Err(SafetensorError::TruncatedPrefix { file_len });
        }
        let mut prefix = [0_u8; 8];
        file.read_exact(&mut prefix)
            .map_err(|source| SafetensorError::Io {
                path: path.to_path_buf(),
                detail: source.to_string(),
            })?;
        let header_len = u64::from_le_bytes(prefix);
        if header_len > MAX_HEADER_BYTES {
            return Err(SafetensorError::HeaderTooLarge {
                header_len,
                maximum: MAX_HEADER_BYTES,
            });
        }
        let data_start = 8_u64
            .checked_add(header_len)
            .ok_or(SafetensorError::OffsetOverflow)?;
        if data_start > file_len {
            return Err(SafetensorError::TruncatedHeader {
                header_len,
                file_len,
            });
        }
        let header_len_usize =
            usize::try_from(header_len).map_err(|_| SafetensorError::HeaderTooLarge {
                header_len,
                maximum: MAX_HEADER_BYTES,
            })?;
        let mut header = vec![0_u8; header_len_usize];
        file.read_exact(&mut header)
            .map_err(|source| SafetensorError::Io {
                path: path.to_path_buf(),
                detail: source.to_string(),
            })?;
        let raw = parse_header(&header)?;
        if raw.len() > MAX_TENSORS {
            return Err(SafetensorError::TooManyTensors {
                actual: raw.len(),
                maximum: MAX_TENSORS,
            });
        }
        let data_len = file_len - data_start;
        let mut ranges = Vec::with_capacity(raw.len());
        let mut tensors = BTreeMap::new();
        for (name, tensor) in raw {
            if name.is_empty() || name.len() > MAX_TENSOR_NAME_BYTES {
                return Err(SafetensorError::InvalidName { name });
            }
            let [start, end] = tensor.data_offsets;
            if start > end || end > data_len {
                return Err(SafetensorError::RangeOutOfBounds {
                    name,
                    start,
                    end,
                    data_len,
                });
            }
            let elements = tensor.shape.iter().try_fold(1_u64, |product, extent| {
                product
                    .checked_mul(*extent)
                    .ok_or(SafetensorError::ShapeOverflow)
            })?;
            let expected_bytes = elements
                .checked_mul(tensor.dtype.byte_width())
                .ok_or(SafetensorError::ShapeOverflow)?;
            if end - start != expected_bytes {
                return Err(SafetensorError::ByteLength {
                    name,
                    actual: end - start,
                    expected: expected_bytes,
                });
            }
            ranges.push((start, end, name.clone()));
            let absolute_start = data_start
                .checked_add(start)
                .ok_or(SafetensorError::OffsetOverflow)?;
            let absolute_end = data_start
                .checked_add(end)
                .ok_or(SafetensorError::OffsetOverflow)?;
            tensors.insert(
                name.clone(),
                SafetensorEntry {
                    name,
                    dtype: tensor.dtype,
                    shape: tensor.shape,
                    file_range: absolute_start..absolute_end,
                },
            );
        }
        ranges.sort_unstable_by_key(|(start, end, _)| (*start, *end));
        for pair in ranges.windows(2) {
            let (_, left_end, left_name) = &pair[0];
            let (right_start, _, right_name) = &pair[1];
            if left_end > right_start {
                return Err(SafetensorError::Overlap {
                    left: left_name.clone(),
                    right: right_name.clone(),
                });
            }
        }
        Ok(Self {
            path: path.to_path_buf(),
            identity: SafetensorShardIdentity {
                header_digest: *blake3::hash(&header).as_bytes(),
                file_len,
                data_start,
            },
            tensors,
        })
    }

    /// Source shard path.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Metadata-only identity used before full immutable weight verification.
    #[must_use]
    pub const fn identity(&self) -> &SafetensorShardIdentity {
        &self.identity
    }

    /// Exact tensor metadata by checkpoint name.
    #[must_use]
    pub fn tensor(&self, name: &str) -> Option<&SafetensorEntry> {
        self.tensors.get(name)
    }

    /// Tensors in canonical name order.
    pub fn tensors(&self) -> impl ExactSizeIterator<Item = &SafetensorEntry> {
        self.tensors.values()
    }
}

/// Expected checkpoint tensor metadata supplied by a model compiler.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SafetensorRequirement<'a> {
    /// Exact checkpoint tensor key.
    pub name: &'a str,
    /// Required stored element representation.
    pub dtype: SafetensorDtype,
    /// Required checkpoint-order dimensions.
    pub shape: &'a [u64],
}

/// One checkpoint tensor resolved through a sharded index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckpointTensor {
    /// Relative shard path from the checkpoint root.
    pub shard: PathBuf,
    /// Validated tensor descriptor.
    pub tensor: SafetensorEntry,
}

/// Trusted full-file BLAKE3 digest for one relative checkpoint shard.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpectedShardDigest<'a> {
    /// Relative shard path exactly as it appears in the shard index.
    pub shard: &'a Path,
    /// Trusted BLAKE3 digest of the complete shard file.
    pub blake3: [u8; 32],
}

/// Content-verified identity for one complete sharded checkpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedCheckpointIdentity {
    manifest_digest: [u8; 32],
    shard_digests: BTreeMap<PathBuf, [u8; 32]>,
    content_digest: [u8; 32],
}

impl VerifiedCheckpointIdentity {
    /// Metadata manifest identity verified before payload hashing.
    #[must_use]
    pub const fn manifest_digest(&self) -> [u8; 32] {
        self.manifest_digest
    }

    /// Trusted full-file shard digests in canonical relative-path order.
    pub fn shard_digests(&self) -> impl ExactSizeIterator<Item = (&Path, &[u8; 32])> {
        self.shard_digests
            .iter()
            .map(|(path, digest)| (path.as_path(), digest))
    }

    /// Identity over the manifest plus every framed relative path and full-file digest.
    #[must_use]
    pub const fn content_digest(&self) -> [u8; 32] {
        self.content_digest
    }
}

/// Validated `model.safetensors.index.json` plus every referenced shard header.
#[derive(Debug, Clone)]
pub struct ShardedSafetensorIndex {
    tensors: BTreeMap<String, CheckpointTensor>,
    shards: BTreeMap<PathBuf, SafetensorIndex>,
    manifest_digest: [u8; 32],
}

impl ShardedSafetensorIndex {
    /// Load a shard index and every referenced metadata header.
    ///
    /// Tensor payloads are never read. The caller may stream or memory-map the
    /// returned absolute ranges after independently verifying immutable shard
    /// content digests supplied by the checkpoint distributor.
    pub fn open(
        checkpoint_root: impl AsRef<Path>,
        index_path: impl AsRef<Path>,
    ) -> Result<Self, SafetensorError> {
        let checkpoint_root = checkpoint_root.as_ref();
        let canonical_root =
            fs::canonicalize(checkpoint_root).map_err(|source| SafetensorError::Io {
                path: checkpoint_root.to_path_buf(),
                detail: source.to_string(),
            })?;
        let index_path = index_path.as_ref();
        let mut index_file = File::open(index_path).map_err(|source| SafetensorError::Io {
            path: index_path.to_path_buf(),
            detail: source.to_string(),
        })?;
        let index_len = index_file
            .metadata()
            .map_err(|source| SafetensorError::Io {
                path: index_path.to_path_buf(),
                detail: source.to_string(),
            })?
            .len();
        if index_len > MAX_SHARD_INDEX_BYTES {
            return Err(SafetensorError::ShardIndexTooLarge {
                actual: index_len,
                maximum: MAX_SHARD_INDEX_BYTES,
            });
        }
        let capacity =
            usize::try_from(index_len).map_err(|_| SafetensorError::ShardIndexTooLarge {
                actual: index_len,
                maximum: MAX_SHARD_INDEX_BYTES,
            })?;
        let mut bytes = Vec::with_capacity(capacity);
        (&mut index_file)
            .take(MAX_SHARD_INDEX_BYTES + 1)
            .read_to_end(&mut bytes)
            .map_err(|source| SafetensorError::Io {
                path: index_path.to_path_buf(),
                detail: source.to_string(),
            })?;
        if bytes.len() as u64 > MAX_SHARD_INDEX_BYTES {
            return Err(SafetensorError::ShardIndexTooLarge {
                actual: bytes.len() as u64,
                maximum: MAX_SHARD_INDEX_BYTES,
            });
        }
        let raw: RawShardIndex = serde_json::from_slice(&bytes)
            .map_err(|error| SafetensorError::ShardIndex(error.to_string()))?;
        if raw.weight_map.len() > MAX_TENSORS {
            return Err(SafetensorError::TooManyTensors {
                actual: raw.weight_map.len(),
                maximum: MAX_TENSORS,
            });
        }
        let mut shard_names = BTreeMap::<PathBuf, Vec<String>>::new();
        for (tensor, shard) in &raw.weight_map {
            if tensor.is_empty() || tensor.len() > MAX_TENSOR_NAME_BYTES {
                return Err(SafetensorError::InvalidName {
                    name: tensor.clone(),
                });
            }
            if shard.is_empty() || shard.len() > MAX_SHARD_PATH_BYTES {
                return Err(SafetensorError::UnsafeShardPath {
                    path: PathBuf::from(shard),
                });
            }
            let shard = PathBuf::from(shard);
            if !safe_relative_path(&shard) {
                return Err(SafetensorError::UnsafeShardPath { path: shard });
            }
            shard_names.entry(shard).or_default().push(tensor.clone());
        }
        let mut shards = BTreeMap::new();
        let mut tensors = BTreeMap::new();
        for (shard, expected_names) in shard_names {
            let candidate = canonical_root.join(&shard);
            let canonical_shard =
                fs::canonicalize(&candidate).map_err(|source| SafetensorError::Io {
                    path: candidate,
                    detail: source.to_string(),
                })?;
            if !canonical_shard.starts_with(&canonical_root) {
                return Err(SafetensorError::UnsafeShardPath {
                    path: shard.clone(),
                });
            }
            let index = SafetensorIndex::open(canonical_shard)?;
            for name in expected_names {
                let tensor = index.tensor(&name).cloned().ok_or_else(|| {
                    SafetensorError::MissingMappedTensor {
                        name: name.clone(),
                        shard: shard.clone(),
                    }
                })?;
                tensors.insert(
                    name,
                    CheckpointTensor {
                        shard: shard.clone(),
                        tensor,
                    },
                );
            }
            for tensor in index.tensors() {
                if !raw.weight_map.contains_key(&tensor.name) {
                    return Err(SafetensorError::UnmappedShardTensor {
                        name: tensor.name.clone(),
                        shard: shard.clone(),
                    });
                }
            }
            shards.insert(shard, index);
        }
        let mut hasher = blake3::Hasher::new();
        hasher.update(&bytes);
        for (shard, index) in &shards {
            hasher.update(shard.to_string_lossy().as_bytes());
            hasher.update(&index.identity.header_digest);
            hasher.update(&index.identity.file_len.to_le_bytes());
        }
        Ok(Self {
            tensors,
            shards,
            manifest_digest: *hasher.finalize().as_bytes(),
        })
    }

    /// Tensor metadata by checkpoint name.
    #[must_use]
    pub fn tensor(&self, name: &str) -> Option<&CheckpointTensor> {
        self.tensors.get(name)
    }

    /// Tensors in canonical name order.
    pub fn tensors(&self) -> impl ExactSizeIterator<Item = (&str, &CheckpointTensor)> {
        self.tensors
            .iter()
            .map(|(name, tensor)| (name.as_str(), tensor))
    }

    /// Referenced shards in canonical relative-path order.
    pub fn shards(&self) -> impl ExactSizeIterator<Item = (&Path, &SafetensorIndex)> {
        self.shards
            .iter()
            .map(|(path, index)| (path.as_path(), index))
    }

    /// Validate model/compiler tensor requirements against resolved shard metadata.
    ///
    /// Additional tensors are allowed because one checkpoint may contain
    /// separately compiled towers. Every supplied requirement remains unique,
    /// present, and exact in dtype and shape.
    pub fn validate_requirements<'a>(
        &self,
        requirements: impl IntoIterator<Item = SafetensorRequirement<'a>>,
    ) -> Result<(), SafetensorError> {
        let mut names = BTreeMap::new();
        for requirement in requirements {
            if names.insert(requirement.name, ()).is_some() {
                return Err(SafetensorError::DuplicateRequirement {
                    name: requirement.name.to_string(),
                });
            }
            let actual = self.tensors.get(requirement.name).ok_or_else(|| {
                SafetensorError::MissingRequiredTensor {
                    name: requirement.name.to_string(),
                }
            })?;
            if actual.tensor.dtype != requirement.dtype {
                return Err(SafetensorError::RequiredDtype {
                    name: requirement.name.to_string(),
                    actual: actual.tensor.dtype,
                    expected: requirement.dtype,
                });
            }
            if actual.tensor.shape != requirement.shape {
                return Err(SafetensorError::RequiredShape {
                    name: requirement.name.to_string(),
                    actual: actual.tensor.shape.clone(),
                    expected: requirement.shape.to_vec(),
                });
            }
        }
        Ok(())
    }

    /// Stream every complete shard through a fixed-size buffer and compare it
    /// with trusted BLAKE3 digests.
    ///
    /// The expected set must name every indexed shard exactly once and no
    /// others. A successful result is safe to use as immutable weight identity.
    pub fn verify_shards<'a>(
        &self,
        expected: impl IntoIterator<Item = ExpectedShardDigest<'a>>,
    ) -> Result<VerifiedCheckpointIdentity, SafetensorError> {
        let mut expected_by_shard = BTreeMap::new();
        for item in expected {
            if expected_by_shard
                .insert(item.shard.to_path_buf(), item.blake3)
                .is_some()
            {
                return Err(SafetensorError::DuplicateShardDigest {
                    shard: item.shard.to_path_buf(),
                });
            }
        }
        for shard in expected_by_shard.keys() {
            if !self.shards.contains_key(shard) {
                return Err(SafetensorError::UnexpectedShardDigest {
                    shard: shard.clone(),
                });
            }
        }
        for shard in self.shards.keys() {
            if !expected_by_shard.contains_key(shard) {
                return Err(SafetensorError::MissingShardDigest {
                    shard: shard.clone(),
                });
            }
        }

        let mut verified = BTreeMap::new();
        for (shard, index) in &self.shards {
            let expected_digest = expected_by_shard[shard];
            let mut file = File::open(index.path()).map_err(|source| SafetensorError::Io {
                path: index.path().to_path_buf(),
                detail: source.to_string(),
            })?;
            let before_len = file
                .metadata()
                .map_err(|source| SafetensorError::Io {
                    path: index.path().to_path_buf(),
                    detail: source.to_string(),
                })?
                .len();
            if before_len != index.identity.file_len {
                return Err(SafetensorError::ShardLengthChanged {
                    shard: shard.clone(),
                    indexed: index.identity.file_len,
                    actual: before_len,
                });
            }
            let mut hasher = blake3::Hasher::new();
            let mut buffer = vec![0_u8; SHARD_VERIFY_BUFFER_BYTES];
            loop {
                let read = file
                    .read(&mut buffer)
                    .map_err(|source| SafetensorError::Io {
                        path: index.path().to_path_buf(),
                        detail: source.to_string(),
                    })?;
                if read == 0 {
                    break;
                }
                hasher.update(&buffer[..read]);
            }
            let actual_digest = *hasher.finalize().as_bytes();
            if actual_digest != expected_digest {
                return Err(SafetensorError::ShardDigestMismatch {
                    shard: shard.clone(),
                    actual: actual_digest,
                    expected: expected_digest,
                });
            }
            verified.insert(shard.clone(), actual_digest);
        }

        let mut hasher = blake3::Hasher::new();
        hasher.update(VERIFIED_CHECKPOINT_IDENTITY_VERSION.as_bytes());
        hasher.update(&[0]);
        hasher.update(&self.manifest_digest);
        for (shard, digest) in &verified {
            let shard = shard
                .to_str()
                .ok_or_else(|| SafetensorError::UnsafeShardPath {
                    path: shard.clone(),
                })?;
            hasher.update(&(shard.len() as u64).to_le_bytes());
            hasher.update(shard.as_bytes());
            hasher.update(digest);
        }
        Ok(VerifiedCheckpointIdentity {
            manifest_digest: self.manifest_digest,
            shard_digests: verified,
            content_digest: *hasher.finalize().as_bytes(),
        })
    }

    /// Digest of index bytes plus validated shard metadata identities.
    #[must_use]
    pub const fn manifest_digest(&self) -> [u8; 32] {
        self.manifest_digest
    }
}

fn safe_relative_path(path: &Path) -> bool {
    !path.as_os_str().is_empty()
        && path
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
}

/// Safetensors ingestion failure.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum SafetensorError {
    /// Shard file access failed.
    #[error("could not access safetensors shard `{path}`: {detail}")]
    Io {
        /// Shard path.
        path: PathBuf,
        /// Operating-system error.
        detail: String,
    },
    /// The eight-byte header-length prefix is absent.
    #[error("safetensors shard is {file_len} bytes; expected an 8-byte header prefix")]
    TruncatedPrefix {
        /// Observed file length.
        file_len: u64,
    },
    /// Header allocation would exceed the ingestion boundary.
    #[error("safetensors header is {header_len} bytes; maximum is {maximum}")]
    HeaderTooLarge {
        /// Declared header bytes.
        header_len: u64,
        /// Allocation boundary.
        maximum: u64,
    },
    /// Declared header extends beyond the file.
    #[error("safetensors header declares {header_len} bytes but file is {file_len} bytes")]
    TruncatedHeader {
        /// Declared header bytes.
        header_len: u64,
        /// Observed file length.
        file_len: u64,
    },
    /// Metadata JSON is malformed or contains a duplicate tensor name.
    #[error("invalid safetensors metadata: {0}")]
    Metadata(String),
    /// Tensor count exceeds the bounded index allocation.
    #[error("safetensors metadata has {actual} tensors; maximum is {maximum}")]
    TooManyTensors {
        /// Observed tensor count.
        actual: usize,
        /// Allocation boundary.
        maximum: usize,
    },
    /// Tensor name is empty or unreasonably large.
    #[error("invalid safetensors tensor name `{name}`")]
    InvalidName {
        /// Invalid name.
        name: String,
    },
    /// Tensor data range is reversed or outside the shard payload.
    #[error("tensor `{name}` range [{start}, {end}) exceeds payload length {data_len}")]
    RangeOutOfBounds {
        /// Tensor name.
        name: String,
        /// Relative range start.
        start: u64,
        /// Relative range end.
        end: u64,
        /// Available payload bytes.
        data_len: u64,
    },
    /// Tensor shape multiplication overflowed.
    #[error("safetensors tensor shape overflows u64 byte arithmetic")]
    ShapeOverflow,
    /// Shape and dtype disagree with the declared byte range.
    #[error("tensor `{name}` contains {actual} bytes; shape and dtype require {expected}")]
    ByteLength {
        /// Tensor name.
        name: String,
        /// Declared range bytes.
        actual: u64,
        /// Required bytes.
        expected: u64,
    },
    /// Two tensor ranges overlap.
    #[error("safetensors tensor ranges overlap: `{left}` and `{right}`")]
    Overlap {
        /// Earlier tensor name.
        left: String,
        /// Later tensor name.
        right: String,
    },
    /// Shard-index allocation exceeds the same bounded metadata policy.
    #[error("safetensors shard index is {actual} bytes; maximum is {maximum}")]
    ShardIndexTooLarge {
        /// Observed index bytes.
        actual: u64,
        /// Allocation boundary.
        maximum: u64,
    },
    /// Shard-index JSON is malformed.
    #[error("invalid safetensors shard index: {0}")]
    ShardIndex(String),
    /// Shard map attempts absolute or parent-directory traversal.
    #[error("unsafe safetensors shard path `{path}`")]
    UnsafeShardPath {
        /// Rejected relative path.
        path: PathBuf,
    },
    /// Weight map names a tensor absent from its assigned shard.
    #[error("weight map assigns tensor `{name}` to `{shard}`, but that shard does not contain it")]
    MissingMappedTensor {
        /// Mapped tensor name.
        name: String,
        /// Assigned shard.
        shard: PathBuf,
    },
    /// A shard contains bytes not owned by the weight map.
    #[error("shard `{shard}` contains unmapped tensor `{name}`")]
    UnmappedShardTensor {
        /// Unexpected tensor name.
        name: String,
        /// Containing shard.
        shard: PathBuf,
    },
    /// Model/compiler requirements repeat one checkpoint key.
    #[error("checkpoint requirements repeat tensor `{name}`")]
    DuplicateRequirement {
        /// Repeated requirement name.
        name: String,
    },
    /// A model/compiler-required tensor is absent.
    #[error("checkpoint is missing required tensor `{name}`")]
    MissingRequiredTensor {
        /// Missing tensor key.
        name: String,
    },
    /// Stored tensor dtype disagrees with the compiled model port.
    #[error("tensor `{name}` has dtype {actual:?}; compiled model requires {expected:?}")]
    RequiredDtype {
        /// Tensor key.
        name: String,
        /// Stored dtype.
        actual: SafetensorDtype,
        /// Required dtype.
        expected: SafetensorDtype,
    },
    /// Stored tensor shape disagrees with the compiled model port.
    #[error("tensor `{name}` has shape {actual:?}; compiled model requires {expected:?}")]
    RequiredShape {
        /// Tensor key.
        name: String,
        /// Stored dimensions.
        actual: Vec<u64>,
        /// Required dimensions.
        expected: Vec<u64>,
    },
    /// Trusted digest input repeats one relative shard.
    #[error("trusted checkpoint digests repeat shard `{shard}`")]
    DuplicateShardDigest {
        /// Repeated relative shard path.
        shard: PathBuf,
    },
    /// Trusted digest input omits an indexed shard.
    #[error("trusted checkpoint digests omit indexed shard `{shard}`")]
    MissingShardDigest {
        /// Missing relative shard path.
        shard: PathBuf,
    },
    /// Trusted digest input names a shard absent from the index.
    #[error("trusted checkpoint digests include unknown shard `{shard}`")]
    UnexpectedShardDigest {
        /// Unknown relative shard path.
        shard: PathBuf,
    },
    /// Shard length changed after metadata indexing.
    #[error("shard `{shard}` changed length from indexed {indexed} bytes to {actual} bytes")]
    ShardLengthChanged {
        /// Relative shard path.
        shard: PathBuf,
        /// Length seen while indexing metadata.
        indexed: u64,
        /// Length seen before content verification.
        actual: u64,
    },
    /// Full shard bytes disagree with the trusted digest.
    #[error("shard `{shard}` BLAKE3 digest does not match the trusted checkpoint identity")]
    ShardDigestMismatch {
        /// Relative shard path.
        shard: PathBuf,
        /// Digest of bytes read from the indexed file.
        actual: [u8; 32],
        /// Trusted expected digest.
        expected: [u8; 32],
    },
    /// File-range arithmetic overflowed.
    #[error("safetensors file offset arithmetic overflowed")]
    OffsetOverflow,
}

fn deserialize_unique_weight_map<'de, D>(
    deserializer: D,
) -> Result<BTreeMap<String, String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    struct UniqueWeightMapVisitor;

    impl<'de> Visitor<'de> for UniqueWeightMapVisitor {
        type Value = BTreeMap<String, String>;

        fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str("a unique tensor-to-shard map")
        }

        fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
        where
            A: MapAccess<'de>,
        {
            let mut weights = BTreeMap::new();
            while let Some(name) = map.next_key::<String>()? {
                let shard = map.next_value::<String>()?;
                if weights.insert(name.clone(), shard).is_some() {
                    return Err(A::Error::custom(format!(
                        "duplicate weight-map tensor `{name}`"
                    )));
                }
            }
            Ok(weights)
        }
    }

    deserializer.deserialize_map(UniqueWeightMapVisitor)
}

fn parse_header(header: &[u8]) -> Result<BTreeMap<String, RawTensor>, SafetensorError> {
    struct HeaderVisitor;
    impl<'de> Visitor<'de> for HeaderVisitor {
        type Value = BTreeMap<String, RawTensor>;

        fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str("a safetensors metadata object")
        }

        fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
        where
            A: MapAccess<'de>,
        {
            let mut tensors = BTreeMap::new();
            while let Some(name) = map.next_key::<String>()? {
                if name == "__metadata__" {
                    map.next_value::<IgnoredAny>()?;
                    continue;
                }
                let tensor = map.next_value::<RawTensor>()?;
                if tensors.insert(name.clone(), tensor).is_some() {
                    return Err(A::Error::custom(format!("duplicate tensor name `{name}`")));
                }
            }
            Ok(tensors)
        }
    }

    let mut deserializer = serde_json::Deserializer::from_slice(header);
    let tensors = deserializer
        .deserialize_map(HeaderVisitor)
        .map_err(|error| SafetensorError::Metadata(error.to_string()))?;
    deserializer
        .end()
        .map_err(|error| SafetensorError::Metadata(error.to_string()))?;
    Ok(tensors)
}

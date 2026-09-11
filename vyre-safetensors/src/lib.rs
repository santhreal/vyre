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

mod error;
mod sharded;
mod verified;

pub use error::SafetensorError;
pub use sharded::ShardedSafetensorIndex;
pub use verified::{
    TransactionalCheckpoint, TransactionalTensorReader, VerifiedShardHandle, VerifiedTensorHandle,
};

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

/// Typed transfer descriptor mapping container entry to byte range and element layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SafetensorTransferDescriptor {
    /// Resource or tensor name.
    pub name: String,
    /// Element data type representation.
    pub dtype: SafetensorDtype,
    /// Ordered dimensions.
    pub shape: Vec<u64>,
    /// Byte offset within shard file.
    pub offset: u64,
    /// Byte length of payload.
    pub length: u64,
}

impl SafetensorEntry {
    /// Map this entry into a typed transfer descriptor without model conventions.
    #[must_use]
    pub fn to_transfer_descriptor(&self) -> SafetensorTransferDescriptor {
        SafetensorTransferDescriptor {
            name: self.name.clone(),
            dtype: self.dtype,
            shape: self.shape.clone(),
            offset: self.file_range.start,
            length: self.file_range.end.saturating_sub(self.file_range.start),
        }
    }
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

    /// Read tensor bytes directly from a shard file, verifying bounds and file length.
    pub fn read_tensor_from_file(
        &self,
        name: &str,
        file: &mut File,
    ) -> Result<Vec<u8>, SafetensorError> {
        let entry = self
            .tensor(name)
            .ok_or_else(|| SafetensorError::MissingRequiredTensor {
                name: name.to_string(),
            })?;
        let current_len = file
            .metadata()
            .map_err(|source| SafetensorError::Io {
                path: self.path.clone(),
                detail: source.to_string(),
            })?
            .len();
        if current_len != self.identity.file_len {
            return Err(SafetensorError::ShardLengthChanged {
                shard: self.path.clone(),
                indexed: self.identity.file_len,
                actual: current_len,
            });
        }
        let byte_len = entry.file_range.end - entry.file_range.start;
        let byte_len_usize =
            usize::try_from(byte_len).map_err(|_| SafetensorError::OffsetOverflow)?;
        let mut buffer = vec![0_u8; byte_len_usize];
        use std::io::Seek;
        file.seek(std::io::SeekFrom::Start(entry.file_range.start))
            .map_err(|source| SafetensorError::Io {
                path: self.path.clone(),
                detail: source.to_string(),
            })?;
        file.read_exact(&mut buffer)
            .map_err(|source| SafetensorError::Io {
                path: self.path.clone(),
                detail: source.to_string(),
            })?;
        Ok(buffer)
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

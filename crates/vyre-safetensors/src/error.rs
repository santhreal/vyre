//! Every way safetensors ingestion refuses.

use crate::*;

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
    /// Shard content was modified after verification or during reading.
    #[error("safetensors shard `{shard}` tensor `{name}` content was modified on disk")]
    ShardContentChanged {
        /// Tensor name.
        name: String,
        /// Relative shard path.
        shard: PathBuf,
    },
    /// Manifest is stale or incompatible.
    #[error("stale or invalid checkpoint manifest: {detail}")]
    StaleManifest {
        /// Rejection detail.
        detail: String,
    },
}

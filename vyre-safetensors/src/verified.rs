//! Handles that only exist once content has been verified.
//!
//! Each handle holds the file descriptor pinned during verification, so a
//! symlink swap or a path redirection after verification cannot change what a
//! later read returns.

use crate::*;

/// Immutable, verified handle to one open safetensors shard.
///
/// Holds the pinned file descriptor opened during verification so that symlink
/// swaps or path redirections after verification cannot affect reads.
#[derive(Debug, Clone)]
pub struct VerifiedShardHandle {
    pub(crate) shard_path: PathBuf,
    pub(crate) file: std::sync::Arc<std::sync::Mutex<File>>,
    pub(crate) identity: SafetensorShardIdentity,
    pub(crate) expected_digest: [u8; 32],
}

impl VerifiedShardHandle {
    /// Relative shard path.
    #[must_use]
    pub fn shard_path(&self) -> &Path {
        &self.shard_path
    }

    /// Immutable metadata identity.
    #[must_use]
    pub const fn identity(&self) -> &SafetensorShardIdentity {
        &self.identity
    }

    /// Verified BLAKE3 content digest.
    #[must_use]
    pub const fn expected_digest(&self) -> &[u8; 32] {
        &self.expected_digest
    }
}

/// Immutable verified handle to one tensor in a verified shard.
#[derive(Debug, Clone)]
pub struct VerifiedTensorHandle {
    pub(crate) shard_path: PathBuf,
    pub(crate) tensor: SafetensorEntry,
    pub(crate) file: std::sync::Arc<std::sync::Mutex<File>>,
    pub(crate) expected_shard_digest: [u8; 32],
    pub(crate) expected_file_len: u64,
    pub(crate) expected_content_digest: [u8; 32],
}

impl VerifiedTensorHandle {
    /// Relative shard path.
    #[must_use]
    pub fn shard(&self) -> &Path {
        &self.shard_path
    }

    /// Tensor metadata.
    #[must_use]
    pub fn tensor(&self) -> &SafetensorEntry {
        &self.tensor
    }

    /// Expected trusted BLAKE3 content digest for the shard containing this tensor.
    #[must_use]
    pub const fn expected_shard_digest(&self) -> &[u8; 32] {
        &self.expected_shard_digest
    }

    /// Expected verified byte length of the shard containing this tensor.
    #[must_use]
    pub const fn expected_file_len(&self) -> u64 {
        self.expected_file_len
    }

    /// Read tensor bytes directly from the open verified file descriptor.
    ///
    /// The descriptor was opened during verification, so a symlink swap or a
    /// rename that replaces the path leaves this read on the verified file. An
    /// in-place rewrite does reach it, so the bytes are hashed against the
    /// digest recorded for this tensor at verification and a read that would
    /// return different content fails instead.
    ///
    /// # Errors
    ///
    /// Returns [`SafetensorError::ShardLengthChanged`] when the shard was
    /// truncated or extended, and [`SafetensorError::ShardContentChanged`]
    /// when the tensor's own bytes no longer hash to the verified digest.
    pub fn read_bytes(&self) -> Result<Vec<u8>, SafetensorError> {
        let mut file = self.file.lock().map_err(|_| SafetensorError::Io {
            path: self.shard_path.clone(),
            detail: "mutex poisoned".to_string(),
        })?;
        let current_len = file
            .metadata()
            .map_err(|source| SafetensorError::Io {
                path: self.shard_path.clone(),
                detail: source.to_string(),
            })?
            .len();
        if current_len != self.expected_file_len {
            return Err(SafetensorError::ShardLengthChanged {
                shard: self.shard_path.clone(),
                indexed: self.expected_file_len,
                actual: current_len,
            });
        }
        let range = &self.tensor.file_range;
        let byte_len = range.end - range.start;
        let byte_len_usize =
            usize::try_from(byte_len).map_err(|_| SafetensorError::OffsetOverflow)?;
        let mut buffer = vec![0_u8; byte_len_usize];
        use std::io::Seek;
        file.seek(std::io::SeekFrom::Start(range.start))
            .map_err(|source| SafetensorError::Io {
                path: self.shard_path.clone(),
                detail: source.to_string(),
            })?;
        file.read_exact(&mut buffer)
            .map_err(|source| SafetensorError::Io {
                path: self.shard_path.clone(),
                detail: source.to_string(),
            })?;
        if *blake3::hash(&buffer).as_bytes() != self.expected_content_digest {
            return Err(SafetensorError::ShardContentChanged {
                name: self.tensor.name.clone(),
                shard: self.shard_path.clone(),
            });
        }
        Ok(buffer)
    }

    /// Open a transactional reader over this tensor.
    #[must_use]
    pub fn reader(&self) -> TransactionalTensorReader<'_> {
        TransactionalTensorReader { handle: self }
    }
}

/// Transactional reader for one verified tensor.
#[derive(Debug)]
pub struct TransactionalTensorReader<'a> {
    pub(crate) handle: &'a VerifiedTensorHandle,
}

impl<'a> TransactionalTensorReader<'a> {
    /// Read exact verified tensor payload bytes.
    pub fn read_bytes(&self) -> Result<Vec<u8>, SafetensorError> {
        self.handle.read_bytes()
    }

    /// Read tensor bytes into an existing buffer.
    pub fn read_into(&self, buf: &mut [u8]) -> Result<(), SafetensorError> {
        let bytes = self.handle.read_bytes()?;
        if buf.len() != bytes.len() {
            return Err(SafetensorError::ByteLength {
                name: self.handle.tensor.name.clone(),
                actual: buf.len() as u64,
                expected: bytes.len() as u64,
            });
        }
        buf.copy_from_slice(&bytes);
        Ok(())
    }
}

/// Content-verified checkpoint holding immutable file handles to every shard.
#[derive(Debug, Clone)]
pub struct TransactionalCheckpoint {
    pub(crate) identity: VerifiedCheckpointIdentity,
    pub(crate) shards: BTreeMap<PathBuf, VerifiedShardHandle>,
    pub(crate) tensors: BTreeMap<String, VerifiedTensorHandle>,
}

impl TransactionalCheckpoint {
    /// Content-verified checkpoint identity.
    #[must_use]
    pub const fn identity(&self) -> &VerifiedCheckpointIdentity {
        &self.identity
    }

    /// Look up verified tensor handle by name.
    #[must_use]
    pub fn tensor(&self, name: &str) -> Option<&VerifiedTensorHandle> {
        self.tensors.get(name)
    }

    /// All verified tensors in canonical name order.
    pub fn tensors(&self) -> impl ExactSizeIterator<Item = (&str, &VerifiedTensorHandle)> {
        self.tensors
            .iter()
            .map(|(name, tensor)| (name.as_str(), tensor))
    }

    /// Shard handle by relative shard path.
    #[must_use]
    pub fn shard(&self, path: &Path) -> Option<&VerifiedShardHandle> {
        self.shards.get(path)
    }

    /// All verified shard handles in canonical relative-path order.
    pub fn shards(&self) -> impl ExactSizeIterator<Item = (&Path, &VerifiedShardHandle)> {
        self.shards.iter().map(|(p, h)| (p.as_path(), h))
    }

    /// Read tensor bytes by name.
    pub fn read_tensor(&self, name: &str) -> Result<Vec<u8>, SafetensorError> {
        let handle = self
            .tensor(name)
            .ok_or_else(|| SafetensorError::MissingRequiredTensor {
                name: name.to_string(),
            })?;
        handle.read_bytes()
    }
    /// Obtain a transactional reader for one tensor by name.
    pub fn tensor_reader(
        &self,
        name: &str,
    ) -> Result<TransactionalTensorReader<'_>, SafetensorError> {
        let handle = self
            .tensor(name)
            .ok_or_else(|| SafetensorError::MissingRequiredTensor {
                name: name.to_string(),
            })?;
        Ok(handle.reader())
    }
}

//! The sharded index and the shards it names.
//!
//! A `model.safetensors.index.json` is validated together with every header it
//! points at, so a checkpoint is accepted or rejected whole.

use crate::*;

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
        self.verify_transactional(expected)
            .map(|checkpoint| checkpoint.identity)
    }

    /// Stream every complete shard through a fixed-size buffer, compare with
    /// trusted BLAKE3 digests, and return a [`TransactionalCheckpoint`] holding
    /// open verified file descriptors.
    pub fn verify_transactional<'a>(
        &self,
        expected: impl IntoIterator<Item = ExpectedShardDigest<'a>>,
    ) -> Result<TransactionalCheckpoint, SafetensorError> {
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
        let mut shard_handles = BTreeMap::new();
        let mut tensor_handles = BTreeMap::new();
        let mut tensor_content: BTreeMap<String, [u8; 32]> = BTreeMap::new();
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
            use std::io::Seek;
            file.seek(std::io::SeekFrom::Start(0))
                .map_err(|source| SafetensorError::Io {
                    path: index.path().to_path_buf(),
                    detail: source.to_string(),
                })?;
            // One sequential pass produces the whole-shard digest and every
            // tensor's content digest together. The per-tensor digests are what
            // let a later read prove the bytes it returns are the bytes that
            // were verified, which a length check alone cannot state: an
            // in-place rewrite of the same length reaches the open descriptor.
            let mut hasher = blake3::Hasher::new();
            let mut tensor_hashers: Vec<(&String, Range<u64>, blake3::Hasher)> = self
                .tensors
                .iter()
                .filter(|(_, checkpoint_tensor)| checkpoint_tensor.shard == *shard)
                .map(|(name, checkpoint_tensor)| {
                    (
                        name,
                        checkpoint_tensor.tensor.file_range.clone(),
                        blake3::Hasher::new(),
                    )
                })
                .collect();
            let mut buffer = vec![0_u8; SHARD_VERIFY_BUFFER_BYTES];
            let mut chunk_start = 0_u64;
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
                let chunk_end = chunk_start.saturating_add(read as u64);
                for (_, range, tensor_hasher) in &mut tensor_hashers {
                    let start = range.start.max(chunk_start);
                    let end = range.end.min(chunk_end);
                    if start < end {
                        let from = usize::try_from(start - chunk_start)
                            .map_err(|_| SafetensorError::OffsetOverflow)?;
                        let to = usize::try_from(end - chunk_start)
                            .map_err(|_| SafetensorError::OffsetOverflow)?;
                        tensor_hasher.update(&buffer[from..to]);
                    }
                }
                chunk_start = chunk_end;
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
            for (name, _, tensor_hasher) in tensor_hashers {
                tensor_content.insert(name.clone(), *tensor_hasher.finalize().as_bytes());
            }
            let shared_file = std::sync::Arc::new(std::sync::Mutex::new(file));
            let shard_handle = VerifiedShardHandle {
                shard_path: shard.clone(),
                file: shared_file.clone(),
                identity: index.identity.clone(),
                expected_digest: actual_digest,
            };
            shard_handles.insert(shard.clone(), shard_handle);
        }

        for (name, checkpoint_tensor) in &self.tensors {
            let shard_handle = &shard_handles[&checkpoint_tensor.shard];
            let expected_content_digest = tensor_content[name];
            tensor_handles.insert(
                name.clone(),
                VerifiedTensorHandle {
                    shard_path: checkpoint_tensor.shard.clone(),
                    tensor: checkpoint_tensor.tensor.clone(),
                    file: shard_handle.file.clone(),
                    expected_shard_digest: shard_handle.expected_digest,
                    expected_file_len: shard_handle.identity.file_len,
                    expected_content_digest,
                },
            );
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

        let identity = VerifiedCheckpointIdentity {
            manifest_digest: self.manifest_digest,
            shard_digests: verified,
            content_digest: *hasher.finalize().as_bytes(),
        };

        Ok(TransactionalCheckpoint {
            identity,
            shards: shard_handles,
            tensors: tensor_handles,
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

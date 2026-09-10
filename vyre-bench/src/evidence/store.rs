//! Content-addressed evidence storage for benchmark receipts.
//!
//! BACKLOG row 95 requires a content-addressed evidence store where two runs of the same
//! workload on the same binaries and device facts produce the same content address, and a
//! change to any recorded field produces a different one.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};
use thiserror::Error;
use vyre_foundation::failure_domain::{reclaim_poisoned_read, reclaim_poisoned_write};

use super::receipt::BenchmarkReceipt;

/// The subsystem every poison report in this module names as the owner.
const OWNER: &str = "benchmark evidence store";

/// The state every poison report in this module names.
const INDEX: &str = "the in-memory receipt index";

/// Maximum allowed byte size for a single benchmark receipt on disk (16 MiB).
pub const MAX_BENCHMARK_RECEIPT_BYTES: u64 = 16 * 1024 * 1024;
/// Errors returned by evidence store operations.
#[derive(Debug, Error)]
pub enum EvidenceStoreError {
    /// File system IO failure.
    #[error("evidence store IO error: {0}")]
    Io(#[from] std::io::Error),
    /// JSON serialization or deserialization failure.
    #[error("evidence store JSON error: {0}")]
    Json(#[from] serde_json::Error),
    /// Requested receipt content address was not found in the store.
    #[error("evidence receipt `{0}` not found in store")]
    NotFound(String),
    /// Content address mismatch between requested address and decoded receipt.
    #[error("content address mismatch: expected `{expected}`, computed `{actual}`")]
    AddressMismatch {
        /// Expected content address hash.
        expected: String,
        /// Computed content address from file contents.
        actual: String,
    },
}

/// Content-addressed storage for versioned benchmark receipts.
pub struct EvidenceStore {
    root_dir: Option<PathBuf>,
    in_memory: RwLock<BTreeMap<String, BenchmarkReceipt>>,
}

impl std::fmt::Debug for EvidenceStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EvidenceStore")
            .field("root_dir", &self.root_dir)
            .finish_non_exhaustive()
    }
}

impl EvidenceStore {
    /// Create an in-memory content-addressed evidence store.
    #[must_use]
    pub fn in_memory() -> Self {
        Self {
            root_dir: None,
            in_memory: RwLock::new(BTreeMap::new()),
        }
    }

    /// Open or create an on-disk content-addressed evidence store at `root_dir`.
    pub fn open(root_dir: impl AsRef<Path>) -> Result<Self, EvidenceStoreError> {
        let path = root_dir.as_ref().to_path_buf();
        fs::create_dir_all(&path)?;
        Ok(Self {
            root_dir: Some(path),
            in_memory: RwLock::new(BTreeMap::new()),
        })
    }

    /// Take the in-memory index for reading, keeping it after a panic.
    ///
    /// A store opened without a root directory holds its receipts here and
    /// nowhere else, so discarding the index would discard the evidence.
    fn read_index(&self) -> RwLockReadGuard<'_, BTreeMap<String, BenchmarkReceipt>> {
        reclaim_poisoned_read(&self.in_memory, OWNER, INDEX)
    }

    /// Take the in-memory index for writing under the same policy as [`Self::read_index`].
    fn write_index(&self) -> RwLockWriteGuard<'_, BTreeMap<String, BenchmarkReceipt>> {
        reclaim_poisoned_write(&self.in_memory, OWNER, INDEX)
    }

    /// Store a benchmark receipt indexed by its content address.
    ///
    /// Returns the computed content address hash string.
    pub fn put(&self, receipt: &BenchmarkReceipt) -> Result<String, EvidenceStoreError> {
        let address = receipt.content_address();
        if let Some(dir) = &self.root_dir {
            let file_path = dir.join(format!("{address}.json"));
            let json = serde_json::to_string_pretty(receipt)?;
            fs::write(file_path, json)?;
        }
        let mut mem = self.write_index();
        mem.insert(address.clone(), receipt.clone());
        Ok(address)
    }

    /// Retrieve a benchmark receipt by its content address.
    ///
    /// Validates that the loaded receipt's computed content address matches
    /// the requested address.
    pub fn get(&self, address: &str) -> Result<BenchmarkReceipt, EvidenceStoreError> {
        {
            let mem = self.read_index();
            if let Some(receipt) = mem.get(address) {
                return Ok(receipt.clone());
            }
        }
        if let Some(dir) = &self.root_dir {
            let file_path = dir.join(format!("{address}.json"));
            if file_path.exists() {
                let content = xtask::output_arg::read_text_bounded(
                    &file_path,
                    MAX_BENCHMARK_RECEIPT_BYTES,
                    "evidence receipt",
                )?;
                let receipt: BenchmarkReceipt = serde_json::from_str(&content)?;
                let computed = receipt.content_address();
                if computed != address {
                    return Err(EvidenceStoreError::AddressMismatch {
                        expected: address.to_string(),
                        actual: computed,
                    });
                }
                return Ok(receipt);
            }
        }
        Err(EvidenceStoreError::NotFound(address.to_string()))
    }

    /// Check if a receipt exists in the store.
    #[must_use]
    pub fn contains(&self, address: &str) -> bool {
        self.get(address).is_ok()
    }

    /// List all unique content addresses in the store.
    pub fn list(&self) -> Result<Vec<String>, EvidenceStoreError> {
        let mut addresses = Vec::new();
        {
            let mem = self.read_index();
            addresses.extend(mem.keys().cloned());
        }
        if let Some(dir) = &self.root_dir {
            if dir.exists() {
                for entry in fs::read_dir(dir)? {
                    let entry = entry?;
                    let path = entry.path();
                    if path.extension().is_some_and(|ext| ext == "json") {
                        if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                            if !addresses.contains(&stem.to_string()) {
                                addresses.push(stem.to_string());
                            }
                        }
                    }
                }
            }
        }
        addresses.sort();
        Ok(addresses)
    }

    /// Find a recorded benchmark receipt matching a specific input cell identity key.
    ///
    /// Enables resumable campaigns to discover whether a cell has already been
    /// executed and stored under the exact same input parameters.
    pub fn find_by_cell_key(
        &self,
        cell_key: &str,
    ) -> Result<Option<BenchmarkReceipt>, EvidenceStoreError> {
        for addr in self.list()? {
            if let Ok(receipt) = self.get(&addr) {
                if receipt.cell_identity_key() == cell_key {
                    return Ok(Some(receipt));
                }
            }
        }
        Ok(None)
    }
}

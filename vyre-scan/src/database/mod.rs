//! Scan database framing, compatibility, and decode budgets.

mod budget;
mod header;

pub use budget::{
    validate_scan_construct_decode_budget, validate_scan_database_decode_budget,
    ScanConstructDecodeBudget, ScanConstructDecodeBudgetEvidence, ScanConstructDecodeShape,
    ScanDatabaseDecodeBudget, ScanDatabaseDecodeBudgetError, ScanDatabaseDecodeBudgetEvidence,
    ScanDatabaseDecodeShape,
};
pub use header::{
    decode_compatible_scan_database_header, decode_scan_database_header,
    decode_scan_database_header_with_compatibility, encode_scan_database_header,
    put_scan_database_header, ScanDatabaseCompatibilityRecord, ScanDatabaseHeader,
    ScanDatabaseMode, ScanDatabaseReaderCompatibility, ScanDatabaseSectionHeader,
    ScanDatabaseSectionKind, UnsupportedScanFeature, MAX_SCAN_DATABASE_SECTIONS,
    MAX_SCAN_DATABASE_UNSUPPORTED_FEATURES, SCAN_DATABASE_HEADER_MAGIC,
    SCAN_DATABASE_HEADER_VERSION,
};

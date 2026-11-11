//! Tokenizer schemas and vocabulary metadata.
//!
//! Owns tokenizer definitions, special token identifiers, vocabulary boundaries,
//! and encoding schemes for frontier model architectures.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use thiserror::Error;

/// Tokenizer algorithm classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TokenizerKind {
    /// Byte-level Byte-Pair Encoding (LLaMA 3, Qwen 2.5, DeepSeek V3/V4).
    ByteLevelBpe,
    /// SentencePiece BPE with byte fallback (LLaMA 2, Mistral).
    SentencePieceBpe,
    /// SentencePiece Unigram (Gemma 1, Gemma 2).
    SentencePieceUnigram,
    /// WordPiece subword tokenization (BERT, ViT).
    WordPiece,
}

/// Error during tokenizer schema validation or sequence token checks.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum TokenizerError {
    /// Token index exceeds declared vocabulary size.
    #[error("Fix: token id {token_id} out of range for vocab size {vocab_size}")]
    TokenOutOfRange {
        /// Out of range token id.
        token_id: u32,
        /// Vocabulary size upper bound.
        vocab_size: u32,
    },
    /// Special token missing or inconsistent.
    #[error("Fix: invalid special token configuration: {reason}")]
    InvalidSpecialTokens {
        /// Reason for inconsistency.
        reason: String,
    },
}

/// Structural schema describing a model's tokenizer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenizerSchema {
    /// Canonical tokenizer algorithm.
    pub kind: TokenizerKind,
    /// Total vocabulary size.
    pub vocab_size: u32,
    /// Beginning of Sequence (BOS) token ID.
    pub bos_token_id: Option<u32>,
    /// End of Sequence (EOS) token ID.
    pub eos_token_id: Option<u32>,
    /// Padding token ID.
    pub pad_token_id: Option<u32>,
    /// Unknown token ID.
    pub unk_token_id: Option<u32>,
    /// Special added tokens mapped to their explicit token IDs.
    pub added_tokens: BTreeMap<String, u32>,
}

impl TokenizerSchema {
    /// DeepSeek V3 / V4 Flash tokenizer schema (129,280 vocabulary, ByteLevelBpe).
    #[must_use]
    pub fn deepseek() -> Self {
        let mut added = BTreeMap::new();
        added.insert("<｜begin of sentence｜>".to_string(), 0);
        added.insert("<｜end of sentence｜>".to_string(), 1);
        added.insert("<｜User｜>".to_string(), 129_281);
        added.insert("<｜Assistant｜>".to_string(), 129_282);

        Self {
            kind: TokenizerKind::ByteLevelBpe,
            vocab_size: 129_280,
            bos_token_id: Some(0),
            eos_token_id: Some(1),
            pad_token_id: Some(1),
            unk_token_id: None,
            added_tokens: added,
        }
    }

    /// LLaMA 3 / 3.1 tokenizer schema (128,256 vocabulary, ByteLevelBpe).
    #[must_use]
    pub fn llama3() -> Self {
        let mut added = BTreeMap::new();
        added.insert("<|begin_of_text|>".to_string(), 128_000);
        added.insert("<|end_of_text|>".to_string(), 128_001);
        added.insert("<|eot_id|>".to_string(), 128_009);

        Self {
            kind: TokenizerKind::ByteLevelBpe,
            vocab_size: 128_256,
            bos_token_id: Some(128_000),
            eos_token_id: Some(128_009),
            pad_token_id: Some(128_004),
            unk_token_id: None,
            added_tokens: added,
        }
    }

    /// Mistral tokenizer schema (32,768 vocabulary, SentencePieceBpe).
    #[must_use]
    pub fn mistral() -> Self {
        let mut added = BTreeMap::new();
        added.insert("<s>".to_string(), 1);
        added.insert("</s>".to_string(), 2);
        added.insert("<unk>".to_string(), 0);

        Self {
            kind: TokenizerKind::SentencePieceBpe,
            vocab_size: 32_768,
            bos_token_id: Some(1),
            eos_token_id: Some(2),
            pad_token_id: None,
            unk_token_id: Some(0),
            added_tokens: added,
        }
    }

    /// Qwen 2.5 tokenizer schema (152,064 vocabulary, ByteLevelBpe).
    #[must_use]
    pub fn qwen2_5() -> Self {
        let mut added = BTreeMap::new();
        added.insert("<|im_start|>".to_string(), 151_644);
        added.insert("<|im_end|>".to_string(), 151_645);
        added.insert("<|endoftext|>".to_string(), 151_643);

        Self {
            kind: TokenizerKind::ByteLevelBpe,
            vocab_size: 152_064,
            bos_token_id: None,
            eos_token_id: Some(151_645),
            pad_token_id: Some(151_643),
            unk_token_id: None,
            added_tokens: added,
        }
    }

    /// Gemma 2 tokenizer schema (256,000 vocabulary, SentencePieceUnigram).
    #[must_use]
    pub fn gemma2() -> Self {
        let mut added = BTreeMap::new();
        added.insert("<bos>".to_string(), 2);
        added.insert("<eos>".to_string(), 1);
        added.insert("<pad>".to_string(), 0);
        added.insert("<unk>".to_string(), 3);

        Self {
            kind: TokenizerKind::SentencePieceUnigram,
            vocab_size: 256_000,
            bos_token_id: Some(2),
            eos_token_id: Some(1),
            pad_token_id: Some(0),
            unk_token_id: Some(3),
            added_tokens: added,
        }
    }

    /// Validate a sequence of token IDs against this schema.
    pub fn validate_tokens(&self, tokens: &[u32]) -> Result<(), TokenizerError> {
        for &token in tokens {
            if token >= self.vocab_size && !self.added_tokens.values().any(|&v| v == token) {
                return Err(TokenizerError::TokenOutOfRange {
                    token_id: token,
                    vocab_size: self.vocab_size,
                });
            }
        }
        Ok(())
    }
}

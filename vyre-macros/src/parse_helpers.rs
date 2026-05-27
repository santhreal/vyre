//! Shared parsers for proc-macro argument surfaces.

use syn::parse::ParseStream;
use syn::{Expr, ExprArray, LitStr};

/// Parse a bracketed array whose entries must all be string literals.
pub(crate) fn parse_litstr_array(
    input: ParseStream<'_>,
    error_message: &'static str,
) -> syn::Result<Vec<LitStr>> {
    let array: ExprArray = input.parse()?;
    array
        .elems
        .into_iter()
        .map(|expr| match expr {
            Expr::Lit(lit) => match lit.lit {
                syn::Lit::Str(value) => Ok(value),
                other => Err(syn::Error::new_spanned(other, error_message)),
            },
            other => Err(syn::Error::new_spanned(other, error_message)),
        })
        .collect()
}


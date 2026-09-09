//! Declarative operation proc-macro generating three identity-joined records:
//! `SemanticDescriptor`, `LoweringProvider`, and `ConformanceProvider`.

use std::collections::BTreeSet;

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
use syn::{parse_macro_input, Expr, Ident, Token};

use crate::arg_parsers::reject_duplicate_key;

struct OpArg {
    key: Ident,
    _colon: Token![:],
    value: Expr,
}

impl Parse for OpArg {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let key: Ident = input.parse()?;
        let colon: Token![:] = input.parse()?;
        let value: Expr = input.parse()?;
        Ok(Self {
            key,
            _colon: colon,
            value,
        })
    }
}

struct OpInput {
    args: Punctuated<OpArg, Token![,]>,
}

impl Parse for OpInput {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let args = Punctuated::parse_terminated(input)?;
        Ok(Self { args })
    }
}

pub(crate) fn vyre_operation_impl(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as OpInput);
    let mut seen = BTreeSet::new();

    let mut id_expr: Option<Expr> = None;
    let mut semantic_version_expr: Option<Expr> = None;
    let mut signature_expr: Option<Expr> = None;
    let mut tier_expr: Option<Expr> = None;
    let mut category_expr: Option<Expr> = None;
    let mut laws_expr: Option<Expr> = None;
    let mut numeric_expr: Option<Expr> = None;
    let mut geometry_expr: Option<Expr> = None;
    let mut effects_expr: Option<Expr> = None;
    let mut capabilities_expr: Option<Expr> = None;
    let mut build_expr: Option<Expr> = None;
    let mut test_inputs_expr: Option<Expr> = None;
    let mut expected_output_expr: Option<Expr> = None;

    for arg in input.args {
        let key_name = match reject_duplicate_key(&mut seen, &arg.key) {
            Ok(k) => k,
            Err(e) => return e.to_compile_error().into(),
        };

        match key_name.as_str() {
            "id" => id_expr = Some(arg.value),
            "semantic_version" => semantic_version_expr = Some(arg.value),
            "signature" => signature_expr = Some(arg.value),
            "tier" => tier_expr = Some(arg.value),
            "category" => category_expr = Some(arg.value),
            "laws" => laws_expr = Some(arg.value),
            "numeric" => numeric_expr = Some(arg.value),
            "geometry_requirements" | "geometry" => geometry_expr = Some(arg.value),
            "explicit_effects" | "effects" => effects_expr = Some(arg.value),
            "explicit_capabilities" | "capabilities" => capabilities_expr = Some(arg.value),
            "build" => build_expr = Some(arg.value),
            "test_inputs" => test_inputs_expr = Some(arg.value),
            "expected_output" => expected_output_expr = Some(arg.value),
            other => {
                return syn::Error::new(
                    arg.key.span(),
                    format!("unknown operation argument `{other}`. Allowed: id, semantic_version, signature, tier, category, laws, numeric, geometry_requirements, effects, capabilities, build, test_inputs, expected_output"),
                )
                .to_compile_error()
                .into()
            }
        }
    }

    let Some(id) = id_expr else {
        return syn::Error::new(
            proc_macro2::Span::call_site(),
            "missing required `id` argument for `vyre_operation!`",
        )
        .to_compile_error()
        .into();
    };

    let semantic_version: TokenStream2 = semantic_version_expr
        .map(|e| quote!(#e))
        .unwrap_or_else(|| quote!(1u32));

    let signature: TokenStream2 = signature_expr
        .map(|e| quote!(#e))
        .unwrap_or_else(|| quote!(None));

    let tier: TokenStream2 = tier_expr
        .map(|e| quote!(#e))
        .unwrap_or_else(|| quote!(::vyre_foundation::operation::OperationTier::Library));

    let category: TokenStream2 = category_expr
        .map(|e| quote!(#e))
        .unwrap_or_else(|| quote!(None));

    let laws: TokenStream2 = laws_expr.map(|e| quote!(#e)).unwrap_or_else(|| quote!(&[]));

    let numeric: TokenStream2 = numeric_expr
        .map(|e| quote!(#e))
        .unwrap_or_else(|| quote!(::vyre_foundation::numeric::NumericContract::EXACT));

    let geometry: TokenStream2 = geometry_expr
        .map(|e| quote!(#e))
        .unwrap_or_else(|| quote!(::vyre_foundation::geometry::GeometryRequirements::agnostic()));

    let explicit_effects: TokenStream2 = effects_expr
        .map(|e| quote!(#e))
        .unwrap_or_else(|| quote!(None));

    let explicit_capabilities: TokenStream2 = capabilities_expr
        .map(|e| quote!(#e))
        .unwrap_or_else(|| quote!(None));

    let build: TokenStream2 = build_expr
        .map(|e| quote!(#e))
        .unwrap_or_else(|| quote!(None));

    let test_inputs: TokenStream2 = test_inputs_expr
        .map(|e| quote!(#e))
        .unwrap_or_else(|| quote!(None));

    let expected_output: TokenStream2 = expected_output_expr
        .map(|e| quote!(#e))
        .unwrap_or_else(|| quote!(None));

    let expanded = quote! {
        const _: () = {
            ::inventory::submit! {
                ::vyre_foundation::operation::SemanticDescriptor {
                    id: #id,
                    semantic_version: #semantic_version,
                    signature: #signature,
                    tier: #tier,
                    category: #category,
                    laws: #laws,
                    numeric: #numeric,
                    geometry_requirements: #geometry,
                    explicit_effects: #explicit_effects,
                    explicit_capabilities: #explicit_capabilities,
                }
            }
            ::inventory::submit! {
                ::vyre_foundation::operation::LoweringProvider {
                    id: #id,
                    build: #build,
                }
            }
            ::inventory::submit! {
                ::vyre_foundation::operation::ConformanceProvider {
                    id: #id,
                    test_inputs: #test_inputs,
                    expected_output: #expected_output,
                }
            }
        };
    };

    expanded.into()
}

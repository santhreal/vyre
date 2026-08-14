use proc_macro::TokenStream;
use quote::quote;
use syn::parse::{Parse, ParseStream};
use syn::{parse_macro_input, Attribute, Ident, Token, Type};

struct FieldDef {
    name: Ident,
    ty: Type,
}

enum VariantData {
    Unit,
    Unnamed(Vec<Type>),
    Named(Vec<FieldDef>),
}

struct AstVariant {
    attrs: Vec<Attribute>,
    ident: Ident,
    data: VariantData,
}

struct AstEnum {
    attrs: Vec<Attribute>,
    name: Ident,
    variants: Vec<AstVariant>,
}

struct AstManifest {
    enums: Vec<AstEnum>,
}

impl AstManifest {
    fn validate(&self) -> syn::Result<()> {
        let mut enum_names = std::collections::BTreeSet::new();
        for ast_enum in &self.enums {
            let enum_name = ast_enum.name.to_string();
            if !enum_names.insert(enum_name.clone()) {
                return Err(syn::Error::new_spanned(
                    &ast_enum.name,
                    format!(
                        "duplicate AST enum `{enum_name}`. Fix: merge the variants into one `{enum_name}` block or rename the second enum."
                    ),
                ));
            }

            let mut variant_names = std::collections::BTreeSet::new();
            for variant in &ast_enum.variants {
                let variant_name = variant.ident.to_string();
                if !variant_names.insert(variant_name.clone()) {
                    return Err(syn::Error::new_spanned(
                        &variant.ident,
                        format!(
                            "duplicate AST variant `{variant_name}` in `{enum_name}`. Fix: keep one `{variant_name}` variant or give each variant a stable unique name."
                        ),
                    ));
                }
            }
        }
        Ok(())
    }
}

impl Parse for AstManifest {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let mut enums = Vec::new();
        while !input.is_empty() {
            let enum_attrs = input.call(Attribute::parse_outer)?;
            let name: Ident = input.parse()?;
            let content;
            syn::braced!(content in input);
            let mut variants = Vec::new();
            while !content.is_empty() {
                let variant_attrs = content.call(Attribute::parse_outer)?;
                let v_ident: Ident = content.parse()?;
                let data = if content.peek(syn::token::Brace) {
                    let fields_content;
                    syn::braced!(fields_content in content);
                    let mut fields = Vec::new();
                    while !fields_content.is_empty() {
                        let f_name: Ident = fields_content.parse()?;
                        fields_content.parse::<Token![:]>()?;
                        let f_ty: Type = fields_content.parse()?;
                        fields.push(FieldDef {
                            name: f_name,
                            ty: f_ty,
                        });
                        if fields_content.peek(Token![,]) {
                            fields_content.parse::<Token![,]>()?;
                        }
                    }
                    VariantData::Named(fields)
                } else if content.peek(syn::token::Paren) {
                    let fields_content;
                    syn::parenthesized!(fields_content in content);
                    let mut fields = Vec::new();
                    while !fields_content.is_empty() {
                        let f_ty: Type = fields_content.parse()?;
                        fields.push(f_ty);
                        if fields_content.peek(Token![,]) {
                            fields_content.parse::<Token![,]>()?;
                        }
                    }
                    VariantData::Unnamed(fields)
                } else {
                    VariantData::Unit
                };
                variants.push(AstVariant {
                    attrs: variant_attrs,
                    ident: v_ident,
                    data,
                });
                if content.peek(Token![,]) {
                    content.parse::<Token![,]>()?;
                }
            }
            enums.push(AstEnum {
                attrs: enum_attrs,
                name,
                variants,
            });
        }
        Ok(AstManifest { enums })
    }
}

#[allow(clippy::too_many_lines)]
pub(crate) fn vyre_ast_registry_impl(item: TokenStream) -> TokenStream {
    let manifest = parse_macro_input!(item as AstManifest);
    if let Err(error) = manifest.validate() {
        return error.to_compile_error().into();
    }

    let mut outputs = Vec::new();

    for ast_enum in manifest.enums {
        let enum_name = &ast_enum.name;
        let enum_attrs = &ast_enum.attrs;

        let variants = ast_enum.variants.iter().map(|v| {
            let ident = &v.ident;
            let attrs = &v.attrs;
            match &v.data {
                VariantData::Unit => quote! { #(#attrs)* #ident },
                VariantData::Unnamed(types) => quote! { #(#attrs)*  #ident(#(#types),*) },
                VariantData::Named(fields) => {
                    let f = fields.iter().map(|f| {
                        let n = &f.name;
                        let t = &f.ty;
                        quote! { #n: #t }
                    });
                    quote! { #(#attrs)* #ident { #(#f),* } }
                }
            }
        });

        // op_id implementation
        let op_ids = ast_enum.variants.iter().map(|v| {
            let ident = &v.ident;
            if ident == "Opaque" {
                quote! {
                    #enum_name::Opaque(ext) => ext.extension_kind().to_string()
                }
            } else {
                let lower_name = format!(
                    "vyre.{}.{}",
                    enum_name.to_string().to_lowercase(),
                    ident.to_string().to_lowercase()
                );
                match &v.data {
                    VariantData::Unit => quote! {
                        #enum_name::#ident => #lower_name.to_string()
                    },
                    VariantData::Unnamed(_) => quote! {
                        #enum_name::#ident ( .. ) => #lower_name.to_string()
                    },
                    VariantData::Named(_) => quote! {
                        #enum_name::#ident { .. } => #lower_name.to_string()
                    },
                }
            }
        });

        let op_id_fn_name = syn::Ident::new(
            &format!("{}_op_id", enum_name.to_string().to_lowercase()),
            proc_macro2::Span::call_site(),
        );

        // Run-time enumeration of the declared variants.
        //
        // The variant list is only knowable at the declaration site, and a
        // downstream crate cannot recover it: these enums are
        // `#[non_exhaustive]`, so no match outside the defining crate is
        // exhaustive and no test outside it can notice a variant it was never
        // told about. Emitting the names here makes "did anyone record a
        // decision for every variant" a checkable property instead of a
        // hand-maintained list that goes stale silently.
        let variant_name_strings = ast_enum
            .variants
            .iter()
            .map(|v| v.ident.to_string())
            .collect::<Vec<_>>();
        let variant_names_const = syn::Ident::new(
            &format!("{}_VARIANT_NAMES", enum_name.to_string().to_uppercase()),
            proc_macro2::Span::call_site(),
        );
        let variant_name_fn = syn::Ident::new(
            &format!("{}_variant_name", enum_name.to_string().to_lowercase()),
            proc_macro2::Span::call_site(),
        );
        let variant_name_arms = ast_enum.variants.iter().map(|v| {
            let ident = &v.ident;
            let name = ident.to_string();
            match &v.data {
                VariantData::Unit => quote! { #enum_name::#ident => #name },
                VariantData::Unnamed(_) => quote! { #enum_name::#ident ( .. ) => #name },
                VariantData::Named(_) => quote! { #enum_name::#ident { .. } => #name },
            }
        });

        // PartialEq implementations
        let partial_eq_arms = ast_enum.variants.iter().map(|v| {
            let ident = &v.ident;
            if ident == "Opaque" {
                // Special case for Opaque
                quote! {
                    (Self::Opaque(left), Self::Opaque(right)) => {
                        left.extension_kind() == right.extension_kind()
                            && left.stable_fingerprint() == right.stable_fingerprint()
                    }
                }
            } else {
                match &v.data {
                    VariantData::Unit => quote! {
                        (Self::#ident, Self::#ident) => true,
                    },
                    VariantData::Unnamed(types) => {
                        let lefts: Vec<_> = (0..types.len()).map(|i| syn::Ident::new(&format!("l{i}"), proc_macro2::Span::call_site())).collect();
                        let rights: Vec<_> = (0..types.len()).map(|i| syn::Ident::new(&format!("r{i}"), proc_macro2::Span::call_site())).collect();
                        let checks = lefts.iter().zip(rights.iter()).map(|(l, r)| quote! { #l == #r });
                        quote! {
                            (Self::#ident(#(#lefts),*), Self::#ident(#(#rights),*)) => { #(#checks)&&* },
                        }
                    },
                    VariantData::Named(fields) => {
                        let lefts: Vec<_> = fields.iter().map(|f| syn::Ident::new(&format!("l_{}", f.name), proc_macro2::Span::call_site())).collect();
                        let rights: Vec<_> = fields.iter().map(|f| syn::Ident::new(&format!("r_{}", f.name), proc_macro2::Span::call_site())).collect();
                        let f_names = fields.iter().map(|f| &f.name);
                        let f_names2 = fields.iter().map(|f| &f.name);
                        let checks = lefts.iter().zip(rights.iter()).map(|(l, r)| quote! { #l == #r });
                        quote! {
                            (Self::#ident { #(#f_names: #lefts),* }, Self::#ident { #(#f_names2: #rights),* }) => { #(#checks)&&* },
                        }
                    }
                }
            }
        });

        outputs.push(quote! {
            #(#enum_attrs)*
            #[allow(missing_docs)]
            #[non_exhaustive]
            #[derive(Debug, Clone)]
            pub enum #enum_name {
                #(#variants),*
            }

            impl PartialEq for #enum_name {
                fn eq(&self, other: &Self) -> bool {
                    match (self, other) {
                        #(#partial_eq_arms)*
                        _ => false,
                    }
                }
            }

            #[must_use]
            pub fn #op_id_fn_name(item: &#enum_name) -> String {
                match item {
                    #(#op_ids,)*
                }
            }

            /// Every declared variant name, in declaration order.
            pub const #variant_names_const: &[&str] = &[#(#variant_name_strings),*];

            /// The declared name of the variant `item` holds.
            #[must_use]
            pub fn #variant_name_fn(item: &#enum_name) -> &'static str {
                match item {
                    #(#variant_name_arms,)*
                }
            }
        });
    }

    let out = quote! { #(#outputs)* };
    out.into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use quote::quote;

    #[test]
    fn ast_manifest_accepts_unit_tuple_and_named_variants() {
        let manifest = syn::parse2::<AstManifest>(quote! {
            Expr {
                Const,
                Unary(u32),
                Binary { left: u32, right: u32 },
            }
        })
        .expect("Fix: AST manifest should parse unit, tuple, and named variants");

        manifest
            .validate()
            .expect("Fix: unique AST enum and variant names should validate");
        assert_eq!(manifest.enums.len(), 1);
        assert_eq!(manifest.enums[0].variants.len(), 3);
    }

    #[test]
    fn ast_manifest_rejects_duplicate_enum_names() {
        let manifest = syn::parse2::<AstManifest>(quote! {
            Expr { Const }
            Expr { Add }
        })
        .expect("Fix: duplicate enum names are a validation error, not a parse error");

        let err = manifest
            .validate()
            .expect_err("Fix: duplicate AST enum names must be rejected");

        assert!(err.to_string().contains("duplicate AST enum"));
        assert!(err.to_string().contains("Fix:"));
    }

    #[test]
    fn ast_manifest_rejects_duplicate_variant_names() {
        let manifest = syn::parse2::<AstManifest>(quote! {
            Expr {
                Const,
                Const,
            }
        })
        .expect("Fix: duplicate variant names are a validation error, not a parse error");

        let err = manifest
            .validate()
            .expect_err("Fix: duplicate AST variant names must be rejected");

        assert!(err.to_string().contains("duplicate AST variant"));
        assert!(err.to_string().contains("Fix:"));
    }
}

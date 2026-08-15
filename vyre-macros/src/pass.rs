use crate::arg_parsers;
use proc_macro::TokenStream;
use quote::quote;
use syn::parse::{Parse, ParseStream};
use syn::{parse_macro_input, Fields, ItemStruct, LitBool, LitStr, Token};

pub(crate) struct PassArgs {
    pub(crate) name: LitStr,
    pub(crate) requires: Vec<LitStr>,
    pub(crate) invalidates: Vec<LitStr>,
    pub(crate) phase: Option<LitStr>,
    pub(crate) boundary_class: Option<LitStr>,
    pub(crate) requires_caps: Vec<LitStr>,
    pub(crate) preserves_abi: Option<LitBool>,
    pub(crate) cost_model_family: Option<LitStr>,
    pub(crate) analyze_always: bool,
    /// The pass reads device facts and implements `transform_for_adapter`.
    pub(crate) adapter_dependent: bool,
}

impl Parse for PassArgs {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let mut name = None;
        let mut requires = Vec::new();
        let mut invalidates = Vec::new();
        let mut phase = None;
        let mut boundary_class = None;
        let mut requires_caps = Vec::new();
        let mut preserves_abi = None;
        let mut cost_model_family = None;
        let mut analyze_always = false;
        let mut adapter_dependent = false;
        let mut seen_keys = std::collections::BTreeSet::new();

        while !input.is_empty() {
            let key: syn::Ident = input.parse()?;
            let key_name = arg_parsers::reject_duplicate_key(&mut seen_keys, &key)?;
            input.parse::<Token![=]>()?;
            match key_name.as_str() {
                "name" => name = Some(input.parse()?),
                "requires" => {
                    requires = arg_parsers::parse_litstr_array(
                        input,
                        "pass metadata arrays accept only string literals. Fix: use [\"analysis_name\"].",
                    )?
                }
                "invalidates" => {
                    invalidates = arg_parsers::parse_litstr_array(
                        input,
                        "pass metadata arrays accept only string literals. Fix: use [\"analysis_name\"].",
                    )?
                }
                "phase" => phase = Some(input.parse()?),
                "boundary_class" => boundary_class = Some(input.parse()?),
                "requires_caps" => {
                    requires_caps = arg_parsers::parse_litstr_array(
                        input,
                        "pass metadata arrays accept only string literals. Fix: use [\"analysis_name\"].",
                    )?
                }
                "adapter_dependent" => {
                    adapter_dependent = input.parse::<LitBool>()?.value;
                }
                "preserves_abi" => preserves_abi = Some(input.parse()?),
                "cost_model_family" => cost_model_family = Some(input.parse()?),
                "analyze" => {
                    let value: LitStr = input.parse()?;
                    if value.value() == "always" {
                        analyze_always = true;
                    } else {
                        return Err(syn::Error::new_spanned(
                            value,
                            "unsupported analyze mode. Fix: use analyze = \"always\" or omit it.",
                        ));
                    }
                }
                _ => {
                    return Err(syn::Error::new(
                        key.span(),
                        "unsupported vyre_pass argument. Fix: use name, requires, invalidates, phase, boundary_class, requires_caps, preserves_abi, cost_model_family, adapter_dependent, or analyze.",
                    ));
                }
            }
            if input.peek(Token![,]) {
                input.parse::<Token![,]>()?;
            }
        }

        validate_unique_string_literals("requires", &requires)?;
        validate_unique_string_literals("invalidates", &invalidates)?;
        validate_unique_string_literals("requires_caps", &requires_caps)?;

        Ok(Self {
            name: name.ok_or_else(|| input.error("missing pass name. Fix: add name = \"...\"."))?,
            requires,
            invalidates,
            phase,
            boundary_class,
            requires_caps,
            preserves_abi,
            cost_model_family,
            analyze_always,
            adapter_dependent,
        })
    }
}

/// One pass metadata enum the attribute lowers a string literal into.
///
/// The accepted strings, the variants they produce, and the diagnostic that
/// lists them are one datum. Three hand-written copies of that list used to
/// exist per enum (the match arms, the error text, and the unit-test case
/// table), so a new variant could be accepted by the macro and never named by
/// the diagnostic or reached by a test.
pub(crate) struct MetadataEnum {
    /// `vyre_pass` argument this enum is written as.
    pub(crate) argument: &'static str,
    /// Enum type name under `::vyre::optimizer`.
    pub(crate) type_name: &'static str,
    /// Accepted attribute strings paired with the variant each lowers to. The
    /// first row is what an omitted argument produces.
    pub(crate) rows: &'static [(&'static str, &'static str)],
}

impl MetadataEnum {
    /// Variant this enum lowers `value` to, or a diagnostic naming every
    /// accepted string.
    pub(crate) fn tokens(&self, value: Option<&LitStr>) -> syn::Result<proc_macro2::TokenStream> {
        let variant = match value {
            None => self.rows[0].1,
            Some(literal) => {
                let text = literal.value();
                let found = self
                    .rows
                    .iter()
                    .find(|(accepted, _)| *accepted == text)
                    .map(|(_, variant)| *variant);
                let Some(variant) = found else {
                    return Err(syn::Error::new_spanned(
                        literal,
                        format!(
                            "unsupported pass {}. Fix: use {}.",
                            self.argument,
                            self.accepted_list()
                        ),
                    ));
                };
                variant
            }
        };
        let type_ident = syn::Ident::new(self.type_name, proc_macro2::Span::call_site());
        let variant_ident = syn::Ident::new(variant, proc_macro2::Span::call_site());
        Ok(quote! { ::vyre::optimizer::#type_ident::#variant_ident })
    }

    /// Accepted strings in declaration order, as prose for a diagnostic.
    fn accepted_list(&self) -> String {
        let mut list = String::new();
        for (index, (accepted, _)) in self.rows.iter().enumerate() {
            if index > 0 {
                list.push_str(", ");
            }
            if index + 1 == self.rows.len() && self.rows.len() > 1 {
                list.push_str("or ");
            }
            list.push_str(accepted);
        }
        list
    }
}

pub(crate) const PASS_PHASE: MetadataEnum = MetadataEnum {
    argument: "phase",
    type_name: "PassPhase",
    rows: &[
        ("unclassified", "Unclassified"),
        ("canonicalization", "Canonicalization"),
        ("scalar_algebra", "ScalarAlgebra"),
        ("loop", "Loop"),
        ("memory", "Memory"),
        ("fusion_cse", "FusionCse"),
        ("sync", "Sync"),
        ("specialization", "Specialization"),
        ("cleanup", "Cleanup"),
        ("dataflow", "Dataflow"),
        ("megakernel", "Megakernel"),
    ],
};

pub(crate) const PASS_BOUNDARY_CLASS: MetadataEnum = MetadataEnum {
    argument: "boundary_class",
    type_name: "PassBoundaryClass",
    rows: &[
        ("unknown", "Unknown"),
        ("abi_preserving", "AbiPreserving"),
        ("abi_changing", "AbiChanging"),
        ("backend_aware", "BackendAware"),
        ("runtime_aware", "RuntimeAware"),
        ("domain_specific", "DomainSpecific"),
    ],
};

pub(crate) const PASS_COST_MODEL_FAMILY: MetadataEnum = MetadataEnum {
    argument: "cost_model_family",
    type_name: "CostModelFamily",
    rows: &[
        ("unknown", "Unknown"),
        ("scalar", "Scalar"),
        ("loop", "Loop"),
        ("memory", "Memory"),
        ("fusion", "Fusion"),
        ("sync", "Sync"),
        ("dataflow", "Dataflow"),
        ("megakernel", "Megakernel"),
    ],
};

fn validate_unique_string_literals(field: &str, values: &[LitStr]) -> syn::Result<()> {
    let mut seen = std::collections::BTreeSet::new();
    for value in values {
        let text = value.value();
        if !seen.insert(text.clone()) {
            return Err(syn::Error::new_spanned(
                value,
                format!(
                    "duplicate vyre_pass {field} entry `{text}`. Fix: list each dependency, invalidation, or capability once."
                ),
            ));
        }
    }
    Ok(())
}

/// Register a unit struct as a `vyre::optimizer::ProgramPass`.
///
/// Expands to (a) a full `ProgramPass` trait impl that forwards to your inherent
/// `analyze` / `transform` methods plus the canonical optimizer
/// fingerprint and (b) an
/// `inventory::submit!` that adds the pass to the global registry so
/// `vyre::optimize()` picks it up automatically.
///
/// # Arguments
///
/// | Argument       | Type        | Meaning                                                             |
/// |----------------|-------------|---------------------------------------------------------------------|
/// | `name`         | string lit  | Stable pass name used in diagnostics / ordering.                    |
/// | `requires`     | `[&str]`    | Pass names that must fire before this one.                          |
/// | `invalidates`  | `[&str]`    | Analyses invalidated when this pass rewrites the program.           |
/// | `phase`        | string lit  | Optional scheduler phase.                                           |
/// | `boundary_class` | string lit | Optional architectural boundary class.                              |
/// | `requires_caps` | `[&str]`   | Optional backend/runtime capabilities required by the pass.          |
/// | `preserves_abi` | bool       | Whether public buffer ABI is preserved. Defaults to true.            |
/// | `cost_model_family` | string lit | Optional cost attribution family.                                |
///
/// # Required inherent methods on the annotated type
///
/// ```ignore
/// fn analyze_impl(program: &Program) -> PassAnalysis;
/// fn transform(program: Program) -> PassResult;
/// ```
///
/// # Example
///
/// ```ignore
/// use vyre::optimizer::{vyre_pass, PassAnalysis, PassResult};
/// use vyre::ir::Program;
///
/// #[vyre_pass(name = "fold_zero_add", requires = [], invalidates = [])]
/// pub struct FoldZeroAdd;
///
/// impl FoldZeroAdd {
///     fn analyze(_program: &Program) -> PassAnalysis { PassAnalysis::RUN }
///     fn transform(program: Program) -> PassResult {
///         // ... real rewrite ...
///         PassResult::from_programs(&program.clone(), program)
///     }
/// }
/// ```
///
/// After expansion, `vyre::optimize(p)` will pick up `FoldZeroAdd` through
/// the `inventory::collect!(ProgramPassRegistration)` entry emitted by the macro.
/// No manual registration needed.
pub(crate) fn vyre_pass_impl(args: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(args as PassArgs);
    let item = parse_macro_input!(item as ItemStruct);
    if !matches!(item.fields, Fields::Unit) {
        return syn::Error::new_spanned(
            &item.ident,
            "#[vyre_pass] supports only unit structs. Fix: move pass state into explicit scheduler/context storage and declare the pass as `pub struct PassName;`.",
        )
        .to_compile_error()
        .into();
    }
    let ident = &item.ident;
    let name = args.name;
    let requires = args.requires;
    let invalidates = args.invalidates;
    let requires_caps = args.requires_caps;
    let phase = match PASS_PHASE.tokens(args.phase.as_ref()) {
        Ok(tokens) => tokens,
        Err(error) => return error.to_compile_error().into(),
    };
    let boundary_class = match PASS_BOUNDARY_CLASS.tokens(args.boundary_class.as_ref()) {
        Ok(tokens) => tokens,
        Err(error) => return error.to_compile_error().into(),
    };
    let cost_model_family = match PASS_COST_MODEL_FAMILY.tokens(args.cost_model_family.as_ref()) {
        Ok(tokens) => tokens,
        Err(error) => return error.to_compile_error().into(),
    };
    let preserves_abi = args.preserves_abi.map(|value| value.value).unwrap_or(true);
    let analyze_body = if args.analyze_always {
        quote! { ::vyre::optimizer::PassAnalysis::RUN }
    } else {
        quote! { Self::analyze_impl(program) }
    };
    // A pass is device-dependent only by saying so. Everything else gets a
    // `transform_for_adapter` that discards the adapter, which is the honest
    // statement that its rewrite is the same program on every device; the
    // alternative, letting a pass pick a profile inside `transform`, is how
    // the whole pipeline came to compile against one profile nobody chose.
    let transform_for_adapter_body = if args.adapter_dependent {
        quote! { Self::transform_for_adapter(program, caps) }
    } else {
        quote! {
            let _ = caps;
            Self::transform(program)
        }
    };
    let metadata = quote! {
        ::vyre::optimizer::PassMetadata {
            name: #name,
            requires: &[#(#requires),*],
            invalidates: &[#(#invalidates),*],
            phase: #phase,
            boundary_class: #boundary_class,
            requires_caps: &[#(#requires_caps),*],
            preserves_abi: #preserves_abi,
            cost_model_family: #cost_model_family,
        }
    };

    quote! {
        #item

        impl ::vyre::optimizer::sealed::Sealed for #ident {}

        impl ::vyre::optimizer::ProgramPass for #ident {
            #[inline]
            fn metadata(&self) -> ::vyre::optimizer::PassMetadata {
                #metadata
            }

            #[inline]
            fn analyze(&self, program: &::vyre::ir::Program) -> ::vyre::optimizer::PassAnalysis {
                #analyze_body
            }

            #[inline]
            fn transform(
                &self,
                program: ::vyre::ir::Program,
            ) -> ::vyre::optimizer::PassResult {
                Self::transform(program)
            }

            #[inline]
            fn transform_for_adapter(
                &self,
                program: ::vyre::ir::Program,
                caps: &::vyre::optimizer::AdapterCaps,
            ) -> ::vyre::optimizer::PassResult {
                #transform_for_adapter_body
            }

            #[inline]
            fn fingerprint(&self, program: &::vyre::ir::Program) -> u64 {
                ::vyre::optimizer::fingerprint_program(program)
            }
        }

        ::inventory::submit! {
            ::vyre::optimizer::ProgramPassRegistration {
                metadata: #metadata,
                factory: || ::std::boxed::Box::new(#ident),
            }
        }
    }
    .into()
}

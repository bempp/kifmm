#![allow(non_upper_case_globals)] // allow top-level consts if any

#[allow(
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals,
    clippy::all
)]
mod bindings {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
    include!(concat!(env!("OUT_DIR"), "/codegen.rs"));
}

#[allow(unused)]
pub use bindings::*;

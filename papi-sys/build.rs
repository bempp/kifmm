use std::env;
use std::fs::File;
use std::io::prelude::*;
use std::path::PathBuf;
use std::process::Command;
use pkg_config;


fn main() {
    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rerun-if-env-changed=PAPI_PREFIX");

    // Detect PAPI prefix (manual override)
    let papi_prefix = std::env::var("PAPI_PREFIX").ok();

    let papi_found = if let Some(prefix) = papi_prefix {
        println!("cargo:rustc-link-search={}", format!("{}/lib", prefix));
        println!("cargo:rustc-link-lib=static=papi"); // ⬅ force static
        Some(vec![format!("-I{}/include", prefix)])
    } else if pkg_config::Config::new().statik(true).probe("papi").is_ok() {
        // pkg-config with static mode
        Some(Vec::new())
    } else {
        None
    };

    if let Some(clang_args) = papi_found {
        let out_path = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());

        bindgen::Builder::default()
            .header("wrapper.h")
            .clang_args(clang_args)
            .allowlist_function("^PAPI_[[:alpha:]_]+")
            .allowlist_var("^PAPI_[[:alpha:]_]+")
            .allowlist_type("^PAPI_[[:alpha:]_]+")
            .generate()
            .expect("Unable to generate PAPI bindings")
            .write_to_file(out_path.join("bindings.rs"))
            .expect("Unable to write PAPI bindings");

        println!("cargo:rustc-cfg=has_papi");
    } else {
        println!("cargo:warning=PAPI not found, skipping bindings");
        let out_path = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());
        std::fs::write(out_path.join("bindings.rs"), "// no papi\n").unwrap();
    }
}

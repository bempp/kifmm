use std::env;
use std::fs::File;
use std::io::prelude::*;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rustc-check-cfg=cfg(has_papi)");
    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rerun-if-env-changed=PAPI_PREFIX");
    println!("cargo:rerun-if-env-changed=PAPI_STATIC");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());

    let papi_prefix = env::var("PAPI_PREFIX").ok();
    let prefer_static = env::var("PAPI_STATIC")
        .map(|v| v == "1" || v.to_lowercase() == "true")
        .unwrap_or(false);

    let papi_found = if let Some(prefix) = papi_prefix {
        println!("cargo:rustc-link-search={}/lib", prefix);

        let static_path = format!("{}/lib/libpapi.a", prefix);
        if prefer_static && std::path::Path::new(&static_path).exists() {
            println!("cargo:rustc-link-lib=static=papi");
        } else {
            println!("cargo:rustc-link-lib=papi"); // fall back to dynamic
        }

        Some(vec![format!("-I{}/include", prefix)])
    } else if pkg_config::Config::new()
        .statik(prefer_static)
        .probe("papi")
        .is_ok()
    {
        // pkg-config already added the link flags
        Some(Vec::new())
    } else {
        None
    };

    if let Some(clang_args) = papi_found {
        bindgen::Builder::default()
            .header("wrapper.h")
            .clang_args(clang_args)
            .allowlist_function("^PAPI_.*")
            .allowlist_var("^PAPI_.*")
            .allowlist_type("^PAPI_.*")
            .generate()
            .expect("Unable to generate PAPI bindings")
            .write_to_file(out_path.join("bindings.rs"))
            .expect("Unable to write PAPI bindings");

        println!("cargo:rustc-cfg=has_papi");

        // Run codegen.sh to dump constants
        let codegen_out = Command::new("sh")
            .arg("codegen.sh")
            .output()
            .expect("failed to run codegen.sh")
            .stdout;

        let mut file = File::create(out_path.join("codegen.rs")).unwrap();
        file.write_all(&codegen_out).unwrap();
        file.sync_all().unwrap();
    } else {
        println!("cargo:warning=PAPI not found, skipping bindings");
        std::fs::write(out_path.join("bindings.rs"), "// no papi\n").unwrap();
    }
}

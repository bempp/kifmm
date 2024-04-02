use std::env;
use std::path::PathBuf;

fn main() {
    // Tell cargo to invalidate the built crate whenever the wrapper changes
    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rustc-link-lib=static=fftw3");
    println!("cargo:rustc-link-lib=static=fftw3f");


    let root = PathBuf::from(std::env::var("DEP_FFTW3_ROOT").unwrap());
    println!("HERE {:?}", root.display());

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());

    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg(format!("-I{}", root.join("api").display())) // Use the determined path
        .allowlist_type("^fftw.*")
        .allowlist_var("^FFTW.*")
        .allowlist_function("^fftw.*")
        .allowlist_type("^fftwf.*")
        .allowlist_function("^fftwf.*")
        .blocklist_type("FILE")
        .blocklist_type("fftw.*_complex")
        .blocklist_function("fftwl_.*")
        .default_enum_style(bindgen::EnumVariation::Rust {
            non_exhaustive: true,
        })
        .generate()
        .expect("Unable to generate bindings");

    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();

    // Determine the target directory within the workspace root
    let target_dir = env::var("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| Path::new(&crate_dir).join("target"));

    // Ensure the target directory exists
    fs::create_dir_all(target_dir).expect("Unable to create target directory");

    // Create the header file path
    let header_path = Path::new(&crate_dir).join("include").join("kifmm_rs.h");

    let config_path = Path::new(&crate_dir).join("cbindgen.toml");
    let config = cbindgen::Config::from_file(config_path).expect("Unable to load cbindgen config");

    // Generate the bindings
    let bindings = cbindgen::Builder::new()
        .with_crate(crate_dir)
        .with_config(config)
        .generate()
        .expect("Unable to generate bindings");

    // Write the bindings to the header file
    bindings.write_to_file(header_path);
}

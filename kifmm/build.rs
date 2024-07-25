use cbindgen;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();

    // Determine the workspace root directory by finding the nearest ancestor directory containing a Cargo.toml
    let workspace_root = find_workspace_root(&crate_dir).expect("Unable to find workspace root");

    // Determine the target directory within the workspace root
    let target_dir = env::var("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| workspace_root.join("target"));

    // Ensure the target directory exists
    fs::create_dir_all(&target_dir).expect("Unable to create target directory");

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

// Helper function to find the workspace root directory
fn find_workspace_root(current_dir: &str) -> Option<PathBuf> {
    let mut dir = PathBuf::from(current_dir);
    while dir.pop() {
        if dir.join("Cargo.toml").exists() {
            return Some(dir);
        }
    }
    None
}

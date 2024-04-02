use std::{
    env::var,
    fs::canonicalize,
    path::{Path, PathBuf},
    process::Command,
};

fn run(command: &mut Command) {
    println!("Running: {:?}", command);
    match command.status() {
        Ok(status) => {
            if !status.success() {
                panic!("`{:?}` failed: {}", command, status);
            }
        }
        Err(error) => {
            panic!("failed to execute `{:?}`: {}", command, error);
        }
    }
}

fn build_fftw(src_dir: &Path, out_dir: &Path, flags: &[&str]) {
    run(
        Command::new(canonicalize(src_dir.join("configure")).unwrap())
            .arg("--with-pic")
            .arg("--enable-static")
            .arg("--disable-doc")
            .arg(format!("--prefix={}", out_dir.display()))
            .args(flags)
            .current_dir(src_dir),
    );

    run(Command::new("make")
        .arg(format!("-j{}", var("NUM_JOBS").unwrap()))
        .current_dir(src_dir));

    run(Command::new("make").arg("install").current_dir(src_dir));
}

fn main() {
    let out_dir = PathBuf::from(var("OUT_DIR").unwrap());
    let src_dir = PathBuf::from(var("CARGO_MANIFEST_DIR").unwrap()).join("fftw-3.3.10");
    let out_src_dir = out_dir.join("src");

    // Out of source build
    fs_extra::dir::copy(
        &src_dir,
        &out_src_dir,
        &fs_extra::dir::CopyOptions {
            overwrite: true,
            skip_exist: false,
            buffer_size: 64000,
            copy_inside: true,
            depth: 0,
            content_only: false,
        },
    )
    .unwrap();

    // Build FFTW
    if !out_dir.join("lib/libfftw3.a").exists() {
        build_fftw(&src_dir, &out_dir, &[]);
    }

    // Build FFTWF
    if !out_dir.join("lib/libfftw3f.a").exists() {
        build_fftw(&src_dir, &out_dir, &["--enable-single"]);
    }

    println!("cargo:rustc-link-search={}", out_dir.join("lib").display());
    println!("cargo:rustc-link-lib=static=fftw3");
    println!("cargo:rustc-link-lib=static=fftw3f");
    println!("cargo:root={}", out_src_dir.display());
}

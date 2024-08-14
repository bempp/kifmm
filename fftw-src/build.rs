use std::{
    env::var,
    fs::{canonicalize, File},
    io::Write,
    path::{Path, PathBuf},
    process::Command,
};

use zip::ZipArchive;

/// Run a shell command
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

/// Download FFTW release archive from FTP server
fn download_unix_archive(archive_path: &Path) -> Result<String, Box<dyn std::error::Error>> {
    let mut conn = ftp::FtpStream::connect("ftp.fftw.org:21")?;
    conn.login("anonymous", "anonymous")?;
    conn.cwd("pub/fftw")?;
    let buf = conn.simple_retr("fftw-3.3.9.zip")?.into_inner();
    let mut f = File::create(archive_path)?;
    f.write_all(&buf).unwrap();

    Ok("success".to_string())
}

/// Extract FFTW release archive
fn extract_unix_archive(archive_path: &PathBuf) -> Result<String, Box<dyn std::error::Error>> {
    let file = File::open(archive_path)?;
    let mut zip = ZipArchive::new(file)?;
    let out_path = PathBuf::from(var("CARGO_MANIFEST_DIR")?);
    zip.extract(&out_path)?;

    Ok("success".to_string())
}

/// Build FFTW library
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
    let src_dir = PathBuf::from(var("CARGO_MANIFEST_DIR").unwrap()).join("fftw-3.3.9");
    let out_src_dir = out_dir.join("src");

    let archive_path = out_dir.join("fftw-3.3.9.zip");

    if !archive_path.exists() {
        // Download and extract archive
        let _ = download_unix_archive(&archive_path);
        let _ = extract_unix_archive(&archive_path);

        // Copy source files, for out of source build
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
    }

    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    let target_feature = std::env::var("CARGO_CFG_TARGET_FEATURE").unwrap_or_default();

    let mut flags = Vec::new();
    if target_arch == "x86_64" {
        if target_feature.contains("avx") {
            flags.push("--enable-avx");
        }
        if target_feature.contains("avx2") {
            flags.push("--enable-avx2");
        }
        if target_feature.contains("fma") {
            flags.push("--enable-fma");
        }
    }

    // Build FFTW
    if !out_dir.join("lib/libfftw3.a").exists() {
        build_fftw(&src_dir, &out_dir, &flags);
    }

    let mut flags = Vec::new();
    if target_arch == "x86_64" {
        if target_feature.contains("sse2") {
            flags.push("--enable-sse2");
        }
    } else if target_arch == "aarch64" && target_feature.contains("neon") {
        flags.push("--enable-neon");
    }

    flags.push("--enable-single");

    // Build FFTWF
    if !out_dir.join("lib/libfftw3f.a").exists() {
        build_fftw(&src_dir, &out_dir, &flags);
    }

    // Link built libraries
    println!("cargo:rustc-link-search={}", out_dir.join("lib").display());
    println!("cargo:rustc-link-lib=static=fftw3");
    println!("cargo:rustc-link-lib=static=fftw3f");

    // Set dependency root to source path
    println!("cargo:root={}", out_src_dir.display());
}

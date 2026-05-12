fn main() {
    // Rebuild if anything under csrc/ changes (new .c, new .h, modified source).
    // The cc crate only auto-tracks files passed to .file(); this widens the net so
    // a new or modified C source always triggers recompilation of q4_dot.
    println!("cargo:rerun-if-changed=csrc");
    println!("cargo:rerun-if-changed=build.rs");

    // Q4 dot product kernel
    let mut build = cc::Build::new();
    build.file("csrc/q4_dot.c");
    build.opt_level(3);

    #[cfg(target_arch = "aarch64")]
    build.flag("-march=armv8.2-a+dotprod");

    #[cfg(target_arch = "x86_64")]
    build.flag("-mavx2");

    build.compile("q4_dot");

    // Ternary dot product kernel (BitNet b1.58)
    let mut build2 = cc::Build::new();
    build2.file("csrc/ternary_dot.c");
    build2.opt_level(3);

    #[cfg(target_arch = "aarch64")]
    build2.flag("-march=armv8.2-a");

    #[cfg(target_arch = "x86_64")]
    build2.flag("-mavx2");

    build2.compile("ternary_dot");
}

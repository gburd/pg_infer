#
# flake.nix - pg_infer development environment
#
# Quick Start:
#   nix develop          # Development shell with all build dependencies
#   nix build            # Build the workspace (excluding pgrx extension)
#   nix flake check      # Run cargo check + clippy
#
{
  description = "pg_infer - query transformer model weights as PostgreSQL relations";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        lib = pkgs.lib;
      in
      {
        devShells.default = pkgs.mkShell {
          nativeBuildInputs = with pkgs; [
            pkg-config
            rustc
            cargo
            clippy
            rustfmt
            rust-analyzer
            cargo-watch
          ];

          buildInputs = with pkgs; [
            # OpenSSL (required by openblas-src build script for network fetch)
            openssl
            openssl.dev

            # OpenBLAS (BLAS backend for infer-compute and infer-inference)
            openblas

            # Common C build tools (needed by various -sys crates)
            cmake
            clang
            llvmPackages.libclang
          ] ++ lib.optionals stdenv.hostPlatform.isLinux [
            # pgrx needs these on Linux
            readline
            zlib
          ] ++ lib.optionals stdenv.hostPlatform.isDarwin (with darwin.apple_sdk.frameworks; [
            Accelerate
            Security
            SystemConfiguration
          ]);

          # Environment variables for -sys crates
          OPENSSL_LIB_DIR = "${pkgs.openssl.out}/lib";
          OPENSSL_INCLUDE_DIR = "${pkgs.openssl.dev}/include";
          OPENBLAS_LIB_DIR = "${pkgs.openblas}/lib";
          LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";

          shellHook = ''
            echo "pg_infer dev shell"
            echo "  cargo check          - type-check workspace"
            echo "  cargo test            - run tests"
            echo "  cargo pgrx install    - install extension into PG"
            echo ""
          '';
        };
      }
    );
}

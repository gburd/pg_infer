#
# flake.nix - pg_infer development environment
#
# Quick start:
#   nix develop              # dev shell: toolchain, BLAS, cargo-pgrx, PG18+PG19
#   nix develop .#pg19       # same, with PostgreSQL 19 as the default target
#   nix build                # build the workspace (the pgrx extension needs a
#                            # live pg_config, so `nix build` covers the crates)
#   nix flake check          # fmt + clippy + the workspace test suite
#
# The extension itself is built with cargo-pgrx against a real server:
#   cargo pgrx init --pg18 $(which pg_config)   # or --pg19
#   cargo pgrx install --release
#
{
  description = "pg_infer - query transformer model weights as PostgreSQL relations";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

    # PostgreSQL 19.0 went GA on 2026-11-19 and nixpkgs took the bump the day
    # after (bacf73b39, "postgresql_19: 19rc2 -> 19.0"), which has not reached
    # the nixos-unstable channel yet -- it still serves 19rc2. pg_infer claims
    # PG19 GA support, so the dev shell has to be able to offer GA rather than
    # a release candidate. Pinned to the bump commit; fold this back into the
    # main nixpkgs input once the channel catches up, which is a one-line
    # deletion plus dropping the `pgNixpkgs` references below.
    nixpkgsPg19.url = "github:NixOS/nixpkgs/bacf73b39";

    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      nixpkgsPg19,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        pgPkgs = nixpkgsPg19.legacyPackages.${system};
        lib = pkgs.lib;

        # Keep every major pg_infer supports available in one shell, so
        # `cargo pgrx init --pg18 ... --pg19 ...` can register both and the
        # pg18/pg19 cargo features can each be exercised.
        postgresqlVersions = {
          pg18 = pkgs.postgresql_18;
          pg19 = pgPkgs.postgresql_19;
        };

        # Matches rust-toolchain.toml's `channel = "stable"`. Nix pins the
        # exact build, which is the point: CI resolves `stable` freshly and
        # has gone red on a new clippy lint before now, so a reproducible
        # shell is the place to notice that first.
        rustToolchain = with pkgs; [
          rustc
          cargo
          clippy
          rustfmt
          rust-analyzer
        ];

        nativeDeps = with pkgs; [
          pkg-config
          cmake
          clang
          llvmPackages.libclang
          # cargo-pgrx must match the `pgrx` dependency's minor or it refuses
          # to operate ("cargo-pgrx 0.20.0 requires pgrx ~0.20"). nixpkgs may
          # lag Cargo.toml; the shellHook checks and says so rather than
          # letting the mismatch surface as a confusing build error.
          cargo-pgrx
        ];

        buildDeps =
          with pkgs;
          [
            # openssl: reached through native-tls via reqwest. NB the release
            # binaries link it DYNAMICALLY, which is why a tarball built on
            # Ubuntu will not load on Fedora -- see the compatibility chapter.
            openssl
            openssl.dev

            # OpenBLAS backs infer-compute and infer-inference.
            openblas

            # oniguruma: onig_sys, reached through `tokenizers`. Its build.rs
            # looks the library up with pkg-config and vendors nothing, so
            # without this `nix build` fails with "Package oniguruma was not
            # found in the pkg-config search path".
            oniguruma
          ]
          ++ lib.optionals stdenv.hostPlatform.isLinux [
            # pgrx builds PostgreSQL from source when asked to `download`,
            # and its configure needs these.
            readline
            zlib
            libxml2
            lz4
            zstd
            icu
            bison
            flex
            perl
          ]
          ++ lib.optionals stdenv.hostPlatform.isDarwin (
            with darwin.apple_sdk.frameworks;
            [
              Accelerate
              Security
              SystemConfiguration
            ]
          );

        commonEnv = {
          OPENSSL_LIB_DIR = "${pkgs.openssl.out}/lib";
          OPENSSL_INCLUDE_DIR = "${pkgs.openssl.dev}/include";
          OPENBLAS_LIB_DIR = "${pkgs.openblas}/lib";
          LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";
        };

        mkShell =
          { pgDefault }:
          pkgs.mkShell (
            commonEnv
            // {
              nativeBuildInputs =
                rustToolchain
                ++ nativeDeps
                ++ [
                  pkgs.cargo-watch
                  # First on PATH, so bare `pg_config` is this shell's major.
                  # NB nixpkgs strips the pg_config BINARY from the server
                  # outputs (they carry only nix-support/pg_config.env) and
                  # exposes it as a separate `postgresql_NN.pg_config`
                  # package. That is what nixpkgs' own buildPgrxExtension
                  # passes to `cargo pgrx init`, so it is the supported path.
                  postgresqlVersions.${pgDefault}.pg_config
                ];

              buildInputs =
                buildDeps
                ++ lib.concatMap (p: [
                  p # initdb, psql, postgres
                  p.dev # headers and pgxs, for the -sys crates
                ]) (lib.attrValues postgresqlVersions);

              # Which server bare `pg_config` resolves to. Set as a
              # derivation attribute rather than in shellHook, because
              # `nix develop --command` does not run shellHook and would
              # otherwise see a different default than an interactive shell.
              shellHook = ''
                # Banner on STDERR, not stdout. A shellHook that prints to
                # stdout corrupts every `$(nix develop --command ...)`
                # capture, which is how scripts and CI ask this shell for a
                # path. Learned by having it swallow a pg_config lookup.
                {
                  echo "pg_infer dev shell"
                  echo "  default server:  $(pg_config --version)  (nix develop .#${
                    if pgDefault == "pg18" then "pg19" else "pg18"
                  } for the other)"
                  echo "  pg18 pg_config:  ${postgresqlVersions.pg18.pg_config}/bin/pg_config"
                  echo "  pg19 pg_config:  ${postgresqlVersions.pg19.pg_config}/bin/pg_config"
                  echo ""

                  # A cargo-pgrx whose MINOR differs from the `pgrx`
                  # dependency refuses to run, and its error names versions
                  # without naming the fix. Say it here instead. nixpkgs
                  # tracks pgrx closely but not instantly.
                  want=$(sed -n 's/^pgrx = "\([0-9.]*\)"/\1/p' Cargo.toml | head -1)
                  have=$(cargo pgrx --version 2>/dev/null | awk '{print $2}')
                  if [ -n "$want" ] && [ -n "$have" ] &&
                     [ "''${want%.*}" != "''${have%.*}" ]; then
                    echo "  WARNING: cargo-pgrx $have, but Cargo.toml wants pgrx $want."
                    echo "           Same minor is required. Install the match with:"
                    echo "             cargo install cargo-pgrx --version $want --locked"
                    echo ""
                  fi

                  echo "  cargo check                      type-check the workspace"
                  echo "  cargo test --workspace --exclude pg_infer"
                  echo "  cargo pgrx init --pg18 ${postgresqlVersions.pg18.pg_config}/bin/pg_config \\"
                  echo "                  --pg19 ${postgresqlVersions.pg19.pg_config}/bin/pg_config"
                  echo "  cargo pgrx install --release     install into a registered server"
                  echo ""
                } >&2
              '';
            }
          );
      in
      {
        devShells = {
          default = mkShell { pgDefault = "pg18"; };
          pg18 = mkShell { pgDefault = "pg18"; };
          pg19 = mkShell { pgDefault = "pg19"; };
        };

        # `nix build`. The pgrx extension is deliberately excluded: building it
        # requires a pg_config for a specific major and writes into that
        # server's directories, which does not fit a pure derivation. The
        # library crates are what a reproducible build can meaningfully cover.
        packages.default = pkgs.rustPlatform.buildRustPackage {
          pname = "pg_infer-crates";
          version =
            (builtins.fromTOML (builtins.readFile ./Cargo.toml)).workspace.package.version
              or (builtins.fromTOML (builtins.readFile ./Cargo.toml)).package.version;
          src = ./.;
          cargoLock.lockFile = ./Cargo.lock;

          nativeBuildInputs = nativeDeps;
          buildInputs = buildDeps;
          inherit (commonEnv)
            OPENSSL_LIB_DIR
            OPENSSL_INCLUDE_DIR
            OPENBLAS_LIB_DIR
            LIBCLANG_PATH
            ;

          cargoBuildFlags = [
            "--workspace"
            "--exclude"
            "pg_infer"
          ];
          # The pgrx tests need a live server, so they run under
          # `cargo pgrx test`, not here.
          doCheck = false;

          meta = {
            description = "pg_infer library crates (extension excluded; needs pg_config)";
            licence = lib.licenses.asl20;
          };
        };

        # `nix flake check`.
        checks = {
          fmt = pkgs.runCommand "pg_infer-fmt" { nativeBuildInputs = rustToolchain; } ''
            cd ${./.}
            cargo fmt --all -- --check
            touch $out
          '';
        };

        # nixfmt-rfc-style is now an alias for nixfmt; use the real name.
        formatter = pkgs.nixfmt;
      }
    );
}

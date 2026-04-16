# RUSTIS build targets — each feature combination gets its own target directory
# so binaries don't overwrite each other.
#
# Binaries land at:  build/<variant>/release/cosmo_sim[.exe]
# Examples:
#   make hydro
#   make hydro-sym
#   make hydro-sym TARGET=x86_64-pc-windows-gnu

TARGET     ?=
TARGET_FLAG = $(if $(TARGET),--target $(TARGET),)
EXE_SUFFIX  = $(if $(filter %-windows-%,$(TARGET)),.exe,)

# Comma trick: needed because commas inside $(call ...) are arg separators
, := ,

define build
	CARGO_TARGET_DIR=build/$(1) cargo build --profile dist --features $(2) $(TARGET_FLAG)
	@printf '\e[32m→ build/%s/dist/cosmo_sim%s\e[0m\n' '$(1)' '$(EXE_SUFFIX)'
endef

.PHONY: hydro hydro-sym hydro-fofr hydro-sym-fofr hydro-mpi hydro-sym-mpi test test-mpi clean

hydro:
	$(call build,hydro,hydro)

hydro-sym:
	$(call build,hydro-sym,hydro$(,)symmetron)

hydro-fofr:
	$(call build,hydro-fofr,hydro$(,)fofr)

hydro-sym-fofr:
	$(call build,hydro-sym-fofr,hydro$(,)symmetron$(,)fofr)

hydro-mpi:
	$(call build,hydro-mpi,hydro$(,)mpi_parallel)

hydro-sym-mpi:
	$(call build,hydro-sym-mpi,hydro$(,)symmetron$(,)mpi_parallel)

test:
	cargo test --features hydro$(,)symmetron$(,)fofr

# Multi-rank MPI integration test: builds the harness=false test binary and
# runs it under mpirun -np 2.  Requires OpenMPI or MPICH to be installed.
test-mpi:
	cargo test --features mpi_parallel --test mpi_ghost_exchange --no-run 2>&1
	@BIN=$$(ls -t target/debug/deps/mpi_ghost_exchange-* 2>/dev/null | head -1); \
	if [ -z "$$BIN" ]; then echo "ERROR: test binary not found" >&2; exit 1; fi; \
	echo "Running: mpirun -np 2 $$BIN"; \
	mpirun -np 2 $$BIN

clean:
	rm -rf build/
	rm -rf build/

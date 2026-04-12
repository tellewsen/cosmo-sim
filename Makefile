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
	CARGO_TARGET_DIR=build/$(1) cargo build --release --features $(2) $(TARGET_FLAG)
	@printf '\e[32m→ build/%s/release/cosmo_sim%s\e[0m\n' '$(1)' '$(EXE_SUFFIX)'
endef

.PHONY: hydro hydro-sym hydro-fofr hydro-sym-fofr hydro-mpi hydro-sym-mpi test clean

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

clean:
	rm -rf build/

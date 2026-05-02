# Ellewsen et al. (2018) reference profiles

Paper: Ellewsen, Falck & Mota 2018, arXiv:1803.04197
Figures 3–5: stacked halo profiles of ρ_gas/ρ̄, T [K], ρ_DM/ρ̄_DM vs r/R₂₀₀
at z=0, for high-mass halos (M > 10^13.5 M☉/h).

## File naming

    ellewsen2018_{variant}_{mass_bin}.csv

where variant ∈ {lcdm, sym_a, sym_b, sym_c, sym_d}
and   mass_bin ∈ {highmass, lowmass}

## CSV format (5 columns, space-separated, header line starts with #)

    # r_R200  rho_gas_rho_mean  T_K  rho_dm_rho_mean
    0.10      42.3              8.1e6   310.0
    ...

Columns must match the output of `extract_profiles.py` exactly:
  r_R200           — r / R₂₀₀  (dimensionless)
  rho_gas_rho_mean — ρ_gas / ρ̄  (gas density / mean gas density)
  T_K              — Temperature in Kelvin
  rho_dm_rho_mean  — ρ_DM / ρ̄_DM (DM density / mean DM density)

Rows with no data should be omitted (the plotter skips NaN automatically).

## Digitizing from the paper

Use a plot digitizer (e.g. WebPlotDigitizer, https://automeris.io/WebPlotDigitizer)
on Figures 3–5 of arXiv:1803.04197. Extract the ΛCDM curve and each Sym variant
separately. Save one CSV per (variant, mass_bin) combination.

The plotter (`plot_halo_profiles.py --reference-dir analysis/reference`) will
overlay the corresponding CSV as a dashed black curve if the file exists,
or silently skip it if not.

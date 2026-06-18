# HRCCS/CCF pRT Retrieval Scaffold

This folder is a clean pivot away from the direct pixel-space likelihood.  It
does not replace the existing xcorr pipeline.  Instead it reuses the trusted
pieces:

- pRT model generation from `retrieval.prt_emission_model`
- the validated `xcorr_processed` model representation
- `exopipe.crosscorrelation.modelCorrelation_weighted`
- `exopipe.crosscorrelation.finalCorr_stack`
- `exopipe.crosscorrelation.template_to_dmag`
- `exopipe.tools.shift2rest`
- `exopipe.tools.orders2keep`

## Validation First

Run this before trusting any retrieval:

```bash
python -m retrieval.hrccs_retrieval.run_validate_xcorr_model \
  /path/to/project \
  --retrieval-config retrieval/configs/mascara1b_fe_smoketest.yaml \
  --k 4 \
  --nights 20240528 \
  --cameras red \
  --output retrieval/results/hrccs_validate_fe \
  --sigma-cut 3.0 \
  --save-per-order
```

This generates one pRT Fe model, processes it into the same xcorr template
format that recovered the Fe detection, runs the existing weighted CCF stack,
and saves:

- `validate_xcorr_model_maps.npz`
- `validate_xcorr_model_summary.json`
- `validate_xcorr_model.png`
- optional per-order maps

The peak should land near the known Fe detection before sampler work.

## Small Fe Grid

Start with Kp/Vsys fixed or narrowly gridded, and only a few atmospheric
points:

```bash
python -m retrieval.hrccs_retrieval.run_fe_grid_retrieval \
  /path/to/project \
  --retrieval-config retrieval/configs/mascara1b_fe_smoketest.yaml \
  --k 4 \
  --nights 20240528 \
  --cameras red \
  --orders 0,1,2 \
  --fixed-kp 198 \
  --fixed-vsys -2 \
  --T-deep-grid 1800,2400,300 \
  --delta-T-grid 1500,3000,500 \
  --logFe-grid -5.5,-3.5,0.5 \
  --output retrieval/results/hrccs_fe_grid \
  --n-jobs 1
```

The default objective is `matched_filter_loglike`, which uses the weighted
data-model correlation, model power, and data power along the planet trail with
an analytic best-fit amplitude.  `ccf_peak_value` exists only as a debugging
objective and is not the paper-grade likelihood.

## Fe Sampler

The first sampler mode fixes Kp/Vsys by default and samples only:

- `T_deep`
- `delta_T_inv`
- `log10_Fe`

```bash
python -m retrieval.hrccs_retrieval.run_fe_sampler \
  /path/to/project \
  --retrieval-config retrieval/configs/mascara1b_fe_smoketest.yaml \
  --k 4 \
  --nights 20240528 \
  --cameras red \
  --orders 0,1,2 \
  --fix-kp 198 \
  --fix-vsys -2 \
  --nlive 100 \
  --n-jobs 1 \
  --output retrieval/results/hrccs_fe_sampler
```

For a quick plumbing test:

```bash
python -m retrieval.hrccs_retrieval.run_fe_sampler \
  /path/to/project \
  --retrieval-config retrieval/configs/mascara1b_fe_smoketest.yaml \
  --k 4 \
  --nights 20240528 \
  --cameras red \
  --orders 0 \
  --fix-kp 198 \
  --fix-vsys -2 \
  --test \
  --n-jobs 1 \
  --output retrieval/results/hrccs_fe_sampler_test
```

Sampling Kp/Vsys is available with `--sample-kp-vsys`, but the recommended
first run is fixed or tightly constrained velocities because the one-night
Kp/Vsys ridge can be degenerate.

## Sampler Parallelism

The sampler is serial by default:

```bash
python -m retrieval.hrccs_retrieval.run_fe_sampler \
  /path/to/project \
  --retrieval-config retrieval/configs/mascara1b_fe_smoketest.yaml \
  --k 4 \
  --orders 0 \
  --fix-kp 198 \
  --fix-vsys -2 \
  --test \
  --n-jobs 1 \
  --output retrieval/results/hrccs_fe_sampler_test_serial
```

Use multiple dynesty likelihood workers on a single node with `--n-jobs`:

```bash
python -m retrieval.hrccs_retrieval.run_fe_sampler \
  /path/to/project \
  --retrieval-config retrieval/configs/mascara1b_fe_smoketest.yaml \
  --k 4 \
  --orders 0 \
  --fix-kp 198 \
  --fix-vsys -2 \
  --test \
  --n-jobs 4 \
  --output retrieval/results/hrccs_fe_sampler_test_parallel4
```

On SLURM, request the same CPU count:

```bash
#SBATCH --cpus-per-task=4

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

python -m retrieval.hrccs_retrieval.run_fe_sampler ... --n-jobs 4
```

The thread environment variables keep each process from spawning its own BLAS
or OpenMP thread team.  This matters when `--n-jobs` is greater than 1.

For timing the likelihood machinery without running dynesty, use:

```bash
python -m retrieval.hrccs_retrieval.run_fe_sampler \
  /path/to/project \
  --retrieval-config retrieval/configs/mascara1b_fe_smoketest.yaml \
  --k 4 \
  --nights 20240528 \
  --cameras red \
  --orders 1,2,3 \
  --fix-kp 198 \
  --fix-vsys -2 \
  --benchmark-likelihood-calls 8 \
  --n-jobs 4 \
  --output retrieval/results/hrccs_likelihood_benchmark_n4
```

## Fe emcee Sampler

The emcee pathway is an alternate posterior sampler. It reuses the same HRCCS
data loading, pRT model generation, xcorr-processed template representation,
uniform YAML priors, fixed/sampled Kp-Vsys handling, and likelihood machinery
as the dynesty sampler. It writes an HDF5 backend every step, so interrupted
runs can be continued with `--resume`.

Small 5D Fe-only test with fixed velocity:

```bash
python -m retrieval.hrccs_retrieval.run_fe_emcee \
  /home/edeibert/projects/def-ldang05/edeibert/mascara1b \
  --retrieval-config retrieval/configs/mascara1b_fe_n1red_guoTP_sampler_expanded.yaml \
  --k 7 \
  --nights 20240528 \
  --cameras red \
  --orders 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 \
  --fix-kp 198 \
  --fix-vsys -2 \
  --n-walkers 48 \
  --n-steps 1000 \
  --burn-in 300 \
  --thin 5 \
  --n-jobs 8 \
  --seed 123 \
  --output retrieval/results/hrccs_emcee_fe_fixedvel_test
```

Small 5D Fe-only test with sampled narrow Kp/Vsys:

```bash
python -m retrieval.hrccs_retrieval.run_fe_emcee \
  /home/edeibert/projects/def-ldang05/edeibert/mascara1b \
  --retrieval-config retrieval/configs/mascara1b_fe_n1red_guoTP_sampler_freevel.yaml \
  --k 7 \
  --nights 20240528 \
  --cameras red \
  --orders 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 \
  --sample-kp-vsys \
  --n-walkers 64 \
  --n-steps 1500 \
  --burn-in 500 \
  --thin 5 \
  --n-jobs 8 \
  --seed 123 \
  --output retrieval/results/hrccs_emcee_fe_freevel_test
```

To continue an interrupted run:

```bash
python -m retrieval.hrccs_retrieval.run_fe_emcee \
  /path/to/project \
  --retrieval-config retrieval/configs/mascara1b_fe_smoketest.yaml \
  --k 4 \
  --orders 0 \
  --fix-kp 198 \
  --fix-vsys -2 \
  --n-walkers 48 \
  --n-steps 500 \
  --resume \
  --n-jobs 8 \
  --output retrieval/results/hrccs_emcee_fe_fixedvel_test
```

If an old backend exists and you want a fresh run, use `--overwrite`; otherwise
the script fails before replacing `fe_hrccs_emcee_backend.h5`.

On SLURM, match `--n-jobs` to `--cpus-per-task` and keep threaded math
libraries to one thread per process:

```bash
#SBATCH --cpus-per-task=8

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

python -m retrieval.hrccs_retrieval.run_fe_emcee ... --n-jobs 8
```

Expected emcee outputs:

- `fe_hrccs_emcee.log`
- `fe_hrccs_emcee_backend.h5`
- `fe_hrccs_emcee_chain.npz`
- `fe_hrccs_emcee_samples.npz`
- `fe_hrccs_emcee_summary.json`
- `fe_hrccs_emcee_corner.png`
- `fe_hrccs_emcee_trace.png`

## Free Two-Point T-P Profile

New staged configs can opt into a free-pressure inversion profile:

```yaml
tp_profile:
  type: free_two_point_inversion
  min_delta_logP: 0.25
  T_upper_bounds: [1000.0, 7000.0]
```

The pressure parameters are `logP_deep` and `logP_upper`, both in log10 bar.
Higher log pressure is deeper, so valid samples must satisfy
`logP_deep > logP_upper + min_delta_logP`. The code does not sort pressure
points internally; invalid samples are rejected. The temperature points are
`(logP_deep, T_deep)` and `(logP_upper, T_deep + delta_T_inv)`. Between the
points the profile is linear in log10 pressure; outside them it is constant at
the nearest endpoint.

For the cleaner direct parameterization, use:

```yaml
tp_profile:
  type: free_two_point_inversion_direct
  min_delta_T: 100.0
  min_delta_logP: 0.25
```

This samples `T_lower`, `T_upper`, `logP_lower`, and `logP_upper`, where
`lower` means deeper/higher pressure. Valid samples must satisfy
`logP_lower > logP_upper + min_delta_logP` and
`T_upper > T_lower + min_delta_T`; invalid samples are rejected by the prior.

The preferred free-pressure test mode is now the Guo-like delta-pressure
parameterization:

```yaml
tp_profile:
  type: free_two_point_inversion_delta
  min_delta_T: 0.0
  min_delta_logP: 0.25
  T_upper_bounds: [0.0, 7000.0]
```

This samples `T_lower`, `delta_T_inv`, `logP_upper`, and `delta_logP`, then
derives:

```text
T_upper = T_lower + delta_T_inv
logP_lower = logP_upper + delta_logP
```

The derived pressure points must satisfy `logP_upper < logP_lower`, both must
fall inside the configured pRT pressure grid, and `delta_logP` must be larger
than `min_delta_logP`. For the staged delta-pressure configs, the pressure grid
runs from `1e-8` to `1` bar, so `logP_lower` is capped at `0.0` unless a future
config explicitly extends the pRT grid deeper.

The HRCCS samplers remain backward-compatible. If `sampler.sampled_parameters`
is absent, they use the old Fe-only parameter list. If it is present, the YAML
list controls atmospheric and nuisance parameters, while the CLI still controls
whether Kp/Vsys are sampled with `--sample-kp-vsys` or inserted from
`--fix-kp`/`--fix-vsys`.

No-beta direct T-P staged configs:

```text
retrieval/configs/mascara1b_fe_twopointTP_direct_nobeta_n1red_freevel_narrow.yaml
retrieval/configs/mascara1b_fe_twopointTP_direct_nobeta_n1red_fixedvel.yaml
retrieval/configs/mascara1b_fe_twopointTP_direct_nobeta_continuum_n1red_freevel_narrow.yaml
```

Delta-pressure staged configs:

```text
retrieval/configs/mascara1b_fe_twopointTP_deltaP_nobeta_continuum_n1red_freevel_narrow.yaml
retrieval/configs/mascara1b_fe_twopointTP_deltaP_nobeta_continuum_n1red_fixedP.yaml
```

Fixed-pressure Narval smoke test:

```bash
python -m retrieval.hrccs_retrieval.run_fe_emcee \
  /home/edeibert/projects/def-ldang05/edeibert/mascara1b \
  --retrieval-config retrieval/configs/mascara1b_fe_twopointTP_deltaP_nobeta_continuum_n1red_fixedP.yaml \
  --k 4 \
  --nights 20240528 \
  --cameras red \
  --orders 2 \
  --sample-kp-vsys \
  --n-walkers 32 \
  --n-steps 5 \
  --burn-in 0 \
  --thin 1 \
  --n-jobs 1 \
  --seed 123 \
  --overwrite \
  --output retrieval/results/hrccs_emcee_deltaP_fixedP_smoke_k4
```

Free-pressure Narval smoke test:

```bash
python -m retrieval.hrccs_retrieval.run_fe_emcee \
  /home/edeibert/projects/def-ldang05/edeibert/mascara1b \
  --retrieval-config retrieval/configs/mascara1b_fe_twopointTP_deltaP_nobeta_continuum_n1red_freevel_narrow.yaml \
  --k 4 \
  --nights 20240528 \
  --cameras red \
  --orders 2 \
  --sample-kp-vsys \
  --n-walkers 32 \
  --n-steps 5 \
  --burn-in 0 \
  --thin 1 \
  --n-jobs 1 \
  --seed 123 \
  --overwrite \
  --output retrieval/results/hrccs_emcee_deltaP_freeP_smoke_k4
```

The run logs and summary JSON include sampled T-P values and derived
`T_upper`/`logP_lower` values. Prior-edge diagnostics still apply only to
sampled parameters.

## Species And Beta

The new species mapping style is:

```yaml
species:
  active_species: [Fe, FeII, Ti, TiII]
  Fe:
    prt_name: Fe
    abundance_parameter: log10_Fe
  FeII:
    prt_name: Fe+
    abundance_parameter: log10_FeII
```

The older `species.line_species` plus `species.prt_names` style still works.

`log_beta` is log10 of a multiplicative noise scale beta. Beta is disabled
unless `log_beta`/`ln_beta` appears in `sampler.sampled_parameters` or in a
top-level `fixed_parameters` mapping:

```yaml
fixed_parameters:
  log_beta: 0.0
```

When disabled, the likelihood is exactly the historical no-beta likelihood.
When fixed, beta is not included in parameter names, corner plots, or trace
plots. When sampled, it is included like any other sampled parameter. The run
log reports `beta mode: disabled`, `beta mode: fixed log_beta=0`, or
`beta mode: sampled log_beta`.

The current implementation supports beta only for `matched_filter_loglike`, as
sigma -> beta*sigma with the beta-dependent Gaussian normalization term. Treat
beta/noise-scale parameters cautiously for matched-filter HRCCS likelihoods
unless the objective normalization is scientifically well-defined. Beta is not
implemented for the `ccf_peak_value` debug objective.

## Continuum Contributors

Continuum/background opacity is YAML-controlled and defaults to empty, matching
old configs:

```yaml
continuum_contributors:
  - yaml_name: H2-H2
    prt_name: H2--H2-NatAbund__BoRi.R831_0.6-250mu
    file: /path/to/input_data/opacities/continuum/collision_induced_absorptions/H2--H2/H2--H2-NatAbund/H2--H2-NatAbund__BoRi.R831_0.6-250mu.ciatable.petitRADTRANS.h5
  - yaml_name: H2-He
    prt_name: H2--He-NatAbund__BoRi.DeltaWavenumber2_0.5-500mu
    file: /path/to/input_data/opacities/continuum/collision_induced_absorptions/H2--He/H2--He-NatAbund/H2--He-NatAbund__BoRi.DeltaWavenumber2_0.5-500mu.ciatable.petitRADTRANS.h5
```

The wrapper currently passes the pRT contributor names `H2-H2`, `H2-He`, and
`H-`; `H2--H2` and `H2--He` are accepted as explicit aliases and logged before
conversion. Unsupported names raise a clear error instead of being ignored.
For CIA contributors, prefer exact `prt_name` plus `file` entries. Generic
strings are allowed only when they resolve to exactly one local file; ambiguous
matches fail before pRT can prompt interactively. The code preflights the
configured pRT `input_data` tree and fails if local files are not visible,
because otherwise pRT may try to auto-download or select defaults through
Keeper/Selenium on Narval. Set
`prt.require_continuum_opacity_files: false` only for an intentional interactive
opacity-install step. H- also requires fixed mass fractions for `H-`, `H`, and
`e-`; keep it disabled unless the local pRT input data and interface have been
verified.

The staged continuum config is:

```text
retrieval/configs/mascara1b_fe_twopointTP_direct_nobeta_continuum_n1red_freevel_narrow.yaml
```

Check the continuum selection before submitting a retrieval:

```bash
python -m retrieval.hrccs_retrieval.check_continuum_opacities \
  /home/edeibert/projects/def-ldang05/edeibert/mascara1b \
  --retrieval-config retrieval/configs/mascara1b_fe_twopointTP_direct_nobeta_continuum_n1red_freevel_narrow.yaml
```

The check prints each YAML name, exact pRT name passed to Radtrans, matched
local file, and whether the selection is unique/non-interactive.

Tiny Narval smoke test for the no-beta direct T-P continuum config:

```bash
python -m retrieval.hrccs_retrieval.run_fe_emcee \
  /home/edeibert/projects/def-ldang05/edeibert/mascara1b \
  --retrieval-config retrieval/configs/mascara1b_fe_twopointTP_direct_nobeta_continuum_n1red_freevel_narrow.yaml \
  --k 7 \
  --nights 20240528 \
  --cameras red \
  --orders 2 \
  --fix-kp 198 \
  --fix-vsys 0 \
  --n-walkers 24 \
  --n-steps 5 \
  --burn-in 0 \
  --thin 1 \
  --n-jobs 1 \
  --seed 123 \
  --overwrite \
  --output retrieval/results/hrccs_emcee_twopointTP_continuum_smoke
```

In `fe_hrccs_emcee.log`, confirm that pRT setup reports:

```text
YAML-requested pRT continuum contributors: ['H2-H2', 'H2-He']
Requested pRT continuum contributors: ['H2--H2-NatAbund__BoRi.R831_0.6-250mu', 'H2--He-NatAbund__BoRi.DeltaWavenumber2_0.5-500mu']
Requested pRT continuum contributors for Radtrans: ['H2--H2-NatAbund__BoRi.R831_0.6-250mu', 'H2--He-NatAbund__BoRi.DeltaWavenumber2_0.5-500mu']
```

## Model Wavelength Padding

Keep pRT model-generation wavelength boundaries padded beyond the selected
data/order wavelengths. These boundaries control the rest-frame pRT template
coverage only; they do not select data, orders, or science wavelength range.
Data selection remains controlled by the run's nights, cameras, orders, SYSREM
iteration, and the existing xcorr order-selection logic.

Tight model boundaries can create velocity-dependent finite-overlap artifacts
near template edges, especially for `matched_filter_loglike`, because different
Kp/Vsys points may see different numbers of finite model pixels. For
MASCARA-1b N1 red directTP tests, changing pRT generation bounds from
`[0.383, 1.0]` to `[0.3, 1.08]` fixed the old-good-equivalent directTP
matched-filter grid, recovering the expected region near Kp = 196.5 km/s and
Vsys = -1.0 km/s.

The directTP/deltaP staged MASCARA-1b configs therefore use:

```yaml
model:
  wavelength_boundaries_micron: [0.3, 1.08]
```

HRCCS scripts now log the selected data wavelength range, configured pRT model
boundaries, blue/red padding, and an approximate Doppler-padding estimate from
Kp/Vsys priors plus BERV. Treat warnings about close model boundaries as a
reason to widen only the model-generation range, not the data selection.

## Kp/Vsys Diagnostic Grid

When an emcee run lands on velocity-prior edges, map the likelihood surface at
fixed atmospheric parameters:

```bash
python -m retrieval.hrccs_retrieval.diagnose_kp_vsys_likelihood_grid \
  /home/edeibert/projects/def-ldang05/edeibert/mascara1b \
  --retrieval-config retrieval/configs/mascara1b_fe_twopointTP_direct_nobeta_n1red_freevel_narrow.yaml \
  --k 7 \
  --nights 20240528 \
  --cameras red \
  --orders 2 \
  --parameters-json retrieval/results/hrccs_emcee_direct_nobeta_freevel/fe_hrccs_emcee_summary.json \
  --kp-min 180 \
  --kp-max 215 \
  --kp-step 1 \
  --vsys-min -15 \
  --vsys-max 10 \
  --vsys-step 1 \
  --output retrieval/results/hrccs_kp_vsys_diagnostic
```

The diagnostic writes `kp_vsys_likelihood_grid.npz`,
`kp_vsys_likelihood_grid.png`, and `kp_vsys_likelihood_grid_summary.json`.

## Matched-Filter Component Diagnostics

If `matched_filter_loglike` lands at a velocity-prior corner while
`ccf_peak_value` still peaks at the Fe detection, decompose the matched-filter
terms before changing retrieval physics. The component diagnostic compares
valid overlap, weighted means, dot products, model norms, analytic amplitude,
and likelihood contributions at selected velocity points.

For the directTP old-good-equivalent Fe test, first write:

```bash
mkdir -p retrieval/results/hrccs_directTP_oldgood_equiv_mf_debug

cat > retrieval/results/hrccs_directTP_oldgood_equiv_mf_debug/directTP_oldgood_equiv_params.json <<'JSON'
{
  "T_lower": 2460.23110505095,
  "T_upper": 5985.870425768106,
  "logP_lower": -1.5,
  "logP_upper": -3.0,
  "log10_Fe": -2.634774406856159
}
JSON
```

Component diagnostic:

```bash
python -m retrieval.hrccs_retrieval.diagnose_matched_filter_components \
  /home/edeibert/projects/def-ldang05/edeibert/mascara1b \
  --retrieval-config retrieval/configs/mascara1b_fe_twopointTP_direct_nobeta_n1red_freevel_narrow.yaml \
  --parameters-json retrieval/results/hrccs_directTP_oldgood_equiv_mf_debug/directTP_oldgood_equiv_params.json \
  --k 4 \
  --nights 20240528 \
  --cameras red \
  --orders 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 \
  --kp-vsys 196.5 -1.0 ccf_peak \
  --kp-vsys 185.0 -12.0 corner \
  --kp-vsys 198.0 0.0 fiducial \
  --output retrieval/results/hrccs_directTP_oldgood_equiv_mf_debug/components
```

Zero-mean model matched-filter grid:

```bash
python -m retrieval.hrccs_retrieval.diagnose_kp_vsys_likelihood_grid \
  /home/edeibert/projects/def-ldang05/edeibert/mascara1b \
  --retrieval-config retrieval/configs/mascara1b_fe_twopointTP_direct_nobeta_n1red_freevel_narrow.yaml \
  --parameters-json retrieval/results/hrccs_directTP_oldgood_equiv_mf_debug/directTP_oldgood_equiv_params.json \
  --k 4 \
  --nights 20240528 \
  --cameras red \
  --orders 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 \
  --kp-min 185 \
  --kp-max 210 \
  --kp-step 0.5 \
  --vsys-min -12 \
  --vsys-max 8 \
  --vsys-step 0.5 \
  --objective matched_filter_loglike_zero_mean_model \
  --output retrieval/results/hrccs_directTP_oldgood_equiv_mf_debug/grid_zero_mean_model
```

Zero-mean data/model matched-filter grid:

```bash
python -m retrieval.hrccs_retrieval.diagnose_kp_vsys_likelihood_grid \
  /home/edeibert/projects/def-ldang05/edeibert/mascara1b \
  --retrieval-config retrieval/configs/mascara1b_fe_twopointTP_direct_nobeta_n1red_freevel_narrow.yaml \
  --parameters-json retrieval/results/hrccs_directTP_oldgood_equiv_mf_debug/directTP_oldgood_equiv_params.json \
  --k 4 \
  --nights 20240528 \
  --cameras red \
  --orders 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 \
  --kp-min 185 \
  --kp-max 210 \
  --kp-step 0.5 \
  --vsys-min -12 \
  --vsys-max 8 \
  --vsys-step 0.5 \
  --objective matched_filter_loglike_zero_mean_data_model \
  --output retrieval/results/hrccs_directTP_oldgood_equiv_mf_debug/grid_zero_mean_data_model
```

Optional normalized weighted-correlation grid:

```bash
python -m retrieval.hrccs_retrieval.diagnose_kp_vsys_likelihood_grid \
  /home/edeibert/projects/def-ldang05/edeibert/mascara1b \
  --retrieval-config retrieval/configs/mascara1b_fe_twopointTP_direct_nobeta_n1red_freevel_narrow.yaml \
  --parameters-json retrieval/results/hrccs_directTP_oldgood_equiv_mf_debug/directTP_oldgood_equiv_params.json \
  --k 4 \
  --nights 20240528 \
  --cameras red \
  --orders 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 \
  --kp-min 185 \
  --kp-max 210 \
  --kp-step 0.5 \
  --vsys-min -12 \
  --vsys-max 8 \
  --vsys-step 0.5 \
  --objective matched_filter_loglike_normalized_ccf \
  --output retrieval/results/hrccs_directTP_oldgood_equiv_mf_debug/grid_normalized_ccf
```

The original `matched_filter_loglike` remains the default. The zero-mean
variants are opt-in and currently do not support beta/noise-scale parameters.
The normalized CCF objective is a diagnostic weighted-correlation score, not a
Gaussian log likelihood.

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

python -m retrieval.hrccs_retrieval.run_fe_sampler ... --n-jobs 4
```

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

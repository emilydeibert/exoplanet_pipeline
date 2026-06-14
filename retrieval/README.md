## Minimal petitRADTRANS Fe Emission Retrieval

The new `retrieval/` folder is isolated from the existing reduction, SYSREM,
and cross-correlation pipeline.  It is a first-pass high-resolution dayside
emission retrieval workflow for MASCARA-1b:

1. Fe-only pRT emission smoke test on one order or wavelength chunk.
2. Fe-only Kp-Vsys likelihood map.
3. Gated Fe-only dynesty sampler, to run only after the grid is validated.

The implementation follows the pRT3 high-resolution workflow concepts, while
using direct `Radtrans.calculate_flux` plus wrapper-level Doppler
shifting/rebinning for this first emission-residual setup.  pRT references:
[high-resolution spectra](https://petitradtrans.readthedocs.io/en/latest/content/notebooks/high_resolution_spectra.html),
[high-resolution SpectralModel retrievals](https://petitradtrans.readthedocs.io/en/latest/content/notebooks/retrieval_spectral_model.html),
and [pRT input_data setup](https://petitradtrans.readthedocs.io/en/latest/content/notebooks/getting_started.html#Configuring-the-input-data-folder).

### Environment

Local conda example:

```bash
conda env create -f environment.yml
conda activate exopipe-prt
pip install -e .
```

If pRT build isolation causes issues, follow the pRT install docs and install
with:

```bash
pip install numpy meson-python ninja
pip install "petitRADTRANS[retrieval]" --no-build-isolation
```

You can also install from `requirements.txt` in an existing environment such as
`astro`.

### pRT Opacity Data

pRT expects its data directory to be named `input_data`.  Configure it either in
`retrieval/configs/mascara1b_fe_smoketest.yaml`:

```yaml
prt:
  input_data_path: /path/to/petitRADTRANS/input_data
```

or with an environment variable:

```bash
export PETITRADTRANS_INPUT_DATA=/path/to/petitRADTRANS/input_data
```

Run the preflight check before spending time on pRT jobs:

```bash
python -m retrieval.check_retrieval_environment \
  --config retrieval/configs/mascara1b_fe_smoketest.yaml
```

On a cluster where opacities must already be local:

```bash
python -m retrieval.check_retrieval_environment \
  --config retrieval/configs/mascara1b_fe_smoketest.yaml \
  --require-local-opacities
```

### Data Config

Edit `retrieval/configs/mascara1b_fe_smoketest.yaml` so `data.paths` points to
your prepared arrays.  Inputs can be `.npy` files or `.npz` files with keys:

```yaml
data:
  wavelength_unit: nm
  paths:
    wavelengths: /path/to/night_red_analysis_ready.npz
    flux: /path/to/night_red_sysrem.npz
    uncertainties: /path/to/night_red_sysrem.npz
    phases: /path/to/night_red_analysis_ready.npz
    barycentric_velocities: /path/to/night_red_analysis_ready.npz
  keys:
    wavelengths: wave
    flux: sysrem
    uncertainties: magerr
    phases: phase
    barycentric_velocities: berv
```

If using one SYSREM iteration from a saved cube, save that iteration as
`(n_orders, n_exposures, n_pixels)` first.  The retrieval loader intentionally
fails loudly on shape mismatches, unit confusion, missing arrays, or all-masked
orders.

### Local Commands

Smoke test on one order:

```bash
python -m retrieval.run_prt_smoketest \
  --config retrieval/configs/mascara1b_fe_smoketest.yaml \
  --order 0
```

Export the current pRT model into the legacy cross-correlation model format:

```bash
python -m retrieval.export_prt_model_for_xcorr \
  --config retrieval/configs/mascara1b_fe_smoketest.yaml \
  --output /path/to/models/Fe_pRT_model.npy \
  --wavelength-min-micron 0.383 \
  --wavelength-max-micron 1.0
```

The exported `.npy` has two columns: wavelength in Angstrom and raw pRT
emission flux.  This matches the existing xcorr loader, which divides column 0
by 10 to get nm before convolution and `template_to_dmag`.

Fe-only Kp-Vsys grid:

```bash
python -m retrieval.run_fe_kp_vsys_grid \
  --config retrieval/configs/mascara1b_fe_smoketest.yaml
```

Tiny 3x3 fake-injection timing grid, useful before a full run:

```bash
python -m retrieval.run_fe_kp_vsys_grid \
  --config retrieval/configs/mascara1b_fe_smoketest.yaml \
  --inject-fake \
  --tiny-grid \
  --n-jobs 1
```

Preparation-filter benchmark on one shifted model cube:

```bash
python -m retrieval.run_fe_kp_vsys_grid \
  --config retrieval/configs/mascara1b_fe_smoketest.yaml \
  --tiny-grid \
  --benchmark-preparation \
  --n-jobs 1
```

Full Fe Kp-Vsys grid in serial debugging mode:

```bash
python -m retrieval.run_fe_kp_vsys_grid \
  --config retrieval/configs/mascara1b_fe_smoketest.yaml \
  --n-jobs 1
```

Parallel Fe grid on a workstation or Compute Canada node:

```bash
python -m retrieval.run_fe_kp_vsys_grid \
  --config retrieval/configs/mascara1b_fe_smoketest.yaml \
  --n-jobs 4
```

Per-order likelihood diagnostics at selected velocity points:

```bash
python -m retrieval.diagnose_fe_order_contributions \
  --config retrieval/configs/mascara1b_fe_smoketest.yaml \
  --points "194,0;210,1;220,1" \
  --output retrieval/results/mascara1b_fe_smoketest/order_diagnostics
```

The first point is used as the delta-log-likelihood reference when multiple
points are supplied.

Injected fake-signal recovery:

```bash
python -m retrieval.run_fe_kp_vsys_grid \
  --config retrieval/configs/mascara1b_fe_smoketest.yaml \
  --inject-fake \
  --injection-kp 198 \
  --injection-vsys -2 \
  --injection-scale 1
```

Fe-only sampler, after inspecting the grid:

```bash
python -m retrieval.run_fe_sampler \
  --config retrieval/configs/mascara1b_fe_smoketest.yaml \
  --confirm-grid-validated
```

Outputs go to `retrieval/results/mascara1b_fe_smoketest/` by default.

### Kp-Vsys Grid Performance

The Fe grid keeps the same velocity convention, BERV handling, wavelength
units, preparation method, and likelihood definition as the serial prototype.
The optimized grid avoids repeated work by caching the instrumentally convolved
rest-frame model once and reusing it for every Kp/Vsys point.  It also
vectorizes exposure shifting/rebinning within each order.

Use `--n-jobs 1` for serial debugging.  Use `--n-jobs N` to parallelize over
Kp/Vsys grid points with Python multiprocessing.  The config equivalents live
under `grid.n_jobs`, `grid.chunksize`, and `grid.multiprocessing_start_method`.

Each grid point logs timings for:

```text
shifted_model_cube
prepare_model_like_data
compute_log_likelihood
```

Timing arrays are saved in the output `.npz` with names such as
`timing_shifted_model_cube`, `timing_prepare_model_like_data`, and
`timing_compute_log_likelihood`.

### Model Preparation Methods

The default model-preparation method is now:

```yaml
preparation:
  method: median_highpass_delta_mag_exact
  highpass_width_pixels: 601
  median_filter_backend: auto
```

This preserves the previous median-filter science behavior.  With
`median_filter_backend: auto`, the code uses `bottleneck.move_median` when
available and falls back to the original `scipy.ndimage.median_filter`
reference.  The literal old reference path remains available as:

```yaml
preparation:
  method: median_highpass_delta_mag_scipy_reference
```

Approximate fast continuum options are available for testing:

```yaml
preparation:
  method: gaussian_highpass_delta_mag_fast
```

or:

```yaml
preparation:
  method: uniform_highpass_delta_mag_fast
```

Use `--benchmark-preparation` to compare each method against the SciPy median
reference.  The benchmark writes `preparation_benchmark.json` and logs elapsed
time, maximum absolute difference, and RMS difference relative to the reference.

Keep the fake-injection grid as the regression check after changing preparation
methods.  For the current Fe injection test, the recovered peak should remain
near `Kp=198`, `Vsys=-2`, with amplitude about `0.001`.

### Compute Canada / Narval

Edit the placeholders in:

```text
retrieval/slurm/prt_smoketest.slurm
retrieval/slurm/fe_kp_vsys_grid.slurm
retrieval/slurm/fe_only_retrieval.slurm
```

Set the account, wall time, memory, CPUs, Python module, environment activation,
and `PETITRADTRANS_INPUT_DATA`.  Submit from the repository root:

```bash
sbatch retrieval/slurm/prt_smoketest.slurm
sbatch retrieval/slurm/fe_kp_vsys_grid.slurm
sbatch retrieval/slurm/fe_only_retrieval.slurm
```

The Fe grid SLURM template maps `SLURM_CPUS_PER_TASK` to `--n-jobs`.  The
Fe-only retrieval job is intentionally gated in Python and requires the
`--confirm-grid-validated` flag already present in the SLURM template.

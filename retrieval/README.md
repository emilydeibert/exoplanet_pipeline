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

Fe-only Kp-Vsys grid:

```bash
python -m retrieval.run_fe_kp_vsys_grid \
  --config retrieval/configs/mascara1b_fe_smoketest.yaml
```

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

The Fe-only retrieval job is intentionally gated in Python and requires the
`--confirm-grid-validated` flag already present in the SLURM template.

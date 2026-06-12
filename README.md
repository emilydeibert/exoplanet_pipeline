# exopipe

`exopipe` is a Python package for high-resolution exoplanet spectroscopy analysis. It provides command-line tools and reusable Python modules for building reproducible workflows around data extraction, preprocessing, detrending, model injection/recovery, and cross-correlation analysis.

This repository is research software developed for the analysis of high-resolution spectroscopic observations of exoplanet atmospheres, with an emphasis on transparent, reproducible, and modular analysis pipelines.

## Features

* Python-based analysis workflows for high-resolution spectroscopic datasets
* Command-line interface for running common pipeline steps
* Tools for data extraction, preprocessing, and organization
* Cross-correlation analysis utilities for exoplanet atmospheric searches
* Support for reproducible analysis using project-folder based workflows
* Modular package structure for easier testing, debugging, and extension

## Installation

Clone the repository:

```bash
git clone https://github.com/emilydeibert/exoplanet_pipeline.git
cd exoplanet_pipeline
```

Install the package in editable mode:

```bash
pip install -e .
```

Editable installation is recommended for development, since changes to the source code will be reflected immediately without reinstalling the package.

## Usage

Pipeline commands can be run from the command line using Python’s module syntax. For example:

```bash
python -m exopipe.cli.extractData path_to_project_folder
```

Additional command-line tools are available under:

```bash
exopipe/cli/
```

A typical workflow consists of organizing the input data into a project directory, running the relevant preprocessing and extraction commands, applying detrending/model-processing steps, and producing cross-correlation outputs for scientific interpretation.

## Repository Structure

```text
exoplanet_pipeline/
├── exopipe/
│   ├── cli/                 # Command-line entry points
│   ├── reduction.py          # Data reduction and preprocessing tools
│   ├── crosscorrelation.py   # Cross-correlation analysis utilities
│   ├── sysrem.py             # SYSREM detrending tools
│   └── tools.py              # Shared helper functions
├── README.md
└── setup.py / pyproject.toml
```

## Example Command

```bash
python -m exopipe.cli.extractData path_to_project_folder
```

Replace `path_to_project_folder` with the path to the project directory containing the relevant input data and configuration files.

## Development Notes

This package is under active development and is primarily intended for research use. Interfaces, command-line options, and expected directory structures may change as the pipeline evolves.

Recommended development workflow:

```bash
git clone https://github.com/emilydeibert/exoplanet_pipeline.git
cd exoplanet_pipeline
pip install -e .
```

Then run commands from the root of the repository or from a working directory where the installed `exopipe` package is available.

## Research Context

High-resolution exoplanet spectroscopy often requires extracting weak planetary signals from noisy, heterogeneous time-series spectra. This package supports workflows involving data cleaning, signal extraction, detrending, model comparison, and cross-correlation analysis.

The codebase reflects practical research needs, including reproducibility, modularity, and the ability to iterate on analysis choices while preserving a clear workflow.

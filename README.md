# exopipe

`exopipe` is a Python package for high-resolution exoplanet spectroscopy analysis. It provides command-line tools and reusable Python modules for building reproducible workflows around data reduction, data extraction, SYSREM detrending, model cross-correlation, and result visualization.

This repository is research software developed for the analysis of high-resolution spectroscopic observations of exoplanet atmospheres, with an emphasis on transparent, reproducible, and modular analysis workflows.

## Features

* Python-based workflows for high-resolution spectroscopic datasets
* Command-line interface for common pipeline steps
* Tools for data reduction, extraction, and organization
* SYSREM detrending for time-series spectroscopic data
* Cross-correlation analysis with model spectra
* Plotting and result-generation utilities
* Modular package structure for easier debugging, testing, and extension

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

## Command-Line Tools

The command-line scripts are located in:

```text
src/exopipe/cli/
```

Current CLI modules include:

```text
src/exopipe/cli/reduceData.py
src/exopipe/cli/extractData.py
src/exopipe/cli/sysremData.py
src/exopipe/cli/correlateModel.py
src/exopipe/cli/getResults.py
```

These scripts can be run using Python’s module syntax. For example:

```bash
python -m exopipe.cli.extractData path_to_project_folder
```

Replace `path_to_project_folder` with the path to the relevant project directory.

## Example Workflow

A typical workflow may include:

```bash
python -m exopipe.cli.reduceData path_to_project_folder
python -m exopipe.cli.extractData path_to_project_folder
python -m exopipe.cli.sysremData path_to_project_folder
python -m exopipe.cli.correlateModel path_to_project_folder
python -m exopipe.cli.getResults path_to_project_folder
```

Depending on the dataset and analysis goals, not every step may be required for every project.

## Repository Structure

```text
exoplanet_pipeline/
├── src/
│   └── exopipe/
│       ├── cli/
│       │   ├── reduceData.py
│       │   ├── extractData.py
│       │   ├── sysremData.py
│       │   ├── correlateModel.py
│       │   └── getResults.py
│       ├── reduction.py
│       ├── crosscorrelation.py
│       ├── sysrem.py
│       └── tools.py
├── README.md
└── pyproject.toml
```

## Development Notes

This package is under active development and is primarily intended for research use. Interfaces, command-line options, and expected directory structures may change as the pipeline evolves.

Recommended development workflow:

```bash
git clone https://github.com/emilydeibert/exoplanet_pipeline.git
cd exoplanet_pipeline
pip install -e .
```

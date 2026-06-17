"""Check HRCCS pRT continuum opacity selections without creating Radtrans."""

from __future__ import annotations

import argparse
from pathlib import Path

from retrieval.prt_emission_model import (
    configured_prt_input_data_path,
    continuum_contributor_specs,
    load_yaml_config,
    resolved_continuum_opacity_specs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project_path", help="Project path kept for CLI symmetry; not read by this check.")
    parser.add_argument("--retrieval-config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.retrieval_config)
    input_data_path = configured_prt_input_data_path(config)
    if input_data_path is None:
        raise RuntimeError(
            "No pRT input_data path is configured. Set prt.input_data_path or "
            "PETITRADTRANS_INPUT_DATA before checking continuum opacities."
        )

    specs = continuum_contributor_specs(config)
    if not specs:
        print("No continuum contributors requested.")
        return

    print(f"Project path: {Path(args.project_path)}")
    print(f"Retrieval config: {args.retrieval_config}")
    print(f"pRT input_data: {input_data_path}")
    print()

    resolved = resolved_continuum_opacity_specs(config, input_data_path)
    for spec in resolved:
        files = list(spec.get("matched_files", []))
        print(f"YAML requested name: {spec['yaml_name']}")
        print(f"pRT name for Radtrans: {spec['prt_name']}")
        if files:
            print(f"local file matched: {files[0]}")
        else:
            print("local file matched: not required for this contributor")
        print(f"unique/non-interactive: {bool(spec.get('unique', False))}")
        print()


if __name__ == "__main__":
    main()

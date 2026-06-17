"""Check that the high-resolution retrieval environment is usable."""

from __future__ import annotations

import argparse
import glob
import importlib
import importlib.metadata
import importlib.util
import os
from pathlib import Path
from typing import Any, Mapping, Optional

from retrieval.prt_emission_model import configured_prt_input_data_path


PACKAGE_CHECKS = [
    ("numpy", "numpy", True),
    ("scipy", "scipy", True),
    ("bottleneck", "Bottleneck", False),
    ("matplotlib", "matplotlib", True),
    ("astropy", "astropy", True),
    ("yaml", "PyYAML", True),
    ("h5py", "h5py", True),
    ("petitRADTRANS", "petitRADTRANS", True),
    ("dynesty", "dynesty", True),
    ("emcee", "emcee", True),
    ("corner", "corner", True),
    ("ultranest", "ultranest", False),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="retrieval/configs/mascara1b_fe_smoketest.yaml",
        help="YAML retrieval config to inspect for pRT input_data and species.",
    )
    parser.add_argument(
        "--require-local-opacities",
        action="store_true",
        help="Exit nonzero if requested local line opacity files are not visible.",
    )
    return parser.parse_args()


def import_status(import_name: str, dist_name: str) -> tuple[bool, str]:
    if importlib.util.find_spec(import_name) is None:
        return False, "not importable"
    try:
        importlib.import_module(import_name)
    except Exception as exc:
        return False, f"import failed: {exc}"
    try:
        version = importlib.metadata.version(dist_name)
    except importlib.metadata.PackageNotFoundError:
        version = "version unknown"
    return True, version


def load_config_if_possible(path: Path) -> tuple[dict[str, Any], Optional[str]]:
    if not path.exists():
        return {}, f"config file does not exist: {path}"
    try:
        import yaml
    except ImportError:
        return {}, "PyYAML is missing, so the YAML config could not be inspected"
    try:
        with path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    except Exception as exc:
        return {}, f"failed to read config: {exc}"
    if not isinstance(config, dict):
        return {}, "config top level is not a mapping"
    return config, None


def pRT_current_input_data_path() -> Optional[str]:
    try:
        from petitRADTRANS.config import petitradtrans_config_parser
    except Exception:
        return None
    try:
        return str(petitradtrans_config_parser.get_input_data_path())
    except Exception:
        return None


def set_pRT_input_data_path(path: Path) -> Optional[str]:
    try:
        from petitRADTRANS.config import petitradtrans_config_parser
    except Exception as exc:
        return f"could not import pRT config parser: {exc}"
    try:
        petitradtrans_config_parser.set_input_data_path(str(path))
    except Exception as exc:
        return f"could not set pRT input_data path: {exc}"
    return None


def requested_species(config: Mapping[str, Any]) -> list[str]:
    species_cfg = config.get("species", {})
    requested = species_cfg.get("active_species", species_cfg.get("line_species", ["Fe"]))
    mapping = species_cfg.get("prt_names", {})
    out = []
    for species in requested:
        entry = species_cfg.get(str(species), {})
        if isinstance(entry, Mapping) and "prt_name" in entry:
            out.append(str(entry["prt_name"]))
        else:
            out.append(str(mapping.get(species, species)))
    return out


def species_opacity_patterns(input_data_path: Path, species_name: str) -> list[str]:
    species_clean = species_name.replace("+", "").replace("_p", "").replace("__", "_")
    short = species_clean.split("__", 1)[0]
    if short.lower() in {"fe", "56fe"}:
        directory = "Fe"
        glob_piece = "*Fe*"
    elif short.lower() in {"ti", "48ti"}:
        directory = "Ti"
        glob_piece = "*Ti*"
    elif short.lower() in {"cr", "52cr"}:
        directory = "Cr"
        glob_piece = "*Cr*"
    else:
        directory = short
        glob_piece = f"*{short}*"

    base = input_data_path / "opacities" / "lines" / "line_by_line"
    return [
        str(base / directory / "**" / f"{glob_piece}.xsec.petitRADTRANS.h5"),
        str(base / directory / "**" / f"{glob_piece}.petitRADTRANS.h5"),
    ]


def find_opacity_files(input_data_path: Path, species_name: str) -> list[Path]:
    matches = []
    for pattern in species_opacity_patterns(input_data_path, species_name):
        matches.extend(Path(match) for match in glob.glob(pattern, recursive=True))
    return sorted(set(matches))


def main() -> None:
    args = parse_args()
    ok = True

    print("Retrieval environment check")
    print("===========================")

    for import_name, dist_name, required in PACKAGE_CHECKS:
        available, detail = import_status(import_name, dist_name)
        label = "OK" if available else ("MISSING" if required else "optional-missing")
        print(f"{label:16s} {import_name:16s} {detail}")
        if required and not available:
            ok = False

    config_path = Path(args.config)
    config, config_error = load_config_if_possible(config_path)
    if config_error:
        print(f"CONFIG WARNING   {config_error}")
    else:
        print(f"OK               config           {config_path}")

    input_data_path = configured_prt_input_data_path(config) if config else None
    if input_data_path is None:
        env_hint = os.environ.get("PETITRADTRANS_INPUT_DATA") or os.environ.get("PRT_INPUT_DATA_PATH")
        if env_hint:
            input_data_path = Path(env_hint).expanduser()

    if input_data_path is not None:
        print(f"INFO             configured pRT input_data {input_data_path}")
        if not input_data_path.exists():
            print("MISSING          pRT input_data path does not exist")
            ok = False
        else:
            print("OK               pRT input_data path exists")
            error = set_pRT_input_data_path(input_data_path)
            if error:
                print(f"PRT WARNING      {error}")
            current = pRT_current_input_data_path()
            if current:
                print(f"INFO             pRT active input_data {current}")

            species_names = requested_species(config) if config else ["Fe"]
            for species_name in species_names:
                matches = find_opacity_files(input_data_path, species_name)
                if matches:
                    print(f"OK               opacity {species_name}: {matches[0]}")
                else:
                    print(
                        f"MISSING          opacity {species_name}: no local lbl files found under "
                        f"{input_data_path / 'opacities' / 'lines' / 'line_by_line'}"
                    )
                    if args.require_local_opacities:
                        ok = False
    else:
        print(
            "INFO             no pRT input_data path configured; pRT will use its own "
            "default unless you set prt.input_data_path or PETITRADTRANS_INPUT_DATA"
        )
        current = pRT_current_input_data_path()
        if current:
            print(f"INFO             pRT active input_data {current}")

    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

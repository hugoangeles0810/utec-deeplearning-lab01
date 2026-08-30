"""Punto de entrada para validar el corpus y congelar sus particiones."""

from __future__ import annotations

import argparse
from pathlib import Path

from anuraset_dl.splits import (
    build_manifests,
    build_report,
    load_preparation_policy,
    optimize_assignments,
    prepare_data,
    serialize_artifacts,
    sha256_file,
    validate_experiment_configs,
    write_artifacts_atomically,
)
from anuraset_dl.utils import load_config


def main() -> None:
    """Ejecuta el pipeline completo de preparación."""
    parser = argparse.ArgumentParser(description="Preparación reproducible de AnuraSet")
    parser.add_argument("--config", required=True, help="Configuración baseline del proyecto")
    parser.add_argument(
        "--force", action="store_true", help="Sobrescribe artefactos existentes diferentes"
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    policy = load_preparation_policy(config)
    validate_experiment_configs(config_path, config)
    metadata_path = Path(config["data"]["metadata"])
    audio_dir = Path(config["data"]["root"]) / "train"

    print("Validando metadatos e inventario de audio...")
    data = prepare_data(metadata_path, audio_dir, config, policy)
    print(f"Optimizando {len(data.recordings):,} grabaciones sin fugas...")
    result = optimize_assignments(data, policy)
    manifests = build_manifests(data, result, policy)
    report = build_report(data, manifests, result, policy, sha256_file(config_path))
    artifacts = serialize_artifacts(manifests, report, policy)
    outcome = write_artifacts_atomically(artifacts, force=args.force)

    counts = ", ".join(f"{name}={len(manifests[name]):,}" for name in policy.split_names)
    print(f"Preparación completada ({outcome}): {counts}")


if __name__ == "__main__":
    main()

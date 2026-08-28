"""Punto de entrada del entrenamiento."""

import argparse

from anuraset_dl.utils import load_config


def main() -> None:
    """Valida la configuración mientras se completa el pipeline de entrenamiento."""
    parser = argparse.ArgumentParser(description="Entrenamiento sobre AnuraSet")
    parser.add_argument("--config", required=True, help="Ruta de la configuración YAML")
    args = parser.parse_args()
    config = load_config(args.config)
    print(f"Configuración válida para: {config.get('experiment', 'experimento sin nombre')}")
    print("El pipeline de entrenamiento todavía no está implementado.")


if __name__ == "__main__":
    main()

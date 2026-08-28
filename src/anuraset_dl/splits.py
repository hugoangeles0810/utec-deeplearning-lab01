"""Creación de particiones sin fuga entre segmentos de una misma grabación."""

import re
from pathlib import Path

_SEGMENT_SUFFIX = re.compile(r"_\d+_\d+\.wav$")


def recording_id(filename: str | Path) -> str:
    """Devuelve el identificador de la grabación que originó un segmento."""
    name = Path(filename).name
    identifier = _SEGMENT_SUFFIX.sub("", name)
    if identifier == name:
        raise ValueError(f"Nombre de segmento no reconocido: {name}")
    return identifier

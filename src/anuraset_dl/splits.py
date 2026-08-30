"""Preparación reproducible de particiones agrupadas y multietiqueta."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf
import yaml
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix, vstack

_SEGMENT_SUFFIX = re.compile(r"_\d+_\d+\.wav$")
_SITE_PREFIX = re.compile(r"^INCT(?P<site>\d+)_")
_SPLIT_NAMES = ("train", "validation")


@dataclass(frozen=True)
class PreparationPolicy:
    """Política ejecutable para crear y auditar los manifiestos."""

    seed: int
    split_names: tuple[str, ...]
    proportions: tuple[Fraction, ...]
    minimum_positive_recordings: tuple[int, ...]
    exploratory_labels: tuple[str, ...]
    exploratory_minimum_positive_recordings: tuple[int, ...]
    expected_metadata_sha256: str
    manifest_paths: tuple[Path, ...]
    report_path: Path


@dataclass(frozen=True)
class PreparedData:
    """Metadatos validados y estadísticas agregadas por grabación."""

    clips: pd.DataFrame
    recordings: pd.DataFrame
    label_columns: tuple[str, ...]
    active_labels: tuple[str, ...]
    excluded_labels: tuple[str, ...]
    exploratory_labels: tuple[str, ...]
    metadata_sha256: str


@dataclass(frozen=True)
class OptimizationResult:
    """Asignación óptima y valores alcanzados por cada objetivo."""

    assignments: dict[str, str]
    objectives: dict[str, int]
    status: str


def recording_id(filename: str | Path) -> str:
    """Devuelve el identificador de la grabación que originó un segmento."""
    name = Path(filename).name
    identifier = _SEGMENT_SUFFIX.sub("", name)
    if identifier == name:
        raise ValueError(f"Nombre de segmento no reconocido: {name}")
    return identifier


def recording_site(identifier: str) -> str:
    """Extrae el sitio desde un identificador AnuraSet."""
    match = _SITE_PREFIX.match(identifier)
    if match is None:
        raise ValueError(f"No se pudo extraer el sitio de: {identifier}")
    return match.group("site")


def sha256_file(path: str | Path) -> str:
    """Calcula SHA-256 mediante lectura por bloques."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_preparation_policy(config: dict[str, Any]) -> PreparationPolicy:
    """Valida y convierte la sección ``preparation`` del baseline."""
    section = config.get("preparation")
    if not isinstance(section, dict):
        raise ValueError("La configuración requiere la sección 'preparation'")
    required = {
        "metadata_sha256",
        "proportions",
        "split_order",
        "minimum_positive_recordings",
        "exploratory_labels",
        "exploratory_minimum_positive_recordings",
        "report",
    }
    missing = sorted(required - section.keys())
    if missing:
        raise ValueError(f"Faltan campos de preparación: {', '.join(missing)}")

    split_names = tuple(section["split_order"])
    if split_names != _SPLIT_NAMES:
        raise ValueError("preparation.split_order debe ser train, validation")
    proportion_map = section["proportions"]
    proportions = tuple(Fraction(str(proportion_map[name])) for name in split_names)
    if any(value <= 0 for value in proportions) or sum(proportions) != 1:
        raise ValueError("Las proporciones deben ser positivas y sumar 1")

    minimum_map = section["minimum_positive_recordings"]
    exploratory_minimum_map = section["exploratory_minimum_positive_recordings"]
    minimum = tuple(int(minimum_map[name]) for name in split_names)
    exploratory_minimum = tuple(int(exploratory_minimum_map[name]) for name in split_names)
    if any(value < 0 for value in (*minimum, *exploratory_minimum)):
        raise ValueError("Las coberturas mínimas no pueden ser negativas")

    exploratory = tuple(section["exploratory_labels"])
    if len(exploratory) != len(set(exploratory)):
        raise ValueError("preparation.exploratory_labels contiene duplicados")
    expected_hash = str(section["metadata_sha256"])
    if re.fullmatch(r"[0-9a-f]{64}", expected_hash) is None:
        raise ValueError("preparation.metadata_sha256 no es un SHA-256 válido")

    return PreparationPolicy(
        seed=int(config["seed"]),
        split_names=split_names,
        proportions=proportions,
        minimum_positive_recordings=minimum,
        exploratory_labels=exploratory,
        exploratory_minimum_positive_recordings=exploratory_minimum,
        expected_metadata_sha256=expected_hash,
        manifest_paths=tuple(Path(config["data"]["splits"][name]) for name in split_names),
        report_path=Path(section["report"]),
    )


def validate_experiment_configs(config_path: str | Path, baseline: dict[str, Any]) -> None:
    """Evita divergencias de taxonomía y rutas entre configuraciones experimentales."""
    expected_data = baseline["data"]
    comparable = ("metadata", "splits", "num_labels", "excluded_labels", "zero_label_policy")
    for path in sorted(Path(config_path).parent.glob("*.yaml")):
        with path.open(encoding="utf-8") as stream:
            candidate = yaml.safe_load(stream)
        for field in comparable:
            if candidate["data"].get(field) != expected_data.get(field):
                raise ValueError(f"{path} difiere del baseline en data.{field}")


def _validate_metadata_table(frame: pd.DataFrame) -> tuple[str, ...]:
    if "filename" not in frame.columns:
        raise ValueError("Los metadatos no contienen la columna filename")
    if frame.empty:
        raise ValueError("Los metadatos están vacíos")
    if frame["filename"].isna().any() or not frame["filename"].map(
        lambda value: isinstance(value, str) and bool(value)
    ).all():
        raise ValueError("filename contiene valores nulos o inválidos")
    if frame["filename"].duplicated().any():
        raise ValueError("Los metadatos contienen nombres de archivo duplicados")

    labels = tuple(column for column in frame.columns if column != "filename")
    if not labels:
        raise ValueError("Los metadatos no contienen columnas de etiquetas")
    if frame.loc[:, labels].isna().any(axis=None):
        raise ValueError("Las etiquetas contienen valores nulos")
    values = set(np.unique(frame.loc[:, labels].to_numpy()))
    if not values <= {0, 1}:
        raise ValueError(f"Las etiquetas deben ser binarias; valores observados: {sorted(values)}")
    return labels


def _validate_audio_files(
    frame: pd.DataFrame,
    audio_dir: Path,
    sample_rate: int,
    clip_seconds: float,
) -> None:
    referenced = set(frame["filename"])
    available = {path.name for path in audio_dir.glob("*.wav")}
    missing = sorted(referenced - available)
    extra = sorted(available - referenced)
    if missing or extra:
        raise ValueError(
            f"Correspondencia CSV/audio inválida: faltantes={len(missing)}, extras={len(extra)}"
        )

    expected_frames = round(sample_rate * clip_seconds)
    failures: list[str] = []
    for filename in sorted(referenced):
        info = sf.info(audio_dir / filename)
        if (
            info.format != "WAV"
            or info.subtype != "PCM_16"
            or info.channels != 1
            or info.samplerate != sample_rate
            or info.frames != expected_frames
        ):
            failures.append(filename)
            if len(failures) == 10:
                break
    if failures:
        raise ValueError(f"Audios con cabecera incompatible: {', '.join(failures)}")


def prepare_data(
    metadata_path: str | Path,
    audio_dir: str | Path,
    config: dict[str, Any],
    policy: PreparationPolicy,
) -> PreparedData:
    """Valida el corpus completo y construye estadísticas por grabación."""
    metadata_path = Path(metadata_path)
    observed_hash = sha256_file(metadata_path)
    if observed_hash != policy.expected_metadata_sha256:
        raise ValueError(
            "La huella de los metadatos cambió; repita la auditoría antes de particionar"
        )
    frame = pd.read_csv(metadata_path)
    labels = _validate_metadata_table(frame)

    excluded = tuple(config["data"]["excluded_labels"])
    unknown_excluded = sorted(set(excluded) - set(labels))
    if unknown_excluded:
        raise ValueError(f"Etiquetas excluidas inexistentes: {', '.join(unknown_excluded)}")
    active = tuple(label for label in labels if label not in excluded)
    if len(active) != config["data"]["num_labels"]:
        raise ValueError("data.num_labels no coincide con la taxonomía activa derivada")
    if not set(policy.exploratory_labels) <= set(excluded):
        raise ValueError("Las etiquetas exploratorias deben estar excluidas de la taxonomía activa")

    identifiers = frame["filename"].map(recording_id)
    sites = identifiers.map(recording_site)
    _validate_audio_files(
        frame,
        Path(audio_dir),
        int(config["data"]["sample_rate"]),
        float(config["data"]["clip_seconds"]),
    )

    clips = frame.copy()
    clips.insert(1, "recording_id", identifiers)
    clips.insert(2, "site", sites)
    clips["active_all_negative"] = clips.loc[:, active].sum(axis=1).eq(0)
    clips["out_of_scope_foreground"] = clips["active_all_negative"] & clips.loc[
        :, labels
    ].sum(axis=1).gt(0)

    grouped = clips.groupby("recording_id", sort=True)
    recordings = grouped[list(labels)].max()
    recordings.insert(0, "site", grouped["site"].first())
    recordings.insert(1, "clip_count", grouped.size())
    recordings.insert(2, "active_all_negative_clips", grouped["active_all_negative"].sum())
    recordings.insert(
        3, "out_of_scope_foreground_clips", grouped["out_of_scope_foreground"].sum()
    )
    if grouped["site"].nunique().gt(1).any():
        raise ValueError("Una grabación fue asociada a más de un sitio")

    return PreparedData(
        clips=clips,
        recordings=recordings,
        label_columns=labels,
        active_labels=active,
        excluded_labels=excluded,
        exploratory_labels=policy.exploratory_labels,
        metadata_sha256=observed_hash,
    )


class _MilpBuilder:
    """Construye las variables y restricciones comunes del MILP."""

    def __init__(self, recordings: pd.DataFrame, policy: PreparationPolicy) -> None:
        self.recordings = recordings
        self.policy = policy
        self.n_recordings = len(recordings)
        self.n_splits = len(policy.split_names)
        self.lower: list[float] = [0.0] * (self.n_recordings * self.n_splits)
        self.upper: list[float] = [1.0] * (self.n_recordings * self.n_splits)
        self.integrality: list[int] = [1] * (self.n_recordings * self.n_splits)
        self.rows: list[dict[int, float]] = []
        self.constraint_lower: list[float] = []
        self.constraint_upper: list[float] = []
        self.stage_variables: dict[str, list[int]] = {}

    @property
    def size(self) -> int:
        return len(self.lower)

    def x_index(self, recording: int, split: int) -> int:
        return recording * self.n_splits + split

    def add_variable(self, lower: float, upper: float, integral: bool) -> int:
        index = self.size
        self.lower.append(lower)
        self.upper.append(upper)
        self.integrality.append(int(integral))
        return index

    def add_constraint(
        self, coefficients: dict[int, float], lower: float, upper: float
    ) -> None:
        self.rows.append(coefficients)
        self.constraint_lower.append(lower)
        self.constraint_upper.append(upper)

    def add_assignment_constraints(self) -> None:
        for recording in range(self.n_recordings):
            self.add_constraint(
                {self.x_index(recording, split): 1.0 for split in range(self.n_splits)},
                1.0,
                1.0,
            )

    def add_minimum_coverage(self, labels: tuple[str, ...], minima: tuple[int, ...]) -> None:
        for label in labels:
            positives = self.recordings[label].to_numpy(dtype=int)
            if int(positives.sum()) < sum(minima):
                raise ValueError(f"La etiqueta {label} no alcanza la cobertura mínima total")
            for split, minimum in enumerate(minima):
                coefficients = {
                    self.x_index(recording, split): float(value)
                    for recording, value in enumerate(positives)
                    if value
                }
                self.add_constraint(coefficients, float(minimum), np.inf)

    def add_exploratory_stage(self, labels: tuple[str, ...]) -> None:
        variables: list[int] = []
        for label in labels:
            positives = self.recordings[label].to_numpy(dtype=int)
            for split, minimum in enumerate(
                self.policy.exploratory_minimum_positive_recordings
            ):
                z_index = self.add_variable(0.0, 1.0, True)
                variables.append(z_index)
                coefficients = {
                    self.x_index(recording, split): float(value)
                    for recording, value in enumerate(positives)
                    if value
                }
                coefficients[z_index] = -float(minimum)
                self.add_constraint(coefficients, 0.0, np.inf)
        self.stage_variables["exploratory_coverage"] = variables

    def add_absolute_deviation_stage(
        self,
        name: str,
        vectors: list[np.ndarray],
        targets: list[tuple[int, ...]] | None = None,
    ) -> None:
        variables: list[int] = []
        denominator = math.lcm(*(fraction.denominator for fraction in self.policy.proportions))
        numerators = tuple(
            int(fraction * denominator) for fraction in self.policy.proportions
        )
        for vector_index, vector in enumerate(vectors):
            total = int(vector.sum())
            for split in range(self.n_splits):
                deviation = self.add_variable(0.0, np.inf, False)
                variables.append(deviation)
                if targets is None:
                    scale = denominator
                    constant = numerators[split] * total
                else:
                    scale = 1
                    constant = targets[vector_index][split]
                expression = {
                    self.x_index(recording, split): float(scale * value)
                    for recording, value in enumerate(vector)
                    if value
                }
                positive = dict(expression)
                positive[deviation] = -1.0
                negative = {index: -value for index, value in expression.items()}
                negative[deviation] = -1.0
                self.add_constraint(positive, -np.inf, float(constant))
                self.add_constraint(negative, -np.inf, float(-constant))
        self.stage_variables[name] = variables

    def matrix(self) -> csr_matrix:
        row_indices: list[int] = []
        column_indices: list[int] = []
        values: list[float] = []
        for row_index, row in enumerate(self.rows):
            for column_index, value in row.items():
                row_indices.append(row_index)
                column_indices.append(column_index)
                values.append(value)
        return csr_matrix(
            (values, (row_indices, column_indices)), shape=(len(self.rows), self.size)
        )


def _largest_remainder_targets(total: int, proportions: tuple[Fraction, ...]) -> tuple[int, ...]:
    exact = [proportion * total for proportion in proportions]
    targets = [value.numerator // value.denominator for value in exact]
    remaining = total - sum(targets)
    priorities = sorted(
        range(len(exact)), key=lambda index: (-(exact[index] - targets[index]), index)
    )
    for index in priorities[:remaining]:
        targets[index] += 1
    return tuple(targets)


def optimize_assignments(data: PreparedData, policy: PreparationPolicy) -> OptimizationResult:
    """Resuelve la asignación mediante objetivos MILP lexicográficos."""
    builder = _MilpBuilder(data.recordings, policy)
    builder.add_assignment_constraints()
    builder.add_minimum_coverage(data.active_labels, policy.minimum_positive_recordings)
    builder.add_exploratory_stage(data.exploratory_labels)

    ones = np.ones(len(data.recordings), dtype=int)
    recording_targets = _largest_remainder_targets(len(data.recordings), policy.proportions)
    builder.add_absolute_deviation_stage("recordings", [ones], [recording_targets])
    builder.add_absolute_deviation_stage(
        "clips", [data.recordings["clip_count"].to_numpy(dtype=int)]
    )
    builder.add_absolute_deviation_stage(
        "active_label_prevalence",
        [data.recordings[label].to_numpy(dtype=int) for label in data.active_labels],
    )
    builder.add_absolute_deviation_stage(
        "sites",
        [
            data.recordings["site"].eq(site).to_numpy(dtype=int)
            for site in sorted(data.recordings["site"].unique())
        ],
    )
    builder.add_absolute_deviation_stage(
        "active_all_negative",
        [data.recordings["active_all_negative_clips"].to_numpy(dtype=int)],
    )
    builder.add_absolute_deviation_stage(
        "out_of_scope_foreground",
        [data.recordings["out_of_scope_foreground_clips"].to_numpy(dtype=int)],
    )

    matrix = builder.matrix()
    lower = list(builder.constraint_lower)
    upper = list(builder.constraint_upper)
    objectives: dict[str, int] = {}
    solution: np.ndarray | None = None
    stage_order = (
        "exploratory_coverage",
        "recordings",
        "clips",
        "active_label_prevalence",
        "sites",
        "active_all_negative",
        "out_of_scope_foreground",
    )
    locked_rows: list[csr_matrix] = []

    for stage in stage_order:
        objective = np.zeros(builder.size)
        direction = -1.0 if stage == "exploratory_coverage" else 1.0
        objective[builder.stage_variables[stage]] = direction
        active_matrix = vstack([matrix, *locked_rows], format="csr")
        result = milp(
            objective,
            integrality=np.asarray(builder.integrality),
            bounds=Bounds(builder.lower, builder.upper),
            constraints=LinearConstraint(active_matrix, lower, upper),
            options={"presolve": True},
        )
        if result.status != 0 or result.x is None:
            raise RuntimeError(
                f"El optimizador no alcanzó optimalidad en {stage}: {result.message}"
            )
        optimum = int(round(float(objective @ result.x)))
        objectives[stage] = -optimum if direction < 0 else optimum
        lock = csr_matrix(objective.reshape(1, -1))
        locked_rows.append(lock)
        lower.append(float(optimum))
        upper.append(float(optimum))
        solution = result.x

    rng = np.random.default_rng(policy.seed)
    tie_break = np.zeros(builder.size)
    tie_break[: len(data.recordings) * len(policy.split_names)] = rng.permutation(
        len(data.recordings) * len(policy.split_names)
    )
    active_matrix = vstack([matrix, *locked_rows], format="csr")
    result = milp(
        tie_break,
        integrality=np.asarray(builder.integrality),
        bounds=Bounds(builder.lower, builder.upper),
        constraints=LinearConstraint(active_matrix, lower, upper),
        options={"presolve": True},
    )
    if result.status != 0 or result.x is None:
        raise RuntimeError(f"Falló el desempate determinista: {result.message}")
    solution = result.x

    assignment_matrix = np.rint(
        solution[: len(data.recordings) * len(policy.split_names)]
    ).astype(int).reshape(len(data.recordings), len(policy.split_names))
    if not np.all(assignment_matrix.sum(axis=1) == 1):
        raise RuntimeError("El optimizador produjo una asignación no integral")
    assignments = {
        identifier: policy.split_names[int(np.argmax(assignment_matrix[index]))]
        for index, identifier in enumerate(data.recordings.index)
    }
    return OptimizationResult(assignments, objectives, "optimal")


def build_manifests(
    data: PreparedData, result: OptimizationResult, policy: PreparationPolicy
) -> dict[str, pd.DataFrame]:
    """Construye y valida manifiestos ordenados a partir de una asignación."""
    manifest_source = data.clips.loc[:, ["filename", "recording_id"]].copy()
    manifest_source["split"] = manifest_source["recording_id"].map(result.assignments)
    if manifest_source["split"].isna().any():
        raise ValueError("Hay clips sin partición asignada")
    manifests = {
        split: manifest_source.loc[manifest_source["split"].eq(split), ["filename", "recording_id"]]
        .sort_values("filename")
        .reset_index(drop=True)
        for split in policy.split_names
    }
    validate_manifests(manifests, data, policy)
    return manifests


def validate_manifests(
    manifests: dict[str, pd.DataFrame], data: PreparedData, policy: PreparationPolicy
) -> None:
    """Comprueba exhaustividad, exclusividad y coberturas de los manifiestos."""
    if set(manifests) != set(policy.split_names):
        raise ValueError("El conjunto de manifiestos no coincide con las particiones configuradas")
    combined = pd.concat(
        [frame.assign(split=split) for split, frame in manifests.items()], ignore_index=True
    )
    if len(combined) != len(data.clips) or set(combined["filename"]) != set(data.clips["filename"]):
        raise ValueError("Los manifiestos no cubren exactamente todos los clips")
    if combined["filename"].duplicated().any():
        raise ValueError("Un clip aparece en más de un manifiesto")
    expected_recordings = data.clips.set_index("filename")["recording_id"]
    observed_recordings = combined.set_index("filename")["recording_id"]
    if not observed_recordings.sort_index().equals(expected_recordings.sort_index()):
        raise ValueError("Un manifiesto contiene un recording_id incorrecto")
    if combined.groupby("recording_id")["split"].nunique().gt(1).any():
        raise ValueError("Una grabación cruza particiones")

    split_by_recording = combined.groupby("recording_id")["split"].first()
    for label in data.active_labels:
        positives = data.recordings.index[data.recordings[label].eq(1)]
        counts = split_by_recording.loc[positives].value_counts()
        for split, minimum in zip(policy.split_names, policy.minimum_positive_recordings):
            if int(counts.get(split, 0)) < minimum:
                raise ValueError(f"Cobertura insuficiente para {label} en {split}")


def _frame_csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def build_report(
    data: PreparedData,
    manifests: dict[str, pd.DataFrame],
    result: OptimizationResult,
    policy: PreparationPolicy,
    config_sha256: str,
) -> dict[str, Any]:
    """Crea el reporte determinista de trazabilidad y balance."""
    summaries: dict[str, Any] = {}
    manifest_hashes: dict[str, str] = {}
    for split in policy.split_names:
        manifest = manifests[split]
        selected = data.clips[data.clips["filename"].isin(manifest["filename"])]
        grouped = selected.groupby("recording_id", sort=False)
        recordings = data.recordings.loc[sorted(manifest["recording_id"].unique())]
        label_summary = {
            label: {
                "positive_clips": int(selected[label].sum()),
                "positive_recordings": int(recordings[label].sum()),
            }
            for label in data.label_columns
        }
        sites = {
            site: {
                "clips": int(selected["site"].eq(site).sum()),
                "recordings": int(recordings["site"].eq(site).sum()),
            }
            for site in sorted(data.recordings["site"].unique())
        }
        summaries[split] = {
            "clips": len(selected),
            "recordings": grouped.ngroups,
            "clip_proportion": len(selected) / len(data.clips),
            "recording_proportion": grouped.ngroups / len(data.recordings),
            "active_all_negative_clips": int(selected["active_all_negative"].sum()),
            "out_of_scope_foreground_clips": int(
                selected["out_of_scope_foreground"].sum()
            ),
            "sites": sites,
            "labels": label_summary,
        }
        manifest_hashes[f"{split}.csv"] = _sha256_bytes(_frame_csv_bytes(manifest))

    return {
        "schema_version": 1,
        "metadata": {
            "path": "dataset/train.csv",
            "sha256": data.metadata_sha256,
            "clips": len(data.clips),
            "recordings": len(data.recordings),
        },
        "config_sha256": config_sha256,
        "policy": {
            "seed": policy.seed,
            "split_order": list(policy.split_names),
            "proportions": {
                split: float(value) for split, value in zip(policy.split_names, policy.proportions)
            },
            "minimum_positive_recordings": dict(
                zip(policy.split_names, policy.minimum_positive_recordings)
            ),
            "exploratory_minimum_positive_recordings": dict(
                zip(policy.split_names, policy.exploratory_minimum_positive_recordings)
            ),
            "active_labels": list(data.active_labels),
            "exploratory_labels": list(data.exploratory_labels),
            "excluded_labels": list(data.excluded_labels),
        },
        "optimizer": {"status": result.status, "objectives": result.objectives},
        "manifests_sha256": manifest_hashes,
        "splits": summaries,
        "validations": {
            "metadata_binary_and_complete": True,
            "metadata_audio_correspondence": True,
            "audio_headers": True,
            "each_clip_exactly_once": True,
            "recording_disjointness": True,
            "main_label_coverage": True,
        },
    }


def serialize_artifacts(
    manifests: dict[str, pd.DataFrame], report: dict[str, Any], policy: PreparationPolicy
) -> dict[Path, bytes]:
    """Serializa todos los artefactos con formato estable."""
    artifacts = {
        path: _frame_csv_bytes(manifests[split])
        for split, path in zip(policy.split_names, policy.manifest_paths)
    }
    artifacts[policy.report_path] = (
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    return artifacts


def write_artifacts_atomically(artifacts: dict[Path, bytes], force: bool = False) -> str:
    """Escribe artefactos validados sin sobrescribir diferencias accidentalmente."""
    different = [
        path
        for path, content in artifacts.items()
        if path.exists() and path.read_bytes() != content
    ]
    if different and not force:
        joined = ", ".join(str(path) for path in different)
        raise FileExistsError(f"Existen artefactos diferentes; use --force: {joined}")
    if all(path.exists() and path.read_bytes() == content for path, content in artifacts.items()):
        return "unchanged"

    temporary: dict[Path, Path] = {}
    try:
        for path, content in artifacts.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
            temporary_path = Path(temporary_name)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(temporary_path, 0o644)
            temporary[path] = temporary_path
        for path, temporary_path in temporary.items():
            os.replace(temporary_path, path)
        return "written"
    finally:
        for temporary_path in temporary.values():
            temporary_path.unlink(missing_ok=True)

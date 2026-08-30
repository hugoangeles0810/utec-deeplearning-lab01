from __future__ import annotations

import json
import tarfile
from pathlib import Path

from anuraset_dl.package_results import package_results
from anuraset_dl.runtime import sha256_file


def test_package_contains_canonical_results_and_excludes_regenerable_data(
    tmp_path: Path,
) -> None:
    files = {
        "configs/baseline.yaml": "experiment: test\n",
        "outputs/checkpoints/test/best.pt": "checkpoint",
        "outputs/checkpoints/test/last.pt": "checkpoint",
        "outputs/metrics/test.json": "{}",
        "outputs/filterbanks/fbrs.pt": "bank",
        "outputs/features/cache/train.npy": "cache",
        "dataset/train.csv": "private data",
        "docs/experiments.md": "# Experimentos\n",
        "pyproject.toml": "[project]\n",
        "uv.lock": "version = 1\n",
    }
    for relative, content in files.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    result = package_results(tmp_path, tmp_path / "exports")

    archive = Path(result["archive"])
    checksum = Path(result["checksum"])
    assert archive.is_file()
    assert checksum.read_text(encoding="utf-8") == (
        f"{sha256_file(archive)}  {archive.name}\n"
    )
    with tarfile.open(archive, "r:gz") as bundle:
        names = set(bundle.getnames())
        assert "outputs/checkpoints/test/best.pt" in names
        assert "outputs/metrics/test.json" in names
        assert "outputs/filterbanks/fbrs.pt" in names
        assert "outputs/features/cache/train.npy" not in names
        assert "dataset/train.csv" not in names
        manifest = json.load(bundle.extractfile("manifest.json"))  # type: ignore[arg-type]
    manifest_paths = {item["path"] for item in manifest["files"]}
    assert "outputs/checkpoints/test/best.pt" in manifest_paths
    assert "dataset/train.csv" not in manifest_paths

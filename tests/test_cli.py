from __future__ import annotations

import json

from cli import main
from core.media import write_image


def test_cli_image_manifest_roundtrip(tmp_path, cover, monkeypatch, capsys) -> None:
    monkeypatch.setenv("DSP_WATERMARK_KEY", "cli-secret-key-2026")
    source = tmp_path / "source.png"
    marked = tmp_path / "marked.png"
    write_image(source, cover)
    assert main(["image", "embed", "--input", str(source), "--output", str(marked), "--message", "hello"]) == 0
    capsys.readouterr()
    manifest = tmp_path / "marked.png.wm.json"
    assert manifest.exists()
    assert main(["image", "extract", "--input", str(marked), "--manifest", str(manifest)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["valid"] is True
    assert output["text"] == "hello"

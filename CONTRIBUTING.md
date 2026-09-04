# Contributing

Contributions are welcome through focused pull requests with tests and a reproducible benchmark result.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Install the system `ffmpeg` executable before running video integration tests.

## Quality gates

```bash
ruff check .
pytest -m "not integration"
pytest -m integration
python generate_assets.py --output-dir /tmp/dsp-watermark-assets
python -m build
```

Do not change a benchmark table without regenerating the corresponding JSON/CSV and image assets. New algorithms should expose a validated configuration, deterministic keyed carrier selection, capacity calculation, and both positive and wrong-key tests.

## Commit scope

Keep transform changes separate from documentation-only updates when possible. Never commit real production secret keys, private media, or generated shell-history examples containing a key.

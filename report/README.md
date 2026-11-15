# EE313 course project report

**Digital Watermarking for AI-Generated Media using DSP Techniques**  
IIT Indore · Digital Signal Processing (EE313)

Team: Naman V Shetty (240002043), Ridhi Shakkar (240002069), Akarsh J (240002007), and Mayur Soni (240002037).

## Compile

Open `main.tex` in a LaTeX editor, or run from this directory:

```sh
pdflatex main.tex
pdflatex main.tex
```

Alternatively, run `tectonic main.tex`. The PDF is generated as `main.pdf`.
The supplied source ZIP can be uploaded directly to Overleaf; select `main.tex` as the main file. Python is not needed to compile the included results. Packages are standard LaTeX packages; no shell escape or external bibliography processor is required.

## Included material

- `main.tex`: report with the team details, DSP derivations, implementation mapping, results, and limitations.
- `figures/`: result plots/illustrations, including an AI-generated forest-and-river landscape, a public-domain rocket photograph, and a pipeline schematic.
- `image_results.tex`, `rocket_results.tex`, `video_results.tex`, `strength_results.tex`: generated LaTeX tables.
- `data/`: raw measured JSON and execution environment metadata.
- `generate_results.py`: regenerates experiments, figures, and table fragments using the project code.

## Reproduce measurements

Install the project with its development dependencies and ensure `ffmpeg` and `ffprobe` are on PATH. From the repository root:

```sh
python -m pytest -q
python -m report.generate_results
```

The script runs image benchmarks on the included nature image and rocket photograph, runs video benchmarks, then adds a quantization-strength sweep. `--reuse` reuses existing video results in `report/data`; it always reruns both image experiments and the strength sweep. The strength sweep is always rerun.

If `ffmpeg.input` is missing, use a clean virtual environment with the dependency **ffmpeg-python==0.2.0**, rather than the different package named `ffmpeg`.

Measurements were regenerated on 15 November 2026 using source commit `0b74cf3`. The nature image was generated for this report and resized to 512×512; the rocket fixture is a public-domain photograph. The video fixture is synthetic. These examples are a small proof-of-concept set. The report includes failed rotation, median-filter, and crop cases and distinguishes blind recovery from recovery using an original reference. Numeric differences from older documentation are expected across environments. If regenerating results, update the numerical discussion in `main.tex` as well as the generated tables.

The source ZIP is self-contained for LaTeX compilation. The Python regeneration script requires the full repository.

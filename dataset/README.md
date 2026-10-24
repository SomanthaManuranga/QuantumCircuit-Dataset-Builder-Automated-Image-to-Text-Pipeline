# Dataset card: Quantum Circuit Figures from arXiv `quant-ph`

A prototype image-to-text dataset of quantum-circuit schematics with structured metadata. It was built by the
pipeline in [`../pipeline`](../pipeline) from a 1,000-paper sample of recent arXiv `quant-ph` papers, together with the
assigned paper list (`paper_list_count_45.csv`).

## Contents

| Path | Contents |
|---|---|
| `metadata/<NNNNN>_arXiv_<id>/<id>_fig_<n>.json` | **250** records, one per circuit figure, from 129 papers |
| `images/<id>_<NNNNN>_arXiv_<id>_figures_names/*.png` | **160** circuit images, from 81 papers |
| `paper_list_count_45.csv` | `arXiv:<id>,<count>` for each of the 26,907 papers in the list: how many circuit figures were found (0 if none) |

Images are grouped by paper. Within a paper folder, image file names are the authors' original names from the LaTeX
source.

## Record schema

| Field | Type | Description |
|---|---|---|
| `arxiv_number` | string | arXiv identifier of the source paper |
| `page_number` | int | page of the figure in the PDF (found with PyMuPDF) |
| `figure_number` | int | figure number in the paper |
| `quantum_gates` | list[string] or `"not specifically mentioned"` | gates named in the caption (regex) |
| `quantum_problem` | list[string] or `"not specifically mentioned"` | algorithm or problem named in the caption (regex) |
| `description` | string | first sentence of the cleaned caption |
| `text_positions` | list[[start, end]] | character offsets of the description in the source document |

Each file holds one object keyed by `<arxiv_number>_fig_<figure_number>`.

## Statistics

| | |
|---|:---:|
| Records with a page number | 250 / 250 |
| Records with at least one gate | 74 (30 %) |
| Records with an algorithm or problem | 53 (21 %) |
| Top gates | CNOT 25 · H 19 · RY 11 · X 10 · CZ 7 · SWAP 6 · Z 6 |
| Top problems | Ansatz 19 · state preparation 9 · VQC 8 · Bell state 7 · Trotter 5 · QEC 3 |

## Known issues

- 90 records have no matching image: the LaTeX source gave no file reference, or the archive used unusual names.
- The gate and problem fields come from the caption text only, so a gate drawn in the figure but not named in the
  caption is missing.
- Some PDF-to-PNG conversions show slight colour shifts.

## Licence and attribution

Images and caption-derived text belong to the authors of the source papers, identified by `arxiv_number`. The
dataset is provided for non-commercial research and teaching only. Please cite the original papers.

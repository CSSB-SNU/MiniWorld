# MiniWorld 0 technical report

## Build

```bash
tectonic docs/paper/main.tex          # -> docs/paper/main.pdf
```

`tectonic` is installed outside the project env (`pixi global install tectonic`,
binary at `~/.pixi/bin/tectonic`) so the `cu128` training env stays frozen.
It is self-contained: LaTeX packages and fonts are fetched on demand into
`~/.cache/Tectonic`. BibTeX runs automatically (natbib + `unsrtnat`); there is
no `biber` in this setup, so **do not switch to biblatex**.

## Design

Single column, one column per page --- deliberately a technical report, not a
paper. 11 pt Libertinus (serif body, sans headings), letterpaper with 1.35 in
side margins.

* accent colour: `accent` = `#1F5673`; change it in one place at the top of
  `main.tex` and headings, captions, links and the title rule all follow
* headings: sans-serif, accent-coloured, thin rule under each `\section`
  (`titlesec`)
* summary box: tinted `tcolorbox`, no border
* captions: sans-bold accent label, left-aligned (`caption`)
* footer: report name left, page number right (`fancyhdr`)
* `\todo{...}` renders in red --- every placeholder is visible in the PDF, and
  the macro can be redefined to `{}` to hide them all for a clean draft

Naming: **MiniWorld 0** is the baseline of the MiniWorld series; the `0` marks
it as the simplest and least advanced configuration. The numeral is space-
separated and accent-coloured **in display use only** (title block); running
text, captions and the footer use plain `MiniWorld~0`. Use the macros rather
than typing the name: `\mw` in prose (non-breaking, so the name never splits
across a line) and `\mwdisplay` for the title.
Corresponding author: **Minkyung Baek** (marked with `\textdagger`).

## Layout

| Path | Tracked | Contents |
|---|---|---|
| `docs/paper/main.tex` | yes | report source |
| `docs/paper/refs.bib` | yes | 27 entries, auto-fetched via DOI content negotiation |
| `docs/paper/main.pdf` | no | build output (gitignored) |
| `docs/refs/*.pdf` | no | 22 reference PDFs, ~169 MB (gitignored) |
| `docs/images/` | yes | existing analysis figures (rank-collapse study) |

## refs.bib

Every entry was resolved from a verified DOI (Crossref / DataCite), not written
by hand. To add one:

```bash
curl -LH "Accept: application/x-bibtex" https://doi.org/<DOI> >> docs/paper/refs.bib
# then rename the cite key to something stable
```

## ESMFold2 citation (resolved)

The atom-transformer design is from **ESMFold2**, released in
`github.com/Biohub/esm` (formerly `evolutionaryscale/esm`), v3.4.0.

* Preprint: Candido et al., *Language Modeling Materializes a World Model of
  Protein Biology*, bioRxiv 2026, DOI `10.64898/2026.06.03.729735` -> key `esmfold2`
* Code/weights: Zenodo DOI `10.5281/zenodo.14219303` -> key `esmcode`
* Local copy: `docs/refs/esmfold2_preprint.pdf` (111 pages)

The algorithm numbers used throughout the MiniWorld source match the preprint
exactly, so they can be cited directly:

| Preprint | MiniWorld |
|---|---|
| Alg. 5 `DiffusionModule`      | `modules/diffusion_module.py` |
| Alg. 6 `SWAAtomEncoder`       | `SWAAtomAttentionEncoder` |
| Alg. 7 `SWAAtomDecoder`       | `SWAAtomAttentionDecoder` |
| Alg. 8 `SWAAtomTransformer`   | `team_gm.modules.blocks.swa_atom_transformer` |
| Alg. 9 `Build3DRoPE`          | `SWAAtomTransformer.build_rope` |
| Alg. 10 `ConfidenceHead`      | `modules/confidence_head.py` |

Reported ESMFold2 FoldBench pass rates (DockQ >= 0.23), for the results table:
antibody-antigen 50% +/- 2 single-sequence, 53% +/- 2 with MSAs, up to 65% at
1000 samples.

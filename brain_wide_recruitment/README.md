# Brain-Wide Recruitment — Question 3

**Contributor:** Arash Kanafchian

This analysis computes visual-response latency after stimulus onset and tests
whether the chronological ordering of anatomically grouped regions is consistent
with sensory-signal propagation through the mouse brain.

## Analysis outputs

- **Population trajectory latency:** first 70% crossing of the
  left-versus-right population distance, aligned with the IBL reference method.
- **Responsive-unit latency:** first sustained 70% crossing of each unit's
  absolute baseline-subtracted response.

## Run

From the project root:

```bash
python -m pip install -r requirements.txt
jupyter lab brain_wide_recruitment/brain_wide_recruitment.ipynb
```

The public IBL/Neuromatch `stimOn` archive is downloaded only when a local cache
is unavailable. Custom paths can be supplied without editing the notebook:

```bash
export OCTAGRAM_DATA_ROOT=/path/to/ibl-data
export OCTAGRAM_ARTIFACT_ROOT=/path/to/output-root
```

Generated Q3 tables, figures, and checkpoints are written to `results/q3/` and
excluded from Git. Complete precomputed results and extended documentation are
available in Arash's standalone repository:

<https://github.com/ArashKanafchian/ibl-q3-brainwide-latency>

## Code map

- `brain_wide_recruitment.ipynb` — end-to-end analysis and interpretation.
- `../src/ibl_q3/statistics.py` — Benjamini–Hochberg correction.
- `../src/ibl_q3/plotting.py` — publication-ready latency figures.
- `../src/ibl_q3/config.py` — portable data/output paths.
- `../src/ibl_q3/artifacts.py` — reusable local or mounted artifacts.

The IBL/Neuromatch data-loading foundation remains credited as upstream code;
the Q3-specific analysis, validation, and reporting are attributed to Arash
Kanafchian.

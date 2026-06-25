# AMCIS

Adjoint Monte-Carlo tool for **neutral wall provenance** from [SOLEDGE-HDG](https://github.com/soledge) solutions.

Given a plasma target point Ω = (R, Z), AMCIS traces neutrals **backward** through the SOL and scores which wall regions (pump, puff, divertor, …) are most likely to supply them. A second tab covers **core fueling**: births inside the separatrix, same logic toward the wall.

---

## Requirements

- Python ≥ 3.10, **micromamba** (or conda/mamba)
- A SOLEDGE-HDG solution file (`.h5`)
- **[HDG_postprocess](https://github.com/soledge/HDG_postprocess)** — same stack used to read and post-process HDG outputs (reference element, atomic data, wall flux, …)

---

## Installation

```bash
git clone https://github.com/Oscar-Brun/AMCIS.git
cd AMCIS

export HDG_POSTPROCESS_PATH=/path/to/HDG_postprocess
bash scripts/install_dev.sh

micromamba activate soledge-amcis
```

The script creates the `soledge-amcis` environment, installs AMCIS in editable mode, and installs HDG_postprocess. If `HDG_POSTPROCESS_PATH` is omitted, the path is inferred from an editable `hdg_postprocess` install.

Compile check (optional):

```bash
python -c "from adjoint_mc.tracker.backward_cython import cython_available; print(cython_available())"
```

Should print `True`.

---

## Usage

**GUI** (default):

```bash
amcis
```

1. Choose your `.h5` file  
2. Set the target (R, Z) or open the **Core fueling** tab  
3. Run  

**Batch** (point target):

```bash
amcis --solution case.h5 --target-r 0.61 --target-z 0.0 --output-dir output/amcis --plot
```

---

## How it works (briefly)

1. **Load** the HDG solution → wall geometry, plasma fields on an (R, Z) grid, optional SOLEDGE wall neutral flux Γ_wall.

2. **Backward MC** — each history starts at the target (or at S_ion-weighted births in the core). The neutral moves backward; weight decreases with ionization (`W ∝ exp(−∫ Σ_ion ds)`). Charge exchange uses a rejection scheme (velocity updated, W unchanged).

3. **Score** wall hits → connectivity **f_k** (% of hits per region/segment). When Γ_wall is available, **f_k^flux** weights segments by emitted flux (preferred metric for “who actually fuels the target”).

4. **Plots** — wall maps, trajectories, and a few HDG context fields (S_ion, n_n, …).

The hot loop is in **Cython + OpenMP**; the GUI runs two pipelines: `AMCIS → Ω` and core fueling inside ψ = 1.

---

## Tests

```bash
pytest tests/ -q
```

(HDG-dependent tests are skipped if no local `.h5` is configured.)

---

## License

See repository settings / add a `LICENSE` file if needed.

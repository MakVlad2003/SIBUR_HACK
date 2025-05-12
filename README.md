## Overview

The **SIBUR\_HACK** repository provides:

1. A **Python package** (`molecule_generation`) implementing two complementary graph‑based molecular generative models:

   * **MoLeR**: a scaffold‑constrained graph generative model for molecule design ([GitHub][1], [arXiv][2])
   * **CGVAE**: a Constrained Graph Variational Autoencoder for de novo molecule generation ([GitHub][3])
2. A set of **CLI tools** to preprocess data, encode molecules, train models, sample new compounds, and visualize results (`molecule_generation/cli`).
3. **Preprocessing scripts** to convert raw trace data into training/testing formats (`molecule_generation/preprocessing`).
4. **Dataset utilities** for in‑memory and JSONL‑based trace datasets (`molecule_generation/dataset`).
5. **Model definitions** and **layer implementations** for both MoLeR and CGVAE architectures (`molecule_generation/models`, `molecule_generation/layers`).
6. A comprehensive suite of **unit and integration tests** using pytest (`molecule_generation/test`).
7. **Jupyter notebooks** under `notebooks/` showcasing:

   * Data cleaning and feature generation (`SIBUR_data_preprocessing.ipynb`)
   * Embedding computation from pretrained models (`compute_embeddings.ipynb`)
   * Baseline CatBoost and neural network experiments (`Exp_MLP.ipynb`, `Final_Model.ipynb`)
   * The official Element 119 baseline setup (`sibur_element_119_Элемент 119. ИИ в Химии. Baseline.ipynb`)

---

## Repository Structure

```
SIBUR_HACK/
├── molecule_generation/         # Core Python package
│   ├── chem/                    # RDKit helpers, motif and valence utils
│   ├── cli/                     # Command‑line interface (train, sample, preprocess…)
│   ├── dataset/                 # Dataset definitions and loaders
│   ├── layers/                  # Network layer implementations (MoLeR & CGVAE)
│   ├── models/                  # High‑level model classes (cgvae.py, moler_vae.py…)
│   ├── preprocessing/           # Data conversion & trace generation scripts
│   ├── test/                    # Pytest test modules
│   ├── utils/                   # Beam search, decoding, training helpers
│   ├── visualisation/           # CLI/HTML visualisers
│   ├── version.py               # Package version
│   └── wrapper.py               # High‑level wrapper exposing CLI entrypoints
├── notebooks/                   # Hackathon notebooks & outputs
│   ├── SIBUR_data_preprocessing.ipynb
│   ├── compute_embeddings.ipynb
│   ├── Exp_MLP.ipynb
│   ├── Final_Model.ipynb
│   └── sibur_element_119…Baseline.ipynb
├── processed/                   # Intermediate CSV datasets (train_final.csv, etc.)
└── .github/
    └── workflows/ci.yml         # Continuous integration via pytest
```

---

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/MakVlad2003/SIBUR_HACK.git
   cd SIBUR_HACK
   ```
2. **Create a conda environment** (Python 3.10 recommended)

   ```bash
   conda create -n sibur_hack python=3.10
   conda activate sibur_hack
   ```
3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   *If no `requirements.txt` is present, install core packages manually:*

   ```bash
   pip install rdkit-pypi torch pytest numpy pandas scikit-learn catboost matplotlib
   ```

---

## Quick Start

### CLI Usage

* **Preprocess raw traces**

  ```bash
  sibur-cli preprocess --input raw_traces.jsonl --output processed/
  ```
* **Train a MoLeR model**

  ```bash
  sibur-cli train --model moler --config configs/moler.yaml
  ```
* **Sample new molecules**

  ```bash
  sibur-cli sample --model-checkpoint path/to/best.ckpt --n-samples 100
  ```
* **Visualise generation**

  ```bash
  sibur-cli visualise --input generated.smi --output viz.html
  ```

*(Use `sibur-cli --help` for full options.)*

### Running Notebooks

Each notebook in `notebooks/` is self‑contained. To launch:

```bash
jupyter lab notebooks/
```

Follow the cells in order to replicate data cleaning, embedding extraction, model training, and final submission generation.

---

## Testing

Run the full test suite:

```bash
pytest -q
```

The CI pipeline (`.github/workflows/ci.yml`) is configured to execute these tests on push.

---

## Contributing

1. Fork the repo
2. Create a feature branch (`git checkout -b feat/your-feature`)
3. Add tests for new functionality
4. Commit and push (`git push origin feat/your-feature`)
5. Open a Pull Request

Please adhere to the existing **Google‑style docstrings** and keep functions under 40 lines.

---

## License & Citation

This project is released under the **MIT License** (add a `LICENSE` file if missing).

If you use **MoLeR** or **CGVAE** components in your work, please cite:

> **MoLeR**: Krzystof Maziraz et al., “Learning to Extend Molecular Scaffolds with Structural Motifs,” *ICLR 2022* ([arXiv][2])
> **MoLeR Implementation**: microsoft/molecule-generation GitHub ([GitHub][1])

---

### Sources Consulted

* Searched for the official **SIBUR\_HACK** repo page; no public listing found (personal/private repo).
* MoLeR GitHub: microsoft/molecule-generation ([GitHub][1])
* MoLeR paper (arXiv) ([arXiv][2])

> *These searches confirmed the origin and details of the MoLeR model integrated into this codebase.*

[1]: https://github.com/microsoft/molecule-generation?utm_source=chatgpt.com "microsoft/molecule-generation: Implementation of MoLeR - GitHub"
[2]: https://arxiv.org/pdf/2103.03864?utm_source=chatgpt.com "[PDF] learning to extend molecular scaffolds - arXiv"
[3]: https://github.com/microsoft/molecule-generation/releases?utm_source=chatgpt.com "Releases · microsoft/molecule-generation - GitHub"

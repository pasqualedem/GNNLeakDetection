
---

# Explainable Fuzzy GNNs for Leak Detection in Water Distribution Networks

This repository contains the implementation of an **Explainable Fuzzy Graph Neural Network (FGNN)** framework for detecting and localizing leaks in **Water Distribution Networks (WDNs)**. By combining **Graph Neural Networks (GNNs)** with **fuzzy logic**, the model achieves both accurate predictions and interpretable, rule-based explanations.

This work is based on the **Hanoi Benchmark Network dataset (LeakDB)** and accompanies the paper:

> *"Explainable Fuzzy GNNs for Leak Detection in Water Distribution Networks"*
> Qusai Khaled, Pasquale De Marinis, Moez Louati, David Ferras, Laura Genga, Uzay Kaymak
> Submitted to the **2025 IFSA World Congress NAFIPS**

---

## 🔍 Overview

The project investigates multiple GNN architectures and proposes a fuzzy-enhanced model that integrates **mutual information** and **fuzzy rules** to support explainable leak detection.

**Main tasks:**

* **Leak Detection** (Graph-level classification): Detect whether a leak exists in the network.
* **Leak Localization** (Node-level classification): Identify the exact node or pipe with the leak.

**Example explanation:**

> *"IF pressure at Node 1 is high AND pressure at Node 2 is low, THEN leak probability at Node 5 is high."*

This rule-based reasoning supports better decision-making for engineers and water utility managers.

---

## 📁 Repository Structure

```
.
├── notebooks/                # Interactive Jupyter notebooks
│   ├── 01_Reshape_data.ipynb
│   ├── 01_Reshape_oldata.ipynb
│   ├── 05_GCN.ipynb
│   ├── 06_GAT.ipynb
│   ├── 11_AnomalyDetectionScenario.ipynb
│   ├── 12_FeatureExtraction.ipynb
│   └── Explain.ipynb
├── scripts/                  # Python source files
│   ├── data.py
│   ├── preprocess.py
│   ├── model.py
│   ├── train.py
│   ├── test.py
│   ├── grid.py
│   ├── loss.py
│   ├── tracker.py
│   └── logger.py
├── .gitignore
└── README.md
```

---

## ⚙️ Setup


1. **Install dependencies:**

Ensure Python 3.8+ is installed. Then install dependencies:

```bash
pip install -r requirements.txt
```

> ℹ️ You may need to install [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) separately depending on your system.

2. **Prepare the dataset:**

* Download the **Hanoi Benchmark Network dataset (LeakDB)**.
* Place it in the `data/` directory or another appropriate path.
* Use the preprocessing notebooks (`01_Reshape_data.ipynb`, etc.) to convert and prepare the data.

---

## 🚀 Usage

### Train and Test

```bash
python train.py
python test.py
```

> Configuration options can be modified directly in the script or through arguments.

### Run Experiments

Explore different GNN architectures and fuzzy logic features 

### Hyperparameter Tuning

Use:

```bash
python grid.py
```

Or refer to `Grid search py` for custom tuning setups.

---

## 📊 Results

Performance metrics and visualizations are provided in the paper. Experimental results (e.g., accuracy, explainability) are saved in the `results/` directory after training.

---

## 📚 Citation

If you use this repository in your research, please cite the following:

```bibtex
@article{khaled2025explainable,
  title={Explainable Fuzzy GNNs for Leak Detection in Water Distribution Networks},
  author={Khaled, Qusai and De Marinis, Pasquale and Louati, Moez and Ferras, David and Genga, Laura and Kaymak, Uzay},
  journal={2025 IFSA World Congress NAFIPS},
  year={2025}
}
```

---

## 🙏 Acknowledgments

This work is supported by the **ILUSTRE project**, funded in part by the **Dutch Research Council (NWO)**.

---

## 📬 Contact

For questions or suggestions, feel free to [open an issue](https://github.com/yourusername/GNNLeakDetection/issues).
or email
qusai.khaled@ieee.org

---

Let me know if you'd like a `requirements.txt` template or badge support (e.g., license, build status) for the README as well.

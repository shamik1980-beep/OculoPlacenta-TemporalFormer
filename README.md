OculoPlacenta-TemporalFormer
Explainable Federated Multimodal AI for Longitudinal Pre-Eclampsia and Fetal Growth Restriction Prediction
OculoPlacenta-TemporalFormer is a research repository for privacy-preserving, trimester-wise prediction of pre-eclampsia and fetal growth restriction using retinal oculomics, placental biomarkers, uteroplacental Doppler, and maternal clinical trajectories.
The repository deliberately separates two evidence levels:
Reproduced public-data benchmark: leakage-resistant analysis of the Bogotá pre-eclampsia cohort.
Proposed multimodal architecture: an unvalidated PyTorch scaffold for future patient-linked, longitudinal, multicentre research.
![Proposed OculoPlacenta-TemporalFormer architecture](docs/assets/model_architecture.png)
Why this project matters
Pre-eclampsia and fetal growth restriction can evolve before severe symptoms become obvious. Maternal history alone may not capture the underlying placental and vascular process. The project investigates whether four complementary signals can improve risk assessment:
Retinal oculomics: non-invasive indicators of maternal microvascular health.
Placental biomarkers: PlGF, sFlt-1, and related angiogenic signals.
Uteroplacental Doppler: measures of vascular resistance and placental perfusion.
Longitudinal clinical data: blood pressure, maternal risk factors, and pregnancy trajectory.
Federated learning is proposed to enable multicentre model development without centralising raw patient records. Explainability, calibration, fairness, missing-modality handling, and external validation are treated as essential design requirements.
Current evidence in the repository
The public Bogotá cohort contains 190 women already diagnosed with pre-eclampsia, including 80 early-onset cases, 110 late-onset cases, and 44 IUGR cases. The benchmark evaluates subtype and IUGR stratification using pre-outcome clinical variables. It does not validate population screening or the complete multimodal model.
Repository map
```text
oculoplacenta-temporalformer/
├── configs/                         Reproducible experiment settings
├── data/                            Local-only data locations and data guidance
├── docs/                            Research, dataset, model, ethics, and FL documentation
├── examples/                        Synthetic TemporalFormer forward-pass example
├── notebooks/                       Step-by-step exploratory notebook
├── results/                         Generated outputs, ignored except placeholder
├── scripts/                         CLI wrappers and verification scripts
├── src/oculoplacenta_temporalformer/
│   ├── benchmark.py                 Bogotá benchmark workflow
│   ├── data.py                      Loading, validation, and checksums
│   ├── preprocessing.py             Leakage-resistant feature pipeline
│   ├── evaluation.py                CV, metrics, and bootstrap intervals
│   ├── plotting.py                  ROC, PR, calibration, and importance plots
│   ├── temporalformer.py            Proposed longitudinal multimodal transformer
│   ├── federated.py                 Weighted FedAvg utility
│   └── cli.py                       Command-line interface
└── tests/                            Unit tests
```
Installation
```bash
git clone https://github.com/USERNAME/oculoplacenta-temporalformer.git
cd oculoplacenta-temporalformer
python -m venv .venv
```
Activate the environment:
```bash
# macOS/Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```
Install the reproducible benchmark:
```bash
pip install -e .
```
Install all research and development components:
```bash
pip install -e ".[dev,deep-learning,notebook]"
```
Dataset preparation
Download `dataframe.csv` from the public `cfarkas/preeclampsia_ml` repository and place it at:
```text
data/raw/dataframe.csv
```
Verify the version used in the manuscript:
```bash
oculoplacenta verify-data data/raw/dataframe.csv
```
Expected SHA-256:
```text
319e79453531aa2bb48cdf3c05099601e965dfce3c15f4e924a25c7909fd69fe
```

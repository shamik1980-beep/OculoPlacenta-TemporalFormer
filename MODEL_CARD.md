# Model Card

## Validated component

The repository reproduces a benchmark using three conventional classifiers:

- L2-penalised logistic regression.
- Shrinkage linear discriminant analysis.
- Low-depth random forest.

These models classify early- versus late-onset pre-eclampsia and stratify IUGR among women already diagnosed with pre-eclampsia.

## Proposed component

`OculoPlacentaTemporalFormer` is an unvalidated PyTorch research prototype. It accepts visit-level retinal embeddings, placental biomarkers, Doppler features, and clinical variables, supports missing modalities, and produces multitask outputs for pre-eclampsia and fetal growth restriction.

## Intended use

- Method development.
- Reproducibility research.
- Prospective protocol design.
- Federated and multimodal simulation.

## Prohibited interpretation

The code must not be used as an autonomous diagnostic or treatment system. It must not be described as clinically validated, regulator-approved, or safe for patient care.

## Performance and limitations

The public-data benchmark reported modest discrimination near AUROC 0.57. This is evidence against relying on the available maternal-history variables alone, not evidence that the full TemporalFormer performs better. The proposed architecture requires patient-linked, longitudinal, multimodal, multicentre validation.

## Fairness and safety

Future validation must report performance and calibration by site, ethnicity, age, BMI, parity, socioeconomic status, device, gestational stage, and missing-modality pattern. Clinical thresholds must be selected using prespecified utility criteria and independently validated.

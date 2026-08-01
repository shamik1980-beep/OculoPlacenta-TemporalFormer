# Federated Learning Design

The repository includes a framework-agnostic weighted FedAvg utility. This is sufficient for unit testing aggregation logic but not for secure clinical federation.

A future multicentre implementation should include:

1. Local data harmonisation and schema validation.
2. Site-specific train/validation partitions with a locked external site.
3. Secure authenticated transport and encrypted model-update exchange.
4. Secure aggregation so the server cannot inspect individual updates.
5. FedAvg as a baseline, followed by FedProx or personalised approaches for non-IID data.
6. Client-level drift, calibration, and subgroup monitoring.
7. Model-update anomaly detection and poisoning resilience.
8. Privacy accounting when differential privacy is applied.
9. Reproducible orchestration with explicit rounds, seeds, checkpoints, and site participation logs.

Federated learning reduces the need to centralise records but does not automatically remove privacy, security, bias, or governance risks.

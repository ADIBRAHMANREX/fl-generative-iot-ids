# Federated Generative Learning for Real-Time IoT Intrusion Detection

Privacy-preserving federated learning framework for IoT botnet/intrusion detection,
using on-device generative models to balance minority attack classes and
differential privacy to protect model updates.

## Dataset
N-BaIoT (Meidan et al., 2018) — UCI ML Repository:
http://archive.ics.uci.edu/ml/datasets/detection_of_IoT_botnet_attacks_N_BaIoT

Not included in this repo due to size (~2GB). Download and place under `data/nbaiot_raw/`
following the structure in `src/data_loader.py`.

## Structure
- `src/` — data loading, FL simulation, generative models, privacy mechanisms
- `notebooks/` — exploratory analysis
- `results/` — metrics, logs, figures (generated, not committed raw data)

## Status
Work in progress.
## Status
- [x] Dataset acquired and preprocessed (9 N-BaIoT devices, per-device train/test splits, ~6.9M rows)
- [x] Python 3.12 venv configured with CUDA-enabled PyTorch (GTX 1650, CUDA 12.1)
- [ ] Centralized baseline model
- [ ] Federated learning (FedAvg) simulation
- [ ] Generative model for class rebalancing
- [ ] Differential privacy integration
- [ ] Final evaluation and writeup
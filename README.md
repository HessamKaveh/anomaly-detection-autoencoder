# Unsupervised Anomaly Detection with Convolutional Autoencoder

A convolutional autoencoder trained only on "normal" samples (MNIST digit 0),
used to detect anomalies (all other digits) via reconstruction error.

## Pipeline
1. Train autoencoder exclusively on digit 0 (normal class)
2. At test time, reconstruct both normal and anomalous digits
3. Anomaly score = per-image reconstruction error (MSE)
4. Evaluate separability with ROC-AUC and error distribution histograms

## Architecture
Convolutional encoder-decoder (3 conv layers down to a 64-dim latent
representation, mirrored transposed-conv decoder)

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
python src/train.py
python src/evaluate.py
```

## Results
- `results/roc_curve.png` — ROC-AUC for anomaly separability
- `results/error_distribution.png` — reconstruction error: normal vs anomaly
- `results/sample_reconstructions.png` — qualitative examples

## Author
Hessam Kaveh — Research Fellow, Italian Institute of Technology


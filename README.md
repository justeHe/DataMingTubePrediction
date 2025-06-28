# Time-Series Degradation Prediction

Welcome to the **Time-Series Degradation Prediction** project. This repository contains PyTorch implementations for predicting degradation trends and Remaining Useful Life (RUL) in time-series data.

## Overview

The project focuses on analyzing time-series degradation patterns using:
- Moving average trend extraction (`Trend_extraction.py`)
- Simple neural network models (`Simple_model.py`)
- LSTM-based prediction models (`Lstm_test.py`)

Key features:
- Transfer learning-inspired training strategy
- Full degradation trajectory characterization
- Automatic threshold detection for failure onset
- GPU optimization with CUDA support

##  Installation

1. **Clone the repository**:
```bash
git clone https://github.com/justeHe/DataMingTubePrediction.git
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

##  Usage

### Trend Extraction
To extract degradation trends from time-series data:
```bash
python Trend_extraction.py --input_file tube3_mode_1-0-0.875.csv
```

### Training the Model
To train the prediction model:
```bash
python Simple_model.py --full_dataset tube1_mode_1-0-0.875_with_trend.csv --partial_dataset tube2_mode_1-0-0.875_with_trend.csv
```

### LSTM-based Prediction
For advanced LSTM-based RUL prediction:
```bash
python Lstm_test.py --full_csv_file tube1_mode_1-0-0.875_with_trend.csv --partial_csv_file tube2_mode_1-0-0.875_with_trend.csv
```

##  Dataset

The model expects time-series data with:
- Timestamps (f1)
- Feature values (f9)
- Extracted trend columns (trend_f9)

## Methodology

This study adopts a transfer learning-inspired training strategy:
1. Uses complete dataset from a single tube to characterize full degradation trajectory
2. Leverages historical data points closest to early phase of target tube for initialization
3. Computes first and second derivatives to identify inflection points
4. Uses last prominent peak as threshold for failure onset

Benefits:
- Accelerated convergence
- Enhanced prediction accuracy
- Improved sensitivity to early-stage degradation patterns

##  Contributing
Feel free to open issues or submit pull requests! Contributions are welcome. 
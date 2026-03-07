# Physiological Signals EDA (Exploratory Data Analysis)

This comprehensive EDA script analyzes physiological signals dataset containing ECG, EMG, GSR, HR, RESP, and marker channels.

## Features

The EDA script provides the following analyses:

### 1. Dataset Overview - [Dataset](https://physionet.org/content/drivedb/1.0.0/)
- Total recordings and duration statistics
- Sampling frequency analysis
- Channel combinations analysis
- Signal channel usage statistics

### 2. Signal Analysis
- **Statistical Analysis**: Mean, standard deviation, skewness, kurtosis, range, variance
- **Distribution Analysis**: Histograms and box plots for each signal
- **Frequency Domain Analysis**: FFT magnitude and Power Spectral Density (PSD)
- **Correlation Analysis**: Correlation matrix between different signals
- **Signal Quality Metrics**: SNR, zero crossings, peak-to-peak, RMS, dynamic range
- **Temporal Analysis**: Signal segments to show temporal variations

### 3. Visualization
- Signal overview plots
- Distribution plots (histograms and box plots)
- Frequency domain plots (FFT and PSD)
- Correlation heatmaps
- Temporal segment plots

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```


## Dataset Information

Based on the analysis, the dataset contains:
- **18 recordings** with total duration of ~22 hours
- **Multiple sampling frequencies**: 15 Hz, 15.5 Hz, 31 Hz
- **Signal types**: ECG, EMG, foot GSR, hand GSR, HR, RESP, marker
- **Variable channel combinations** across recordings
- **Long-duration recordings** (25+ minutes each)

## Output

The script generates:
1. **Console output** with statistical summaries
2. **Interactive plots** for visual analysis
3. **DataFrames** with detailed metrics

## Key Insights

1. **Physiological Signals**: The dataset contains multiple physiological signals suitable for stress analysis, emotion recognition, or health monitoring applications.

2. **Mixed Sampling Rates**: Different recordings use different sampling frequencies, which may require resampling for consistent analysis.

3. **Variable Channel Sets**: Not all recordings contain the same channels, requiring careful handling for multi-signal analysis.

4. **Long Duration**: The recordings are long enough for temporal analysis and pattern recognition.

5. **Signal Quality**: The script provides quality metrics to assess signal integrity and identify potential artifacts.

## Customization

You can modify the script to:
- Change the number of sample recordings to analyze
- Adjust plot parameters and styles
- Add additional statistical measures
- Focus on specific signal types
- Export results to files

## Dependencies

- `numpy`: Numerical computations
- `pandas`: Data manipulation and analysis
- `matplotlib`: Plotting and visualization
- `seaborn`: Enhanced plotting styles
- `wfdb`: Reading WFDB format files
- `scipy`: Signal processing and statistical functions

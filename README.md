# Weight Loss Tracking and Analysis
This is a homework for "Použití počítačů ve fyzice (NOFY 084) " created by Jakub Mikeš.
## Overview
This script helps you analyze your weight loss journey using scientific methods. It uses Orthogonal Distance Regression (ODR) to fit a weight loss model to your data, handles uncertainties, and provides visual feedback on your progress.

### Features
- ODR Fitting: Fits a weight loss model considering uncertainties.
- Visualization: Plots weight loss data with uncertainty ranges and target goals.
- Weekly Analysis: Calculates and displays weekly median weights.
- Long-term Targets: Projects weight loss targets over 12 weeks.

### Requirements
- Python 3.x
- Libraries: numpy, pandas, scipy, matplotlib, uncertainties, tabulate
Install the required libraries using: 
```
pip install numpy pandas scipy matplotlib tabulate uncertainties
```
- An Excel file (data.xlsx) with Date and Weight columns (see the example file).

### Usage
- Prepare Your Data: Ensure data.xlsx has the required columns.
- Run the Script: Execute the script to analyze your weight loss journey and generate a plot (weight_loss.png) using:
```
python script.py
```

### Output
- Console: Displays weekly progress, estimated starting/current weight, and long-term targets.
- Plot: Visualizes weight loss with a fitted curve and uncertainty ranges.

### Note
- For best results, weigh yourself every day under the same conditions, ideally right after waking.
- It is recommended to use this tool alongside a balanced diet and exercise plan.

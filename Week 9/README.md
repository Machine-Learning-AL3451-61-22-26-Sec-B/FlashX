# Local Weighted Regression Visualization

## Introduction
This Streamlit application demonstrates Local Weighted Regression (LWR) applied to a sine function. LWR is a non-parametric regression method used for fitting a model to data points. Unlike traditional regression methods, LWR assigns different weights to each data point based on its proximity to the point of interest.

## Dataset
The dataset used in this visualization consists of pairs of \( X \) and \( Y \) values, where \( X \) represents values from 0 to \( 6\pi \) and \( Y \) represents the corresponding sine values, \( \sin(X) \). These data points are plotted on a scatter plot.

## Tau Selection
Tau (\(\tau\)) is a hyperparameter that determines the bandwidth of the kernel function used in LWR. It controls the influence of nearby data points on the prediction at a specific point. Users can select the tau value using a slider. The default range for tau is from 0.01 to 1.0, with a default value of 0.1.

## Visualization
The visualization consists of two main components:

1. **Scatter Plot:** This plot displays the original data points generated from the sine function \( Y = \sin(X) \). Each point represents a pair of \( X \) and \( Y \) values.

2. **Local Weighted Regression (LWR) Plot:** This plot overlays the LWR prediction on top of the scatter plot. The LWR prediction is represented by a red line. The tau value selected by the user determines the smoothness of the LWR curve.

## Usage
To use the application, follow these steps:

1. Adjust the tau value using the slider to change the bandwidth of the LWR kernel.
2. Observe how the LWR prediction changes based on the selected tau value.
3. Analyze the relationship between the original data points and the LWR prediction.

## Note
Local Weighted Regression is useful for capturing non-linear relationships in data and is particularly effective when dealing with noisy data or when the underlying relationship between variables is complex. However, selecting an appropriate tau value is crucial, as it directly impacts the smoothness of the fitted curve and the model's performance.

## Links:
- [Streamlit App](https://infosquad9.streamlit.app/)
- [Medium Article](https://medium.com/@pravinelango66/locally-weighted-regression-algorithm-51967dd21ccc)


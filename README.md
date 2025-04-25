# Weather-Forecasting

Weather forecasting with graph neural networks and convolutional neural networks

## Abstract

- Weather forecasting has a profound impact on everyday life and emergency planning. Traditional numerical weather prediction (NWP) models, such as those from ECMWF, involve solving complex and computationally expensive partial differential equations. Recently, machine learning models—particularly Graph Neural Networks (GNNs) and Convolutional Neural Networks (CNNs)—have emerged as compelling alternatives for spatio-temporal modeling, offering comparable accuracy with reduced computational overhead [1, 2, 3].

- In this project, using three-month ERA5 reanalysis dataset, we develop a GNN model to predict hourly surface temperatures (a regression problem) and a CNN model to predict raining or not (a classification problem) across the continental U.S. Although the primary targets are temperature (t2m) and total precipitation (tp), the models also leverage other variables including dew point, radiation, wind features et al. We found that the GNN model accurately predicted temperatures across the U.S. with low errors, outperforming the persistence baseline by leveraging spatio-temporal patterns in the ERA5 dataset. On the other hand, for rain prediction, the CNN model correctly predicts 77.41% of the rain events while the GNN model performs worse than random guess.

## Dataset

- The weather dataset from European Centre for Medium-Range Weather Forecasts (ECMWF)(https://cds.climate.copernicus.eu/) is spato-temporal. We restrict the spatial region to the US continent (excluding Hawaii and Alaska), the grid spacing is 0.25 degree. The temporal duration is from 01/01/2025 to 03/31/2025. 
Dimensions:  valid_time (2160), latitude (101), longitude (237)..

- Data Variables: tp (total precipitation), slhf (surface latent heat flux), sshf (surface sensible heat flux), ssrd (downward short-wave radiation), strd (downward long-wave radiation), u10 and v10 (10-meter wind components), d2m (2-meter dewpoint temperature), t2m (2-meter temperature), sp (surface pressure), tcc (total cloud cover), stl1 (soil temperature level), blh (boundary layer height),  q (specific humidity), t (temperature), msl (mean sea level pressure), tcwv (total column water vapor), cape (convective available potential energy).

## Model Architecture

### WeatherGNN
- Architecture: 3 Graph Convolutional Network layers (GCNConv) with residual connections and dropout (0.3).
- Dimensions: 15 input features → 128 → 128 → 1 output.
- Loss Function: L1 loss.
- Training: 35 epochs with early stopping, Adam optimizer, and learning rate scheduler.
- 
We employ iterative prediction, where the model’s own outputs are used as inputs for subsequent time steps. To benchmark our results, we compared our GNN’s performance to a persistence model, which assumes static weather conditions (i.e., tomorrow = today)

### CNN_RainPrediction
- Architecture: 3 layers of Conv3d with MaxPooling and Dropout.
- Dimensions: 10 input features, 128 hidden dimensions, 2 outputs.
- Loss Function: A combination of F1 loss, Focal loss and Cross Entropy loss..
- Training: 20 epochs with early stopping, Adam optimizer, and learning rate scheduler.
- 
We also compare rain prediction models based on graph neural networks.


## Key Results

### Temperature Prediction
- High temporal fidelity, capturing both short-term and evolving weather dynamics.
- Robust spatial generalization, adapting well across different U.S. regions.
- Superior to baseline, indicating value added by modeling spatiotemporal dependencies.
- Scalable design, ready for extension to additional variables or longer forecast horizons.

   
### Rain Prediction
- The GNN model gives very inaccurate rain predictions.  (F1: 0.2653, Accuracy: 0.8063, Precision: 0.2544, Recall: 0.2772)
- The CNN based model gives much better prediction scores. (F1: 0.7679, Accuracy: 0.9388, Precision: 0.7712, Recall: 0.7646)
- Reducing the window size only slightly improves the performance scores because fewer data points are labeled as raining with decreasing window size. 
- See the attached figures: confusion matrix for CNN model on test dataset and sample visualization of rain prediction on the spatial map. 
<img src="https://github.com/user-attachments/assets/029effd3-f226-46b2-9edb-c269603d114e" width = "600">
<img src="https://github.com/user-attachments/assets/e33de3ce-946f-452c-af6e-a6530846d4af" width = "800">

  
## References: 

- 1.Accurate medium-range global weather forecasting with 3D neural networks
 Bi, Kaifeng, et al. "Accurate medium-range global weather forecasting with 3D neural networks." Nature, vol. 619, 2023, pp. 533–538. https://www.nature.com/articles/s41586-023-06185-3 

- 2.Forecasting Global Weather with Graph Neural Networks
Ryan Keisler, “Forecasting Global Weather with Graph Neural Networks”, arXiv: 2202.07575v1. https://arxiv.org/abs/2202.07575

- 3.HiSTGNN: Hierarchical Spatio-temporal Graph Neural Networks for Weather Forecasting (arXiv)
 Wang, Zhaonan, et al. HiSTGNN: Hierarchical Spatio-Temporal Graph Neural Networks for Weather Forecasting. arXiv preprint arXiv:2301.10569, 2023. https://arxiv.org/abs/2301.10569.

🌲 Tree Detection from Hyperspectral Imagery Using Point-Based XGBoost in Python

📄 Overview:
This project presents a robust and efficient approach for tree canopy detection using high-resolution hyperspectral imagery and a point-based XGBoost classification workflow implemented in Python.

The method integrates the spectral richness of hyperspectral data (~1 m resolution) with the power of XGBoost to distinguish tree vs. non-tree classes with high accuracy. Training samples were manually digitized in QGIS and exported as CSV points, forming the foundation for model training and evaluation.

The complete pipeline includes data preprocessing, feature extraction, model training, raster classification, accuracy assessment, and export of final outputs in multiple formats.

📊 Project Details:
| Attribute           | Details                                                                     |
| ------------------- | --------------------------------------------------------------------------- |
| **Title**           | Tree Canopy Mapping from Hyperspectral Imagery Using Point-Based XGBoost    |
| **Imagery Source**  | NEON Surface Directional Reflectance (Hyperspectral)                        |
| **Platform**        | NEON AOP (Airborne Observation Platform)                                    |
| **Resolution**      | \~1 m                                                                       |
| **Year of Imagery** | 2014 (JERC Site)                                                            |
| **Sample Type**     | Point-based (Tree/Non-Tree), digitized in QGIS                              |
| **Model**           | XGBoost (Scikit-learn + XGBoost)                                            |
| **Programming**     | Python                                                                      |
| **Outputs**         | GeoTIFF (classified map), CSV (spectral signatures), Pickle (trained model) |
| **Validation**      | Accuracy, RMSE, R² Score, MBE, MAE                                          |

⚙️ Dependencies:
      Install via pip or conda:pip install geopandas rasterio xgboost scikit-learn numpy matplotlib shapely joblib pandas

| Library      | Purpose                                              |
| ------------ | ---------------------------------------------------- |
| geopandas    | Handling shapefiles & point datasets                 |
| rasterio     | Reading/writing raster imagery                       |
| scikit-learn | Preprocessing, metrics, evaluation                   |
| xgboost      | Training & prediction                                |
| numpy        | Numerical operations                                 |
| pandas       | Tabular dataset handling                             |
| matplotlib   | Visualization (classification maps, spectral curves) |
| shapely      | Geometry processing                                  |
| joblib       | Model persistence (save/load)                        |

🚀 Workflow:graph TD
A[🎯 Define Objective] --> B[🛰️ Load Hyperspectral Image]
B --> C[📉 Preprocess & Handle Nodata]
C --> D[🧭 Digitize Training Points in QGIS]
D --> E[💾 Export Points as CSV]
E --> F[📥 Extract Pixel Spectra]
F --> G[🧠 Train XGBoost Model + Tuning]
G --> H[🗺️ Classify Entire Image]
H --> I[🧹 Post-process & Validate]
I --> J[📤 Export GeoTIFF + CSV + Model]
J --> K[📊 Evaluate Accuracy & Metrics]

📌 Model Parameters:
| Parameter             | Range/Value | Description                   |
| --------------------- | ----------- | ----------------------------- |
| **n\_estimators**     | 1000–2000   | Number of boosting rounds     |
| **max\_depth**        | 8–12        | Maximum tree depth            |
| **learning\_rate**    | 0.01–0.2    | Step size shrinkage           |
| **subsample**         | 0.6–1.0     | Fraction of samples per tree  |
| **colsample\_bytree** | 0.6–1.0     | Fraction of features per tree |
| **random\_state**     | 42          | Reproducibility               |

📈 Evaluation Metrics:
| Metric   | Value  | Description                                                   |
| -------- | ------ | ------------------------------------------------------------- |
| Accuracy | 0.9355 | \~93.6% of pixels classified correctly                        |
| RMSE     | 0.20   | Average magnitude of error                                    |
| R² Score | \~0.80 | \~80% of variance explained                                   |
| MBE      | 0.032  | Slight positive bias (minor overestimation)                   |
| MAE      | 0.065  | Average absolute difference between prediction & ground truth |

📦 Final Outputs:
| Output Type        | Format  | Description                             |
| ------------------ | ------- | --------------------------------------- |
| **Classified Map** | GeoTIFF | Binary raster: 1 = Tree, 0 = Non-Tree   |
| **Spectral Data**  | CSV     | Reflectance values for training samples |
| **Trained Model**  | .pkl    | Serialized XGBoost model for inference  |

🌈 Why XGBoost + Hyperspectral Data?
      Hyperspectral Data:
        Hundreds of narrow spectral bands capture subtle surface properties
        Enables vegetation vs. non-vegetation separation with high precision
        ~1 m resolution supports fine-scale canopy mapping
      XGBoost Classifier:
        Handles high-dimensional data efficiently
        Built-in regularization prevents overfitting
        Performs implicit feature selection
        Scalable and faster than many traditional ML methods
      Combined Strength:
        Hyperspectral richness fused with XGBoost’s learning capacity yields a scalable, accurate, and computationally efficient solution for tree canopy mapping.

🌐 Data Source:
      Platform: NEON AOP (Airborne Observation Platform)
      Dataset: NEON Surface Directional Reflectance (NEON_D_HYPERSPECTRAL)
      Resolution: ~1m
      Year: 2014 (JERC Site)

📌 Conclusion:
      This project demonstrates the potential of combining hyperspectral imagery with XGBoost classification for tree canopy mapping.
      With carefully prepared reflectance datasets, labeled training samples, and optimized hyperparameters, we achieved:
      ✅ Accuracy: 93.55%
      ✅ RMSE: 0.20
      ✅ R²: ~0.80
      ✅ MBE: 0.032
      ✅ MAE: 0.065
      The resulting outputs—classified GeoTIFF, spectral dataset, and trained model—are applicable for ecological monitoring, forest inventory, and environmental change detection.

🌍 Applications:
      Vegetation / canopy cover mapping
      Land use / land cover (LULC) classification
      Crop health and stress monitoring
      Environmental change analysis

👤 Author: Subham Roy

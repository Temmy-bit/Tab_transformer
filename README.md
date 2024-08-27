# TabTransformer

TabTransformer is an extension of a project originally developed by Amazon, aimed at improving the efficacy of machine learning models on price prediction datasets using transformer encoders and various scaling techniques.

### Overview

This study focuses on transforming categorical features using encoders and scaling numerical features before feeding them into different machine-learning algorithms. Our best-performing model, Gradient Boosting with robust scaling, achieved an R2 score of 0.8879 and RMSE of 28026.68. Linear Regression with z-score scaling also showed good performance with an R2 score of 0.8188 and RMSE of 35629.

### Methodology

![Data Preparation](https://github.com/user-attachments/assets/96b503f9-0a43-4756-b5f1-eae991aea80e)

- **Datasets**: We used openly accessible datasets from Kaggle, including the Advanced Regression Techniques and Flight Price Prediction datasets.
- **Feature Processing**: Categorical features were encoded using transformer blocks, and numerical features were scaled using Min-Max, Z-Score, and Robust Scaling techniques.
- **Model Training**: Models were trained using various algorithms, including Random Forest, Gradient Boosting, XGBRegressor, LGBMRegressor, CatBoost, StackingCV, and Linear Regression.

### Results

The following table summarizes the performance of different models across the three scaling methods:

| **Model Name**      | **R2 Score (Min-Max Scaling)** | **MSE (Min-Max Scaling)** | **RMSE (Min-Max Scaling)** | **R2 Score (Z-Score Scaling)** | **MSE (Z-Score Scaling)** | **RMSE (Z-Score Scaling)** | **R2 Score (Robust Scaling)** | **MSE (Robust Scaling)** | **RMSE (Robust Scaling)** |
|---------------------|--------------------------------|---------------------------|----------------------------|--------------------------------|---------------------------|----------------------------|-------------------------------|--------------------------|---------------------------|
| RandomForest        | 0.8688                         | 9.19E+08                  | 30318.76                   | 0.8759                         | 8.70E+08                  | 29488.81                   | 0.8705                        | 9.07E+08                 | 30115.97                  |
| GradientBoost       | 0.8737                         | 8.85E+08                  | 29743.98                   | 0.8848                         | 8.07E+08                  | 28408.21                   | **0.8879**                   | **7.85E+08**             | **28026.68**              |
| XGBRegressor        | 0.8681                         | 9.24E+08                  | 30391.73                   | 0.8681                         | 9.24E+08                  | 30391.73                   | 0.8735                        | 8.86E+08                 | 29765.94                  |
| LGBMRegressor       | 0.8807                         | 8.36E+08                  | 28912.51                   | 0.8778                         | 8.56E+08                  | 29260.25                   | 0.8782                        | 8.53E+08                 | 29206.21                  |
| Catboost            | **0.8868**                     | **7.93E+08**              | **28160.37**               | **0.8868**                     | **7.93E+08**              | **28165.89**               | 0.8866                        | 7.94E+08                 | 28179.39                  |
| StackingCV          | 0.8791                         | 8.47E+08                  | 29101.01                   | 0.8728                         | 8.91E+08                  | 29848.30                   | 0.8808                        | 8.35E+08                 | 28902.57                  |
| LinearRegression    | 0.8190                         | 1.27E+09                  | 35608.49                   | 0.8188                         | 1.27E+09                  | 35629.00                   | 0.8193                        | 1.27E+09                 | 35579.53                  |

### Conclusion

Gradient Boosting with robust scaling showed the best performance in this study, making it a strong choice for price prediction tasks. Other models like CatBoost and LGBMRegressor also performed well, particularly with different scaling methods.

# Crop Yield Prediction Model

This repository contains a machine learning pipeline for predicting crop yield based on various agricultural and environmental features. The model utilizes feature engineering techniques to enhance prediction accuracy and employs a Stacking Regressor for robust ensemble learning.

## Project Overview

The main objective is to build a predictive model that can accurately estimate crop yield based on key features. The dataset includes information on temperature ranges, pollinator counts, rainfall effects, and yield-related characteristics.

### Key Steps

1. **Data Preprocessing**: 
    - Remove unnecessary columns (`id`, `Row#`).
    - Perform feature engineering to create derived features like `SeedToFruitMassRatio`, `BeeActivityIndex`, `RainImpactFactor`, and `CloneSizeFruitMass`.

2. **Modeling**:
    - Use a stacking approach with models such as Ridge, Lasso, DecisionTreeRegressor, BaggingRegressor, and RandomForestRegressor.
    - Apply feature selection using `RFECV` and scaling using `StandardScaler` within a `Pipeline`.

3. **Training and Validation**:
    - The model is trained using K-Fold cross-validation.
    - ROC-AUC scores are calculated for training and validation sets to assess model performance.

4. **Prediction and Submission**:
    - Predictions are generated for the test dataset.
    - Results are saved as a submission file for evaluation.

## Requirements

- Python 3.x
- Scikit-learn, Pandas, Numpy, Seaborn, Matplotlib

## Installation

Clone the repository and install dependencies:
```bash
git clone https://github.com/yourusername/crop-yield-prediction.git
cd crop-yield-prediction
pip install -r requirements.txt
```

## Usage

To train the model and make predictions:
1. Place the training dataset (`train.csv`) in the project folder.
2. Run the main script to train the model and generate predictions:
   ```bash
   python main.py
   ```
3. The submission file (`Haqnazar_submission_colab28.csv`) will be generated for evaluation.

## Results and Evaluation

ROC-AUC and MAE scores are used to measure performance. Feature importances are visualized to understand which variables contribute most to yield prediction.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Sample License File (LICENSE)

```plaintext
MIT License

Copyright (c) 2024 Eshonqulov Haqnazar

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

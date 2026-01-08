# Diabetes Prediction Solution - Playground S5E12

## Questions
[Playground S5E12](https://www.kaggle.com/competitions/playground-series-s5e12/overview)


## Prerequisites
1. **Data**: You need `train.csv`, `test.csv`, and `sample_submission.csv` in the `data/` directory.
   - If you have `kaggle.json` set up:
     ```bash
     kaggle competitions download -c playground-series-s5e12
     unzip playground-series-s5e12.zip -d data
     ```

2. **Environment**: Use the `hello-env` conda environment.
   ```bash
   conda activate hello-env
   ```
   Dependencies: `pandas`, `numpy`, `xgboost`, `scikit-learn`, `matplotlib`.

## Running the Solution
Run the script to train the XGBoost model and generate a submission:
```bash
python solve_diabetes.py
```
This will train the model using 5-fold cross-validation.

## Output
- `submission.csv`: The file to upload to Kaggle (generated in the root directory).
- Console output: Model CV scores for each fold.

## Approach
- **Model**: XGBoost Classifier.
- **Validation**: Stratified K-Fold Cross-Validation.
- **Metric**: ROC AUC.

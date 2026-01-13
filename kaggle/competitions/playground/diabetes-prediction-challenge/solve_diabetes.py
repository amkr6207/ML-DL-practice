import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import os

def main():
    # 1. Load Data
    print("Loading data...")
    if not os.path.exists('data/train.csv') or not os.path.exists('data/test.csv'):
        print("Error: train.csv or test.csv not found in data/ directory. Please ensure dataset is downloaded and unzipped into 'data/'.")
        return

    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    submission = pd.read_csv('data/sample_submission.csv')

    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")

    # 2. Preprocessing
    print("Preprocessing...")
    target = 'diagnosed_diabetes'
    drop_cols = ['id', target]
    
    # Identify features
    features = [c for c in train.columns if c not in drop_cols]
    
    # Combine for consistent encoding
    train['is_train'] = 1
    test['is_train'] = 0
    df = pd.concat([train.drop(columns=[target]), test], axis=0).reset_index(drop=True)
    
    # Handle Categoricals (Label Encoding)
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = df[col].astype(str)
        df[col] = le.fit_transform(df[col])
        
    # Split back
    X = df[df['is_train'] == 1].drop(columns=['is_train'])
    X_test = df[df['is_train'] == 0].drop(columns=['is_train'])
    y = train[target]
    
    # 3. Model Training (XGBoost)
    print("Training Model...")
    folds = 5
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    
    # XGBoost Parameters
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'learning_rate': 0.05,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'n_estimators': 2000,
        'random_state': 42,
        'n_jobs': -1,
        'device': 'cpu'
    }

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        clf = xgb.XGBClassifier(**params, early_stopping_rounds=100)
        
        clf.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=100
        )
        
        val_preds = clf.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_preds
        
        fold_auc = roc_auc_score(y_val, val_preds)
        print(f"Fold {fold+1} AUC: {fold_auc:.5f}")
        
        test_preds += clf.predict_proba(X_test)[:, 1] / folds

    # 4. Evaluation
    overall_auc = roc_auc_score(y, oof_preds)
    print(f"\nOverall CV AUC: {overall_auc:.5f}")
    
    # 5. Submission
    submission[target] = test_preds
    submission.to_csv('submission.csv', index=False)
    print("Submission saved to submission.csv")

if __name__ == "__main__":
    main()

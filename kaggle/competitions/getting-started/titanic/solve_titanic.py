import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import os
import warnings

warnings.filterwarnings('ignore')

def main():
    # 1. Load Data
    print("Loading data...")
    if not os.path.exists('data/train.csv') or not os.path.exists('data/test.csv'):
        print("Error: train.csv or test.csv not found in data/ directory.")
        return

    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    submission = pd.read_csv('data/gender_submission.csv')

    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")

    # 2. Preprocessing & Feature Engineering
    print("Preprocessing...")
    
    # Store PassengerId for submission
    test_ids = test['PassengerId']
    
    target = 'Survived'
    drop_cols = ['cv_fold_id'] # placeholder
    
    # Combine for consistent processing
    train['is_train'] = 1
    test['is_train'] = 0
    test[target] = np.nan
    
    df = pd.concat([train, test], axis=0).reset_index(drop=True)
    
    # Impute Missing Values
    df['Age'] = df['Age'].fillna(df.groupby(['Pclass', 'Sex'])['Age'].transform('median'))
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    
    # Feature Engineering
    # Title Extraction
    df['Title'] = df['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
    # Group rare titles
    rare_titles = ['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
    df['Title'] = df['Title'].replace(rare_titles, 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    
    # Family Size
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = 1
    df['IsAlone'].loc[df['FamilySize'] > 1] = 0
    
    # Drop columns that are difficult to use directly
    cols_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    df = df.drop(columns=cols_to_drop)
    
    # Encoding
    cat_cols = ['Sex', 'Embarked', 'Title']
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        
    # Split back
    X = df[df['is_train'] == 1].drop(columns=['is_train', target])
    y = df[df['is_train'] == 1][target].astype(int)
    X_test = df[df['is_train'] == 0].drop(columns=['is_train', target])
    
    print(f"Features: {X.columns.tolist()}")
    
    # 3. Model Training (XGBoost)
    print("Training Model...")
    folds = 5
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    
    oof_preds = np.zeros(len(X))
    test_preds = [] # List to store preds from each fold
    
    # XGBoost Parameters (Tuned for small data)
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss', # Logloss is better for probas, but we check accuracy
        'learning_rate': 0.05,
        'max_depth': 4,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'n_estimators': 1000,
        'random_state': 42,
        'n_jobs': -1
    }

    fold_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        clf = xgb.XGBClassifier(**params, early_stopping_rounds=100)
        
        clf.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Predict Proba for OOF
        val_probs = clf.predict_proba(X_val)[:, 1]
        val_preds_binary = (val_probs > 0.5).astype(int)
        
        acc = accuracy_score(y_val, val_preds_binary)
        fold_accuracies.append(acc)
        print(f"Fold {fold+1} Accuracy: {acc:.5f}")
        
        # Test predictions (Probability)
        test_preds.append(clf.predict_proba(X_test)[:, 1])

    # 4. Evaluation
    avg_acc = np.mean(fold_accuracies)
    print(f"\nOverall CV Accuracy: {avg_acc:.5f}")
    
    # Average test probabilities
    avg_test_probs = np.mean(test_preds, axis=0)
    final_preds = (avg_test_probs > 0.5).astype(int)
    
    # 5. Submission
    submission = pd.DataFrame({
        'PassengerId': test_ids,
        'Survived': final_preds
    })
    
    submission.to_csv('submission.csv', index=False)
    print("Submission saved to submission.csv")

if __name__ == "__main__":
    main()

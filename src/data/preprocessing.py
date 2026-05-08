from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

import pandas as pd

def clean_data(df):

    df = df.copy()
    
    df.drop(columns=['Unnamed: 0'], inplace=True)
    
    df.columns = [
    'Limit_Bal',
    'Gender',
    'Education',
    'Marriage',
    'Age',
    'RPay_stat_sep',
    'RPay_stat_aug',
    'RPay_stat_jul',
    'RPay_stat_jun',
    'RPay_stat_may',
    'RPay_stat_apr',
    'Bill_amt_sep',
    'Bill_amt_aug',
    'Bill_amt_jul',
    'Bill_amt_jun',
    'Bill_amt_may',
    'Bill_amt_apr',
    'Pay_amt_sep',
    'Pay_amt_aug',
    'Pay_amt_jul',
    'Pay_amt_jun',
    'Pay_amt_may',
    'Pay_amt_apr',
    'Default_Payment'
    ]
    
    df = df.iloc[1:]  # Keeps all rows except index 0
    
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    
    df['Education'] = df['Education'].replace([0,5,6],4)

    df['Marriage'] = df['Marriage'].replace(0,3)

    return df



def split_features_target(df):

    X = df.drop('Default_Payment', axis=1)

    y = df['Default_Payment']

    return X, y



def split_data(X, y):

    return train_test_split(

        X,

        y,

        test_size=0.2,

        random_state=42,

        stratify=y
    )



def scale_data(X_train, X_test):

    scaler = RobustScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    return (
        X_train_scaled,
        X_test_scaled,
        scaler
    )
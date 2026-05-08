import pandas as pd



def create_engineered_features(df):

    df_engineered = df.copy()

    bill_cols = [
    'Bill_amt_sep',
    'Bill_amt_aug',
    'Bill_amt_jul',
    'Bill_amt_jun',
    'Bill_amt_may',
    'Bill_amt_apr'
    ]

    payment_cols = [
    'Pay_amt_sep',
    'Pay_amt_aug',
    'Pay_amt_jul',
    'Pay_amt_jun',
    'Pay_amt_may',
    'Pay_amt_apr'
    ]

    repayment_cols = [
    'RPay_stat_sep',
    'RPay_stat_aug',
    'RPay_stat_jul',
    'RPay_stat_jun',
    'RPay_stat_may',
    'RPay_stat_apr'
    ]

    df_engineered['Avg_Repayment_Status'] = (
        df_engineered[repayment_cols]
        .mean(axis=1)
    )


    df_engineered['Max_Repayment_Status'] = (
        df_engineered[repayment_cols]
        .max(axis=1)
    )

    df_engineered['Delayed_Months_Count'] = (
        df_engineered[repayment_cols] > 0
        ).sum(axis=1)
    
    df_engineered['Payment_Ratio'] = (
        df_engineered['Total_Payment_Amount'] /
        (df_engineered['Total_Bill_Amount'] + 1)
    )


    df_engineered['Credit_Utilization_Ratio'] = (
        df_engineered['Total_Bill_Amount'] /
        (df_engineered['Limit_Bal'] + 1)
    )
    
    df_engineered['Med_Payment_Amount'] = (
        df_engineered[payment_cols]
        .median(axis=1)
    )


    df_engineered['Total_Payment_Amount'] = (
        df_engineered[payment_cols]
        .sum(axis=1)
    )
    
    df_engineered['Med_Bill_Amount'] = (
        df_engineered[bill_cols]
        .median(axis=1)
    )


    df_engineered['Total_Bill_Amount'] = (
        df_engineered[bill_cols]
        .sum(axis=1)
    )

    df_engineered.drop(
        columns=(
            bill_cols +
            payment_cols +
            repayment_cols
        ),
        inplace=True
    )

    return df_engineered
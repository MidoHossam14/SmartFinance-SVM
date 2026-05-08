import streamlit as st

from preprocess_input import (
    preprocess_user_input
)

from predict import (
    load_model,
    predict_default
)

from utils import (
    interpret_prediction
)


# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(

    page_title='Credit Default Prediction',

    page_icon='💳',

    layout='centered'
)


# ==========================================
# Load Trained Baseline Pipeline
# ==========================================

model = load_model(
    '../artifacts/tuned_baseline_pipeline.pkl'
)


# ==========================================
# Title
# ==========================================

st.title('💳 Credit Default Prediction System')

st.markdown(
    'Predict whether a customer may default '
    'on the next credit card payment.'
)

st.divider()


# ==========================================
# User Inputs
# ==========================================

st.subheader('Customer Financial Information')

limit_balance = st.number_input(
    'Credit Limit',
    min_value=0,
    value=50000
)

sex = st.selectbox(
    'Gender',
    [1, 2],
    help='1 = Male, 2 = Female'
)

education = st.selectbox(
    'Education Level',
    [1, 2, 3, 4],
    help='1 = Graduate School, 2 = University, 3 = High School, 4 = Others'
)

marriage = st.selectbox(
    'Marital Status',
    [1, 2, 3],
    help='1 = Married, 2 = Single, 3 = Others'
)

age = st.number_input(
    'Age',
    min_value=18,
    max_value=100,
    value=30
)


# ==========================================
# Repayment Status Inputs
# ==========================================

st.subheader('Repayment Status History')

repay_sep = st.slider(
    'Repayment Status September',
    min_value=-2,
    max_value=8,
    value=0
)

repay_aug = st.slider(
    'Repayment Status August',
    min_value=-2,
    max_value=8,
    value=0
)

repay_jul = st.slider(
    'Repayment Status July',
    min_value=-2,
    max_value=8,
    value=0
)

repay_jun = st.slider(
    'Repayment Status June',
    min_value=-2,
    max_value=8,
    value=0
)

repay_may = st.slider(
    'Repayment Status May',
    min_value=-2,
    max_value=8,
    value=0
)

repay_apr = st.slider(
    'Repayment Status April',
    min_value=-2,
    max_value=8,
    value=0
)


# ==========================================
# Bill Amount Inputs
# ==========================================

st.subheader('Bill Statement Amounts')

bill_amt1 = st.number_input('Bill Amount September', value=5000)
bill_amt2 = st.number_input('Bill Amount August', value=5000)
bill_amt3 = st.number_input('Bill Amount July', value=5000)
bill_amt4 = st.number_input('Bill Amount June', value=5000)
bill_amt5 = st.number_input('Bill Amount May', value=5000)
bill_amt6 = st.number_input('Bill Amount April', value=5000)


# ==========================================
# Payment Amount Inputs
# ==========================================

st.subheader('Previous Payment Amounts')

pay_amt1 = st.number_input('Payment Amount September', value=2000)
pay_amt2 = st.number_input('Payment Amount August', value=2000)
pay_amt3 = st.number_input('Payment Amount July', value=2000)
pay_amt4 = st.number_input('Payment Amount June', value=2000)
pay_amt5 = st.number_input('Payment Amount May', value=2000)
pay_amt6 = st.number_input('Payment Amount April', value=2000)


# ==========================================
# Prediction Button
# ==========================================

if st.button('Predict Default Risk'):

    user_data = {

        'LIMIT_BAL': limit_balance,

        'SEX': sex,

        'EDUCATION': education,

        'MARRIAGE': marriage,

        'AGE': age,

        'Repay_Sep': repay_sep,

        'Repay_Aug': repay_aug,

        'Repay_Jul': repay_jul,

        'Repay_Jun': repay_jun,

        'Repay_May': repay_may,

        'Repay_Apr': repay_apr,

        'Bill_amt1': bill_amt1,

        'Bill_amt2': bill_amt2,

        'Bill_amt3': bill_amt3,

        'Bill_amt4': bill_amt4,

        'Bill_amt5': bill_amt5,

        'Bill_amt6': bill_amt6,

        'Pay_amt1': pay_amt1,

        'Pay_amt2': pay_amt2,

        'Pay_amt3': pay_amt3,

        'Pay_amt4': pay_amt4,

        'Pay_amt5': pay_amt5,

        'Pay_amt6': pay_amt6
    }

    input_df = preprocess_user_input(
        user_data
    )

    prediction = predict_default(

        model,

        input_df
    )

    title, description = interpret_prediction(
        prediction
    )

    st.divider()

    st.subheader('Prediction Result')

    if prediction == 1:

        st.error(title)

    else:

        st.success(title)

    st.write(description)
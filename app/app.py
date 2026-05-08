import pandas as pd
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

    layout='wide'
)


# ==========================================
# Custom Styling
# ==========================================

st.markdown(
    '''
    <style>

    .main {
        background-color: #0E1117;
    }

    h1, h2, h3 {
        color: white;
    }

    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #2563EB;
        color: white;
        font-size: 16px;
        font-weight: bold;
    }

    .stMetric {
        background-color: #111827;
        padding: 15px;
        border-radius: 12px;
    }

    </style>
    ''',
    unsafe_allow_html=True
)


# ==========================================
# Load Model
# ==========================================

model = load_model(
    '../artifacts/tuned_baseline_pipeline.pkl'
)


# ==========================================
# Header
# ==========================================

st.title('💳 Credit Default Prediction System')

st.markdown(
    'Professional Machine Learning System for '
    'Predicting Credit Card Default Risk'
)

st.divider()


# ==========================================
# Tabs
# ==========================================

single_tab, batch_tab = st.tabs([
    'Single Prediction',
    'Batch Prediction'
])


# ==========================================
# Single Prediction Tab
# ==========================================

with single_tab:

    st.subheader('Customer Financial Information')

    col1, col2, col3 = st.columns(3)

    with col1:

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
            'Education',
            [1, 2, 3, 4],
            help='1 = Graduate School, 2 = University, 3 = High School, 4 = Others'
        )

        marriage = st.selectbox(
            'Marriage',
            [1, 2, 3],
            help='1 = Married, 2 = Single, 3 = Others'
        )

        age = st.number_input(
            'Age',
            min_value=18,
            max_value=100,
            value=30
        )


    with col2:

        st.markdown('### Repayment Status')

        repay_sep = st.slider(
            'September',
            -2,
            8,
            0
        )

        repay_aug = st.slider(
            'August',
            -2,
            8,
            0
        )

        repay_jul = st.slider(
            'July',
            -2,
            8,
            0
        )

        repay_jun = st.slider(
            'June',
            -2,
            8,
            0
        )

        repay_may = st.slider(
            'May',
            -2,
            8,
            0
        )

        repay_apr = st.slider(
            'April',
            -2,
            8,
            0
        )


    with col3:

        st.markdown('### Bill Amounts')

        bill_amt1 = st.number_input(
            'Bill September',
            value=5000
        )

        bill_amt2 = st.number_input(
            'Bill August',
            value=5000
        )

        bill_amt3 = st.number_input(
            'Bill July',
            value=5000
        )

        bill_amt4 = st.number_input(
            'Bill June',
            value=5000
        )

        bill_amt5 = st.number_input(
            'Bill May',
            value=5000
        )

        bill_amt6 = st.number_input(
            'Bill April',
            value=5000
        )


    st.markdown('### Payment Amounts')

    pay_col1, pay_col2, pay_col3 = st.columns(3)

    with pay_col1:

        pay_amt1 = st.number_input(
            'Payment September',
            value=2000
        )

        pay_amt2 = st.number_input(
            'Payment August',
            value=2000
        )

    with pay_col2:

        pay_amt3 = st.number_input(
            'Payment July',
            value=2000
        )

        pay_amt4 = st.number_input(
            'Payment June',
            value=2000
        )

    with pay_col3:

        pay_amt5 = st.number_input(
            'Payment May',
            value=2000
        )

        pay_amt6 = st.number_input(
            'Payment April',
            value=2000
        )


    if st.button('Predict Default Risk'):

        user_data = {

            'Limit_Bal': limit_balance,
            'Sex': sex,
            'Education': education,
            'Marriage': marriage,
            'Age': age,

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

        if prediction == 1:

            st.error(title)

        else:

            st.success(title)

        st.info(description)


# ==========================================
# Batch Prediction Tab
# ==========================================

with batch_tab:

    st.subheader('Batch Prediction Using CSV File')

    st.markdown(
        'Upload a dataset containing the same '
        'features used during model training.'
    )

    uploaded_file = st.file_uploader(
        'Upload CSV File',
        type=['csv']
    )

    if uploaded_file is not None:

        uploaded_df = pd.read_csv(
            uploaded_file
        )

        predictions = model.predict(
            uploaded_df
        )

        uploaded_df['Prediction'] = predictions

        uploaded_df['Risk_Label'] = (
            uploaded_df['Prediction']
            .map({
                0: 'Low Risk',
                1: 'High Risk'
            })
        )

        st.success('Predictions Completed Successfully')

        metric_col1, metric_col2 = st.columns(2)

        with metric_col1:

            st.metric(
                'High Risk Customers',
                (predictions == 1).sum()
            )

        with metric_col2:

            st.metric(
                'Low Risk Customers',
                (predictions == 0).sum()
            )

        st.dataframe(
            uploaded_df,
            use_container_width=True
        )

        csv_file = uploaded_df.to_csv(
            index=False
        ).encode('utf-8')

        st.download_button(

            label='Download Predictions CSV',

            data=csv_file,

            file_name='credit_default_predictions.csv',

            mime='text/csv'
        )
import joblib



def load_model(model_path):

    model = joblib.load(model_path)

    return model



def predict_default(model, input_df):

    prediction = model.predict(input_df)[0]

    return prediction
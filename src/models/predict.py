import joblib



def load_pipeline(path):

    return joblib.load(path)



def predict_new_data(
    pipeline,
    new_data
):

    predictions = pipeline.predict(new_data)

    return predictions
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC



def build_engineered_pipeline():

    pipeline = Pipeline([

        ('scaler', RobustScaler()),

        ('svm', SVC())
    ])

    return pipeline
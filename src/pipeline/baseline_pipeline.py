from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC



def build_baseline_pipeline():

    pipeline = Pipeline([

        ('scaler', RobustScaler()),

        ('svm', SVC())
    ])

    return pipeline
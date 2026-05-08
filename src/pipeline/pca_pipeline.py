from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC



def build_pca_pipeline():

    pipeline = Pipeline([

        ('scaler', RobustScaler()),

        ('pca', PCA()),

        ('svm', SVC())
    ])

    return pipeline
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold
)



def train_model(
    pipeline,
    param_grid,
    X_train,
    y_train
):

    cv_strategy = StratifiedKFold(

        n_splits=3,

        shuffle=True,

        random_state=42
    )

    grid_search = GridSearchCV(

        estimator=pipeline,

        param_grid=param_grid,

        scoring='f1',

        cv=cv_strategy,

        n_jobs=-1,

        verbose=1
    )

    grid_search.fit(
        X_train,
        y_train
    )

    return grid_search
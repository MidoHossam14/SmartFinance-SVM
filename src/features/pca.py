from sklearn.decomposition import PCA



def apply_pca(
    X_train_scaled,
    X_test_scaled,
    n_components=0.95
):

    pca = PCA(
        n_components=n_components,
        random_state=42
    )

    X_train_pca = pca.fit_transform(
        X_train_scaled
    )

    X_test_pca = pca.transform(
        X_test_scaled
    )

    return (
        X_train_pca,
        X_test_pca,
        pca
    )
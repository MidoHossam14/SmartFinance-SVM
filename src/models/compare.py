import pandas as pd



def compare_models(
    baseline_metrics,
    pca_metrics,
    engineered_metrics
):

    comparison_df = pd.DataFrame({

        'Model': [

            'Baseline SVM',

            'PCA + SVM',

            'Engineered Features + SVM'
        ],

        'Accuracy': [

            baseline_metrics['Accuracy'],

            pca_metrics['Accuracy'],

            engineered_metrics['Accuracy']
        ],

        'Precision': [

            baseline_metrics['Precision'],

            pca_metrics['Precision'],

            engineered_metrics['Precision']
        ],

        'Recall': [

            baseline_metrics['Recall'],

            pca_metrics['Recall'],

            engineered_metrics['Recall']
        ],

        'F1_Score': [

            baseline_metrics['F1_Score'],

            pca_metrics['F1_Score'],

            engineered_metrics['F1_Score']
        ]
    })

    return comparison_df
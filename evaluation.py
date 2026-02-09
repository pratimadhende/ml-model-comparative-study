from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
import pandas as pd

def evaluate_models(models, X, y, preprocessor):
    """
    Evaluate models using 5-fold cross-validation.

    Metrics:
    - Accuracy
    - Precision
    - Recall
    - F1-score

    Mean and standard deviation describe performance consistency.
    """

    scoring = ["accuracy", "precision", "recall", "f1"]
    results = []

    for name, model in models.items():
        pipeline = Pipeline([
            ("preprocessing", preprocessor),
            ("model", model)
        ])

        cv_results = cross_validate(
            pipeline,
            X,
            y,
            cv=5,
            scoring=scoring
        )

        results.append({
            "Model": name,
            "Accuracy (Mean)": cv_results["test_accuracy"].mean(),
            "F1-score (Mean)": cv_results["test_f1"].mean(),
            "F1-score (Std)": cv_results["test_f1"].std()
        })

    return pd.DataFrame(results)
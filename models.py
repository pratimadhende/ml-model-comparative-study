from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

def get_models():
    """
    Define models for comparative study.

    - Logistic Regression: linear baseline
    - SVM: margin-based model
    - Random Forest: non-linear ensemble
    """

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Support Vector Machine": SVC(kernel="rbf"),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
    }

    return models
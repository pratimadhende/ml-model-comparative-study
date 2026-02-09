from data_loading import load_dataset
from preprocessing import preprocess_data
from models import get_models
from evaluation import evaluate_models
from visualization import plot_model_performance 

# Load dataset (relative path)
path = "Telco_Churn.csv"
df = load_dataset(path)

# Preprocess
X, y, preprocessor = preprocess_data(df)

# Models
models = get_models()

# Evaluation
results = evaluate_models(models, X, y, preprocessor)

print("\nModel Comparison Results:\n")
print(results)

# Visualization (single plot)
plot_model_performance(results)
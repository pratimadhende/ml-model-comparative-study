import matplotlib.pyplot as plt
import os 

def plot_model_performance(results_df):
    """
    Plot mean F1-score for each model.

    A single visualization is used to support comparison
    without adding unnecessary graphical complexity.

    model comparison figure is automatically saved
    """

    models = results_df["Model"]
    f1_scores = results_df["F1-score (Mean)"]

    #create results dictionary if it does not exist
    os.makedirs("results",exist_ok=True)

    plt.figure()
    plt.bar(models, f1_scores)
    plt.xlabel("Model")
    plt.ylabel("Mean F1-score")
    plt.title("Model Comparison based on Mean F1-score")
    plt.xticks(rotation=20)
    plt.tight_layout()

    #save figure
    save_path="results\model_f1_comparison.png"
    plt.savefig(save_path)

    #show plot 
    plt.show()

    print(f"figure saved at: {save_path}")
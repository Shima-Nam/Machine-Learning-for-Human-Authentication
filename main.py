import os
import pickle
from src.data_loader import load_and_prepare_data
from src.feature_selection import select_features_dendrogram
from src.model_pipeline import run_pipeline
from src.visualization import plot_accuracy_vs_features, plot_best_accuracy_vs_test_size

DATA_PATH = "data/features.csv"
RESULTS_PATH = "results/results.pkl"

# -----------------------------------------------------------------------------
# Function: main
# Main entry point for the pipeline: loads data, selects features, runs the
# model pipeline, saves and loads results, and visualizes the results.
# -----------------------------------------------------------------------------
def main():
    print("Loading and preparing data...")
    # Data validation: check if data file exists
    if not os.path.isfile(DATA_PATH):
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")
    df = load_and_prepare_data(DATA_PATH)
    print("Data loaded.")

    print("Selecting features...")
    selected_features_names, scaler = select_features_dendrogram(df, plot=False)
    print(f"Selected {len(selected_features_names)} features.")

    print("Preparing data for pipeline...")
    # Data validation: check if selected features exist in DataFrame
    missing_features = [f for f in selected_features_names if f not in df.columns]
    if missing_features:
        raise ValueError(f"The following selected features are missing in the DataFrame: {missing_features}")
    X_sel = df[selected_features_names].copy()
    X_sel['Subject_ID'] = df.Subject_ID
    feature_nb = [10, 20, 31]
    if any(nb > len(selected_features_names) for nb in feature_nb):
        raise ValueError("feature_nb contains a value greater than the number of selected features.")

    print("Running model pipeline (this may take a while)...")
    results = run_pipeline(X_sel, scaler, feature_nb=feature_nb)
    print("Pipeline finished.")

    # Ensure results directory exists
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    print("Saving results...")
    with open(RESULTS_PATH, "wb") as f:
        pickle.dump(results, f)
    print("Results saved.")

    print("Loading results for visualization...")
    if not os.path.isfile(RESULTS_PATH):
        raise FileNotFoundError(f"Results file not found: {RESULTS_PATH}")
    with open(RESULTS_PATH, "rb") as f:
        loaded_results = pickle.load(f)

    test_sizes = [0.6, 0.4, 0.2]
    print("Plotting accuracy vs features...")
    plot_accuracy_vs_features(loaded_results, feature_nb, test_sizes)
    print("Plotting best accuracy vs test size...")
    plot_best_accuracy_vs_test_size(loaded_results, feature_nb)
    print("All done!")

if __name__ == "__main__":
    main()
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Function: plot_accuracy_vs_features
# Plots accuracy as a function of the number of features for each test size.
#
# Args:
#   loaded_results (dict): Results dictionary from the pipeline.
#   feature_nb (list): List of feature counts used in the experiment.
#   test_sizes (list): List of test set sizes corresponding to the results.
# -----------------------------------------------------------------------------
def plot_accuracy_vs_features(loaded_results, feature_nb, test_sizes):
    # Data validation: check input types and lengths
    if not isinstance(loaded_results, dict):
        raise ValueError("loaded_results must be a dictionary.")
    if not isinstance(feature_nb, list) or not feature_nb:
        raise ValueError("feature_nb must be a non-empty list.")
    if not isinstance(test_sizes, list) or not test_sizes:
        raise ValueError("test_sizes must be a non-empty list.")
    if len(loaded_results) != len(test_sizes):
        raise ValueError("Length of loaded_results and test_sizes must match.")

    # Iterate over each test size and its corresponding results
    for idx, (test_size, subsets) in enumerate(loaded_results.items()):
        feature_counts = []
        accuracies = []
        # Collect accuracy for each feature subset
        for feature_idx, data in subsets.items():
            if feature_idx >= len(feature_nb):
                raise IndexError(f"feature_idx {feature_idx} out of range for feature_nb.")
            if 'test_score' not in data:
                raise KeyError(f"'test_score' not found in results for feature_idx {feature_idx}.")
            feature_counts.append(feature_nb[feature_idx])
            accuracies.append(data['test_score'])
        # Plot accuracy vs number of features for this test size
        plt.plot(feature_counts, accuracies, marker='o', label=f"Test size: {test_sizes[idx]}")
    plt.xlabel("Number of Features")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Number of Features")
    plt.legend()
    plt.grid(True)
    plt.show()

# -----------------------------------------------------------------------------
# Function: plot_best_accuracy_vs_test_size
# Plots the best accuracy achieved for each test size.
#
# Args:
#   loaded_results (dict): Results dictionary from the pipeline.
#   feature_nb (list): List of feature counts used in the experiment.
# -----------------------------------------------------------------------------
def plot_best_accuracy_vs_test_size(loaded_results, feature_nb):
    # Data validation: check input types and lengths
    if not isinstance(loaded_results, dict):
        raise ValueError("loaded_results must be a dictionary.")
    if not isinstance(feature_nb, list) or not feature_nb:
        raise ValueError("feature_nb must be a non-empty list.")

    test_sizes = []
    best_accuracies = []
    # Iterate over each test size and find the best accuracy among feature subsets
    for test_size, subsets in loaded_results.items():
        if not isinstance(subsets, dict) or not subsets:
            raise ValueError("Each value in loaded_results must be a non-empty dictionary.")
        test_sizes.append(test_size)
        # Find the best accuracy among all feature subsets for this test size
        best_acc = None
        for feature_idx in subsets:
            if feature_idx >= len(feature_nb):
                raise IndexError(f"feature_idx {feature_idx} out of range for feature_nb.")
            if 'test_score' not in subsets[feature_idx]:
                raise KeyError(f"'test_score' not found in results for feature_idx {feature_idx}.")
            acc = subsets[feature_idx]['test_score']
            if best_acc is None or acc > best_acc:
                best_acc = acc
        if best_acc is None:
            raise ValueError(f"No valid accuracy found for test_size {test_size}.")
        best_accuracies.append(best_acc)
    # Plot best accuracy vs test size
    plt.plot(test_sizes, best_accuracies, marker='s', linestyle='-', color='r')
    plt.xticks(range(len(test_sizes)), labels=[str(ts) for ts in test_sizes])
    plt.xlabel("Test Size")
    plt.ylabel("Best Accuracy")
    plt.title("Accuracy vs Test Size")
    plt.grid(True)
    plt.show()
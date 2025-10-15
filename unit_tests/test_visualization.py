import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest
from src.visualization import plot_accuracy_vs_features, plot_best_accuracy_vs_test_size

def test_plot_accuracy_input_validation():
    with pytest.raises(ValueError):
        plot_accuracy_vs_features("not a dict", [10,20], [0.6])

def test_plot_calls_show(monkeypatch):
    called = {"show": False}
    def fake_show():
        called["show"] = True
    monkeypatch.setattr(plt, "show", fake_show)
    # simple fake results dict
    fake = {0: {0: {"test_score": 0.5}}}
    plot_accuracy_vs_features(fake, [10], [0.6])
    plot_best_accuracy_vs_test_size(fake, [10])
    assert called["show"] is True
"""Diagnostics for chronological overfitting and underfitting."""
from __future__ import annotations


def diagnose_fit(train: dict, validation: dict, test: dict) -> dict:
    """Report generalization gaps without treating one season as proof."""
    train_accuracy = float(train.get("winner_accuracy", 0.0))
    validation_accuracy = float(validation.get("winner_accuracy", 0.0))
    test_accuracy = float(test.get("winner_accuracy", 0.0))
    gap = train_accuracy - validation_accuracy
    return {
        "train_validation_accuracy_gap": gap,
        "validation_test_accuracy_gap": validation_accuracy - test_accuracy,
        "overfit_warning": gap >= 0.20,
        "underfit_warning": train_accuracy < 0.55,
        "generalization_warning": abs(validation_accuracy - test_accuracy) >= 0.20,
    }

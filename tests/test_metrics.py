import math

import numpy as np

from src.evaluation.metrics import classification_metrics


def test_known_confusion_matrix():
    # 4 malignant (3 caught), 6 benign (1 false alarm)
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    p = np.array([.9, .8, .7, .2, .6, .1, .1, .3, .4, .2])
    m = classification_metrics(y, p)
    assert (m["tp"], m["fn"], m["tn"], m["fp"]) == (3, 1, 5, 1)
    assert m["recall"] == 3 / 4
    assert m["specificity"] == 5 / 6
    assert math.isclose(m["fpr"], 1 / 6)
    assert m["precision"] == 3 / 4
    assert math.isclose(m["f1"], 0.75)
    assert m["accuracy"] == 0.8


def test_malignant_is_positive_class():
    y = np.array([1, 1, 0, 0])
    m = classification_metrics(y, np.array([0.9, 0.9, 0.9, 0.9]))   # always "malignant"
    assert m["recall"] == 1.0 and m["specificity"] == 0.0


def test_roc_auc_uses_malignant_probability():
    y = np.array([0, 0, 1, 1])
    assert classification_metrics(y, np.array([.1, .2, .8, .9]))["roc_auc"] == 1.0
    assert classification_metrics(y, np.array([.9, .8, .2, .1]))["roc_auc"] == 0.0


def test_undefined_metrics_are_nan():
    m = classification_metrics(np.array([0, 0, 0]), np.array([.1, .2, .3]))   # no malignant
    assert math.isnan(m["recall"]) and math.isnan(m["roc_auc"]) and math.isnan(m["precision"])
    assert m["specificity"] == 1.0
    e = classification_metrics(np.array([], int), np.array([]))
    assert e["n"] == 0 and math.isnan(e["accuracy"])


def test_nonfinite_outputs_flagged():
    m = classification_metrics(np.array([0, 1]), np.array([np.nan, .5]))
    assert m["finite_outputs"] is False and math.isnan(m["accuracy"])

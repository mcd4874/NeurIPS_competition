def roc_auc_curve_compute_fn( y_targets,y_preds):
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        raise RuntimeError("This contrib module requires sklearn to be installed.")

    y_true = y_targets
    y_pred = y_preds[:, 1]
    # print(y_true.shape)
    # print(y_pred.shape)
    return roc_auc_score(y_true, y_pred)
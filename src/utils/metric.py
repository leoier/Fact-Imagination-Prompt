from sklearn import metrics

def get_cls_metrics(true_lbs, pred_lbs, detailed: bool = False):
    """
    Get classification metrics including precision, recall and f1

    Parameters
    ----------
    true_lbs: true labels
    pred_lbs: predicted labels
    detailed: Whether get detailed result report instead of micro-averaged one

    Returns
    -------
    Metrics if not detailed else Dict[str, Metrics]
    """
    if not detailed:
        p = metrics.precision_score(true_lbs, pred_lbs, average="weighted", zero_division=0)
        r = metrics.recall_score(true_lbs, pred_lbs, average="weighted", zero_division=0)
        f = metrics.f1_score(true_lbs, pred_lbs, average="weighted", zero_division=0)
        return {"precision": p, "recall": r, "f1": f}

    else:
        metric_dict = dict()
        report = metrics.classification_report(true_lbs, pred_lbs, output_dict=True, zero_division=0)
        for tp, results in report.items():
            if tp in ["accuracy", "macro avg", "weighted avg"]:
                continue
            metric_dict[tp] = {
                "precision": results["precision"],
                "recall": results["recall"],
                "f1": results["f1-score"],
            }
        return metric_dict

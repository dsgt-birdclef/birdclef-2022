import lightgbm as lgb
from sklearn.multioutput import MultiOutputClassifier


class SubmitClassifier:
    """Class for serialization."""

    def __init__(self, label_encoder, onehot_encoder, classifier):
        self.label_encoder = label_encoder
        self.onehot_encoder = onehot_encoder
        self.classifier = classifier


def train(train_ds: tuple, **kwargs):
    # TODO: parameter sweep
    # TODO: should this support a custom metric? Is it even possible to
    # implement mixup with this style of training?

    # https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html
    # https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html#sklearn.multioutput.MultiOutputClassifier
    bst = MultiOutputClassifier(
        lgb.LGBMClassifier(num_leaves=31, class_weight="balanced", **kwargs)
    )

    # NOTE: we cannot use early stopping, because a validation dataset that
    # is set on the multioutputclassifier will always be in the wrong shape.
    # is there a better way to perform a multi-label classification?
    # callbacks=[lgb.early_stopping(stopping_rounds=10)],
    bst.fit(train_ds[0], train_ds[1])
    return bst

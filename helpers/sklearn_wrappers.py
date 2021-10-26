import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression

class ItemSelector(BaseEstimator, TransformerMixin):
    """
    Transform text features
    """

    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None, *parg, **kwarg):
        return self

    def transform(self, X):
        # NULL replacement
        return X[self.key].fillna('NULL').astype(str)


class NullCheck(BaseEstimator, TransformerMixin):
    """
    Transform text features
    """

    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None, *parg, **kwarg):
        return self

    def transform(self, X):
        # NULL replacement
        return (X[self.key].isnull()*1).values.reshape(-1, 1)


class LogisticRegressionCustom():
    def __init__(self, base_model):
        self.base_model = base_model
        self.models = []
        self.model_parametes = {'C': 0.7,
                          'fit_intercept': False,
                          'penalty': 'l2',
                          'max_iter': 1000,
                          'random_state': 42,
                          'class_weight': 'balanced',
                          'solver': 'lbfgs'}
        self.cv_parameters = {'n_splits' : 5, 
                              'shuffle' : True,
                              'random_state' : 42}
        self.predict_threshold_ = 0.5

    def fit(self, X_train, y_train):
        cv = KFold(**self.cv_parameters)
        y_oof = np.zeros(y_train.shape[0])
        for train_ids, valid_ids in cv.split(X_train):
            model = self.base_model(**self.model_parametes)
            model.fit(X_train[train_ids], y_train[train_ids])
            y_oof[valid_ids] = model.predict_proba(X_train[valid_ids])[:, 1]
            auc_train = round(roc_auc_score(
                y_train[train_ids], model.predict_proba(X_train[train_ids])[:, 1]), 3)
            auc_test = round(roc_auc_score(
                y_train[valid_ids], y_oof[valid_ids]), 3)
            print(f'Train AUC: {auc_train}, Test AUC: {auc_test}')
            self.models.append(model)

        self.oof_predictions = y_oof
        self.labels = y_train

        print('Out of Fold AUC on full test data: {}'.format(
            round(roc_auc_score(y_train, y_oof), 3)))

    def predict_proba(self, X):
        y = np.zeros(X.shape[0])

        for m in self.models:
            y += m.predict_proba(X)[:, 1] / len(self.models)
        return y

    def predict_as_number(self, X):
        return (self.predict_proba(X) >= self.predict_threshold_)*1

    def predict_as_name(self, X):
        return ['accommodation' if x >= self.predict_threshold_ else 'restaurant' for x in self.predict_proba(X)]

def plot_PR_curve(y_labels, y_preds, y_limit=1.0):
    '''Plot average precision chart'''

    from sklearn.metrics import precision_recall_curve, average_precision_score
    import matplotlib.pyplot as plt
    from funcsigs import signature

    average_precision = average_precision_score(y_labels, y_preds)
    print('Average precision-recall score: {0:0.2f}'.format(average_precision))
    precision, recall, _ = precision_recall_curve(y_labels, y_preds)
    step_kwargs = ({'step': 'post'} if 'step' in signature(
        plt.fill_between).parameters else {})
    plt.figure(figsize=(10, 10))
    plt.step(recall, precision, color='r', alpha=0.4, where='post')
    plt.fill_between(recall, precision, alpha=0.4,
                     color='r', **step_kwargs)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, y_limit])
    plt.xlim([0.0, 1.0])
    plt.title(
        '2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
    plt.show()


def plot_roc_auc_curve(y_labels, y_preds):
    import sklearn.metrics as metrics
    # calculate the fpr and tpr for all thresholds of the classification
    fpr, tpr, _ = metrics.roc_curve(y_labels, y_preds)
    roc_auc = metrics.auc(fpr, tpr)

    import matplotlib.pyplot as plt
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
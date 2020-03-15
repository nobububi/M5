import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier as RFC


class LgbBinaryClassifier:
    def __init__(self, params=None):
        self.model = None

        if params is None:
            params = {
                "objective": "binary",
                'metric': 'auc',
                "nround": 500,
                "learning_rate": 0.01,
                "max_depth": 6,
                "num_leaves": 127,
                "random_state": 1
            }
        self.params = params

    def _adjust_data(self, dataset):
        lgb_dataset = lgb.Dataset(
            data=np.array(dataset.drop()),
            label=np.array(dataset.get_target())
        )
        return lgb_dataset

    def train(self, train_dataset, valid_dataset=None, desc=False):
        self.model = lgb.train(
            params=self.params,
            train_set=self._adjust_data(train_dataset),
            valid_sets=self._adjust_data(valid_dataset) if valid_dataset is not None else None
        )
        if desc:
            print("\n\n##################")
            print(self.get_model_info(train_dataset))
            print("##################")

    def predict(self, dataset):
        return self.model.predict(dataset.drop())

    def get_model_info(self, dataset):
        importance = pd.DataFrame(
            self.model.feature_importance(),
            index=dataset.drop().columns,
            columns=['importance']
        )
        return importance


class RfcBinaryClassifier:
    def __init__(self, params=None):
        self.model = None

    def train(self, train_dataset, valid_dataset=None, desc=False):
        self.model = RFC(random_state=1, n_jobs=-1).fit(
            X=train_dataset.drop().replace(np.inf, np.nan).fillna(0),
            y=train_dataset.get_target()
        )

    def predict(self, dataset):
        return self.model.predict_proba(dataset.drop().replace(np.inf, np.nan).fillna(0))[:, 1]

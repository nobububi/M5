import numpy as np
import pandas as pd
import lightgbm as lgb


class LgbLambdaLank:
    def __init__(self, params=None):
        self.model = None
        if params is None:
            params = {
                'objective': 'lambdarank',
                'metric': 'ndcg',
                "ndcg_at": 20,
                "nround": 500,
                "learning_rate": 0.01,
                "max_depth": 6,
                "num_leaves": 127,
                "random_state": 1
            }
        self.params = params

    @staticmethod
    def _adjust_data(dataset):
        dataset.data.sort_values(["KaiinID", "AuctionID"], inplace=True)
        if {"watch_actioned", "bid_actioned"} - set(dataset.data.columns) == set([]):
            label = np.array(dataset.data[["watch_actioned", "bid_actioned"]].astype(int)).max(axis=1)
        else:
            label = None

        weight = (
            np.stack([
                np.array(dataset.data["watch_actioned"].astype(int)),
                (np.array(dataset.data["bid_actioned"]).astype(int) * 2),
                np.ones((dataset.data.shape[0],))
            ], 1).max(axis=1)
        )

        group = (
            dataset.data[["KaiinID", "AuctionID"]]
            .groupby("KaiinID", as_index=False)
            .count()
            .sort_values("KaiinID")["AuctionID"]
        )

        lgb_dataset = lgb.Dataset(
            data=np.array(dataset.drop()),
            label=label,
            weight=weight,
            group=group
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
        dataset.data.sort_values(["KaiinID", "AuctionID"], inplace=True)
        return self.model.predict(
            data=np.array(dataset.drop()),
            group=np.array(dataset.data[["KaiinID", "AuctionID"]].groupby("KaiinID", as_index=False).count()
                           .sort_values("KaiinID")["AuctionID"])
        )

    def get_model_info(self, dataset):
        importance = pd.DataFrame(
            self.model.feature_importance(),
            index=dataset.drop().columns,
            columns=['importance']
        )
        return importance

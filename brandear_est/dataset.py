import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split


class DataSet:
    """
    モデルのインプットになるデータをクラス化。
    データ側で持っておくべき情報を保持する。
    """

    def __init__(self, data, drop_cols, target_col):
        """
        data:
            データ
        drop_cols:
            予測の際削除するカラム。
        target:
            予測対象のカラム。
        """
        self.data = data.copy()
        self.drop_cols = drop_cols
        self.target_col = target_col
        self.pred = None

    def drop(self):
        """予測用に不要カラムを削除"""
        drop_cols = [col for col in self.drop_cols + [self.target_col] + ["pred"]
                     if col in self.data]
        return self.data.drop(drop_cols, axis=1)

    def get_target(self):
        """予測対象データを取得"""
        return self.data[self.target_col]

    def set_pred(self, pred):
        """予測値を保持"""
        self.data["pred"] = pred

    def add_target_encode(self, cat_col):
        # 学習データの変換後の値を格納する配列を準備
        buf = np.repeat(np.nan, self.data.shape[0])

        # 学習データを分割
        kf = KFold(n_splits=4, shuffle=True, random_state=72)
        if isinstance(cat_col, str):
            for idx_1, idx_2 in kf.split(self.data):
                # out-of-foldで各カテゴリにおける目的変数の平均を計算
                target_mean = self.data.iloc[idx_1][[cat_col, self.target_col]].groupby(cat_col)[self.target_col].mean()
                # 変換後の値を一時配列に格納
                buf[idx_2] = self.data[cat_col].iloc[idx_2].map(target_mean)
            # 変換後のデータで元の変数を置換
            self.data[cat_col + "_target_mean"] = buf

        elif isinstance(cat_col, list):
            for idx_1, idx_2 in kf.split(self.data):
                # out-of-foldで各カテゴリにおける目的変数の平均を計算
                target_mean = self.data.iloc[idx_1][cat_col + [self.target_col]].groupby(cat_col, as_index=False)[
                    self.target_col].mean()
                # 変換後の値を一時配列に格納
                #                 import pdb;pdb.set_trace()
                buf[idx_2] = self.data[cat_col].iloc[idx_2].merge(target_mean, on=cat_col, how="left").fillna(0)[
                    "watch_actioned"]
                # 変換後のデータで元の変数を置換
                self.data["_".join(cat_col) + "_target_mean"] = buf

    @classmethod
    def under_sampling(cls, dataset, rate=10):
        positive_targets = dataset.data[dataset.data[dataset.target_col] == 1]
        sampled_negative = (
            dataset.data[dataset.data[dataset.target_col] != 1]
            .sample(n=positive_targets.shape[0] * rate)
        )
        sampled_data = pd.concat([positive_targets, sampled_negative])
        return cls(data=sampled_data, drop_cols=dataset.drop_cols, target_col=dataset.target_col)

    @classmethod
    def train_test_split(cls, dataset):
        train_data, test_data = train_test_split(dataset.data)
        return (cls(data=train_data, drop_cols=dataset.drop_cols, target_col=dataset.target_col),
                cls(data=test_data, drop_cols=dataset.drop_cols, target_col=dataset.target_col))

    @classmethod
    def gen_from_pkls(cls, paths, drop_cols, target_col):
        return [cls(data=pd.read_pickle(path).fillna(0), drop_cols=drop_cols, target_col=target_col)
                for path in paths]


def target_encode_for_test(train_dataset, test_dataset, cat_col):
    if isinstance(cat_col, str):
        test_dataset.data[f"{cat_col}_target_mean"] = (
            test_dataset.data[cat_col].map(
                train_dataset.data[[cat_col, train_dataset.target_col]].groupby(cat_col)[
                    train_dataset.target_col].mean()
            )
        )
    elif isinstance(cat_col, list):
        test_dataset.data = (
            test_dataset.data.merge(
                train_dataset.data[cat_col + [train_dataset.target_col]]
                    .groupby(cat_col, as_index=False)[train_dataset.target_col].mean().rename(
                    columns={train_dataset.target_col: "_".join(cat_col) + "_target_mean"}),
                on=cat_col, how="left"
            ).fillna(0)
        )

import re

import pandas as pd


def add_datepart(df: pd.DataFrame, field_name: str,
                 prefix: str = None, drop: bool = True, time: bool = True, date: bool = True):
    """
    Helper function that adds columns relevant to a date in the column `field_name` of `df`.
    from fastai: https://github.com/fastai/fastai/blob/master/fastai/tabular/transform.py#L55
    dtのカラム(field_name)から年月、月初などの特徴量を作成する関数
    """
    df_datepart = df.copy()
    field = df_datepart[field_name]
    prefix = re.sub('[Dd]ate$', '', field_name)
    attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Is_month_end', 'Is_month_start']
    if time:
        attr = attr + ['Hour', 'Minute']
    for n in attr:
        df_datepart[prefix + n] = getattr(field.dt, n.lower())
    if drop:
        df_datepart.drop(field_name, axis=1, inplace=True)

    return df_datepart


def add_time_features(dataset, feature_df, time_col, prefix, oldest_dtime):
    tmp_time_col = f"Tmp{time_col}Delta"
    key_cols = ["KaiinID", "AuctionID"]
    feature_df[tmp_time_col] = feature_df[time_col].apply(lambda d: (oldest_dtime - d).days)
    time_features = (
        feature_df
        .groupby(key_cols)[tmp_time_col]
        .agg(["count", "max", "min"])
        .rename(columns={"count": f"{prefix}_ua_cnt", "max": f"{prefix}_ua_newest", "min": f"{prefix}_ua_oldest"})
    )
    time_features[f"{prefix}_period"] = time_features[f"{prefix}_ua_newest"] - time_features[f"{prefix}_ua_oldest"]
    output = dataset.merge(time_features, on=key_cols, how="left")
    return output


def cross_counts(df, col_set, col_name=None):
    if col_name is not None:
        cnt_col_name = col_name
    elif isinstance(col_set, str):
        cnt_col_name = col_set + "_cnt"
    elif isinstance(col_set, list):
        cnt_col_name = "_".join(col_set) + "_cnt"
    else:
        raise ValueError
    cnts = (
        df.groupby(col_set, as_index=False).size().reset_index()
        .rename(columns={0: cnt_col_name})
    )
    return cnts


def merge_features(df1, df2, key, prefix):
    df2.columns = [prefix + "_" + column if column not in key else column
                   for column in df2.columns]
    return df1.merge(df2, on=key, how="left")


def add_cross_counts(df, feature_df, prefix, col_sets):
    print("##################")
    print("start cross count")
    df_copy = df.copy()
    print(col_sets)
    for col_set in col_sets:
        print(col_set)
        cnts = cross_counts(df=feature_df, col_set=col_set)
        df_copy = merge_features(df_copy, cnts, key=col_set, prefix=prefix).fillna(0)
    return df_copy

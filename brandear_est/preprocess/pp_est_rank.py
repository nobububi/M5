import datetime

import pandas as pd

import brandear_est.utils as utils
import brandear_est.feature_engineering as ff
from .brandear_common import *


def add_user_feature(df, feature_df, col_prefix):
    user_feature = calc_user_feature(feature_df)
    user_feature.columns = (
        [col_prefix + "_KaiinID_" + col if col != "KaiinID" else "KaiinID" for col in user_feature.columns]
    )
    return df.merge(user_feature, on="KaiinID", how="left").fillna(0)


def calc_user_feature(feature_df):
    user_feature = (
        feature_df.groupby("KaiinID")
        .agg({
            "AuctionID": {"AuctionID_cnt": "count"},
            "SaishuppinKaisuu": {
                "SaishuppinKaisuu_mean": "mean",
                "SaishuppinKaisuu_sum": "sum"},
            "SankouKakaku": {
                "SankouKakaku_mean": "mean",
                "SankouKakaku_sum": "sum"}
        })
    ).fillna(0)
    user_feature.columns = user_feature.columns.droplevel(0)
    user_feature = user_feature.reset_index()
    return user_feature


def build_target_candidate(dataset_types, inputs_type, data_dict, dset_to_period,
                           rank_th, rank_weekly_th, input_est_weekly_dir, output_dir):
    for dset_type in dataset_types:

        print(inputs_type)

        period = dset_to_period[dset_type]

        # データの時系列整理
        watch_arranged, bid_arranged, bid_success_arranged, auction_arranged, target_actions = (
            arrange_inputs(
                watch=data_dict["watch"],
                bid=data_dict["bid"],
                bid_success=data_dict["bid_success"],
                auction=data_dict["auction"],
                period=period
            )
        )

        if dset_type == "submission":
            target_users = data_dict["sub_users"]
        else:
            target_users = target_actions[["KaiinID"]].drop_duplicates()

        # inputとする候補の分岐
        # 商品紐付けのみ使う
        if "rank_weekly" not in inputs_type and "Shouhin" in inputs_type:
            dataset_base = extract_similar_aucs(
                target_users=target_users,
                auction=auction_arranged,
                actions=pd.concat(
                    [watch_arranged[["KaiinID", "ShouhinID"]],
                     bid_arranged[["KaiinID", "ShouhinID"]]]).drop_duplicates(),
                period=period
            )
        # 両方使う
        elif "rank_weekly" in inputs_type and "Shouhin" in inputs_type:
            ranked_weekly = pd.read_pickle(input_est_weekly_dir + f"/watch_{dset_type}_{rank_th}.pkl")
            ranked_weekly = ranked_weekly.query(f"rank <= {rank_weekly_th}")
            similar_aucs = extract_similar_aucs(
                target_users=target_users,
                auction=auction_arranged,
                actions=pd.concat(
                    [watch_arranged[["KaiinID", "ShouhinID"]], bid_arranged[["KaiinID", "ShouhinID"]]]
                ).drop_duplicates(),
                period=period
            )
            similar_aucs = (
                utils.left_anti_join(similar_aucs, bid_success_arranged[["AuctionID"]].drop_duplicates(),
                                     "AuctionID", "AuctionID")
            )
            dataset_base = (
                pd.concat(
                    [ranked_weekly[["AuctionID", "KaiinID"]], similar_aucs[["AuctionID", "KaiinID"]]]).drop_duplicates()
            )

        #         dataset_base = dataset_base.sample(frac=0.00001)
        else:
            raise ValueError()

        if "targets" in inputs_type:
            dataset_base = (
                pd.concat([dataset_base, target_actions[["KaiinID", "AuctionID"]]], sort=False).drop_duplicates()
            )

        # 正解付与
        dataset_base = dataset_base.merge(target_actions, on=["KaiinID", "AuctionID"], how="left").fillna(0)

        # オークション情報付与
        dataset_base_a = dataset_base.merge(
            auction_arranged[[col for col in auction_arranged.columns
                              if col not in dataset_base.columns] + ["AuctionID"]],
            on="AuctionID", how="left"
        )

        # クロス集計
        w_cate_col = ["AuctionID", "BrandID", "ItemShouID", "ShouhinID"]
        b_cate_col = ["AuctionID", "BrandID", "ItemShouID", "ShouhinID"]

        def add_cate_with_user(cate_col):
            cate_with_user = [["KaiinID", col] for col in cate_col]
            return cate_col + cate_with_user

        w_cnt_colsets = add_cate_with_user(w_cate_col)
        b_cnt_colsets = add_cate_with_user(b_cate_col)

        dataset_base_cwb = dataset_base_a

        cross_confs = ([
            [watch_arranged, "watch", w_cnt_colsets],
            [bid_arranged, "bid", b_cnt_colsets]
        ])

        for cross_conf in cross_confs:
            dataset_base_cwb = ff.add_cross_counts(
                df=dataset_base_cwb, feature_df=cross_conf[0], prefix=cross_conf[1], col_sets=cross_conf[2]
            )

        # ユーザーの特徴量付与
        # 何回watch/bid/successしたか
        # 再出品回数、価格の平均、分散、今回との割合
        dataset_base_u = dataset_base_cwb

        dataset_base_u = add_user_feature(df=dataset_base_u, feature_df=watch_arranged, col_prefix="watch")

        for col in ["SaishuppinKaisuu", "SankouKakaku"]:
            dataset_base_u[f"watch_KaiinID_rate_mean_to_{col}"] = (
                    dataset_base_u[f"watch_KaiinID_{col}_mean"] / dataset_base_u[col]
            )

            # 時間系の特緒量付与
        oldest_dtime = period["oldest"]

        def calc_timedelta(df, dtime_col, delta_col):
            df[delta_col] = df[dtime_col].swifter.apply(lambda d: (oldest_dtime - d).days)

        calc_timedelta(dataset_base_u, "CreateDate", "Auction_elapsed_days")
        calc_timedelta(watch_arranged, "TourokuDate", "watch_elapsed_day")

        def agg_time_feature(df, agg_key, na_value):
            time_agg = df.groupby(agg_key).agg({
                "watch_elapsed_day": {
                    f"{agg_key}_watch_elapsed_day_min": "min"
                }
            }).fillna(na_value)
            time_agg.columns = time_agg.columns.droplevel(0)
            time_agg = time_agg.reset_index()
            return time_agg

        w_k_d = agg_time_feature(df=watch_arranged, agg_key="KaiinID", na_value=999)
        w_a_d = agg_time_feature(df=watch_arranged, agg_key="AuctionID", na_value=999)

        dataset_base_u = (
            dataset_base_u.merge(w_k_d, on="KaiinID", how="left").fillna(999)
            .merge(w_a_d, on="AuctionID", how="left").fillna(0)
        )

        # お気に入り合計に対する該当オークションのお気に入り数

        dataset_base_b = dataset_base_u
        dataset_base_b["watch_BrandID_KaiinID_rate"] = (
                dataset_base_b["watch_KaiinID_BrandID_cnt"] / dataset_base_b["watch_KaiinID_AuctionID_cnt_y"]
        ).fillna(0)

        dataset_base_b["watch_ItemShouID_KaiinID_rate"] = (
                dataset_base_b["watch_KaiinID_ItemShouID_cnt"] / dataset_base_b["watch_KaiinID_AuctionID_cnt_y"]
        ).fillna(0)

        for cat_col in ["BrandID", "ItemShouID"]:
            brand_ave = (
                auction_arranged[[cat_col, "SankouKakaku"]].groupby(cat_col, as_index=False).mean()
                .rename(columns={"SankouKakaku": f"{cat_col}_SankouKakaku"})
            )
            dataset_base_b = dataset_base_b.merge(brand_ave, on=cat_col, how="left")
            dataset_base_b[f"SankouKakaku_rate_to_{cat_col}"] = (
                    dataset_base_b["SankouKakaku"] / dataset_base_b[f"{cat_col}_SankouKakaku"]
            )

        now = datetime.datetime.now().strftime("%Y%m%d%H%M")
        # be.df2pkl(dataset_base_b, output_dir, f"{dset_type}_feature_{rank_th}_{now}.pkl")

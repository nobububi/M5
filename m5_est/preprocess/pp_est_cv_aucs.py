import pandas as pd

import brandear_est.feature_engineering as ff
from .brandear_common import *


def build_dataset_input(auction, bid_success, watch, bid, period, dset_type, output_path):
    watch_arranged, bid_arranged, bid_success_arranged, auction_arranged, target_actions = (
        arrange_inputs(watch=watch, bid=bid, bid_success=bid_success, auction=auction, period=period)
    )

    aucs_with_cnts = (
        auction_arranged.merge(
            pd.concat([
                auction_arranged[(auction_arranged["CreateDate"] > period["newest"] - relativedelta(months=1))][
                    ["AuctionID"]],
                watch_arranged[(watch_arranged["TourokuDate"] > period["newest"] - relativedelta(months=1))][
                    ["AuctionID"]]
            ]).drop_duplicates(), on="AuctionID", how="inner")
    )

    # 落札しているオークションを除外
    aucs_with_cnts = utils.left_anti_join(aucs_with_cnts, bid_success_arranged, "AuctionID", "AuctionID")

    # 正解データ付与
    target_aucs = target_actions[["AuctionID", "watch_actioned", "bid_actioned"]].groupby("AuctionID",
                                                                                          as_index=False).max()
    aucs_with_cnts = aucs_with_cnts.merge(target_aucs, on="AuctionID", how="left").fillna(0)

    aucs_with_cnts["Brand_SankouKakaku"] = aucs_with_cnts["SankouKakaku"].map(
        aucs_with_cnts[["BrandID", "SankouKakaku"]].groupby("BrandID")["SankouKakaku"].mean()
    ).fillna(0)

    aucs_with_cnts["inv_rate_to_Brand_SankouKakaku"] = aucs_with_cnts["Brand_SankouKakaku"] / aucs_with_cnts[
        "SankouKakaku"]

    # クロス集計
    # クロス集計用にオークションデータ結合
    newest_dtime, oldest_dtime = (period["newest"], period["oldest"])
    cross_conf = {
        "watch": watch_arranged,
        "bid": bid_arranged,
        "1m_watch": extract_recent_data(watch_arranged, "TourokuDate", oldest_dtime, 30),
        "1m_bid": extract_recent_data(bid_arranged, "ShudouNyuusatsuDate", oldest_dtime, 30)
    }

    col_sets = [["AuctionID"], ["ShouhinID"], ["BrandID"], ["LineID"], ["ItemShouID"]]

    for prefix, feature_df in cross_conf.items():
        aucs_with_cnts = ff.add_cross_counts(aucs_with_cnts, feature_df, prefix=prefix, col_sets=col_sets)

    # 経過日数の特徴量
    aucs_with_time = aucs_with_cnts
    aucs_with_time["elapsed_days"] = aucs_with_time["CreateDate"].apply(lambda d: (newest_dtime - d).days)

    dtime_feat_confs = [["watch", "TourokuDate", watch_arranged],
                        ["bid", "ShudouNyuusatsuDate", bid_arranged]]
    for dtime_feat_conf in dtime_feat_confs:
        prefix = dtime_feat_conf[0]
        last_action = (
            dtime_feat_conf[2][["AuctionID", dtime_feat_conf[1]]]
            .groupby("AuctionID", as_index=False).min()
            .rename(columns={dtime_feat_conf[1]: f"{prefix}_elapsed_days"})
        )
        last_action[f"{prefix}_elapsed_days"] = last_action[f"{prefix}_elapsed_days"].swifter.apply(
            lambda d: (oldest_dtime - d).days)
        aucs_with_time = aucs_with_time.merge(last_action, on="AuctionID", how="left").fillna(0)

    # utils.df2pkl(aucs_with_time, output_path, f"{dset_type}_feature.pkl")

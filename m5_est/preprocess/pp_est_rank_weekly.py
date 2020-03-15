import pandas as pd

import brandear_est.utils as utils
import brandear_est.feature_engineering as ff
from .brandear_common import *


def cross_auc_users(actions, auc_dataset, period, target_users, target_col, dset_type, target_actions):

    if dset_type == "submission":
        # 有望オークションと、ユーザーとのクロスジョインから候補作成
        users_auc_cands_cross = utils.cross_join(target_users, auc_dataset)
    else:
        # クロスジョイン後正解データをジョイン
        users_auc_cands_cross = (
            utils.cross_join(target_users, auc_dataset)
            .merge(target_actions, on=["KaiinID", "AuctionID"], how="left")
            .fillna(0)
        )

    return users_auc_cands_cross


def build_target_candidate(dataset_inputs, valid_auc_th, auc_attr_col, data_dict, dset_to_period, output_path):
    target_col = "watch_actioned"
    for dset_type, dataset_input in dataset_inputs.items():

        period = dset_to_period[dset_type]

        auc_dataset = dataset_input.query(f"pred >= {valid_auc_th}")[auc_attr_col]

        watch_arranged, bid_arranged, bid_success_arranged, auction_arranged, target_actions = (
            arrange_inputs(
                watch=data_dict["watch"],
                bid=data_dict["bid"],
                bid_success=data_dict["bid_success"],
                auction=data_dict["auction"],
                period=period
            )
        )

        # 特徴量作成
        w_k_cnts = ff.cross_counts(watch_arranged, "KaiinID")
        w_ka_cnts = ff.cross_counts(watch_arranged, ["KaiinID", "AuctionID"])
        w_ks_cnts = ff.cross_counts(watch_arranged, ["KaiinID", "ShouhinID"])
        w_kb_cnts = ff.cross_counts(watch_arranged, ["KaiinID", "BrandID"])
        w_ki_cnts = ff.cross_counts(watch_arranged, ["KaiinID", "ItemShouID"])

        actions = (
            pd.concat([watch_arranged, bid_arranged], sort=False)[["AuctionID", "KaiinID", "ShouhinID"]]
            .drop_duplicates().drop("AuctionID", axis=1)
        )

        if dset_type != "submission":
            targets = target_actions.sort_values("KaiinID").reset_index(drop=True)[["KaiinID"]].drop_duplicates()
        else:
            targets = data_dict["sub_users"]

        for g, split_targets in targets.groupby(targets["KaiinID"] // 5000):

            print(dset_type, "分割", g, "番目")
            if (g > 11) & (dset_type == "valid_for_train"):
                continue

            splited_candidates = cross_auc_users(
                actions=actions,
                auc_dataset=auc_dataset[auc_attr_col],
                period=period,
                target_users=split_targets,
                target_col=target_col,
                dset_type=dset_type,
                target_actions=target_actions
            )

            splited_candidates_feat = (
                splited_candidates
                .merge(w_k_cnts, on="KaiinID", how="left")
                .merge(w_ka_cnts, on=["KaiinID", "AuctionID"], how="left")
                .merge(w_ks_cnts, on=["KaiinID", "ShouhinID"], how="left")
                .merge(w_kb_cnts, on=["KaiinID", "BrandID"], how="left")
                .merge(w_ki_cnts, on=["KaiinID", "ItemShouID"], how="left")
                .drop(["ShouhinID", "BrandID", "ItemShouID"], axis=1)
                .fillna(0)
            )

            splited_candidates_feat["BrandID_KaiinID_rate"] = (
                    splited_candidates_feat["KaiinID_BrandID_cnt"] / splited_candidates_feat["KaiinID_cnt"]
            ).fillna(0)
            splited_candidates_feat["ItemShouID_KaiinID_rate"] = (
                    splited_candidates_feat["KaiinID_ItemShouID_cnt"] / splited_candidates_feat["KaiinID_cnt"]
            ).fillna(0)

            splited_candidates_feat.drop(["KaiinID_BrandID_cnt", "KaiinID_ItemShouID_cnt"], axis=1, inplace=True)

            # utils.df2comp_pkl(splited_candidates_feat, output_path + "/" + dset_type + "/", f"split_cands_{g}.pkl")

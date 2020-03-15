

import pandas as pd


def comple_submit_auc(df):
    target_users = df.groupby("KaiinID", as_index=False).count().query("score < 20")["KaiinID"].tolist()
    if not target_users:
        return df
    else:
        candidate_aucs = (
            df[["AuctionID", "score"]]
            .groupby("AuctionID", as_index=False).mean().sort_values("score", ascending=False).iloc[:40, :]
        )
        candidate_aucs["score"] -= 999
        buf = []
        for user in target_users:
            candidate_aucs_tmp = candidate_aucs.copy()
            candidate_aucs_tmp["KaiinID"] = user
            buf.append(candidate_aucs_tmp)
        df_comple = pd.concat(buf)
        df_colmled = pd.concat([df, df_comple], sort=False).groupby(["KaiinID", "AuctionID"], as_index=False).max()
        return df_colmled


def adjust_sub_form(users, pred, drop=False):
    sub_data = users.merge(pred, on="KaiinID", how="left")[["KaiinID", "AuctionID", "score"]]
    sub_data = comple_submit_auc(sub_data)
    sub_data.sort_values(['KaiinID', 'score'], ascending=[True, False], inplace=True)
    sub_data['rank'] = sub_data.groupby('KaiinID')['score'].cumcount()
    sub_valid = sub_data.query("rank < =19")
    sub_valid = sub_valid.sort_values(['KaiinID', 'score'], ascending=[True, False])
    if drop:
        sub_valid.drop(["score", "rank"], axis=1, inplace=True)

    return sub_valid


def get_cheat_pred(data, target_actions):
    actiones = data[["KaiinID", "AuctionID"]].copy()
    scored_targets = (
        stack_target_actions(target_actions).groupby(["KaiinID", "AuctionID"], as_index=False).max())
    cheat_pred = (
        actiones.merge(scored_targets, on=["KaiinID", "AuctionID"], how="left")
        .fillna(0).sort_values(["KaiinID", "score"], ascending=["True", "False"]))
    return cheat_pred


def stack_target_actions(target_actions):
    watch_target = target_actions.query("(watch_actioned == 1)")[["KaiinID", "AuctionID"]]
    bid_target = target_actions.query("(bid_actioned == 1)")[["KaiinID", "AuctionID"]]
    watch_target["score"] = 1
    bid_target["score"] = 2
    stacked_target_actions = pd.concat([watch_target, bid_target], sort=False)
    return stacked_target_actions

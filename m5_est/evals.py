import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve


from .utils import *


def stack_target_actions(target_actions):
    watch_target = target_actions.query("(watch_actioned == 1)")[["KaiinID", "AuctionID"]]
    bid_target = target_actions.query("(bid_actioned == 1)")[["KaiinID", "AuctionID"]]
    watch_target["score"] = 1
    bid_target["score"] = 2
    stacked_target_actions = pd.concat([watch_target, bid_target], sort=False)
    return stacked_target_actions


def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    return np.sum((2 ** r - 1) / np.log2(np.arange(2, r.size + 2)))


def ndcg_at_k(r, k):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max


def calc_dcgs(y_true, y_pred, k=20):
    y_pred_cp = y_pred.copy()

    actione_true = stack_target_actions(y_true)
    actione_true["rank"] = 100

    y_pred_cp['rank'] = y_pred_cp.groupby('KaiinID')['AuctionID'].cumcount()

    scored_pred = (
        y_pred_cp.merge(actione_true[["KaiinID", "AuctionID", "score"]], on=["KaiinID", "AuctionID"],
                        how="left").fillna(0))

    unchoiced_actiones = (
        left_anti_join(actione_true, y_pred_cp, ["KaiinID", "AuctionID"], ["KaiinID", "AuctionID"]))

    scored_actiones = (
        pd.concat([scored_pred, unchoiced_actiones], sort=False)
        .sort_values(["KaiinID", "rank"], ascending=["True", "True"]))

    dcgs = scored_actiones.groupby("KaiinID")["score"].apply(lambda s: ndcg_at_k(s.tolist(), k=k))

    return dcgs


def calc_ndcg(y_true, y_pred, k=20):
    dcgs = calc_dcgs(y_true, y_pred, k=k)
    return dcgs.mean()


def plot_roc_curve(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    plt.plot(fpr, tpr, marker='o')
    plt.xlabel('FPR: False positive rate')
    plt.ylabel('TPR: True positive rate')
    plt.grid()


def plot_tpr_fpr(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    th_df = pd.DataFrame({"fpr": fpr, "tpr": tpr, "thresholds": thresholds})
    plt.plot(th_df["thresholds"], th_df["fpr"])
    plt.plot(th_df["thresholds"], th_df["tpr"])

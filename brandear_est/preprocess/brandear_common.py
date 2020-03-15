from dateutil.relativedelta import relativedelta

import brandear_est.utils as utils


def build_auction_mst(auction, itemshou, genre, brand, color, line):
    itemshou = rename(itemshou, prefix="ItemShow")
    genre = rename(genre, prefix="Genre")
    brand = rename(brand, prefix="Brand")
    color = rename(color, prefix="Color")
    line = rename(line, prefix="Line")

    genre_mst = (
        genre[["GenreID", "ItemShouID", "CategoryID"]]
        .merge(itemshou[["ItemShouID", "ItemDaiID"]], on="ItemShouID", how="inner")
    )

    auction_mst = (
        auction
        .merge(genre_mst, on="GenreID", how="left")
        .merge(brand[["BrandID", "BrandCreateDate"]], on="BrandID", how="left")
        .merge(color[["ColorID", "ItemColorID"]], on="ColorID", how="left")
        .merge(line[["LineID", "ItemLineID", "LineCreateDate"]], on="LineID", how="left")
        .fillna(0)
    )
    auction_mst = utils.to_datetime(auction_mst[sorted(list(auction_mst.columns))])

    return auction_mst


def rename(df, prefix):
    target_columns = ["ModifyDate", "CreateDate"]
    for target_column in target_columns:
        if target_column in df.columns:
            df.rename(columns={target_column: prefix + target_column}, inplace=True)
    return df


def extract_target_actions(watch, bid, period):
    watch_actioned = (
        watch.loc[(watch["TourokuDate"] >= period["oldest"]) & (watch["TourokuDate"] < period["newest"]),
                  ["KaiinID", "AuctionID"]]
    )
    bid_actioned = (
        bid.loc[(bid["ShudouNyuusatsuDate"] >= period["oldest"]) & (bid["ShudouNyuusatsuDate"] < period["newest"]),
                ["KaiinID", "AuctionID"]]
    )

    watch_actioned["watch_actioned"] = 1
    bid_actioned["bid_actioned"] = 1

    target_actions = (
        watch_actioned
        .merge(bid_actioned, on=["KaiinID", "AuctionID"], how="outer")
        .drop_duplicates()
        .fillna(0)
    )
    return target_actions


def arrange_dtime_condition(watch, bid, bid_success, auction, period):
    retval = ((
        watch[watch["TourokuDate"] <= period["oldest"]],
        bid[bid["ShudouNyuusatsuDate"] <= period["oldest"]],
        bid_success[bid_success["RakusatsuDate"] < period["oldest"]],
        auction[auction["CreateDate"] < period["newest"]]
    ))
    return retval


def arrange_inputs(watch, bid, bid_success, auction, period):
    target_actions = extract_target_actions(watch, bid, period)

    watch_t, bid_t, bid_success_t, auction_t = (
        arrange_dtime_condition(watch, bid, bid_success, auction, period)
    )

    arranged_inputs = (
        watch_t.merge(auction_t, on="AuctionID", how="left"),
        bid_t.merge(auction, on="AuctionID", how="left"),
        bid_success_t,
        auction_t,
        target_actions
    )
    return arranged_inputs


def extract_recent_data(df, date_col, base_dtime, days):
    oldest_dtime = base_dtime - relativedelta(days=days)
    return df[df[date_col] > oldest_dtime]


def extract_similar_aucs(target_users, auction, actions, period):
    similar_aucs = (
        actions.merge(target_users, on="KaiinID", how="inner")
        .merge(auction, on="ShouhinID")
    )
    return similar_aucs

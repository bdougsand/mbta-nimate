#!/usr/bin/env python3
import argparse
import calendar
import codecs
import csv
from datetime import datetime, timedelta
from io import BytesIO, TextIOWrapper
import os
import sys
from urllib import parse
from urllib.request import urlopen
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import pytz


def get_updates(dt, local_copy=None):
    """Retrieves the vehicle updates for the given date from
    the archive site and read them into a pandas DataFrame.
    :param dt: a date-like object
    """
    # This reads the gzipped tarfile into memory, extracts the file,
    # and reads it directly into a dataframe.
    csv_filename = dt.strftime("%Y-%m-%d.csv")
    tarname = f"{csv_filename}.gz"
    local_path = local_copy and os.path.join(local_copy, tarname)
    url = f"http://mbta-history.apptic.xyz/%Y/%m/{tarname}"
    if local_path:
        if not os.path.exists(local_path):
            print(f"Downloading vehicle updates for {dt}")
            url = dt.strftime(url)
            filedata = BytesIO(urlopen(url).read())
            with open(local_path, "wb") as out:
                out.write(filedata.read())
        return pd.read_csv(local_path, dtype="unicode")

    return pd.read_csv(url, dtype="unicode")


# def dumb_quotes(error):
#     print("Failed:", error.object[error.start:error.end])
#     raise error

# codecs.register_error("dumbquotes", dumb_quotes)


def _mbta_feed_urls():
    """Returns a generator of (feed_start_date, feed_end_date, archive_url)
    tuples from the MBTA's archived feeds site.
    """
    u = urlopen("https://www.mbta.com/gtfs_archive/archived_feeds.txt")
    # Because there are smart quotes in the feed!!
    data = u.read().replace(b"\xe2\x80\x9d", b'"')
    for l in csv.DictReader(TextIOWrapper(BytesIO(data))):
        yield (datetime.strptime(l["feed_start_date"], "%Y%m%d"),
               datetime.strptime(l["feed_end_date"], "%Y%m%d"),
               l["archive_url"])


FEED_URLS = None


def mbta_feed_urls():
    global FEED_URLS
    if not FEED_URLS:
        FEED_URLS = list(_mbta_feed_urls())
    return FEED_URLS


def mbta_feed_urls_for(range_start=None, range_end=None):
    eastern = pytz.timezone("US/Eastern")
    range_start = range_start or datetime.now()
    range_end = range_end or range_start
    for start, end, url in mbta_feed_urls():
        if range_start.tzinfo:
            start = start.astimezone(eastern)
            end = end.astimezone(eastern)
        if start <= range_end:
            if end < range_start:
                continue
            yield url


def mbta_feed_url_for(when):
    """Get the URL for the MBTA's GTFS feed active at the datetime `when`.

    """
    return next(mbta_feed_urls_for(when, when), None)


def get_zip(url="http://www.mbta.com/uploadedfiles/MBTA_GTFS.zip", save_to=None):
    with urlopen(url) as u:
        data = BytesIO(u.read())
    if save_to:
        with open(save_to, "wb") as out:
            out.write(data.read())
        data.seek(0)

    return zipfile.ZipFile(data)


def get_zip_item(feed, name):
    data = TextIOWrapper(BytesIO(feed.read(name + ".txt")),
                         encoding="utf-8", line_buffering=True)
    # Specify 'unicode' as the datatype so that Pandas
    # doesn't try to infer cell types. It will get it
    # wrong anyway.
    return pd.read_csv(data, dtype="unicode")


def convert_clock_time(row, timezone=pytz.timezone("US/Eastern")):
    y, M, d = map(int, row.trip_start.split("-"))
    dt = timezone.localize(datetime(y, M, d))
    h, m, s = map(int, row.arrival_time.split(":", 2))
    # This is here to avoid DST issues
    # Trips that originate on one day and continue into the 
    # next day have scheduled arrival and departure times of
    # greater than 24 hours. Simply adding hours will result
    # in incorrect dates on days when DST begins or ends,
    # since 
    if h >= 24:
        return timezone.normalize(
            dt.replace(minute=m, second=s) +
            timedelta(days=1, hours=h % 24)
        )
    return dt.replace(hour=h, minute=m, second=s)


def add_hour_start(group):
    start_time = group.scheduled_arrival_time.min()
    group["hour_start"] = start_time.hour
    group["schedule_offset"] = (group.scheduled_arrival_time - start_time).dt.total_seconds()
    return group


def reshape(stops):
    return stops.groupby("stop_sequence").delay.median()


def prepare_frame(dt, route_id, stop_times, trips, local_copy=None,
                  timezone="US/Eastern"):
    df = get_updates(dt, local_copy)
    rt_stop_times = pd.merge(
        stop_times[["trip_id", "stop_sequence", "arrival_time"]],
        trips[["trip_id", "route_id", "direction_id"]],
        on="trip_id")
    rt_stop_times = rt_stop_times[rt_stop_times.route_id == route_id]
    df = pd.merge(df, rt_stop_times, on=["trip_id", "stop_sequence"])
    df = df[df.status == "STOPPED_AT"]
    df["timestamp"] = pd.to_datetime(df.timestamp)\
                        .dt.tz_localize("UTC")\
                        .dt.tz_convert("US/Eastern")
    df["scheduled_arrival_time"] = df.apply(convert_clock_time, axis=1)
    df["delay"] = np.round(
        (df.timestamp - df.scheduled_arrival_time).dt.total_seconds()/60, 1)
    return df.groupby("trip_id").apply(add_hour_start)


def render_frame(args, df):
    f, ax = plt.subplots(figsize=(9, 6))
    data = df.groupby("hour_start").apply(reshape).unstack(level=1)
    return sns.heatmap(data, ax=ax, vmax=args.max_delay, vmin=args.min_delay)


_cached_feed = None


def get_feed(args, when):
    global _cached_feed

    url = mbta_feed_url_for(when)
    file_name = os.path.basename(url)

    if _cached_feed and _cached_feed[0] == file_name:
        return (True, _cached_feed[1])

    if args.save_feed:
        local_path = os.path.join(args.feed_path, file_name)
        if os.path.exists(local_path):
            feed = zipfile.ZipFile(open(local_path, "rb"))
        else:
            print(f"Downloading feed to {local_path}")
            feed = get_zip(url, local_path)
    else:
        feed = get_zip(url)

    _cached_feed = (file_name, feed)
    return (False, feed)


def parse_date(datestr):
    return datetime.strptime(datestr, "%Y-%m-%d")


def month_range(when=None):
    when = when or datetime.now()
    now = when.replace(hour=0, minute=0, second=0, microsecond=0)
    (_, days) = calendar.monthrange(now.year, now.month)
    return now.replace(day=1), now.replace(day=days)


def run(args):
    when = args.from_date
    os.makedirs(args.image_path, exist_ok=True)

    while when <= args.to_date:
        (cached, feed) = get_feed(args, when)
        if not cached:
            stop_times = get_zip_item(feed, "stop_times")
            trips = get_zip_item(feed, "trips")
        try:
            figure = render_frame(args,
                                prepare_frame(when, args.route_id,
                                                stop_times, trips,
                                                (args.save_update_files and
                                                args.updates_path)))
            image_path = os.path.join(args.image_path, when.strftime("%Y%m%d.png"))
            print(f"Saving image to {image_path}")
            figure.figure.savefig(image_path)
        except:
            pass
        when += timedelta(days=1)


def do_main(args):
    from_default, to_default = month_range()
    parser = argparse.ArgumentParser()
    parser.add_argument("route_id")
    parser.add_argument("--no-save", "-U",
                        default=True,
                        dest="save_update_files",
                        action="store_false",
                        help=("Don't use or save local copies of the "
                              "vehicle updates archives. Open them in "
                              "memory"))
    parser.add_argument("--updates-path", default=".")
    parser.add_argument("--no-save-feed", "-F", dest="save_feed", default=True,
                        action="store_false",
                        help=("Don't look for or save local copies of the "
                              "MBTA feed(s)"))
    parser.add_argument("--feed-path", default=".")
    parser.add_argument("--from-date", "-f", type=parse_date,
                        help=("Start date, formatted as YYYY-mm-dd. "
                              "Defaults to the start of the current "
                              "month."))
    parser.add_argument("--to-date", "-t", default=to_default,
                        type=parse_date,
                        help=("End date, formatted as YYYY-mm-dd. "
                              "Defaults to the end of the start "
                              "month, or today, if the month is "
                              "ongoing."))
    parser.add_argument("--image-path", default="./frames",
                        help=("Where to save the intermediate frame "
                              "images"))
    parser.add_argument("--max-delay", default=60,
                        help=("Cap the max delay value at this number of "
                              "minutes"))
    parser.add_argument("--min-delay", default=-60)

    if len(args) == 0:
        parser.print_help()
    else:
        parsed_args = parser.parse_args(args)
        if not parsed_args.to_date:
            (_, end) = month_range(parsed_args.from_date)
            today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            parsed_args.to_date = min(end, today)
        run(parsed_args)


if __name__ == "__main__":
    do_main(sys.argv[1:])

# Essential principals:
# - Vectorized operations
# - Grouping
# - Joining/merging

# - Cleanup
# - Data types (show some examples of where these might matter)


# ?? Go from multindex -> 2d dataframe?
# Coercing Series with MultiIndex to ndarray

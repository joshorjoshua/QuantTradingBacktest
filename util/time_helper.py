from datetime import *


def str_timedelta(datestring, days):
    dt = datetime.strptime(datestring, '%Y%m%d').date()
    delta = timedelta(days)

    new_dt = dt + delta

    str_new_dt = new_dt.strftime('%Y%m%d')

    return str_new_dt
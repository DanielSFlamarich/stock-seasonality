from datetime import datetime, timedelta

from dateutil.relativedelta import relativedelta


def date_range(start, end, step=7, date_format="%Y-%m-%d", mode="day"):
    """
    Creates generator with a range of dates.
    The dates occur every 7th day (default).

    :param start: the start date of the date range
    :param end: the end date of the date range
    :param step: the step size of the dates
    :param date_format: the string format of the dates inputted and returned
    :param mode: indicates the mode to increase time delta : can be "day" or "month"
    """
    start_date = datetime.strptime(str(start), date_format)
    end_date = datetime.strptime(str(end), date_format)

    if mode == "day":
        num_days = (end_date - start_date).days
        for d in range(0, num_days + step, step):
            date_i = start_date + timedelta(days=d)
            yield date_i.strftime(date_format)
    elif mode == "month":
        num_months = (end_date.year - start_date.year) * 12 + (
            end_date.month - start_date.month
        )
        for d in range(0, num_months + step, step):
            date_i = start_date + relativedelta(months=d)
            yield date_i.strftime(date_format)


def print_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

Added :mod:`dateutil.relativedelta_compat`, a diagnostic and regression
module that finds start dates where applying a chain of ``relativedelta``
objects differs from applying their merged sum or from applying them in
reverse order.  Each reported interval records whether the divergence involves
absolute-field overrides, month-end clamping, leap days or weekday
interactions.  Ranges longer than 400 years are handled for purely relative
rules by tiling the Gregorian cycle instead of rescanning every day.

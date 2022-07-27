# Copyright 2022 CSIRO
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implementation of a tools for dealing with :py:class:`TimeInterval`."""

from functools import reduce
from typing import List, Sequence, Iterable
from math import isclose



class TimeInterval:
    """An interval of time.

    :ivar float start: The start of the interval.
    :ivar float end: The end of the interval.

    :param float start: The start of the interval.
    :param float end: The end of the interval.

    Other arguments passed by ``**kwargs`` are set on the instance.

    :todo: Consider whether the values in `**kwargs` should be kept in a
        separate dictionary and `__getattr__` overriden.
    """

    def __init__(self, start, end, **kwargs):
        assert(start <= end)
        self.start  = start
        self.end    = end

        # TODO: Might be better to keep this in a separate dict and override __getattr__.
        for k, v in kwargs.items():
            setattr(self, k, v)


    def __repr__(self):
        args = ('name', 'shedding', 'box')

        argstr = reduce(
            lambda s, a: s + ('' if not hasattr(self, a) else ', {}={}'.format(
                a, getattr(self, a))),
            args,
            '{}, {}'.format(self.start, self.end))

        return 'TimeInterval({})[{}]'.format(argstr, self.length)


    def starts_before(self, other):
        """Does this interval start before *other*?

        :param TimeInterval other: Another interval.

        :rtype: bool
        """
        return self.start <  other.start


    def starts_after(self, other):
        """Does this interval start after *other*?

        :param TimeInterval other: Another interval.

        :rtype: bool
        """
        return self.start >= other.end


    def ends_before(self, other):
        """Does this interval end before the end of *other*?

        :param TimeInterval other: Another interval.

        :rtype: bool
        """
        return self.end   <= other.start


    def ends_after(self, other):
        """Does this interval end after the end of *other*?

        :param TimeInterval other: Another interval.

        :rtype: bool
        """
        return self.end   >  other.end


    def overlaps(self, other):
        """Does this interval overlap *other*?

        :param TimeInterval other: Another interval.

        :rtype: bool
        """
        return not (self.ends_before(other) or self.starts_after(other))


    def contains(self, other):
        """Does this interval contain *other*?

        :param TimeInterval other: Another interval.

        :rtype: bool
        """
        return other.start >= self.start and other.end <= self.end


    @property
    def length(self):
        """The length of this interval."""
        return self.end - self.start


    @property
    def midpoint(self):
        """The midpoint of this interval."""
        return .5 * (self.end + self.start)


    def shift_to_start_of(self, other, shorten):
        """Shift this interval so it overlaps with the start of *other* and
        shorten to be contained by *other* if required.

        After the call, ``self.start == other.start``, and if if *shorten* is
        *true*, then ``self.length <= other.length``.

        :param TimeInterval other: Another interval.

        :param bool shorten: Should this interval be shortened if it is longer
            than *other*?
        """
        self.shift_by(other.start - self.start)
        if self.ends_after(other):
            if shorten:
                self.end = other.end
            else:
                raise Exception("intervals overlap")


    def shift_to_end_of(self, other, shorten):
        """Shift this interval so it overlaps with the end of *other* and
        shorten to be contained by *other* if required.

        After the call, ``self.end == other.end``, and if if *shorten* is
        *true*, then ``self.length <= other.length``.

        :param TimeInterval other: Another interval.

        :param bool shorten: Should this interval be shortened if it is longer
            than *other*?
        """
        self.shift_by(other.end - self.end)
        if self.starts_before(other):
            if shorten:
                self.start = other.start
            else:
                raise Exception("intervals overlap")


    def shift_by(self, shift):
        """Shift this interval by *shift*.

        :param float shift: The amount to shift this interval by.
        """
        self.start += shift
        self.end   += shift


    def shift_to_nearest_end(self, other, shorten):
        """Shift this interval so it overlaps with one of the ends of *other*
        and shorten to be contained by *other* if required.

        After the call, if ``self.midpoint < other.midpoint``, then
        ``self.start == other.start``, otherwise ``self.end == other.end``. If
        *shorten* is *true*, then ``self.length <= other.length``.

        :param TimeInterval other: Another interval.

        :param bool shorten: Should this interval be shortened if it is longer
            than *other*?

        :return: ``TimeInterval(self.end, other.end)`` if
            ``self.length < other.length`` after being shifted and
            (potentially) shortened. Otherwise, return *None*.

        :rtype: TimeInterval
        """
        if self.midpoint < other.midpoint:
            self.shift_to_start_of(other, shorten)
            if self.end < other.end:
                return TimeInterval(self.end, other.end)
            else:
                return None
        else:
            self.shift_to_end_of(other, shorten)
            if self.start > other.start:
                return TimeInterval(other.start, self.start)
            else:
                return None


    def does_not_overlap_any(self, intervals):
        """Does this interval overlap any interval in *intervals*?

        :rtype: bool
        """
        return not any(self.overlaps(i) for i in intervals)


    def remove(self, interval):
        """Remove *interval* from this interval and return the rest as a list.

        **pre-condition**: ``assert self.contains(interval)``

        :param TimeInterval interval: The interval to remove.

        :return: List of remaining intervals.

        :rtype: List[TimeInterval]
        """
        assert self.contains(interval)

        if self.start == interval.start:
            if self.end == interval.end:
                return None
            return [TimeInterval(interval.end, self.end)]

        if self.end == interval.end:
            return [TimeInterval(self.start, interval.start)]

        return [
            TimeInterval(self.start, interval.start),
            TimeInterval(interval.end, self.end)]



def get_overlapping_interval(i1, i2):
    """Return the interval that overlaps *i1* and *i2* (the union of *i1* and
    *i2*).

    :param TimeInterval i1: An interval.
    :param TimeInterval i2: An interval.

    :rtype: TimeInterval.
    """
    if i1.overlaps(i2):
        return TimeInterval(max(i1.start, i2.start), min(i1.end, i2.end))
    return None



def overlap_length(i1, i2):
    """Return the length of the interval that overlaps *i1* and *i2* (i.e. the
    length of the interval that is the union of *i1* and *i2*).

    :param TimeInterval i1: An interval.
    :param TimeInterval i2: An interval.

    :rtype: float.
    """
    oi = get_overlapping_interval(i1, i2)
    return 0. if oi is None else oi.length



def intermediate_intervals(
        intervals: Iterable[TimeInterval]) -> List[TimeInterval]:
    """Create Periods for each 'gap' in *intervals*.

    :param intervals: The intervals to produce 'fill in' intervals for.

    :return: A list of intervals corresponding to the 'gaps' in *intervals*. When
        merged with *intervals*, the result would cover the time interval
        ``(intervals[0].start, intervals[-1].end]``.

    :rtype: List[TimeInterval]
    """

    return reduce(
        lambda ps, p12: ps + ([TimeInterval(
            p12[0].end,
            p12[1].start)] if p12[1].start > p12[0].end else []),
        zip(intervals[:-1], intervals[1:]), [])



def merge_interval(
        interval: TimeInterval,
        others: List[TimeInterval],
        copy: bool = True) -> List[TimeInterval]:
    """Merge *interval* into *others*.

    Assumes that *interval* does not overlap any interval in *others*, and that
        *others* is ordered.

    :param interval: A interval.

    :param others: A list of intervals.

    :param copy: Should we copy others first? If False, then *others* will
        be modified directly.

    :return: *others* with *interval* merged into it.
    """

    if len(others) == 0:
        return [interval]

    # copy others so as to not modify the input
    if copy:
        others = others[:]

    if interval.length <= 0.:
        return others

    ois = enumerate(others)
    i, o = next(ois)

    if interval.start < o.start:
        others.insert(0, interval)
        return others

    last = o
    for i, o in ois:
        if last.end <= interval.start < o.start:
            others.insert(i, interval)
            return others
        last = o

    assert interval.start >= others[-1].end
    others.insert(len(others), interval)
    return others



def merge_intervals(
        intervals: List[TimeInterval],
        others: List[TimeInterval]) -> List[TimeInterval]:
    """Merge each interval in *intervals* into *others*.

    This calls :py:func:`merge_interval` for each interval in *intervals*.

    :param intervals: A list of intervals.

    :param others: A list of intervals.

    :rtype: List[TimeInterval]
    """

    # copy others so as to not modify the input
    others = others[:]

    for interval in intervals:
        merge_interval(interval, others, False)

    return others



def no_overlaps(
        intervals: Iterable[TimeInterval],
        other_intervals: Iterable[TimeInterval] = None):
    """Check that intervals do not overlap.

    If *other_intervals* is *None*, then check that the intervals in
    *intervals* are sorted and do not overlap with each other. Otherwise, check
    that each interval in *intervals* does not overlap with any interval in
    *other_intervals*.

    :param intervals: An iterable.
    :param other_intervals: An iterable.

    :rtype: bool
    """
    if other_intervals is None:
        # remember that all short circuits
        return all(a.end <= b.start for a, b in zip(intervals, intervals[1:]))

    return all(not i.overlaps(j) for i in intervals for j in other_intervals)



def are_sorted(intervals):
    """Check *intervals* are sorted.

    This just calls :py:func:`no_overlaps` on *intervals*.
    """
    return no_overlaps(intervals)



def no_gaps(intervals):
    """Check that there are no gaps between the intervals in *intervals*.

    Checks that the ends of intervals are close as per :py:func:`math.isclose`.

    Assumes that the intervals are sorted.
    """
    assert are_sorted(intervals)
    # remember that all short circuits
    return all(isclose(a.end, b.start) for a, b in zip(intervals, intervals[1:]))

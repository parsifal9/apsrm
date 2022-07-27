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

"""Contains the implementation of :py:class:`Box`."""

from math import floor
from copy import copy
from pprint import pformat
from typing import Union, Any
import numpy as np
from .interval import (
    TimeInterval,
    intermediate_intervals,
    get_overlapping_interval,
    merge_interval)
from ._pathogen import concentration_at_time

class Box:
    """Represents a (closed) space (room or HVAC system).

    :ivar volume: Volume of the box.
    :ivar use: The use of the room. The appropriate type for this depends on
        how box types are distinguished, which is up to the implementer of the
        :py:class apsrm.GatheringGenerator`\ s and alike. We have tended to
        use :py:class:`enum.Enum` (or subclasses thereof), but it might be more
        convenient to simply use strings.
    :ivar max_occupancy: Maximum occupancy.
    :ivar Union[None, apsrm.ventilation.VentilationSubSystem] ventilation_system:
        The ventilation system that services this box. This is initially set to
        *None*, but will be updated when the ventilation system is created.
        See, for instance, the constructor of
        :py:class:`apsrm.ventilation.PseudoVentilationSystem`.
    :ivar list[apsrm.interval.TimeInterval] infected_intervals: A list of
        intervals in which there is at least one infected agent in the box.
    :ivar float total_exposure_risk: The expected number of infections to occur
        due to exposure in this box. This is set through successive calls to
        :py:meth:`add_exposure_risk` from
        :py:meth:`apsrm.Person.live_with_pathogen`.

    :param volume: Volume of the box.
    :param use: The use of the room.
    :param max_occupancy: Maximum occupancy.
    :param dict[str, Any] \\**kwargs: Items set as attributes on the instance.
    """

    def __init__(self,
            volume:float,
            use: Any,
            max_occupancy: Union[int, None],
            **kwargs):
        self.volume = volume
        self.use = use
        self.max_occupancy = None \
            if max_occupancy is None or np.isnan(max_occupancy) \
            else floor(max_occupancy)

        # These will be set by the .ventilation.create_ventilation_system
        self.ventilation_system = None

        self.reset(True)

        for k, v in kwargs.items(): setattr(self, k, v)



    def reset(self, full):
        """Reset the state of this box.

        Varous attributes of objects in the system hold transient state, of
        which there are two kinds: those that hold the state relevant to a
        single period, and those that hold state that 'accumulate' state over
        multiple periods. The parameter *full* describes which ones to reset. If
        it is false, then only those that are relevant to a single period are
        reset, if true, then the latter are reset also.

        :param bool full: Whether to reset state that accumulates over multiple
            periods.
        """
        self.infected_intervals = []
        self.total_exposure_risk = 0.



    def add_infected_interval(self, interval, _loop = 0):
        """Add an infected interval to this box.

        The interval is expected to have the following attributes set:

        * **person** (:py:class:`apsrm.Person`): The infected person.
        * **shedding** (:py:class:`float`): The amount the infected person is shedding.

        :param apsrm.interval.TimeInterval interval: The interval to add.
        :param int _loop: Depth of recursion. This is an implemtation detail
            and should not be set by user code.
        """

        # don't modify the input
        interval = copy(interval)

        # it is OK to recurse once to deal with segments of a interval that don't
        # overlap any existing segments, but never more than that.
        assert _loop <= 1

        if len(self.infected_intervals) == 0:
            # no intervals have been added yet
            new_interval = TimeInterval(interval.start, interval.end)
            new_interval.infected_people = [interval.person]
            new_interval.shedding = interval.shedding
            self.infected_intervals = [new_interval]
            return

        overlapping_indexes = [i for i, ti in enumerate(self.infected_intervals) \
            if ti.overlaps(interval)]

        if len(overlapping_indexes) == 0:
            # we don't have any overlapping intervals, so merge this one
            new_interval = TimeInterval(
                interval.start,
                interval.end,
                infected_people = [interval.person],
                shedding = interval.shedding)

            merge_interval(
                new_interval,
                self.infected_intervals,
                False)

            return

        # we have some overlapping intervals
        offset = 0
        for index in overlapping_indexes:
            old_interval = self.infected_intervals[index + offset]

            if old_interval.start < interval.start:
                # split the existing interval in two.
                self.infected_intervals[index + offset] = TimeInterval(
                    old_interval.start,
                    interval.start,
                    infected_people = old_interval.infected_people[:],
                    shedding=old_interval.shedding)
                offset += 1
                self.infected_intervals.insert(index + offset, TimeInterval(
                    interval.start,
                    old_interval.end,
                    infected_people = old_interval.infected_people[:],
                    shedding=old_interval.shedding))

            elif old_interval.start > interval.start:
                # add a new interval before the existing interval
                new_interval = TimeInterval(
                    interval.start,
                    old_interval.start,
                    infected_people = [interval.person],
                    shedding = interval.shedding)
                self.infected_intervals.insert(index + offset, new_interval)
                # adjust interval because we might end up looping and we need
                # to take account of what we have done
                interval.start = old_interval.start
                offset += 1

            # the overlapping interval: replace old interval
            new_interval = get_overlapping_interval(interval, old_interval)
            new_interval.infected_people = old_interval.infected_people[:] + [interval.person]
            new_interval.shedding = old_interval.shedding + interval.shedding
            self.infected_intervals[index + offset] = new_interval
            interval.start = new_interval.end

            if interval.end < old_interval.end:
                # then we need to add another interval to account of the end of the old interval
                offset += 1
                new_interval = TimeInterval(interval.end, old_interval.end)
                new_interval.infected_people = old_interval.infected_people[:]
                new_interval.shedding = old_interval.shedding
                self.infected_intervals.insert(index + offset, new_interval)

                # we know where done here, so we can...
                return

            if interval.end == old_interval.end:
                return

            else:
                interval.start = old_interval.end

        self.add_infected_interval(interval, _loop + 1)



    def add_exposure_risk(self, risk):
        """Add exposure risk to this box.

        :param float risk: The risk to add.
        """
        self.total_exposure_risk += risk



    def concentration_at_time(self, time, pathogen_dieoff_rate):
        """Retrieve the airborne viral concentration at time *time*.

        :param float time: The time (within period) to retrieve for.
        :param float pathogen_dieoff_rate: The die-off rate of airborne
            pathotgen.

        :rtype: float
        """
        return self.ventilation_system.concentration_at_time(self, time)



    def __repr__(self):
        if self.infected_intervals is not None:
            return 'Box({}, {}, {}, {})'.format(
                self.volume,
                self.use,
                self.max_occupancy,
                pformat(self.infected_intervals))

        return 'Box({}, {})'.format(self.volume, self.max_occupancy)

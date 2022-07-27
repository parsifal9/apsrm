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

import numpy as np
from ..config import END_OF_PERIOD_TIME
from ..interval import (
    get_overlapping_interval,
    intermediate_intervals,
    merge_intervals)
from .._pathogen import (
    ingestion_by_time,
    concentration_at_time)
from ._base import VentilationSubSystem



class PseudoVentilationSystem(VentilationSubSystem):
    def __init__(
            self,
            boxes,
            external_ventilation = None,
            external_ventilation_outflow = None,
            internal_filtering_volume = 0.,
            internal_filtering_efficiency = 0.,
            operates_full_period = False,
            assumptions_checked = False,
            max_day_start_time = None,
            min_day_end_time = None):

        if operates_full_period:
            self._max_day_start_time = 0.
            self._min_day_end_time = END_OF_PERIOD_TIME
        else:
            self._max_day_start_time = max_day_start_time
            self._min_day_end_time = min_day_end_time

        # would use None, but that means something else
        self._start_time = False
        self._end_time = False

        self.pathogen_dieoff_rate = None
        self.boxes = boxes

        for index, box in enumerate(boxes):
            box.ventilation_system = self
            box.ventilation_system_index = index

        n_boxes = len(boxes)

        if not assumptions_checked:
            if external_ventilation is not None:
                if external_ventilation_outflow is not None:
                    raise Exception('cannot specify both external_ventilation and external_ventilation_outflow')
                external_ventilation_outflow = external_ventilation

            elif external_ventilation_outflow is None:
                external_ventilation_outflow = 0.

        def as_array(a):
            a = np.asarray(a)
            if a.size == 1: return np.full(n_boxes, a)
            assert len(a) == n_boxes
            return a

        self.internal_filtering_volume = as_array(internal_filtering_volume)
        self.internal_filtering_efficiency = as_array(internal_filtering_efficiency)
        self.external_acph = as_array([vol / box.volume for vol, box in zip(
            as_array(external_ventilation_outflow),
            self.boxes)])


    @property
    def start_time(self):
        if self._start_time is False:
            day_starts = [i.start for bx in self.boxes for i in bx.infected_intervals]
            self._start_time = self._max_day_start_time if len(day_starts) == 0 else min(day_starts)
            if self._max_day_start_time is not None:
                self._start_time = min(self._start_time, self._max_day_start_time)
        return self._start_time


    @property
    def end_time(self):
        if self._end_time is False:
            day_ends = [i.end for bx in self.boxes for i in bx.infected_intervals]
            self._end_time = self._min_day_end_time if len(day_ends) == 0 else max(day_ends)
            if self._min_day_end_time is not None:
                self._end_time = max(self._end_time, self._min_day_end_time)
        return self._end_time


    def calculate_concentrations(
            self,
            period,
            pathogen_dieoff_rate,
            day_start = None,
            day_end = None):

        self.pathogen_dieoff_rate = pathogen_dieoff_rate

        for box in self.boxes:
            self._calculate_shedding_through_period(box, period)

        if day_start is None:
            day_start = self.start_time

        if day_end is None:
            day_end = self.end_time

        return day_start, day_end


    def _calculate_shedding_through_period(self, box, period):
        """Calculate the temporal concentration of airborne pathogen through
        the current period.
        """

        if len(box.infected_intervals) == 0:
            # then we have no intervals
            return

        i_intervals = intermediate_intervals(box.infected_intervals)

        for i in i_intervals:
            i.shedding = 0.

        box.infected_intervals = merge_intervals(
            i_intervals,
            box.infected_intervals)

        box_index = box.ventilation_system_index

        # calculate concentrations at the start of each period.
        for index, interval in enumerate(box.infected_intervals):
            if index == 0:
                interval.C0 = 0.
            else:
                interval.C0 = concentration_at_time(
                    t  = last_interval.length,
                    S  = last_interval.shedding,
                    G  = self.external_acph[box_index],
                    V  = box.volume,
                    Vf = self.internal_filtering_volume[box_index],
                    pf = self.internal_filtering_efficiency[box_index],
                    pd = self.pathogen_dieoff_rate,
                    C0 = last_interval.C0)
            last_interval = interval


    def reset(self, full):
        """Reset the state of this ventilation system.

        Varous attributes of objects in the system hold transient state, of
        which there are two kinds: those that hold the state relevant to a
        single period, and those that hold state that 'accumulate' state over
        multiple periods. The parameter *full* describes which ones to reset. If
        it is false, then only those that are relevant to a single period are
        reset, if true, then the latter are reset also.

        :param bool full: Whether to reset state that accumulates over multiple
            periods.
        """
        # would use None, but that means something else
        self._start_time = False
        self._end_time = False


    def live_with_pathogen(self, person, interval):
        box = interval.box
        box_index = box.ventilation_system_index

        # accumulate risk through the period
        return ingestion_by_time(
            t  = interval.length,
            S  = interval.shedding,
            G  = self.external_acph[box_index],
            V  = box.volume,
            Vf = self.internal_filtering_volume[box_index],
            pf = self.internal_filtering_efficiency[box_index],
            pd = self.pathogen_dieoff_rate,
            f  = interval.ingestion_filtering,
            p  = interval.breathing_rate,
            C0 = interval.C0,
            D0 = 0.,
            t0 = interval.t_to_start)


    def concentration_at_time(self, box, t):
        if len(box.infected_intervals) == 0:
            return 0.

        index = np.searchsorted(np.asarray([i.start for i in box.infected_intervals]), t)
        if index == 0:
            # time is before start of first interval
            return 0.
        interval = box.infected_intervals[index-1]

        box_index = box.ventilation_system_index

        if t > interval.end:
            # should only happen for last interval
            C0 = concentration_at_time(
                t  = interval.length,
                S  = interval.shedding,
                G  = self.external_acph[box_index],
                V  = box.volume,
                Vf = self.internal_filtering_volume[box_index],
                pf = self.internal_filtering_efficiency[box_index],
                pd = self.pathogen_dieoff_rate,
                C0 = interval.C0)
            return concentration_at_time(
                t  = t - interval.end,
                S  = 0.,
                G  = self.external_acph[box_index],
                V  = box.volume,
                Vf = self.internal_filtering_volume[box_index],
                pf = self.internal_filtering_efficiency[box_index],
                pd = self.pathogen_dieoff_rate,
                C0 = C0)
        return concentration_at_time(
            t  = t - interval.start,
            S  = interval.shedding,
            G  = self.external_acph[box_index],
            V  = box.volume,
            Vf = self.internal_filtering_volume[box_index],
            pf = self.internal_filtering_efficiency[box_index],
            pd = self.pathogen_dieoff_rate,
            C0 = interval.C0)

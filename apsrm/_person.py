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

"""Contains the implementation of :py:class:`Person`."""

from copy import copy
from functools import reduce
from math import exp, floor
import numpy as np
from scipy.stats import (
    bernoulli,
    norm as normal,
    rv_discrete as categorical)
from .config import END_OF_PERIOD_TIME
from .interval import (
    TimeInterval,
    intermediate_intervals,
    get_overlapping_interval,
    no_overlaps,
    no_gaps)



_lens = np.arange(-30., 30., 1., dtype='float')
_pdfs = np.diff(normal(0., 10.).cdf(_lens))
_pdfs /= np.sum(_pdfs)
_DAY_START_END_NOISE_IN_MINUTES = categorical(values=[_lens[1:], _pdfs])



class Person:
    """A Person.

    :ivar int age: The age of this person.

    Items in *\\**kwargs* are set as attributes on this person.
    """

    #: The efficiency of the mask at protecting a susceptible person. Used to
    #: proportionally reduce the virus they breath in. This is intended as a
    #: default that could be overriden on instances.
    ingestion_filter_efficiency = 0.

    #: The efficiency of the mask at reducing the virus shed by an infectious
    #: person. Used to proportionally reduce the virus they breath out. This
    #: is intended as a default that could be overriden on instances.
    shedding_filter_efficiency  = 0.

    def __init__(self, age, **kwargs):
        self.age = age
        self._generators = list()
        for k, v in kwargs.items(): setattr(self, k, v)
        self.reset(True)


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
        self.work_day = None
        if full:
            self.time_infected = None
            self.incubation_period = None
            self.is_honest = True
            self.vaccine_history = []
            self.starting_intervals = []


    @property
    def period_infected(self) -> int:
        """The period this person became infected."""
        return None if self.time_infected is None else floor(self.time_infected)


    def vaccinate(self, time, vaccine):
        """Vaccinate this person.

        This appends a tuple containing *time* and *vaccine* to this persons
        vaccine history.

        :param float time: The time of the vaccination.
        :param apsrm.Vaccine vaccine: The vaccine used.
        """
        self.vaccine_history.append((time, vaccine))


    def _notify_of_infection(self, time):
        """Meant to be overloaded by sub classes to respond to infection
        notifications."""
        pass


    def infect(self, time, pathogen):
        """Set the time at which this person becomes infected.

        .. note:: This does not check if the individual has already been
            infected.

        :param float time: Time since the beginning of the simulation. ``None``
            specifies that the person has never been infected.
        """
        self.time_infected = time
        self.incubation_period = pathogen.incubation_period(self)
        self.is_honest = pathogen.is_honest(self)
        self._notify_of_infection(time)


    def is_infected_by(self, time):
        return self.time_infected is not None and self.time_infected <= time


    def shows_symptoms_by(self, time):
        return self.time_infected is not None and \
            (time - self.time_infected) >= self.incubation_period


    def add_generator(self, generator):
        self._generators.append(generator)


    def add_generators(self, generators):
        self._generators += generators


    def active_during_period(self, period):
        return True


    def generate_schedule(self,
            period: int,
            gatherings: list,
            emissions_calculator):
        """Generate a schedule of activities for a period.

        :param period: Period to generate the schedule for.

        :param gatherings: List of gatherings this Person is already attending.

        :param emissions_calculator: A class with various methods required for
            calculating how much an infectious individual will shed throughout
            the period.
        """

        if not self.active_during_period(period):
            return []

        # TODO: Warn when we drop an interval...
        # because it overlaps with self.starting_intervals).
        my_intervals = [TimeInterval(
            g.start,
            g.end,
            gathering=g,
            box=g.box) for g in gatherings if g.does_not_overlap_any(
                self.starting_intervals)] + self.starting_intervals

        self.starting_intervals = []

        my_intervals.sort(key=lambda i: i.start)
        assert no_overlaps(my_intervals)

        work_day = self._generate_work_day(period)

        if work_day is None:
            assert len(my_intervals) > 0
            work_day = TimeInterval(my_intervals[ 0].start, my_intervals[-1].end)
        elif len(my_intervals) > 0:
            work_day.start = min(my_intervals[ 0].start, work_day.start)
            work_day.end   = max(my_intervals[-1].end,   work_day.end)

        self.work_day = work_day

        # get the available intervals
        if len(my_intervals) == 0:
            available_intervals = [work_day]

        else:
            start_interval = [] if my_intervals[0].start <= work_day.start else \
                [TimeInterval(work_day.start, my_intervals[0].start)]

            end_interval = [] if my_intervals[-1].end >= work_day.end else \
                [TimeInterval(my_intervals[-1].end, work_day.end)]

            middle_intervals = [] if len(my_intervals) < 2 else \
                intermediate_intervals(my_intervals)

            available_intervals = start_interval + middle_intervals + end_interval

            if len(available_intervals) == 0:
                # TODO: check that this makes sense (just bunged in here in a rush)
                available_intervals = [work_day]

        # generate all items in schedule
        for g in self._generators:
            my_intervals, available_intervals = \
                g.create_intervals(self, my_intervals, available_intervals)

        assert no_overlaps(my_intervals)
        assert no_gaps(my_intervals)

        my_intervals = [i for i in my_intervals if
            (i.box is not None and i.length > 0.)]

        # add this person to every box
        for interval in my_intervals:
            interval.person = self

        if self.is_infected_by(period):
            if len(self.vaccine_history):
                vaccination_time, vaccine = self.vaccine_history[-1]
                relative_infectiousness = vaccine.relative_infectiousness(
                    self, period - vaccination_time)
            else:
                relative_infectiousness = 1.

            for interval in my_intervals:
                interval.shedding = relative_infectiousness * emissions_calculator.emissions(
                    period,
                    self,
                    interval,
                    getattr(interval, 'gathering', None))

        else:
            for interval in my_intervals:
                interval.breathing_rate = emissions_calculator.breathing_rate(
                    period,
                    self,
                    interval,
                    getattr(interval, 'gathering', None))

        # ensure that all intervals end before the end of the period and keep
        # any residuals for the next

        def splitter(interval):
            if interval.end <= END_OF_PERIOD_TIME:
                return interval, None

            if interval.start > END_OF_PERIOD_TIME:
                return None, interval

            end = interval.end
            interval.end = END_OF_PERIOD_TIME

            next_periods_interval = copy(interval)
            next_periods_interval.start = 0.
            next_periods_interval.end = end - END_OF_PERIOD_TIME

            return interval, next_periods_interval

        my_intervals = [splitter(i) for i in my_intervals]
        self.starting_intervals = [i[1] for i in my_intervals if i[1] is not None]

        return [i[0] for i in my_intervals if i[0] is not None]


    def live_with_pathogen(
            self,
            period,
            pathogen,
            schedule,
            emissions_calculator):
        """Calculate exposure, the resulting risk and potentially infect this person.

        :param int period: The period to 'live through'.
        :param eipboxes.Pathogen pathogen: The pathogen to 'live with'.
        :param list[apsrm.TimeInterval] schedule: This person's schedule
            through the period.
        """
        if self.is_infected_by(float(period)):
            # TODO: raise exception or return?
            return

        all_intervals = []
        for appointment in schedule:
            box = appointment.box
            intervals_that_overlap = [bi for bi in box.infected_intervals if bi.overlaps(appointment)]
            for oi in intervals_that_overlap:
                overlap                = get_overlapping_interval(appointment, oi)
                overlap.box            = box
                overlap.breathing_rate = appointment.breathing_rate
                overlap.ingestion_filtering = \
                    emissions_calculator.ingestion_filtering_in_box(self, appointment)
                overlap.shedding       = oi.shedding
                overlap.C0             = getattr(oi, 'C0', None)
                overlap.t_to_start     = overlap.start - oi.start
                all_intervals.append(overlap)

        ingestions = [i.box.ventilation_system.live_with_pathogen(self, i)
            for i in all_intervals]

        for ingestion, interval in zip(ingestions, all_intervals):
            interval.box.add_exposure_risk(
                pathogen.probability_of_infection(ingestion))

        pr_infection = pathogen.probability_of_infection(sum(ingestions))

        if len(self.vaccine_history):
            vaccination_time, vaccine = self.vaccine_history[-1]
            relative_susceptibility = vaccine.relative_susceptibility(
                self, period - vaccination_time)
        else:
            relative_susceptibility = 1.

        # TODO: Whether *time* is the period or the time within the period
        # needs to be determined (see the previous todo).
        if bernoulli.rvs(relative_susceptibility * pr_infection, size=1)[0] == 1:
            self.infect(period + .5, pathogen)

        return pr_infection


    def _generate_work_day(self, period):
        return TimeInterval(
            8.5 + _DAY_START_END_NOISE_IN_MINUTES.rvs(size=1)[0]/60.,
            17. + _DAY_START_END_NOISE_IN_MINUTES.rvs(size=1)[0]/60.)


    def __repr__(self):
        # TODO: Add the kwargs from __init__
        return 'Person({})'.format(self.age)

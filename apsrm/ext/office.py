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

"""Utilities for offices... and maybe other workplaces.

The classes implemented here are used for generating schedules for individuals.
There are two main types of generators: generators for gatherings (where people
do something together, like meetings), and generators for indivual activites
(where people do things as individuals, like have lunch or go to the toilet).

Generators for Individuals
--------------------------

These extend :py:class:`.PeriodGenerator` and generate activites for a single
individual. :py:class:`WorkingGenerator` 'fills in the gaps' around other
activities. :py:class:`EgressGenerator` tacks a trip through the foyer onto the
beginning and end of the working day as an agent enters and leaves the
building.The others things like lunch breaks and toilet breaks. The other
generators e.g. :py:class:`LunchGenerator`) give some choice to the agent over
when they do these things and can shuffle them around the gatherings and other
activities they are scheduled to participate.

Generators for Gatherings
-------------------------

These extend :py:class:`.GatheringGenerator` and generate gatherings where
attended by multiple agent. These are run before the
:py:class:`.PeriodGenerator`. :py:class:`MeetingGenerator` generates meetings.
In other workplaces, different types of gatherings would be appropriate. For
example, in a domestic building site, you might use gatherings to form work
teams who work together as a group and move from room to room. Other examples
might be delivery drivers travelling together in a truck, or medical
professionals performing an operation.
"""

import random
import abc
from math import floor
from functools import reduce
from collections import defaultdict
import numpy as np
from scipy.stats import (
    bernoulli,
    rv_discrete as categorical,
    norm as normal)
from .._box import Box
from .._workplace import (
    Gathering,
    GatheringGenerator,
    PeriodGenerator)
from .._person import Person
from ..interval import (
    TimeInterval,
    merge_interval,
    overlap_length)


def _select_and_merge_best_interval(best_interval, current_intervals, available_intervals):
    if len(available_intervals) == 0:
        # The other option is to split an existing interval
        # TODO: warning?
        return current_intervals, available_intervals

    # we don't want ot modify the input
    available_intervals = available_intervals[:]

    # tuples containing:
    #   - index of interval,
    #   - fractional length of overlap
    #   - distance between mid points
    overlaps = [(
        index,
        overlap_length(best_interval, interval) / best_interval.length,
        abs(best_interval.midpoint - interval.midpoint)
        ) for index, interval in enumerate(available_intervals)]

    class dist:
        def __init__(self, i, a, b): self.i, self.a, self.b = i, a, b
        def __lt__(self, other): return self.a < other.a or self.b < other.b
        def __repr__(self): return 'dist({}, {}, {})'.format(self.i, self.a, self.b)

    def loss(i, overlap, distance):
        return \
            dist(i, 0, -overlap) if distance < .5 \
            else dist(i, 1, distance) if overlap  > 0. \
            else dist(i, 2, distance)

    best_available_index = sorted(loss(*o) for o in overlaps)[0].i
    best_available_interval = available_intervals[best_available_index]
    best_is_contained = best_available_interval.contains(best_interval)
    if best_is_contained:
        available_intervals.pop(best_available_index)
        available_intervals[best_available_index:best_available_index] = \
            best_available_interval.remove(best_interval)

    else:
        best_available_resid = best_interval.shift_to_nearest_end(
            best_available_interval, True)
        if best_available_resid is None:
            available_intervals.pop(best_available_index)
        else:
            available_intervals[best_available_index] = best_available_resid

    current = merge_interval(best_interval, current_intervals)
    return current, available_intervals


def _GEN_MEETING_LENGTHS(weights, modes):
    lens = np.arange(0, 122, 2, dtype='float') / 60.
    cdfs = [np.diff(m.cdf(lens)) for m in modes]
    wght = np.matmul(np.asarray(weights), cdfs)
    return lens[1:], wght / np.sum(wght)


class MeetingGenerator(GatheringGenerator):
    """Generate meetings."""

    def __init__(self, gathering_type, worker_types, box_types):
        self._gathering_type = gathering_type
        self._worker_types = worker_types
        self._box_types = box_types

    _ml, _ml_weights = _GEN_MEETING_LENGTHS(
        [.2, .35, .3, .15],
        [normal(20./60., 3./60.),
         normal(45./60., 6./60.),
         normal(1.     , 9./60.),
         normal(1.5    , 9./60.)])
    day_start = 8.5
    day_end   = (16.  + .51/60.)
    _number_of_meetings_generator = categorical(values=[
        [1, 2, 3, 4, 5],
        [.2, .2, .2, .2, .2]])

    def _meeting_length(self):
        return np.random.choice(self._ml, 1, p=self._ml_weights)[0]

    def create_gatherings(self, workplace, period, current_gatherings):
        """Must return a list of gatherings."""

        def meetings_for_day(box):
            res = []
            start = self.day_start
            end   = start
            while end < self.day_end:
                length = self._meeting_length()
                end = start + length
                res.append(Gathering(
                    start, end,
                    box,
                    gathering_type=self._gathering_type))
                start = end
            return res

        class MeetingManager:
            def __init__(self, person, n_meetings_to_attend, meetings):
                self.person = person
                self.n_meetings_to_attend = n_meetings_to_attend
                self.n_meetings_attended  = 0
                self.unattended_meetings  = meetings[:]
                self.attended_meetings    = []

            def __call__(self):
                self.attended_meetings += [m for m in self.unattended_meetings \
                    if len(m) > 0]

                self.unattended_meetings = [m for m in self.unattended_meetings \
                    if len(m) == 0]

                self.attended_meetings = [m for m in self.attended_meetings \
                    if m.remaining_places > 0]

                meetings = self.attended_meetings \
                    if len(self.attended_meetings) else self.unattended_meetings

                if len(meetings):
                    meeting_index = random.randrange(len(meetings))
                    meeting = meetings.pop(meeting_index)
                    meeting.add_participant(self.person)
                    self.n_meetings_attended += 1

                    self.attended_meetings = [m for m in self.attended_meetings \
                        if not m.overlaps(meeting)]
                    self.unattended_meetings = [m for m in self.unattended_meetings \
                        if not m.overlaps(meeting)]

                return self._is_done()

            def _is_done(self):
                return self.n_meetings_attended >= self.n_meetings_to_attend or \
                    (len(self.unattended_meetings) == 0 and \
                        len(self.attended_meetings) == 0)

        # get the meeting rooms
        meeting_rooms = [b for b in workplace.boxes if b.use in self._box_types]
        if len(meeting_rooms) == 0: raise Exception('no meeting rooms in workplace')

        # get the workers
        workers = [p for p in workplace.persons if p.role in self._worker_types]
        if len(workers) == 0: raise Exception('no workers in workplace')

        # list of all meetings
        available_meetings = reduce(lambda ms, b: ms + meetings_for_day(b), meeting_rooms, [])

        # list of tuples of workers, how many meetings they will attend, and
        # all meetings they could attend
        worker_meeting_counts = [MeetingManager(w, c, available_meetings) for w, c in zip(
            workers, self._number_of_meetings_generator.rvs(size=len(workers)))
                if w.active_during_period(period)]

        # allocate each worker to meetings to fill up quotas
        while len(worker_meeting_counts) > 0:
            worker_index = random.randrange(len(worker_meeting_counts))
            if worker_meeting_counts[worker_index]():
                worker_meeting_counts.pop(worker_index)

        # filter out meetings with no paticipants
        return [m for m in available_meetings if len(m) > 0]


class GatheringVisits(GatheringGenerator):
    def __init__(self,
            visitor_class,
            gathering_types_to_visit,
            generators,
            n_visitors_in_period):
        self._gathering_types_to_visit = gathering_types_to_visit
        self._visitor_class = visitor_class
        self._generators = generators
        self._n_visitors_in_period = n_visitors_in_period

    def create_gatherings(self, workplace, period, current_gatherings):
        """Must return a list of gatherings."""

        visitable_gatherings = [g for g in current_gatherings \
            if g.gathering_type in self._gathering_types_to_visit]

        n_visitors = self._n_visitors_in_period(visitable_gatherings)
        assert len(n_visitors) == len(visitable_gatherings)

        for g, n in zip(visitable_gatherings, n_visitors):
            for i in range(n):
                visitor = self._visitor_class(48)
                visitor.add_generators(self._generators)
                workplace.add_visitor(visitor)
                g.add_participant(visitor, force_participation=True)


class LunchGenerator(PeriodGenerator):
    LUNCH_DIST =    normal(12.5  , 20./60.)
    DURATION_DIST = normal(40./60,  5./60.)

    @classmethod
    def start_time_generator(cls, *args, **kwargs):
        return cls.LUNCH_DIST.rvs(size=1)[0]

    @classmethod
    def duration_generator(cls, *args, **kwargs):
        return cls.DURATION_DIST.rvs(size=1)[0]

    def __init__(self,
            start_time_generator = None,
            duration_generator = None,
            proportion_that_leave_workplace=None):

        if start_time_generator is not None:
            start_time_generator = start_time_generator

        if duration_generator is not None:
            duration_generator = duration_generator

        if proportion_that_leave_workplace is not None \
            and not 0. <= proportion_that_leave_workplace <= 1.:
                raise Exception('proportion that leave workplace must be in [0, 1]')

        self._proportion_that_leave_workplace = proportion_that_leave_workplace

    def start_time(self, person):
        return max(0., self.start_time_generator(person))

    def duration(self, person):
        return max(0., self.duration_generator(person))

    def create_intervals(self, person, current_intervals, available_intervals):
        box = person.kitchen \
            if self._proportion_that_leave_workplace is None \
            else (None if bernoulli.rvs(self._proportion_that_leave_workplace, size=1)[0] \
            else person.kitchen)
        start = self.start_time(person)
        duration = self.duration(person)
        best_interval = TimeInterval(start, start + duration, box=box)
        new_curr, new_avail = _select_and_merge_best_interval(
            best_interval, current_intervals, available_intervals)
        return new_curr, new_avail


class RepeatingBreaks(PeriodGenerator):
    @abc.abstractmethod
    def time_until_next_interval(self, first_break):
        pass

    @abc.abstractmethod
    def interval_length(self):
        pass

    @abc.abstractmethod
    def interval_box(self, person):
        pass

    def create_intervals(self, person, current_intervals, available_intervals):
        """Can modify avilable_intervals."""

        if len(available_intervals) == 0:
            # TODO: This is bad, do the right thing
            return current_intervals, available_intervals

        day_start = min(available_intervals[0].start, person.work_day.start)
        day_end   = max(available_intervals[-1].end,  person.work_day.end)

        best_intervals = []
        last_end = day_start
        first_break = True
        while last_end < day_end:
            brk_start = last_end + self.time_until_next_interval(first_break)
            first_break = False
            if brk_start > day_end: break
            last_end = brk_start + self.interval_length()
            if last_end > day_end:
                d = last_end - brk_start
                brk_start += d
                last_end  += d
            best_intervals.append(TimeInterval(brk_start, last_end, box=self.interval_box(person)))

        for best_interval in best_intervals:
            current_intervals, available_intervals = _select_and_merge_best_interval(
                best_interval,
                current_intervals,
                available_intervals)

        return current_intervals, available_intervals


class ToiletBreakGenerator(RepeatingBreaks):
    BREAK_INTERVAL_DIST = normal(4.    , 1.)
    BREAK_LENGTH_DIST =   normal(3./60.,  .75/60.)

    def time_until_next_interval(self, first_break):
        """Length of time between toilet breaks."""
        mult = .5 if first_break else 1.
        return max(1., mult * self.BREAK_INTERVAL_DIST.rvs(size=1)[0])

    def interval_length(self):
        return max(1./60., self.BREAK_LENGTH_DIST.rvs(size=1)[0])

    def interval_box(self, person):
        return person.toilet

class EgressGenerator(PeriodGenerator):
    def __init__(self, egress_space):
        self.box = egress_space

    def create_intervals(self, person, current_intervals, available_intervals):
        """Generate intervals for the time it takes to get from the entry to
        the usual place of work or vice versa.
        """
        arrive = TimeInterval(
            current_intervals[0].start - 5./60.,
            current_intervals[0].start,
            box = self.box)
        leave  = TimeInterval(
            current_intervals[-1].end,
            current_intervals[-1].end + 5./60.,
            box = self.box)
        current = [arrive] + current_intervals + [leave]
        return current, available_intervals


class WorkingGenerator(PeriodGenerator):
    """Generate periods in which the person is working."""
    def create_intervals(self, person, current_intervals, available_intervals):
        """Can modify avilable_intervals"""

        # fill in gaps with work.
        current = reduce(
            lambda ps, p12: ps + [p12[0]] + ([TimeInterval(
                p12[0].end,
                p12[1].start,
                box = person.work_box)] if (p12[0].end != p12[1].start) else []),
            zip(current_intervals[:-1], current_intervals[1:]), []) + \
                [current_intervals[-1]]

        if len(available_intervals) > 0:
            start = [] if available_intervals[0].start >= current[0].start else [
                TimeInterval(
                    available_intervals[0].start,
                    current[0].start,
                    box = person.work_box)]

            end = [] if available_intervals[-1].end <= current[-1].end else [
                TimeInterval(
                    current[-1].end,
                    available_intervals[-1].end,
                    box = person.work_box)]

            current = start + current + end

        return current, []


def add_ventilation_matrix(
        workplace,
        external_acph,
        air_cleaner_filtering_efficiency,
        air_cleaner_filtering_volume,
        air_cleaner_box_types = None,
        ventilation_matrix = None,
        hvac_acph = None,
        inter_box_acph = None,
        hvac_box_type = None,
        hvac_return_filtering_efficiency = None):

    if ventilation_matrix is not None:
        assert hvac_acph is not None
        assert inter_box_acph is not None
        assert hvac_box_type is not None
        assert hvac_return_filtering_efficiency is not None

        hvac_boxes = [box for box in workplace.boxes if box.use == hvac_box_type]
        if len(hvac_boxes):
            assert len(hvac_boxes) == 1
            hvac_box = hvac_boxes[0]
        else:
            hvac_box = Box(1., hvac_box_type, 0, name='hvac')
            workplace.add_box(hvac_box)

        n_boxes = len(workplace.boxes)

        s = ventilation_matrix.shape
        vm = np.zeros((s[0]+1, s[1]+1), dtype=np.float64)
        # implicitly assumes that the hvac_box index is vm.shape[0] - 1
        vm [0:-1, 0:-1] = inter_box_acph * ventilation_matrix

        vm[:, hvac_box.box_index] = vm[hvac_box.box_index, :] = \
            hvac_acph * np.array([b.volume for b in workplace.boxes])

        vm[hvac_box.box_index, hvac_box.box_index] = 0.

        # external ventilation. Note that we do all ventilation via the HVAC system.
        external_ventilation = np.zeros(n_boxes, dtype=np.float64)
        external_ventilation[hvac_box.box_index] = \
            external_acph * sum(box.volume for box in workplace.boxes)

        # stand alone air filters
        filtration_volume     = np.full(n_boxes, air_cleaner_filtering_volume, dtype=np.float64)
        filtration_efficiency = np.full(n_boxes, air_cleaner_filtering_efficiency, dtype=np.float64)
        if air_cleaner_box_types is None:
            filtration_volume[hvac_box.box_index]     = 0.
            filtration_efficiency[hvac_box.box_index] = 0.

        else:
            for box in workplace.boxes:
                if box.use not in air_cleaner_box_types:
                    filtration_volume[box.box_index] = 0.
                    filtration_efficiency[box.box_index] = 0.

        # add the hvac to the workplace
        workplace.set_ventilation_properties(
            ventilation_matrix               = vm,
            hvac_box_type                    = hvac_box_type,
            external_ventilation             = external_ventilation,
            internal_filtering_volume        = filtration_volume,
            internal_filtering_efficiency    = filtration_efficiency,
            hvac_return_filtering_efficiency = hvac_return_filtering_efficiency)

    else:
        # add the hvac to the workplace
        workplace.set_ventilation_properties(
            external_ventilation             = external_ventilation,
            internal_filtering_volume        = filtration_volume,
            internal_filtering_efficiency    = filtration_efficiency)


def populate_workplace(
        workplace,
        box_types,
        worker_types,
        frac_of_max_occupancy,
        receptionist_count,
        proportion_leaving_for_lunch):
    """Populate the workplace with workers."""

    assert 0. < frac_of_max_occupancy <= 1.

    lunch_generator        = LunchGenerator(proportion_leaving_for_lunch)
    toilet_break_generator = ToiletBreakGenerator()
    working_generator      = WorkingGenerator()

    # split data into room types
    grouped_boxes = defaultdict(list)
    for box in workplace.boxes: grouped_boxes[box.use].append(box)

    # assume that there are enough seats for all workers... if there
    # are not, then this will raise. Check with Kamran this is the case
    # (or read more closely).
    reception      = grouped_boxes.pop(box_types.FOYER)
    kitchens       = grouped_boxes.pop(box_types.KITCHEN)
    toilets        = grouped_boxes.pop(box_types.TOILET)
    open_plans     = grouped_boxes.pop(box_types.OPEN_PLAN, None)
    closed_offices = grouped_boxes.pop(box_types.OFFICE, None)

    # should only be one reception
    assert len(reception) == 1
    reception = reception[0]

    generators = [
        lunch_generator,
        toilet_break_generator,
        working_generator,
        EgressGenerator(reception)]

    def add_generators(person):
        for g in generators: person.add_generator(g)
        workplace.add_person(person)

    def create_persons_in_boxes(boxes, worker_type):
        if boxes is None:
            return 0

        total_capacity = floor(frac_of_max_occupancy * sum(o.max_occupancy for o in boxes))
        if total_capacity == 0:
            return 0

        capacities = [[b, b.max_occupancy] for b in boxes if b.max_occupancy > 0]

        for w in range(total_capacity):
            box_index = random.randrange(len(capacities))
            work_box = capacities[box_index][0]
            capacities[box_index][1] -= 1
            if capacities[box_index][1] == 0:
                capacities.pop(box_index)
            person = Person(
                48,
                role     = worker_type,
                work_box = work_box,
                kitchen  = random.sample(kitchens, 1)[0],
                toilet   = random.sample(toilets, 1)[0])
            add_generators(person)

        return total_capacity

    # put workers in spaces
    total_capacity =  create_persons_in_boxes(open_plans, worker_types.OPEN_WORKER)
    total_capacity += create_persons_in_boxes(closed_offices, worker_types.OFFICE_WORKER)

    # and throw in a couple of receptionists for good measure
    for w in range(receptionist_count):
        person = Person(
            48,
            role     = worker_types.RECEPTIONIST,
            work_box = reception,
            kitchen  = random.sample(kitchens, 1)[0],
            toilet   = random.sample(toilets, 1)[0])
        add_generators(person)

    total_capacity += receptionist_count

    return total_capacity

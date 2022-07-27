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

"""Contains the implementation of :py:class:`Workplace`."""

from functools import reduce
from itertools import chain, repeat
from abc import ABC, abstractmethod
from random import sample
from math import floor
import numpy as np
from .config import END_OF_PERIOD_TIME
from .interval import TimeInterval
from .ventilation import create_ventilation_system



class EveryoneInfected(Exception):
    """Raised if no one in a workplace is susceptible.

    :ivar int period: The period in which this exception is raised.

    :param int period: The period in which this exception is raised.
    """

    def __init__(self, period):
        self.period = period
        super().__init__(f'Everyone infected at period {period}')



class Gathering(TimeInterval):
    """A group of individuals in a apsrm.Box.

    :ivar set[apsrm.Person] participants: Participants scheduled to attend
        the gathering.  Participants should be added using :py:meth:`add_partipants`
        or :py:meth:`add_partipants`, both of which check that there is space left
        to add them.

    :ivar apsrm.Box box: The box the gathering will take place in.

    :param float start: The start time of the gathering.
    :param float end: The end time of the gathering.
    :param apsrm.Box box: The box the gathering will take place in.

    *\\*args* and *\\**kwargs* are passed to the base class constructor.
    """
    def __init__(self, start, end, box, *args, **kwargs):
        super().__init__(start, end, *args, **kwargs)
        self.box = box
        self.participants = set()

    def add_participant(self, participant, force_participation=False):
        """Add a *participant* to the gathering if there is space left in the
        gathering, or if *force_participation* is true.

        The maximum number of participants is specified by
        ``self.box.max_occupancy``.

        :param apsrm.Person participant: The participant to add.

        :param bool force_participation: Fore the participant to join the
            meeting even if ``self.box.max_occupancy`` is exceeded.

        :raises Exception: If there is not enough space left in the meeting
            (and *force_participation* is not true).
        """

        # Note that self.participants is a set, and if *participant* is already
        # an attendee, they will not be added a second time... so rather than
        # check the number of participants first, we add the 'new' participant
        # and check the size of the resulting set.
        self.participants.add(participant)
        if not force_participation and self.box.max_occupancy < len(self.participants):
            self.participants.remove(participant)
            raise Exception('box capacity reached, cannot add paticipant')

    def add_participants(self, participants, force_participation=False):
        """Add *participants* to the gathering if there is space left in the
        gathering, or if *force_participation* is true.

        The maximum number of participants is specified by
        ``self.box.max_occupancy``.

        :param apsrm.Person participant: The participant to add.

        :param bool force_participation: Force the participant to join the
            meeting even if ``self.box.max_occupancy`` is exceeded.

        :raises Exception: If there is not enough space left in the meeting (and
            *force_participation* is not true).
        """

        # Note that self.participants is a set, and if a participant is already
        # an attendee, they will not be added a second time... so rather than
        # check the number of participants first, we add the 'new' participants
        # and check the size of the resulting set.
        self.participants.update(participants)
        if not force_participation and self.box.max_occupancy < len(self.participants):
            self.participants -= participants
            raise Exception('box capacity reached, cannot add paticipants')

    @property
    def remaining_places(self):
        """The number of places remaining."""
        if self.box.max_occupancy is None: return None
        return self.box.max_occupancy - len(self.participants)

    def __len__(self):
        """The number of participants currently scheduled to attend the gathering."""
        # TODO: Are the semantics here 'correct'...
        #: or am I just trying to be clever?
        return len(self.participants)




class GatheringGenerator(ABC):
    """Base class for generators of gatherings.

    Gathering generators generate lists of gatherings (:py:class:`Gathering`)
    and should be added to workplaces using
    :py:meth:`apsrm.Workplace.add_generator`. At the start of the method
    :py:meth:`apsrm.Workplace.run_period`, the method
    :py:meth:`create_gatherings` will be called on all gathering generators
    added to the workplace in the order in which they are added to the
    workplace, each being passed the list of gatherings returned by the one
    before it (the first being passed an empty list).  Generators would
    typically add to or otherwise modify that list and return it.
    """

    @abstractmethod
    def create_gatherings(self, workplace, period, current_gatherings):
        """Generate gatherings.

        :param apsrm.Workplace workplace: The workplace associated with a
            gathering. Note that gathering need not be scheduled in a box
            within the workplace.

        :param int period: The period in which gatherings are being created.

        :param list[apsrm.Gathering] current_gatherings: A list of
            gatherings that have already been scheduled. Note there is nothing
            stopping a generator from modifying that list.

        :return: A list of gatherings.
        :rtype: list[apsrm.Gathering]
        """
        pass



class PeriodGenerator(ABC):
    """Generate intervals for an individual.

    Instances of this class can be added to :py:class:`apsrm.Person` via the
    method :py:meth:`apsrm.Person.add_generator`. These will be called in
    the order they are added from the method
    :py:meth:`apsrm.Person.generate_schedule`."""

    @abstractmethod
    def create_intervals(self, person, current_intervals, available_intervals):
        """Generate intervals for an individual.

        :param apsrm.Person person: Person to create activities for.
        :param list[apsrm.interval.TimeInterval] current_intervals:
            Previously scheduled activities.
        :param list[apsrm.interval.TimeInterval] available_intervals:
            Available intervals in which activities can be scheduled.

        :return: Two lists of intervals: the first is the (updated) activities,
            and the second is the updated available intervals. These will be
            passed as arguments *current_intervals* and *available_intervals*
            to the next period generator.

        :rtype: tuple[list[apsrm.interval.TimeInterval], list[apsrm.interval.TimeInterval]]
        """
        pass



class Workplace:
    """Represents a workplace.

    The 'main entry point' for the framework is :py:meth:`run_period`. The
    source of that method is fairly self explanitory, so please have a look
    at that to understand most of how this framework works.

    :ivar bool operates_full_period: Does this workplace operate for full periods?
    :ivar list[apsrm.Box] boxes: rooms and HVAC systems,
    :ivar set[apsrm.Person] persons: people who remain in the workplace
        through multiple periods,
    :ivar ventilation_system: The ventilation system in the workplace.
    :ivar dict[apsrm.Person, list[apsrm.interval.TimeInterval]] infected_schedules:
        Lists of intervals describing the schedules of the infected workers in
        the office for a period. The intervals contained in the lists have extra
        attributes, including ... .
    :ivar dict[apsrm.Person, list[apsrm.interval.TimeInterval]] raw_schedules:
        Lists of intervals describing the schedules of all workers in the office
        for a period. The intervals contained in the lists have extra
        attributes, including ... .
    :ivar float day_start: The time the day starts in the office.
    :ivar float day_end: The time the day ends in the office.
    :ivar set[apsrm.Person] visitors: visitors to the workplace in a single
        period.
    """
    def __init__(self, box_types, worker_types, operates_full_period=False):
        self._box_types = box_types
        self._worker_types = worker_types
        self._generators = list()
        self.operates_full_period = operates_full_period
        self.boxes = list()
        self.persons = set()
        self.ventilation_system = None
        self.reset(True)


    def reset(self, full=False):
        """Reset the state of this workplace.

        Varous attributes of objects in the system hold transient state, of
        which there are two kinds: those that hold the state relevant to a
        single period, and those that hold state that 'accumulate' state over
        multiple periods. The parameter *full* describes which ones to reset. If
        it is false, then only those that are relevant to a single period are
        reset, if true, then the latter are reset also.

        :param bool full: Whether to reset state that accumulates over multiple
            periods.
        """
        self.infected_schedules = None
        self.raw_schedules = None
        self.day_start = None
        self.day_end = None
        self.visitors = set()

        if self.ventilation_system is not None:
            self.ventilation_system.reset(full)

        for box in self.boxes:
            box.reset(full)

        for person in self.persons:
            person.reset(full)


    def set_ventilation_properties(
            self,
            ventilation_matrix = None,
            hvac_box_type = None,
            external_ventilation = None,
            external_ventilation_outflow = None,
            external_ventilation_inflow = None,
            internal_filtering_volume = 0.,
            internal_filtering_efficiency = 0.,
            hvac_return_filtering_efficiency = 0.,
            force_standard_hvac_system = False):
        """Set the ventilation properties for this workplace.

        :return: The previously set ventilation system.
        :rtype: apsrm.ventilation.VentilationSystem
        """

        vs = self.ventilation_system

        # Note that all the conditions on ventilation_matrix are checked in
        # create_ventilation_system.
        self.ventilation_system = create_ventilation_system(
            boxes = self.boxes,
            ventilation_matrix = ventilation_matrix,
            hvac_box_type = hvac_box_type,
            external_ventilation = external_ventilation,
            external_ventilation_outflow = external_ventilation_outflow,
            external_ventilation_inflow = external_ventilation_inflow,
            internal_filtering_volume = internal_filtering_volume,
            internal_filtering_efficiency = internal_filtering_efficiency,
            hvac_return_filtering_efficiency = hvac_return_filtering_efficiency,
            force_standard_hvac_system = force_standard_hvac_system)

        return vs


    def remove_ventilation_system(self):
        # TODO: Should reset be called on vs before returning?
        # Answer: Probably not, as that may erase data we want to use later.
        # Doing things in this order ensures it does not get reset.
        vs = self.ventilation_system
        self.ventilation_system = None
        self.reset(True)
        return vs


    def add_box(self, box):
        if box not in self.boxes:
            try:
                is_meeting_room = box.use == self._box_types.MEETING
            except AttributeError:
                is_meeting_room = False

            if is_meeting_room:
                if box.max_occupancy is None:
                    raise Exception('meeting room with unspecified capacity detected')
                if box.max_occupancy == 0:
                    raise Exception('meeting room with zero capacity detected')

            box.box_index = len(self.boxes)
            self.boxes.append(box)
        else:
            # TODO: emit a warning
            pass


    def add_person(self, person):
        # TODO: should we be checking if this person is already added?
        self.persons.add(person)


    def add_visitor(self, person):
        # TODO: should we be checking if this person is already added?
        self.visitors.add(person)


    def add_generator(self, generator):
        self._generators.append(generator)


    def run_period(
            self,
            period,
            pathogen,
            emissions_calculator,
            raw_schedules = None):

        if raw_schedules is None:
            # generate meetings
            meeting_schedules = self._generate_gatherings(period)

            # generate schedules for all people
            raw_schedules = {p: p.generate_schedule(
                period,
                [ms for ms in meeting_schedules if p in ms.participants],
                emissions_calculator) for p in chain(self.persons, self.visitors)}

        # add infected intervals to boxes
        infected_schedules = {p: s for p, s in raw_schedules.items() \
            if p.is_infected_by(period)}

        self.raw_schedules = raw_schedules
        self.infected_schedules = infected_schedules

        for schedule in infected_schedules.values():
            for interval in schedule:
                interval.box.add_infected_interval(interval)

        # calculate viral concentrations in boxes
        if self.ventilation_system is None:
            self.ventilation_system = create_ventilation_system(
                self.boxes,
                operates_full_period = self.operates_full_period)

        # prepare the shedding rates and time intervals for the boxes
        self.day_start, self.day_end = \
            self.ventilation_system.calculate_concentrations(
                period,
                pathogen.dieoff_rate)

        # run uninfected people through boxes
        prs_of_infection = [person.live_with_pathogen(
            period,
            pathogen,
            schedule,
            emissions_calculator) \
            for person, schedule in raw_schedules.items() \
                if not person.is_infected_by(period)]

        if len(prs_of_infection) == 0:
            raise EveryoneInfected(period)

        return max(prs_of_infection)


    def _do_infect_random_persons(
            self,
            persons,
            pathogen,
            time,
            n,
            ignore_infection_status):

        assert(n >= 0)

        # TODO: emit a warning
        if n == 0:
            return 0

        if np.isscalar(time):
            times = repeat(time, n)
        else:
            # TODO: emit a warning if ignore_infection_status was False
            ignore_infection_status = True
            times = chain(*repeat(time, n))
            n *= len(time)

        if not ignore_infection_status:
            persons = [p for p in persons if not p.is_infected_by(time)]
        n_infectible = len(persons)

        if n_infectible == 0:
            return 0

        if n_infectible < n:
            # TODO: emit a warning
            n = n_infectible

        for p, t in zip(sample(persons, n), times):
            p.infect(t, pathogen)

        return n


    def infect_random_persons(self,
            pathogen,
            time=-1,
            n=1,
            ignore_infection_status=False):
        return self._do_infect_random_persons(self.persons,
            pathogen, time, n, ignore_infection_status)


    def infect_random_visitors(self,
            pathogen,
            time=-1,
            n=1,
            ignore_infection_status=False):
        return self._do_infect_random_persons(self.visitors,
            pathogen, time, n, ignore_infection_status)


    def count_infected_in_period(self, period, include_visitors=False):
        # TODO: implies closed on left... is that correct?
        res = sum(p.time_infected is not None and p.period_infected == period
            for p in self.persons)

        if include_visitors:
            res += sum(p.time_infected is not None and p.period_infected == period
                for p in self.visitors)

        return res


    def count_infected(self, include_visitors=False):
        res = sum(p.time_infected is not None and p.time_infected >= 0
            for p in self.persons)

        if include_visitors:
            res += sum(p.time_infected is not None and p.time_infected >= 0
                for p in self.visitors)

        return res


    def run_testing(self, test, period, only_symptomatic=True, proportion_to_test=None):
        """Run testing.

        If *only\_symptomatic* is *True*, then no random testing will be done,
        and only those that are honest (i.e., :py:meth:`Person.is_honest` is
        ``True``) will be tested.

        Otherwise, if *proportion_to_test* is ``None``, then everyone will be
        tested.

        Otherwise, the fraction specified by ``proportion_to_test`` will be
        sampled randomly from the workforce.

        :param test: The test to use.

        :param float period: The current period.

        :param bool only_symptomatic: Whether only symptomatic (and honest)
            individuals should be tested.

        :param proportion_to_test: The proportion of the population to test, or
            ``None``. If ``None``, then all workers will be tested. If greater
            than or equal to one, then all workers are tested. If less than
            zero, and exception is raised.
        """
        def test_and_set(person, period):
            person.detected = test(person, period)
            return person.detected

        if only_symptomatic:
            return any(test_and_set(p, period) for \
                p in self.persons if p.shows_symptoms_by(period) and p.is_honest)
        else:
            if proportion_to_test is None:
                return any(test_and_set(person, period) for person in self.persons)
            else:
                if proportion_to_test >= 1.:
                    return any(test_and_set(person, period) for person in self.persons)
                elif proportion_to_test < 0.:
                    raise Exception('proportion to test must be greater than zero.')
                else:
                    return any(set(
                        sample(self.persons, floor(len(self.persons) * proportion_to_test)) \
                        + [p for p in self.persons if p.shows_symptoms_by(period) and p.is_honest]))


    def _generate_gatherings(self, period):
        def reducer(gatherings, gen):
            ngatherings = gen.create_gatherings(self, period, gatherings)
            return gatherings if ngatherings is None else (gatherings + ngatherings)
        ret = reduce(reducer, self._generators, [])
        return ret

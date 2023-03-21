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

from enum import Enum
from math import floor
from typing import Callable
from scipy.stats import poisson
from apsrm import Person, Gathering, Workplace, Box, GatheringGenerator
from apsrm.ext.office import MeetingGenerator
from apsrm.ext.simulation import EmissionsCalculator


class WORKER_TYPE(Enum):
    CHORUS = 1

class BOX_TYPE(Enum):
    HALL = 1
    HVAC = 8

class GATHERING_TYPE(Enum):
    MEETING = 1


class Singer(Person):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _generate_work_day(self, period):
        return None

class ChoralPractice(GatheringGenerator):
    def __init__(self, box):
        self.box = box

    def create_gatherings(self, workplace, period, current_gatherings):
        persons = workplace.persons
        gathering = Gathering(start=9., end=13., box=self.box)
        for person in workplace.persons:
            gathering.add_participant(person)

        return [gathering]



# Set the Model up
def create_workplace(n_singers):
    # create the workplace
    workplace = Workplace(BOX_TYPE, WORKER_TYPE)
    hall = Box(180., BOX_TYPE.HALL , None, name='hall')
    workplace.add_box(hall)

    workplace.add_generator(ChoralPractice(hall))

#    if add_visitors:
#        # must be added after MeetingGenerator instance
#        workplace.add_generator(GatheringVisits(
#            Visitor,
#            [GATHERING_TYPE.MEETING],
#            [egress_generator],
#            visitor_count_generator))
#
#    # working_generator and egress_generator must be added last and in this order.
#    # TODO: might want to add two more setters on person for these
##    generators = [lunch_generator, toilet_break_generator, working_generator, egress_generator]
#    generators = [working_generator]
#    def configure_person(person, **kwargs):
##         person.kitchen = kitchen
##        person.toilet  = toilet
#
#        person.add_generators(generators)
#
#        for k, v in kwargs: setattr(person, k, v)
#        workplace.add_person(person)
#
##    for i in range(n_receptionists):
##        configure_person(Person(48, role=WORKER_TYPE.RECEPTIONIST, work_box=foyer))
#    for i in range(n_workers):
#        configure_person(Person(48, role=WORKER_TYPE.CHORUS, work_box=hall))
    for i in range(n_singers):
        workplace.add_person(Singer(48, role=WORKER_TYPE.CHORUS))

    return workplace



# Emissions for Activities
#
# People breath at different rates doing different activites. Infected people
# also emit different volumes of virus particles and the size of the droplets of
# spittle change (changing the relative rate of aerosol and fomite emissions).
#
# We model this based on the type of worker and the room they are in and
# potentially based on other characteristics (e.g. the number of people in a
# gathering). For example, a worker sitting at their desk is assumed to be
# *breathing*, but a worker in a meeting room is either *breathing* or
# *speaking*, and the relative amount of time they do each is determined by the
# number of people in the meeting.
_activity_weights = {
    # for workers
    (WORKER_TYPE.CHORUS, BOX_TYPE.HALL) : 'singing_voiced'}

_mask_wearing = {
    (WORKER_TYPE.CHORUS, BOX_TYPE.HALL) : False}

def create_emissions_calculator(pathogen):
    return EmissionsCalculator(pathogen, _activity_weights, _mask_wearing)

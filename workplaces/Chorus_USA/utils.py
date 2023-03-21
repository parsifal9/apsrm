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
from apsrm import (
    Person,
    Workplace,
    Box)
from apsrm.ext.office import (
#    MeetingGenerator,
#    GatheringVisits,
#    LunchGenerator,
#    ToiletBreakGenerator,
#    EgressGenerator,
    WorkingGenerator)
from apsrm.ext.simulation import EmissionsCalculator


#AVERAGE_VISITORS_PER_MEETING = .25


class WORKER_TYPE(Enum):
    CHORUS       = 1

class BOX_TYPE(Enum):
    OPEN_PLAN      = 1
    HVAC      = 8

class GATHERING_TYPE(Enum):
    MEETING = 1


class Visitor(Person):
    def __init__(self, *args, **kwargs):
        kwargs['role'] = WORKER_TYPE.VISITOR
        super().__init__(*args, **kwargs)

    def _generate_work_day(self, period):
        return None


#_VISITOR_COUNT_DIST = poisson(AVERAGE_VISITORS_PER_MEETING)

#def visitor_count_generator(gatherings):
#    return _VISITOR_COUNT_DIST.rvs(size=len(gatherings))


# ### Set the Model up
#
# This is a simple function that generates a workplace with a specific number
# of workers and receptionists. In the future we would expand the number of
# worker types (e.g. cleaners and security guards), include visitors and
# customers, and potentially have people leave the office to do things like
# shopping through the day.
def create_workplace(
        n_workers,
        n_receptionists=0,            #no receptionist
        open_plan_area_per_worker = 7.,
        open_plan_ceiling_height = 4.5,
        open_plan_total_space = 180,
        meeting_room_capacity = None,
        add_visitors = False):

    # create the workplace
    workplace = Workplace(BOX_TYPE, WORKER_TYPE)

    # create the period generators
#    lunch_generator        = LunchGenerator()
#    toilet_break_generator = ToiletBreakGenerator()
    working_generator      = WorkingGenerator()

    # create the boxes
#    if open_plan_total_space is None:
#        open_plan_total_space = n_workers * open_plan_area_per_worker
#    open_plan_total_space *= open_plan_ceiling_height

#    if meeting_room_capacity is None:
#        meeting_room_capacity = min(20, max(5, .5 * n_workers))

    hall        = Box(180., BOX_TYPE.OPEN_PLAN , None, name='hall')

    # add boxes to the workplace
    workplace.add_box(hall)

    #egress_generator = EgressGenerator(foyer)

    # workplace.add_generator(MeetingGenerator(
    #     GATHERING_TYPE.MEETING,
    #     [WORKER_TYPE.WORKER],
    #     [BOX_TYPE.MEETING]))

    if add_visitors:
        # must be added after MeetingGenerator instance
        workplace.add_generator(GatheringVisits(
            Visitor,
            [GATHERING_TYPE.MEETING],
            [egress_generator],
            visitor_count_generator))

    # working_generator and egress_generator must be added last and in this order.
    # TODO: might want to add two more setters on person for these
#    generators = [lunch_generator, toilet_break_generator, working_generator, egress_generator]
    generators = [working_generator]
    def configure_person(person, **kwargs):
#         person.kitchen = kitchen
#        person.toilet  = toilet

        person.add_generators(generators)

        for k, v in kwargs: setattr(person, k, v)
        workplace.add_person(person)

#    for i in range(n_receptionists):
#        configure_person(Person(48, role=WORKER_TYPE.RECEPTIONIST, work_box=foyer))
    for i in range(n_workers):
        configure_person(Person(48, role=WORKER_TYPE.CHORUS, work_box=hall))

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
    (WORKER_TYPE.CHORUS, BOX_TYPE.OPEN_PLAN)   : 'singing_voiced'
    }



_mask_wearing = {
    (WORKER_TYPE.CHORUS, BOX_TYPE.OPEN_PLAN)  : False}



def create_emissions_calculator(pathogen):
    return EmissionsCalculator(pathogen, _activity_weights, _mask_wearing)

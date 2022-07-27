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
from apsrm import (
    Person,
    Workplace,
    Box)
from apsrm.ext.office import (
    MeetingGenerator,
    GatheringVisits,
    LunchGenerator,
    ToiletBreakGenerator,
    EgressGenerator,
    WorkingGenerator)

import pytest


class WORKER_TYPE(Enum):
    WORKER       = 1
    RECEPTIONIST = 2


class BOX_TYPE(Enum):
    FOYER        = 1
    ELEVATOR     = 2
    OFFICE       = 3
    OPEN_PLAN    = 4
    TOILET       = 5
    MEETING      = 6
    KITCHEN      = 7
    HVAC         = 8


class GATHERING_TYPE(Enum):
    MEETING = 1


@pytest.fixture
def workplace(
        n_workers=40,
        n_receptionists=1,
        open_plan_area_per_worker = 7.,
        open_plan_ceiling_height = 3.,
        open_plan_total_space = None,
        meeting_room_capacity = None):

    # create the workplace
    workplace = Workplace(BOX_TYPE, WORKER_TYPE)

    # create the period generators
    lunch_generator        = LunchGenerator()
    toilet_break_generator = ToiletBreakGenerator()
    working_generator      = WorkingGenerator()

    # create the boxes
    if open_plan_total_space is None:
        open_plan_total_space = n_workers * open_plan_area_per_worker
    open_plan_total_space *= open_plan_ceiling_height

    if meeting_room_capacity is None:
        meeting_room_capacity = min(20, max(5, .5 * n_workers))

    kitchen      = Box(150., BOX_TYPE.KITCHEN, None, name='kitchen')
    toilet       = Box(50.,  BOX_TYPE.TOILET, None, name='toilet')
    meeting_room = Box(75.,  BOX_TYPE.MEETING, meeting_room_capacity, name='meeting')
    open_plan    = Box(open_plan_total_space,  BOX_TYPE.OPEN_PLAN, None, name='open plan')
    foyer        = Box(200., BOX_TYPE.FOYER, None, name='foyer')

    # add boxes to the workplace
    workplace.add_box(kitchen)
    workplace.add_box(toilet)
    workplace.add_box(meeting_room)
    workplace.add_box(open_plan)
    workplace.add_box(foyer)

    egress_generator = EgressGenerator(foyer)

    workplace.add_generator(MeetingGenerator(
        GATHERING_TYPE.MEETING,
        [WORKER_TYPE.WORKER],
        [BOX_TYPE.MEETING]))

    # working_generator and egress_generator must be added last and in this order.
    # TODO: might want to add two more setters on person for these
    generators = [lunch_generator, toilet_break_generator, working_generator, egress_generator]
    def configure_person(person, **kwargs):
        person.kitchen = kitchen
        person.toilet  = toilet

        person.add_generators(generators)

        for k, v in kwargs: setattr(person, k, v)
        workplace.add_person(person)

    for i in range(n_receptionists):
        configure_person(Person(48, role=WORKER_TYPE.RECEPTIONIST, work_box=foyer))
    for i in range(n_workers):
        configure_person(Person(48, role=WORKER_TYPE.WORKER, work_box=open_plan))

    return workplace

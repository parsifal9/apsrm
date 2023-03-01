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
from apsrm import Box, Gathering, GatheringGenerator, Person, Workplace


class WORKER_TYPE(Enum):
    CHORUS = 1


class BOX_TYPE(Enum):
    HALL = 1
    SANCTUARY = 2
    HVAC = 3


class Singer(Person):
    def _generate_work_day(self, period):
        return None


class ChoralPractice(GatheringGenerator):
    def __init__(self, n_singers, hall, sanctuary=None):
        self.n_singers = n_singers
        self.hall = hall
        self.sanctuary = sanctuary

    def create_gatherings(self, workplace, period, current_gatherings):

        gatherings = []
        if self.sanctuary is not None:

            # all rehersing together
            gathering = Gathering(start=18.5, end=19.25, box=self.hall)
            gathering.add_participants(workplace.persons)
            gatherings.append(gathering)

            # all taking a break
            # this is not quite correct, as we are currently setup to workout how people of a specific
            # type behave in a particular box. I'll modify things to allow for this this shortly
            end_of_break = 19.25 + 10./60.
            gathering = Gathering(start=19.25, end=end_of_break, box=self.hall)
            gathering.add_participants(workplace.persons)
            gatherings.append(gathering)

            # split into two groups
            # Note that we don't know which group the infected person goes into here. One should really
            # pick that person (by scanning through all persons and checking their status), then put
            # them in the first group explicitly.
            n_singers_1 = 42
            assert len(workplace.persons) > n_singers_1
            persons = [p for p in workplace.persons]
            infected_index = [i for i, p in enumerate(persons) if p.period_infected is not None][0]
            if infected_index >= n_singers_1:
                persons[0], persons[infected_index] = persons[infected_index], persons[0]
            group_1 = persons[0:n_singers_1]
            group_2 = persons[n_singers_1:len(persons)]

            gathering = Gathering(start=end_of_break, end=20.25, box=self.hall)
            gathering.add_participants(group_1)
            gatherings.append(gathering)

            gathering = Gathering(start=end_of_break, end=20.25, box=self.sanctuary)
            gathering.add_participants(group_2)
            gatherings.append(gathering)

            # back in hall until the end
            gathering = Gathering(start=20.25, end=21.00, box=self.hall)
            gathering.add_participants(workplace.persons)
            gatherings.append(gathering)

        else:
            gathering = Gathering(start=18.5, end=21.00, box=self.hall)
            gathering.add_participants(workplace.persons)
            gatherings.append(gathering)

        return gatherings


# Set the Model up
def create_workplace(n_singers, single_box):
    # create the workplace
    workplace = Workplace(BOX_TYPE, WORKER_TYPE)

    if single_box:
        hall = Box(810., BOX_TYPE.HALL , None, name='hall')
        workplace.add_box(hall)

        sanctuary = None

    else:
        hall = Box(180., BOX_TYPE.HALL , None, name='hall')
        workplace.add_box(hall)

        sanctuary = Box(150., BOX_TYPE.SANCTUARY, None, name='sanctuary')
        workplace.add_box(sanctuary)

    workplace.add_generator(ChoralPractice(n_singers, hall, sanctuary))

    for _ in range(n_singers):
        workplace.add_person(Singer(65, role=WORKER_TYPE.CHORUS))

    return workplace


class EmissionsCalculator:
    def __init__(self, breathing_rate):
        self._breathing_rate = breathing_rate

    def emissions(self, time, person, interval, gathering):
        return 970.

    def breathing_rate(self, time, person, interval, gathering):
        return self._breathing_rate

    def person_wears_mask_in(self, person, interval):
        return False

    def shedding_filtering_in_box(self, person, interval):
        return 0.

    def ingestion_filtering_in_box(self, person, interval):
        return 0.


def create_emissions_calculator(breathing_rate):
    return EmissionsCalculator(breathing_rate)

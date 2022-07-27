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

"""Can we simlulate a day?"""

import numpy as np
from apsrm import Box
from apsrm.ext.simulation import create_pathogen, EmissionsCalculator

def create_activity_weights(workplace):
    worker_type = workplace._worker_types
    box_type = workplace._box_types

    return {
        # for workers
        (worker_type.WORKER, box_type.OFFICE)   : 'breathing',
        (worker_type.WORKER, box_type.OPEN_PLAN): 'breathing',
        (worker_type.WORKER, box_type.TOILET)   : 'breathing',
        (worker_type.WORKER, box_type.FOYER)    : 'breathing_nose_mouth',
        (worker_type.WORKER, box_type.MEETING)  : lambda person, gathering: (
            ('speech_intermediate',      1./len(gathering)),
            ('breathing',           1. - 1./len(gathering))),
        (worker_type.WORKER, box_type.KITCHEN)      : (
            ('speech_intermediate', .2),
            ('breathing', .8)),

        # for receptionists
        (worker_type.RECEPTIONIST, box_type.OPEN_PLAN): 'breathing',
        (worker_type.RECEPTIONIST, box_type.TOILET)   : 'breathing',
        (worker_type.RECEPTIONIST, box_type.FOYER)    : 'breathing',
        (worker_type.RECEPTIONIST, box_type.KITCHEN)  : (
            ('speech_intermediate', .2),
            ('breathing', .8))
    }, {
        (worker_type.WORKER, box_type.KITCHEN)  : False,
        (worker_type.RECEPTIONIST, box_type.KITCHEN)  : False
    }


def test_smoke(workplace):
    hvac = Box(1., workplace._box_types.HVAC, 0, name='hvac')
    workplace.add_box(hvac)

    boxes = workplace.boxes
    n_boxes = len(boxes)

    # Note that this just contains volume because we multiply this by apprpriate things below.
    ventilation_matrix = np.zeros((n_boxes, n_boxes))
    ventilation_matrix[:,hvac.box_index] \
            = ventilation_matrix[hvac.box_index,:] \
            = np.array([b.volume for b in boxes])
    ventilation_matrix[hvac.box_index, hvac.box_index] = 0.

    external_ventilation = np.zeros(n_boxes)
    external_ventilation[hvac.box_index] = sum([b.volume for b in boxes]) - hvac.volume

    pathogen = create_pathogen('delta')
    activity_weights, mask_wearing = create_activity_weights(workplace)

    emissions_calculator =  EmissionsCalculator(pathogen, activity_weights, mask_wearing)

    for i in range(-4, 0):
        workplace.infect_random_persons(pathogen, i)

    workplace.run_period(4, pathogen, emissions_calculator)

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

import os
from enum import Enum
from math import floor
import numpy as np
import pandas as pd
from scipy.stats import poisson
from apsrm import (
    Person,
    Workplace,
    Box)
from apsrm.ext.office import (
    EgressGenerator,
    MeetingGenerator,
    GatheringVisits)
from apsrm.ext.simulation import (
    EmissionsCalculator,
    create_workplace as _create_workplace)



AVERAGE_VISITORS_PER_MEETING = .25



class WORKER_TYPE(Enum):
    OPEN_WORKER   = 1
    OFFICE_WORKER = 2
    RECEPTIONIST  = 3
    VISITOR       = 4



class BOX_TYPE(Enum):
    FOYER         = 1
    ELEVATOR      = 2
    OFFICE        = 3
    OPEN_PLAN     = 4
    TOILET        = 5
    MEETING       = 6
    KITCHEN       = 7
    HVAC          = 8
    PLANT_ROOM    = 9
    RECEPTION     = 10
    ALCOVE        = 11
    CORRIDOR      = 12
    STAIR_WELL    = 13
    LIFT          = 14
    PLANT         = 15
    MISC          = 16



class Visitor(Person):
    def __init__(self, *args, **kwargs):
        kwargs['role'] = WORKER_TYPE.VISITOR
        super().__init__(*args, **kwargs)

    def _generate_work_day(self, period):
        return None



_VISITOR_COUNT_DIST = poisson(AVERAGE_VISITORS_PER_MEETING)
def visitor_count_generator(gatherings):
    return _VISITOR_COUNT_DIST.rvs(size=len(gatherings))



class GATHERING_TYPE(Enum):
    MEETING = 1



def _check_symetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)



def _load_workplace(
        add_hvac,
        add_visitors,
        ceiling_height = 3.):
    """Ingest the data prepared about the workplace structure."""

    box_type_map = {
        'alcove'       : BOX_TYPE.ALCOVE,
        'corridor'     : BOX_TYPE.CORRIDOR,
        'female toilet': BOX_TYPE.TOILET,
        'foyer'        : BOX_TYPE.FOYER,
        'kitchen'      : BOX_TYPE.KITCHEN,
        'lift'         : BOX_TYPE.ELEVATOR,
        'male toilet'  : BOX_TYPE.TOILET,
        'meeting room' : BOX_TYPE.MEETING,
        'office'       : BOX_TYPE.OFFICE,
        'open plan'    : BOX_TYPE.OPEN_PLAN,
        'plant room'   : BOX_TYPE.PLANT,
        'reception'    : BOX_TYPE.RECEPTION,
        'stair well'   : BOX_TYPE.STAIR_WELL,
        'utilities'    : BOX_TYPE.MISC}

    # load the data on the boxes
    boxes_data = pd.read_csv('boxes.csv').iloc[:,1:]

    box_types = np.unique(np.concatenate((
        boxes_data['room1.use'].values,
        boxes_data['room2.use'].values)))
    assert all([b in box_type_map for b in box_types])

    from_room = np.concatenate((boxes_data['room1'].values, boxes_data['room2'].values))
    u, index, inverse_from = np.unique(from_room, return_index=True, return_inverse=True)

    # prepare the data for the boxes
    refined_boxes_data = pd.DataFrame({
        'volume': ceiling_height * np.concatenate((
            boxes_data['room1.area'].values,
            boxes_data['room2.area'].values))[index],
        'use': np.array([box_type_map[u] for u in np.concatenate((
            boxes_data['room1.use'].values,
            boxes_data['room2.use'].values))[index]]),
        'max_occupancy': np.concatenate((
            boxes_data['room1.occupancy'].values,
            boxes_data['room2.occupancy'].values))[index],
        'name': np.concatenate((
            boxes_data['room1'].values,
            boxes_data['room2'].values))[index]})

    # construct the boxes and add them to the workplace
    workplace = Workplace(BOX_TYPE, WORKER_TYPE)

    boxes = [Box(**kw) for kw in refined_boxes_data.to_dict(orient='records')]
    for box in boxes: workplace.add_box(box)

    workplace.add_generator(MeetingGenerator(
        GATHERING_TYPE.MEETING,
        [WORKER_TYPE.OPEN_WORKER, WORKER_TYPE.OFFICE_WORKER],
        [BOX_TYPE.MEETING]))

    # must be added after MeetingGenerator instance
    foyers = [b for b in workplace.boxes if b.use == BOX_TYPE.FOYER]
    assert len(foyers) == 1
    egress_generator = EgressGenerator(foyers[0])

    if add_visitors:
        workplace.add_generator(GatheringVisits(
            Visitor,
            [GATHERING_TYPE.MEETING],
            [egress_generator],
            visitor_count_generator))

    if add_hvac:
        # Create HVAC properties. Note that we are assuming that airflow is
        # symetric, as per Rob's method of calculating airflow.
        to_room    = np.concatenate((boxes_data['room2'].values, boxes_data['room1'].values))
        mp         = {r:i for i, r in enumerate(u)}
        inverse_to = np.array([mp[r] for r in to_room])
        n_boxes = len(boxes)

        # ventilation matrix
        ventilation_matrix = np.zeros((n_boxes, n_boxes), dtype=np.float64)
        ventilation_matrix[inverse_from, inverse_to] = ceiling_height * np.concatenate((
            boxes_data['room1.area'].values * boxes_data['ACPH.2.1'].values,
            boxes_data['room2.area'].values * boxes_data['ACPH.1.2'].values))

        assert _check_symetric(ventilation_matrix)

        # dump some a description of the boxes and ventilation matrix for
        # inclusion in a latex document.
        if False:
            OUTPUT_BASE_DIR = '../../outputs/eveleigh'
            def opath(p):
                output_base_dir = OUTPUT_BASE_DIR if os.path.exists(OUTPUT_BASE_DIR) else '.'
                return os.path.join(output_base_dir, p)

            # save room characteristics
            reverse_box_type_map = {v:k for k, v in box_type_map.items()}
            reverse_box_type_map[BOX_TYPE.MISC] = 'misc'
            nrow = refined_boxes_data.shape[0]
            rows_per_col = floor(nrow/3)
            refined_boxes_data['desc'] = [reverse_box_type_map[box.use] for box in boxes]
            first = refined_boxes_data[['name', 'desc', 'volume']][:rows_per_col]
            second = refined_boxes_data[['name', 'desc', 'volume']][rows_per_col:(2*rows_per_col)]
            third = refined_boxes_data[['name', 'desc', 'volume']][(2*rows_per_col):nrow]
            df_latex = pd.concat(
                [first.reset_index(drop=True), second.reset_index(drop=True), third.reset_index(drop=True)],
                axis=1).to_latex(
                escape=False,
                index=False,
                header=['Room', 'Use', 'Volume (\\unit{\\cubic\\meter})'] * 3,
                float_format='%0.0f',
                caption='Characteristics of rooms used in the Eveleigh model. ' + \
                        'Note that none of these exchange air with the outside atmosphere directly; ' + \
                        'there is an also a single HVAC unit (not included).',
                label='tab:eveleigh:boxes',
                position='htbp')

            with open(opath('boxes.tex'), 'w') as out:
                out.write(df_latex)

            # save ventilation matrix in sparse format for report.
            df = pd.DataFrame({
                'from': from_room,
                'to'  : to_room,
                'volume': ceiling_height * np.concatenate((
                    boxes_data['room1.area'].values * boxes_data['ACPH.2.1'].values,
                    boxes_data['room2.area'].values * boxes_data['ACPH.1.2'].values))})

            nrow = df.shape[0]
            rows_per_col = floor(nrow/3)
            first = df[:rows_per_col]
            second = df[rows_per_col:(2*rows_per_col)]
            third = df[(2*rows_per_col):nrow]
            df_latex = pd.concat(
                [first.reset_index(drop=True), second.reset_index(drop=True), third.reset_index(drop=True)],
                axis=1).to_latex(
                    escape=False,
                    index=False,
                    float_format='%0.0f',
                    header=['From', 'To', r'Rate (\unit{\cubic\meter\per\hour})'] * 3,
                    caption='Inter-room air exchange used in the Eveleigh model.',
                    label='tab:eveleigh:flows',
                    position='htbp')

            with open(opath('flows.tex'), 'w') as out:
                out.write(df_latex)

    else:
        ventilation_matrix = None

    return workplace, ventilation_matrix



_AIR_CLEANER_BOX_TYPES = [
    BOX_TYPE.OFFICE,
    BOX_TYPE.OPEN_PLAN,
    BOX_TYPE.MEETING,
    BOX_TYPE.KITCHEN]

def create_workplace(add_visitors=False, **kwargs):
    return _create_workplace(
        workplace_loader = lambda add_hvac: _load_workplace(
            add_hvac = add_hvac,
            add_visitors = add_visitors),
        box_types = BOX_TYPE,
        worker_types = WORKER_TYPE,
        hvac_box_type = BOX_TYPE.HVAC,
        air_cleaner_box_types = _AIR_CLEANER_BOX_TYPES,
        **kwargs)



# Emissions for Activities
#
# People breath at different rates doing different activites. Infected people
# also emit different volumes of virus particles and the size of the droplets
# of spittle change (changing the relative rate of aerosol and fomite
# emissions).
#
# We model this based on the type of worker and the room they are in and
# potentially based on other characteristics (e.g. the number of people in a
# gathering). For example, a worker sitting at their desk is assumed to be
# *breathing*, but a worker in a meeting room is either *breathing* or
# *speaking*, and the relative amount of time they do each is determined by the
# number of people in the meeting.
_activity_weights = {
    # for workers
    (WORKER_TYPE.OFFICE_WORKER, BOX_TYPE.OFFICE)   : 'breathing',
    (WORKER_TYPE.OFFICE_WORKER, BOX_TYPE.OPEN_PLAN): 'breathing',
    (WORKER_TYPE.OFFICE_WORKER, BOX_TYPE.TOILET)   : 'breathing',
    (WORKER_TYPE.OFFICE_WORKER, BOX_TYPE.FOYER)    : 'breathing_nose_mouth',
    (WORKER_TYPE.OFFICE_WORKER, BOX_TYPE.MEETING)  : lambda person, gathering: (
        ('speech_intermediate',      1./len(gathering)),
        ('breathing',           1. - 1./len(gathering))),
    (WORKER_TYPE.OFFICE_WORKER, BOX_TYPE.KITCHEN)      : (
        ('speech_intermediate', .2),
        ('breathing', .8)),

    # for workers
    (WORKER_TYPE.OPEN_WORKER, BOX_TYPE.OFFICE)   : 'breathing',
    (WORKER_TYPE.OPEN_WORKER, BOX_TYPE.OPEN_PLAN): 'breathing',
    (WORKER_TYPE.OPEN_WORKER, BOX_TYPE.TOILET)   : 'breathing',
    (WORKER_TYPE.OPEN_WORKER, BOX_TYPE.FOYER)    : 'breathing_nose_mouth',
    (WORKER_TYPE.OPEN_WORKER, BOX_TYPE.MEETING)  : lambda person, gathering: (
        ('speech_intermediate',      1./len(gathering)),
        ('breathing',           1. - 1./len(gathering))),
    (WORKER_TYPE.OPEN_WORKER, BOX_TYPE.KITCHEN)      : (
        ('speech_intermediate', .2),
        ('breathing', .8)),

    # for visitors
    (WORKER_TYPE.VISITOR, BOX_TYPE.FOYER)    : 'breathing_nose_mouth',
    (WORKER_TYPE.VISITOR, BOX_TYPE.MEETING)  : lambda person, gathering: (
        ('speech_intermediate',      1./len(gathering)),
        ('breathing',           1. - 1./len(gathering))),

    # for receptionists
    (WORKER_TYPE.RECEPTIONIST, BOX_TYPE.OPEN_PLAN): 'breathing',
    (WORKER_TYPE.RECEPTIONIST, BOX_TYPE.TOILET)   : 'breathing',
    (WORKER_TYPE.RECEPTIONIST, BOX_TYPE.FOYER)    : 'breathing',
    (WORKER_TYPE.RECEPTIONIST, BOX_TYPE.KITCHEN)  : (
        ('speech_intermediate', .2),
        ('breathing', .8))
    }



_mask_wearing = {
    (WORKER_TYPE.OFFICE_WORKER, BOX_TYPE.KITCHEN) : False,
    (WORKER_TYPE.OPEN_WORKER, BOX_TYPE.KITCHEN)   : False,
    (WORKER_TYPE.RECEPTIONIST, BOX_TYPE.KITCHEN)  : False}



def create_emissions_calculator(pathogen):
    return EmissionsCalculator(pathogen, _activity_weights, _mask_wearing)

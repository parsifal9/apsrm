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

import re
from enum import Enum, IntEnum
from functools import reduce
from itertools import chain
from copy import copy
import numpy as np
import pandas as pd
from scipy.stats import (
    bernoulli,
    uniform,
    norm as normal,
    rv_discrete as categorical)
from apsrm import (
    Box,
    Workplace,
    Person,
    PeriodGenerator)
from apsrm.interval import TimeInterval
from apsrm.ext.simulation import (
    EmissionsCalculator)


class WORKER_TYPE(Enum):
    WORKER       = 1
    RECEPTIONIST = 2
    VISITOR      = 3


class BOX_TYPE(Enum):
    FOYER        = 1
    CAFE         = 2
    THEATRE      = 3
    GALLERY      = 4
    DRUM_SEGMENT = 5
    HVAC         = 6
    CLASSROOM    = 7


# box ids for internal use
class BOX(IntEnum):
    FOYER =      0
    CAFE =       1
    JAPAN_THTR = 2
    G1 =         3
    G2 =         4
    G3 =         5
    G4 =         6
    G5 =         7
    G6 =         8
    G7 =         9
    G8 =         10
    G5_BDR =     11


class DIRECTION(IntEnum):
    DOWN = 0
    UP   = 1


# names as per inputs to internal ids
BOX_NAME_TO_BOX_ENUM = {
    'foyer':      BOX.FOYER,
    'cafe':       BOX.CAFE,
    'japan_thtr': BOX.JAPAN_THTR,
    'G1':         BOX.G1,
    'G2':         BOX.G2,
    'G3':         BOX.G3,
    'G4':         BOX.G4,
    'G5':         BOX.G5,
    'G6':         BOX.G6,
    'G7':         BOX.G7,
    'G7M':        BOX.G7,
    'G8':         BOX.G8,
    'G5_BDR':     BOX.G5_BDR}


HUMAN_BOX_TYPES = [
    BOX_TYPE.FOYER,
    BOX_TYPE.CAFE,
    BOX_TYPE.GALLERY,
    BOX_TYPE.THEATRE]


# types to human friendly names
_BOX_NAME_MAP = {
    BOX.FOYER:      'Foyer',
    BOX.CAFE:       'Cafe',
    BOX.JAPAN_THTR: 'Japan Theatre',
    BOX.G1:         'Gallery 1',
    BOX.G2:         'Gallery 2',
    BOX.G3:         'Gallery 3',
    BOX.G4:         'Gallery 4',
    BOX.G5:         'Gallery 5',
    BOX.G6:         'Gallery 6',
    BOX.G7:         'Gallery 7',
    BOX.G8:         'Gallery 8',
    BOX.G5_BDR:     'Blue Door Room'}


_INFLOW_ROWS_REGEX = re.compile('oa_(.+)')
_NOT_BOX_REGEX = re.compile('(e|oa)_(.+)')


_OPENING_TIME = 9.
_CLOSING_TIME = 17.
_ARRIVE_TIME_DIST = uniform(
    loc=_OPENING_TIME, scale=_CLOSING_TIME - _OPENING_TIME - 2.)


_DURATION_IN_BOX = [
    normal(10./60., 3./60.),
    normal( 3./60., 1./60.)]


# Trip from top to bottom or bottom to top?
_PR_DOWN = .8
_UP_DOWN_DIST = bernoulli(_PR_DOWN)


# Do they go to the cafe?
_PR_CAFE_VIST = .7
_CAFE_VISIT_DIST = bernoulli(_PR_CAFE_VIST)


# Do they eat in the foyer?
_DINING_BOXES = [BOX.FOYER, BOX.CAFE]
_EAT_AT_DIST = categorical(values = ([0, 1, 2], [.55, .35, .1]))
def eat_at():
    i = _EAT_AT_DIST.rvs(size=1)[0]
    return None if i == len(_DINING_BOXES) else _DINING_BOXES[i]


# Do they go to Mini-Q?
_PR_MINI_Q = .5
_MINI_Q_VISIT_DIST = bernoulli(_PR_MINI_Q)

_HUMAN_CONNECTIONS_DICT = {
    # TODO: how to deal with exit?
    BOX.FOYER: {
        DIRECTION.DOWN: [(BOX.G1, .80), (BOX.G6, .10), (BOX.G8, .10)],
        DIRECTION.UP  : [(BOX.G7, .70), (BOX.G6, .10), (BOX.G8, .20)]
    },
    #BOX.CAFE: {
    #    DIRECTION.DOWN: [(BOX.FOYER, 1.)],
    #    DIRECTION.UP  : [(BOX.FOYER, 1.)]
    #},
    # should gift shop be here?

    BOX.G1: {
        DIRECTION.DOWN: [(BOX.G2, .90), (BOX.FOYER, .10)],
        DIRECTION.UP  : [(BOX.G2, .10), (BOX.FOYER, .70), (BOX.G7, .20)]
    },
    BOX.G2: {
        DIRECTION.DOWN: [(BOX.G3, .90), (BOX.G1, .10)],
        DIRECTION.UP  : [(BOX.G3, .10), (BOX.G1, .90)]
    },
    BOX.G3: {
        DIRECTION.DOWN: [(BOX.G4, .90), (BOX.G2, .10)],
        DIRECTION.UP  : [(BOX.G4, .10), (BOX.G2, .90)]
    },
    BOX.G4: {
        DIRECTION.DOWN: [(BOX.G5, .90), (BOX.G3, .10)],
        DIRECTION.UP  : [(BOX.G5, .10), (BOX.G3, .90)]
    },
    # forget B5_BDR for now
    BOX.G5: {
        DIRECTION.DOWN: [(BOX.G7, .90), (BOX.G4, .10)],#, ('G5_BDR', .02)],
        DIRECTION.UP  : [(BOX.G7, .10), (BOX.G4, .90)] #, ('G5_BDR', .02)]
    },
    BOX.G6: {
        DIRECTION.DOWN: [(BOX.G8, .30), (BOX.G7, .20), (BOX.FOYER, .50)],
        DIRECTION.UP  : [(BOX.G8, .30), (BOX.G7, .60), (BOX.FOYER, .10)]
    },
    BOX.G7: {
        DIRECTION.DOWN: [(BOX.G8, .30), (BOX.G5, .10), (BOX.G6, .20), (BOX.FOYER, .40)],
        DIRECTION.UP  : [(BOX.G8, .05), (BOX.G5, .80), (BOX.G6, .05), (BOX.FOYER, .10)]
    },
    BOX.G8: {
        DIRECTION.DOWN: [(BOX.FOYER, .90), (BOX.G7, .10)],
        DIRECTION.UP  : [(BOX.FOYER, .10), (BOX.G7, .90)]
    },
    # forget B5_BDR for now
    #'G5_BDR': {
    #    DIRECTION.DOWN: [(BOX.G5, 1.0)],
    #    DIRECTION.UP  : [(BOX.G5, 1.0)]
    #},
}


# a faster datastructure to work with
_HUMAN_CONNECTIONS = np.ndarray((len(BOX), len(DIRECTION)), dtype=object)
for k, v in _HUMAN_CONNECTIONS_DICT.items():
    for k1, v1 in v.items():
        _HUMAN_CONNECTIONS[k, k1] = categorical(values=(
            [v for v, p in v1],
            [p for v, p in v1]))


_BOX_STAFF = [
    (BOX.FOYER, 6, WORKER_TYPE.RECEPTIONIST), # includes 2 at ticketing desk
    (BOX.CAFE, 3, WORKER_TYPE.WORKER), # assumed by Simon
    (BOX.JAPAN_THTR, 1, WORKER_TYPE.WORKER),
    (BOX.G1, 1, WORKER_TYPE.WORKER),
    (BOX.G2, 1, WORKER_TYPE.WORKER),
    (BOX.G3, 1, WORKER_TYPE.WORKER),
    (BOX.G4, 1, WORKER_TYPE.WORKER),
    (BOX.G5, 1, WORKER_TYPE.WORKER),
    (BOX.G6, 1, WORKER_TYPE.WORKER),
    #(BOX.G7, 3, WORKER_TYPE.WORKER), # assumed by Simon
    (BOX.G8, 1, WORKER_TYPE.WORKER)]
    #(BOX.G5_BDR, 2)


# sizes of groups
_MAX_GROuP_SIZE = 5
_GROUP_SIZES = (
    [i for i in range(1, _MAX_GROuP_SIZE + 1)],
    [1/float(_MAX_GROuP_SIZE)] * _MAX_GROuP_SIZE)
_GROUP_SIZE_DIST = categorical(values=_GROUP_SIZES)
_MEAN_GROUP_SIZE = _GROUP_SIZE_DIST.mean()


_ACTIVITY_WEIGHTS = {
    # for visitors
    (WORKER_TYPE.VISITOR, BOX_TYPE.FOYER): (
        ('speech_intermediate', .2),
        ('breathing', .8)),
    (WORKER_TYPE.VISITOR, BOX_TYPE.GALLERY): (
        ('speech_intermediate', .2),
        ('breathing', .8)),
    (WORKER_TYPE.VISITOR, BOX_TYPE.CAFE): (
        ('speech_intermediate', .2),
        ('breathing', .8)),
    (WORKER_TYPE.VISITOR, BOX_TYPE.THEATRE): (
        ('speech_intermediate', .2),
        ('breathing', .8)),

    # for workers
    (WORKER_TYPE.WORKER, BOX_TYPE.GALLERY): (
        ('speech_intermediate', .2),
        ('breathing', .8)),
    (WORKER_TYPE.WORKER, BOX_TYPE.CAFE): (
        ('speech_intermediate', .2),
        ('breathing', .8)),
    (WORKER_TYPE.WORKER, BOX_TYPE.THEATRE): 'speech_loud',

    # for receptionists
    (WORKER_TYPE.RECEPTIONIST, BOX_TYPE.FOYER): (
        ('speech_intermediate', .2),
        ('breathing', .8))
    }


def _internal_name_to_human_name(n):
    is_hvac = False
    is_drum = False
    is_mezanine = 'G7M' in n

    if n.startswith('h_'):
        is_hvac = True
        n = n[2:]
    elif n.startswith('dr_'):
        is_drum = True
        n = n[3:]

    n = _BOX_NAME_MAP[BOX_NAME_TO_BOX_ENUM[n]]

    if is_mezanine: n += ' Mezanine'
    if is_drum: n += ' Drum'
    if is_hvac: n += ' HVAC'

    return n


class NotifierMixin:
    def _notify_of_infection(self, time):
        pass


class Visitor(NotifierMixin, Person):
    def __init__(self, trip, *args, **kwargs):
        assert len(trip) > 0
        self._trip = [copy(interval) for interval in trip]
        self._trip_interval = TimeInterval(trip[0].start, trip[-1].end)
        kwargs['role'] = WORKER_TYPE.VISITOR
        super().__init__(*args, **kwargs)

    def _generate_work_day(self, period):
        return self._trip_interval


class VisitGenerator(PeriodGenerator):
    def create_intervals(self, person, current_intervals, available_intervals):
        assert len(current_intervals) == 0
        return person._trip, []


class Worker(NotifierMixin, Person):
    def _generate_work_day(self, period):
        return TimeInterval(_OPENING_TIME, _CLOSING_TIME)


class WorkGenerator(PeriodGenerator):
    def create_intervals(self, person, current_intervals, available_intervals):
        return [TimeInterval(
            _OPENING_TIME,
            _CLOSING_TIME,
            box=person.work_box)], []


def load_boxes(filename):
    """Load the boxes.

    Boxes are sorted by name.
    """

    def make_box(row):
        name     = row.pop('room')
        volume   = row.pop('volume')
        box_type = BOX_TYPE[row.pop('use').upper()]
        if box_type is BOX_TYPE.HVAC: volume   = 1.
        return Box(volume, box_type, None, name=name, **row)

    df = pd.read_csv(filename, comment='#')
    return sorted(
        [make_box(row) for row in df.to_dict(orient="records")],
        key=lambda b: b.name)


def load_ventilation_matrix(filename, boxes):
    flows_data = pd.read_csv(filename, comment='#')
    box_name_to_index = {box.name: box.box_index for i, box in enumerate(boxes)}
    n_boxes = len(boxes)

    # Create HVAC properties. Note that we are assuming that airflow is
    # symetric, as per Rob's method of calculating airflow.
    ventilation_matrix = np.zeros((n_boxes, n_boxes), dtype=np.float64)
    ventilation_matrix[
        [box_name_to_index[n] for n in flows_data['from'].values],
        [box_name_to_index[n] for n in flows_data['to'].values]] \
            = flows_data['volume'].values

    ventilation_properties =  {
        'ventilation_matrix': ventilation_matrix,
        'external_ventilation_outflow': [box.ventilation_out for box in boxes],
        'external_ventilation_inflow': [box.ventilation_in for box in boxes],
        'hvac_box_type': BOX_TYPE.HVAC}

    # write the ventilation matrix to disk
    ventilation_d = np.zeros((n_boxes + 1, n_boxes + 1), dtype=np.float64)
    ventilation_d[
        [box_name_to_index[n] for n in flows_data['from'].values],
        [box_name_to_index[n] for n in flows_data['to'].values]] \
            = flows_data['volume'].values
    ventilation_d[0:n_boxes, n_boxes] = ventilation_properties['external_ventilation_outflow']
    ventilation_d[n_boxes, 0:n_boxes] = ventilation_properties['external_ventilation_inflow']
    ventilation_df = pd.DataFrame(ventilation_d)
    ventilation_df.index = [b.name for b in boxes] + ['outside']
    ventilation_df.columns = [b.name for b in boxes] + ['outside']
    ventilation_df.to_csv('ventilation-matrix.csv')

    return ventilation_properties


def generate_trip_for_group(
        group_size,
        box_map,
        boxes_to_drop = None,
        ban_eating_in_foyer = False):
    arrive_time = _ARRIVE_TIME_DIST.rvs(size=1)[0]
    up_or_down = DIRECTION.DOWN if _UP_DOWN_DIST.rvs(size=1)[0] else DIRECTION.UP
    box_visit_counts = np.zeros(len(BOX), dtype='int')
    trip_done = False
    trip_started = False

    current_box = BOX.FOYER
    trip = [(current_box, 5./60.)]

    # generate the time intervals
    if boxes_to_drop is None:
        boxes_to_drop = []

    while not trip_done:
        loop_count = 0
        while True:
            current_box = _HUMAN_CONNECTIONS[current_box, up_or_down].rvs(size=1)[0]
            if current_box not in boxes_to_drop:
                # hopefully this does introduce any infinite loops!
                break
            loop_count += 1
            if loop_count > 100:
                raise Exception('too many attempts to find next box: possible infinite loop')

        n_visits_to_box = box_visit_counts[current_box]

        if current_box != BOX.FOYER:
            # We had people spending too long in the foyer, this reduces that problem
            if n_visits_to_box < len(_DURATION_IN_BOX):
                trip.append((current_box, max(0., _DURATION_IN_BOX[n_visits_to_box].rvs(size=1)[0])))
            box_visit_counts[current_box] += 1

        if not trip_started:
            trip_started = current_box == BOX.G4

        trip_done = current_box == BOX.FOYER and trip_started

    assert current_box == BOX.FOYER
    trip.append((current_box, 3./60.))

    if BOX.G6 not in boxes_to_drop:
        # if they haven't been to Mini-Q... then go there for half and hour
        if box_visit_counts[BOX.G6] == 0 and _MINI_Q_VISIT_DIST.rvs(size=1)[0] == 1:
            trip.append((BOX.G6, .5))

    # do they go to the cafe?
    if _CAFE_VISIT_DIST.rvs(size=1)[0] == 1:
        # TODO: add randomness here

        # order and wait for food
        trip.append((BOX.CAFE, 10./60.))

        # where do they eat
        # eat in foyer (I guess they leave if banned from the foyer).
        if not ban_eating_in_foyer:
            eat_in = eat_at()
            if eat_in is not None:
                trip.append((eat_in, 15./60., False))

    trip = reduce(
        lambda ints, trp: ints + [TimeInterval(
            ints[-1].end if len(ints) > 0 else arrive_time,
            (ints[-1].end if len(ints) > 0 else arrive_time) + trp[1],
            box = box_map[trp[0]],
            wearing_mask = len(trp) > 2 and trp[2])],
        (t for t in trip if t[1] > 0.), [])

    return group_size, trip


def generate_visitors_for_period(
        n_visitors,
        box_map,
        generators,
        remaining_seats,
        *args,
        **kwargs):
    def gen():
        tot = 0
        while True:
            group_size = int(_GROUP_SIZE_DIST.rvs(size=1)[0])
            nxt_tot = tot + group_size

            if nxt_tot < n_visitors:
                tot = nxt_tot
                yield group_size
                continue

            yield n_visitors - tot
            break

    groups = (generate_trip_for_group(group_size, box_map, *args, **kwargs) for group_size in gen())

    # Add trips to Japan Theatre.
    show_times = [11, 14]
    remaining_seats = [remaining_seats, remaining_seats]
    wait_at_door = 5./60.
    length_of_show = .5

    def check_can_go(intervals):
        for interval in intervals:
            for i, st in enumerate(show_times):
                if interval.start < st < interval.end:
                    return True, st, i
            return False, None, None

    class GroupTransformer:
        def __init__(self):
            self.done = False

        def __call__(self, interval, show_time):
            if self.done:
                yield interval

            else:
                if interval.end < show_time:
                    yield interval

                else:
                    if interval.start < (show_time - wait_at_door):
                        yield TimeInterval(
                            interval.start, show_time - wait_at_door,
                            box=interval.box,
                            wearing_mask = interval.wearing_mask)

                        yield TimeInterval(
                            show_time - wait_at_door, show_time,
                            box=box_map[BOX.FOYER],
                            wearing_mask=True)

                    elif interval.start < show_time:
                        yield TimeInterval(
                            interval.start, show_time,
                            box=box_map[BOX.FOYER],
                            wearing_mask=True)

                    yield TimeInterval(
                        show_time, show_time + length_of_show,
                        box=box_map[BOX.JAPAN_THTR],
                        wearing_mask=True)

                self.done = True


    def transform_group(group):
        can_go, show_time, show_time_index = check_can_go(group[1])
        if not can_go or group[0] > remaining_seats[show_time_index]:
            return group
        gt = GroupTransformer()
        new_trip = [i for gv in group[1] for i in gt(gv, show_time)]
        remaining_seats[show_time_index] -= group[0]

        # new_trip could contain overlaps, so fix that
        class Shifter:
            def __init__(self):
                self.overlap_length = 0.

            def __call__(self, i1s, i2):
                # assumes only one overlapping pair
                if self.overlap_length <= 0.:
                    i1 = i1s[-1]
                    if i2.start < i1.end:
                        self.overlap_length = i1.end - i2.start

                i2.start += self.overlap_length
                i2.end += self.overlap_length

                i1s.append(i2)
                return i1s

        new_trips = reduce(Shifter(), new_trip[1:], [new_trip[0]])

        return group[0], new_trips


    # generate the visitors
    def make_visitors(group_size, trip):
        for i in range(group_size):
            visitor = Visitor(trip, age=48)
            visitor.add_generators(generators)
            yield visitor

    return chain(*(make_visitors(*transform_group(group)) for group in groups))


def create_workplace(**kwargs):
    workplace = Workplace(BOX_TYPE, WORKER_TYPE)

    # load the boxes
    for box in load_boxes('boxes.csv'): workplace.add_box(box)
    box_enum_to_box_map = {BOX_NAME_TO_BOX_ENUM[box.name]: box \
        for box in workplace.boxes if box.name in BOX_NAME_TO_BOX_ENUM}

    # create the workers
    work_generator = WorkGenerator()
    for boxe, count, wtype in _BOX_STAFF:
        for i in range(count):
            worker = Worker(
                48,
                role=wtype,
                work_box=box_enum_to_box_map[boxe])
            worker.add_generator(work_generator)
            workplace.add_person(worker)

    # load the ventilation properties
    ventilation_properties = \
        load_ventilation_matrix('flows.csv', workplace.boxes)

    workplace.set_ventilation_properties(
        **ventilation_properties,
        **kwargs)

    box_enum_to_box_map = {BOX_NAME_TO_BOX_ENUM[box.name]: box \
        for box in workplace.boxes if box.name in BOX_NAME_TO_BOX_ENUM}

    return (
        workplace,
        box_enum_to_box_map,
        ventilation_properties['ventilation_matrix'],
        ventilation_properties['external_ventilation_outflow'],
        ventilation_properties['external_ventilation_inflow'])


def create_emissions_calculator(pathogen):
    return EmissionsCalculator(pathogen, _ACTIVITY_WEIGHTS, {})






if __name__ == "__main__":
    import os
    import sys
    from math import floor, ceil



    OUTPUT_BASE_DIR = '../../outputs/questacon'
    def opath(p):
        bd = OUTPUT_BASE_DIR if os.path.exists(OUTPUT_BASE_DIR) else '.'
        return os.path.join(bd, p)



    def _save_human_connections(filename):
        # Will require \usepackage{multirow} in preamble of document
        indent = 4
        with open(filename, 'w') as of:
            of.write(r"""\begin{table}[ht]
\centering
\caption{Transition Probabilities Between Galleries.}
\label{tab:questacon:transition-prs}
\begin{tabular}{llrlr}
\toprule
{} & \multicolumn{2}{c}{Down} & \multicolumn{2}{c}{Up} \\
Room & Room & Pr & Room & Pr \\
    """)

            for k, v in _HUMAN_CONNECTIONS_DICT.items():
                of.write('\\midrule\n')
                max_rows = max(len(v1) for v1 in v.values())

                if max_rows > 1:
                    of.write('\\multirow{{{}}}{{*}}{{{}}} &\n{}'.format(
                        max_rows,
                        _BOX_NAME_MAP[k],
                        ' ' * (indent + 5)))
                else:
                    of.write('{} &\n{}'.format(_BOX_NAME_MAP[k], ' ' * (indent + 5)))

                vds = v[DIRECTION.DOWN] + ([('', '{}')] * (max_rows - len(v[DIRECTION.DOWN])))
                vus = v[DIRECTION.UP]   + ([('', '{}')] * (max_rows - len(v[DIRECTION.UP])))

                of.write('\n{}{{}} & '.format(' ' * indent).join('{} & {} & {} & {} \\\\'.format(
                    _BOX_NAME_MAP.get(vdn, '{}'), vdp,
                    _BOX_NAME_MAP.get(vun, '{}'), vup) for
                        (vdn, vdp), (vun, vup) in zip(vds, vus)) + '\n')

            of.write(r"""\bottomrule
\end{tabular}
\end{table}
    """)
    _save_human_connections(opath('transition-prs.tex'))



    def _save_box_durations(filename):
        # Will require \usepackage{multirow} in preamble of document
        with open(filename, 'w') as of:
            of.write(r"""\begin{table}[ht]
\centering
\caption{Distributions of Durations in Each Gallery.}
\label{tab:questacon:gallery-durations}
\begin{tabular}{lrr}
\toprule
Visit Number & Mean (minutes) & Standard Deviation (minutes) \\
\midrule
""")

            for visit, dist in enumerate(_DURATION_IN_BOX):
                of.write('{} & {:0.0f} & {:0.0f} \\\\'.format(
                    visit + 1, 60. * dist.mean(), 60. * dist.std()))

            of.write(r"""\bottomrule
\end{tabular}
\end{table}
""")
    _save_box_durations(opath('gallery-durations.tex'))



    def _save_box_staff(filename):
        with open(filename, 'w') as of:
            of.write(pd.DataFrame({
                "Box Names": [_BOX_NAME_MAP[i[0]] for i in _BOX_STAFF],
                "Staff Counts": [i[1] for i in _BOX_STAFF]}).to_latex(
                    index=False,
                    caption="Counts of staff in each public area.",
                    label="tab:questacon:staff-box-counts",
                    position='htbp'))
    _save_box_staff(opath('staff-box-counts.tex'))



    def _write_files(
            names,
            volume,
            ventilation_matrix,
            external_ventilation_outflow,
            external_ventilation_inflow):

        # boxes.csv
        df = pd.DataFrame({
            'room': names,
            'volume': volume,
            'ventilation_out': external_ventilation_outflow,
            'ventilation_in': external_ventilation_inflow})
        df.to_csv('boxes.csv', index=False)
        df['room'] = df['room'].str.replace('_', '\\_')
        df_latex = df.to_latex(
            escape = False,
            index = False,
            float_format='%0.0f',
            header=[
                'Room',
                'Volume (\\unit{\\cubic\\meter})',
                'External Venting out (\\unit{\\cubic\\meter\\per\\hour})',
                'External Venting in (\\unit{\\cubic\\meter\\per\\hour})'],
            caption='Characteristics of rooms used in the Questacon model.',
            label='tab:questacon:boxes',
            position='htbp')

        with open(opath('boxes.tex'), 'w') as out:
            out.write(df_latex)

        # sparse ventilation matrix
        positive_inds = np.nonzero(ventilation_matrix)
        names = np.array(names)
        df = pd.DataFrame({
            'from': names[positive_inds[0].astype(int)],
            'to'  : names[positive_inds[1].astype(int)],
            'volume': ventilation_matrix[positive_inds]})
        df.to_csv('flows.csv', index=False)

        # save ventilation matrix in sparse format for report.
        df['from'] = [_internal_name_to_human_name(n) for n in df['from']]
        df['to'] = [_internal_name_to_human_name(n) for n in df['to']]

        nrow = df.shape[0]
        first = df[:floor(nrow/2)]
        last = df[floor(nrow/2):]
        df = pd.concat([first.reset_index(drop=True), last.reset_index(drop=True)], axis=1)
        df_latex = df.to_latex(
            escape=False,
            index=False,
            float_format='%0.0f',
            header=['From', 'To', r'Rate (\unit{\cubic\meter\per\hour})'] * 2,
            caption='Inter-room air exchange used in the Questacon model.',
            label='tab:questacon:flows',
            position='htbp')

        with open(opath('flows.tex'), 'w') as out:
            out.write(df_latex)

    if len(sys.argv) > 1 and sys.argv[1] == 'save':
        workplace_data = create_workplace()

        workplace = workplace_data[0]

        # save the data in nicer formats
        _write_files(
            [b.name for b in workplace.boxes],
            [b.volume for b in workplace.boxes],
            workplace_data[2],
            workplace_data[3],
            workplace_data[4])

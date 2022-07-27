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

import numpy as np
import networkx as nx
from ..config import END_OF_PERIOD_TIME
from ._base import VentilationSystem
from ._standard_ventilation import StandardVentilationSystem
from ._pseudo_ventilation import PseudoVentilationSystem



def _get_subsystems(ventilation_matrix):
    g = nx.Graph()
    g.add_nodes_from([n for n in range(ventilation_matrix.shape[0])])
    g.add_edges_from([e for e in zip(*np.nonzero(ventilation_matrix))])
    return [list(s) for s in nx.connected_components(g)]



class HybridVentilationSystem(VentilationSystem):
    def __init__(
            self,
            ventilation_matrix,
            boxes,
            subsystem_box_inds,
            hvac_box_type = None,
            external_ventilation = None,
            external_ventilation_outflow = None,
            external_ventilation_inflow = None,
            internal_filtering_volume = 0.,
            internal_filtering_efficiency = 0.,
            hvac_return_filtering_efficiency = 0.,
            operates_full_period = False,
            assumptions_checked = False,
            max_day_start_time = None,
            min_day_end_time = None):

        if operates_full_period:
            self._max_day_start_time = 0.
            self._min_day_end_time = END_OF_PERIOD_TIME
        else:
            self._max_day_start_time = max_day_start_time
            self._min_day_end_time = min_day_end_time

        self._start_time = False
        self._end_time = False

        # TODO: Add checks and stuff here.
        # This would be straight forward (cut and paste from
        # ./ventilation.create_ventilation_system) but I wanted to wait for
        # to see if any of that changes.
        assert assumptions_checked

        separate_boxes = [i[0] for i in subsystem_box_inds if len(i) == 1]
        connected_systems = [i for i in subsystem_box_inds if len(i) > 1]

        def make_subsystem(box_inds, disconnected):
            my_boxes = [boxes[i] for i in box_inds]

            if disconnected:
                return PseudoVentilationSystem(
                    my_boxes,
                    external_ventilation_outflow = external_ventilation_outflow[box_inds],
                    internal_filtering_volume = internal_filtering_volume[box_inds],
                    internal_filtering_efficiency = internal_filtering_efficiency[box_inds],
                    operates_full_period = operates_full_period,
                    assumptions_checked = assumptions_checked,
                    max_day_start_time = max_day_start_time,
                    min_day_end_time = min_day_end_time)

            my_ventilation_matrix = ventilation_matrix[box_inds][:,box_inds]

            return StandardVentilationSystem(
                ventilation_matrix = my_ventilation_matrix,
                boxes = my_boxes,
                hvac_box_type = hvac_box_type,
                external_ventilation_outflow = external_ventilation_outflow[box_inds],
                external_ventilation_inflow = None \
                    if external_ventilation_inflow is None \
                    else external_ventilation_inflow[box_inds],
                internal_filtering_volume = internal_filtering_volume[box_inds],
                internal_filtering_efficiency = internal_filtering_efficiency[box_inds],
                hvac_return_filtering_efficiency = hvac_return_filtering_efficiency[box_inds],
                operates_full_period = operates_full_period,
                assumptions_checked = assumptions_checked,
                max_day_start_time = max_day_start_time,
                min_day_end_time = min_day_end_time)

        sub_systems = [make_subsystem(ssbi, False) for ssbi in connected_systems]

        if len(separate_boxes) > 0:
            sub_systems.append(make_subsystem(separate_boxes, True))

        self.sub_systems = sub_systems


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
        self._start_time = False
        self._end_time = False
        for system in self.sub_systems:
            system.reset(full)


    @property
    def start_time(self):
        if self._start_time is False:
            day_starts = [t for t in (system.start_time for system in self.sub_systems) if t is not None]
            self._start_time = self._max_day_start_time if len(day_starts) == 0 else min(day_starts)
            if self._max_day_start_time is not None:
                self._start_time = min(self._start_time, self._max_day_start_time)
        return self._start_time


    @property
    def end_time(self):
        if self._end_time is False:
            day_ends = [t for t in (system.end_time for system in self.sub_systems) if t is not None]
            self._end_time = self._min_day_end_time if len(day_ends) == 0 else max(day_ends)
            if self._min_day_end_time is not None:
                self._end_time = max(self._end_time, self._min_day_end_time)
        return self._end_time


    def calculate_concentrations(
            self,
            period,
            pathogen_dieoff_rate):

        day_start = self.start_time
        day_end = self.end_time

        # I think re-calculalting start, end is redundant
        day_starts_ends = [
            system.calculate_concentrations(period, pathogen_dieoff_rate, day_start, day_end)
                for system in self.sub_systems]

        day_starts = [se[0] for se in day_starts_ends if se[0] is not None]
        day_start = 0. if len(day_starts) == 0 else min(day_starts)

        day_ends = [se[1] for se in day_starts_ends if se[1] is not None]
        day_end = END_OF_PERIOD_TIME if len(day_ends) == 0 else max(day_ends)

        return day_start, day_end

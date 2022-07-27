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
from nptyping import NDArray, Shape, Float
from ._pseudo_ventilation import PseudoVentilationSystem
from ._standard_ventilation import StandardVentilationSystem
from ._hybrid_ventilation import (
    _get_subsystems,
    HybridVentilationSystem)



def create_ventilation_system(
        boxes,
        ventilation_matrix = None,
        hvac_box_type = None,
        external_ventilation = None,
        external_ventilation_outflow = None,
        external_ventilation_inflow = None,
        internal_filtering_volume = 0.,
        internal_filtering_efficiency = 0.,
        hvac_return_filtering_efficiency = 0.,
        operates_full_period = False,
        force_standard_hvac_system = False):
    """Factory method for creating a ventilation system.

    :param list[apsrm.Box] boxes: The boxes in the workplace. This would
        contain both the rooms and (optionally) the HVAC systems.

    :param NDArray[Shape["n,n"], Float] ventilation_matrix: Square matrix describing the
        airflow between *boxes*. Rows and columns must be ordered to correspond
        with the ordering of *boxes*.

    :param typing.Any hvac_box_type: A value to compare with
        :py:attr:`apsrm.Box.use` to determine if that box is and HVAC system.
    """

    n_boxes = len(boxes)

    if ventilation_matrix is None:
        if force_standard_hvac_system:
            ventilation_matrix = np.zeros((n_boxes, n_boxes))
        else:
            return PseudoVentilationSystem(
                boxes,
                external_ventilation = external_ventilation,
                external_ventilation_outflow = external_ventilation_outflow,
                internal_filtering_volume = internal_filtering_volume,
                internal_filtering_efficiency = internal_filtering_efficiency,
                operates_full_period = operates_full_period)

    vm_shape = ventilation_matrix.shape

    # assumptions about the ventilation_matrix
    vm_shape = ventilation_matrix.shape

    # assumptions about the ventilation_matrix
    if len(vm_shape) != 2:
        raise Exception('ventilation_matrix must two dimensional')

    if vm_shape[0] != vm_shape[1]:
        raise Exception('ventilation_matrix must be square')

    if n_boxes != vm_shape[0]:
        raise Exception('boxes and ventilation_matrix have incompatible shapes')

    if not np.all(np.isclose(np.diag(ventilation_matrix), np.zeros(n_boxes))):
        raise Exception('diagonal of ventilation_matrix must be zero: {}'.format(
            np.diag(ventilation_matrix)))

    if not np.all(ventilation_matrix >= 0.):
        raise Exception('ventilation_matrix must be non-negative')

    # assumptions about other arguments
    if not 0. <= hvac_return_filtering_efficiency < 1.:
        raise Exception('hvac_return_filtering_efficiency must be in [0,1)')

    if np.any(hvac_return_filtering_efficiency > 0.) and hvac_box_type is None:
        raise Exception('hvac_box_type cannot be None if hvac_return_filtering_efficiency is not zero')

    # TODO: check internal_filtering_volume and internal_filtering_efficiency are OK.
    if external_ventilation is not None:
        if external_ventilation_outflow is not None:
            raise Exception('cannot specify both external_ventilation and external_ventilation_outflow')
        if external_ventilation_inflow is not None:
            raise Exception('cannot specify both external_ventilation and external_ventilation_inflow')
        external_ventilation_outflow = external_ventilation
        external_ventilation = None

    elif external_ventilation_outflow is None:
        external_ventilation_outflow = 0.

    elif external_ventilation_inflow is None:
        raise Exception('must specify external_ventilation_inflow if external_ventilation_outflow is specified')

    def as_array(a, n):
        a = np.asarray(a)
        if a.size == 1: return np.full(len(boxes), a)
        if len(a) != n_boxes:
           raise Exception('{} must be scalar or have length equal to the number of boxes'.format(n))
        return a

    hvac_return_filtering_efficiency = as_array(hvac_return_filtering_efficiency, 'hvac_return_filtering_efficiency')
    internal_filtering_volume = as_array(internal_filtering_volume, 'internal_filtering_volume')
    internal_filtering_efficiency = as_array(internal_filtering_efficiency, 'internal_filtering_efficiency')
    external_ventilation_outflow = as_array(external_ventilation_outflow, 'external_ventilation_outflow')
    if external_ventilation_inflow is not None:
        external_ventilation_inflow = as_array(external_ventilation_inflow, 'external_ventilation_inflow')

    if not force_standard_hvac_system:
        subsystem_box_inds = _get_subsystems(ventilation_matrix)

    if force_standard_hvac_system or len(subsystem_box_inds) == 1:
        # then all boxes are connected
        return StandardVentilationSystem(
            ventilation_matrix = ventilation_matrix,
            boxes = boxes,
            hvac_box_type = hvac_box_type,
            external_ventilation_outflow = external_ventilation_outflow,
            external_ventilation_inflow = external_ventilation_inflow,
            internal_filtering_volume = internal_filtering_volume,
            internal_filtering_efficiency = internal_filtering_efficiency,
            hvac_return_filtering_efficiency = hvac_return_filtering_efficiency,
            operates_full_period = operates_full_period,
            assumptions_checked = True)

    if len(subsystem_box_inds) == n_boxes:
        # then no boxes are connected
        return PseudoVentilationSystem(
            boxes,
            external_ventilation_outflow = external_ventilation_outflow,
            internal_filtering_volume = internal_filtering_volume,
            internal_filtering_efficiency = internal_filtering_efficiency,
            operates_full_period = operates_full_period,
            assumptions_checked = True)

    return HybridVentilationSystem(
        ventilation_matrix = ventilation_matrix,
        boxes = boxes,
        subsystem_box_inds = subsystem_box_inds,
        hvac_box_type = hvac_box_type,
        external_ventilation_outflow = external_ventilation_outflow,
        external_ventilation_inflow = external_ventilation_inflow,
        internal_filtering_volume = internal_filtering_volume,
        internal_filtering_efficiency = internal_filtering_efficiency,
        hvac_return_filtering_efficiency = hvac_return_filtering_efficiency,
        operates_full_period = operates_full_period,
        assumptions_checked = True)

    return vs

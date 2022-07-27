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
from math import isclose
from scipy.integrate import trapezoid, solve_ivp
from scipy.interpolate import interp1d
from scipy import sparse
from ..config import END_OF_PERIOD_TIME
from ..interval import get_overlapping_interval
from ._base import VentilationSubSystem



USE_SPARSE = True



class step_func:
    def __init__(self, t_start, t_end, y):
        assert len(t_start.shape) == len(t_end.shape)
        assert len(t_start.shape) == len(y.shape)
        assert len(t_start.shape) == 1
        assert t_start.shape[0] == t_end.shape[0]
        assert t_start.shape[0] == y.shape[0]
        self.t_start = t_start
        self.t_end = t_end
        self.y = y

    def __call__(self, t):
        index = np.searchsorted(self.t_start, t)

        if index == 0:
            return 0.

        if t < self.t_end[index-1]:
            return self.y[index-1]

        return 0.



class ConcentrationSeries:
    def __init__(self, t, y):
        assert len(t.shape) == len(y.shape)
        assert len(t.shape) == 1
        assert t.shape[0] == y.shape[0]
        self._interp = interp1d(t, y)
        self.t = t
        self.y = y


    def at_time(self, t):
        if t < self.t[0]:
            return 0.
        if t > self.t[-1]:
            return 0.
            #raise Exception(
            #    'concentration cannot be estimated beyond time {}'.format(
            #        self.t[-1]))
        return self._interp(t)


    def integrate(self, t_range):
        # assumes t_range is sorted

        t_range = np.asarray(t_range)

        assert len(t_range.shape) == 1
        assert t_range.shape[0]   == 2

        indexes = np.searchsorted(self.t, t_range)

        if indexes[0] == self.y.shape[0] or indexes[1] == 0:
            # no overlap
            return 0.

        if indexes[0] == 0 and indexes[1] == self.y.shape[0]:
            # this is contained - integrate the whole range
            return trapezoid(self.y, self.t)

        if indexes[0] == 0:
            # estimate the top only
            ytop = self._interp(t_range[1])
            return trapezoid(
                np.concatenate((self.y[:indexes[1]], [ytop])),
                np.concatenate((self.t[:indexes[1]], [t_range[1]])))

        if indexes[1] == self.y.shape[0]:
            # estimate the bottom only
            ybot = self._interp(t_range[0])
            return trapezoid(
                np.concatenate(([ybot],       self.y[indexes[0]:])),
                np.concatenate(([t_range[0]], self.t[indexes[0]:])))

        # this contains the range
        ybot = self._interp(t_range[0])
        ytop = self._interp(t_range[1])
        return trapezoid(
            np.concatenate(([ybot], self.y[indexes], [ytop])),
            np.concatenate(([t_range[0]], self.t[indexes], [t_range[1]])))



def balance_matrix(a, check_only=False):
    assert len(a.shape) == 2
    assert a.shape[0] == a.shape[1]

    a     = np.copy(a)
    rsums = np.sum(a, axis=0)
    csums = np.sum(a, axis=1)

    if not np.allclose(rsums, csums):
        if check_only:
            raise Exception('matrix not balanced')
        for i in range(a.shape[0]):
            rsum = np.sum(a[i,:])
            csum = np.sum(a[:,i])
            if not isclose(rsum, csum):
                a[(i+1):, i] *= (rsum - np.sum(a[:(i+1), i])) / np.sum(a[(i+1):, i])

    return a



class _EmptySolution:
    def __init__(self, n_boxes, day_start, day_end):
        self.y = np.zeros((n_boxes, 2))
        self.t = np.array([day_start, day_end])



class StandardVentilationSystem(VentilationSubSystem):
    def __init__(
            self,
            ventilation_matrix,
            boxes,
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

        # If external_ventilation_return is None, the it is (implicitly) assumed to
        # equal external_ventilation_outflow. It is only used for checking that the
        # mass balance makes sense.

        if operates_full_period:
            self._max_day_start_time = 0.
            self._min_day_end_time = END_OF_PERIOD_TIME
        else:
            self._max_day_start_time = max_day_start_time
            self._min_day_end_time = min_day_end_time

        self._operates_full_period = operates_full_period
        self.boxes = boxes

        for index, box in enumerate(boxes):
            box.ventilation_system = self
            box.ventilation_system_index = index

        n_boxes = len(boxes)

        check = lambda o: np.isscalar(o) or o.size == 1 or len(o) == n_boxes

        if not assumptions_checked:
            # Some of this stuff we don't want to do a second time. For
            # instance, we it may legitimately turn out to be the case that
            # both external_ventilation and external_ventilation_outflow are
            # None.

            vm_shape = ventilation_matrix.shape

            # assumptions about the ventilation_matrix
            if len(vm_shape) != 2:
                raise Exception('ventilation_matrix must have dimension 2x2')

            if n_boxes != vm_shape[0]:
                raise Exception('boxes and ventilation_matrix have incompatible shapes')

            if n_boxes != vm_shape[1]:
                raise Exception('ventilation_matrix must be square')

            if not np.all(np.isclose(np.diag(ventilation_matrix), np.zeros(n_boxes))):
                raise Exception('diagonal of ventilation_matrix must be zero: {}'.format(
                    np.diag(ventilation_matrix)))

            if not np.all(ventilation_matrix >= 0.):
                raise Exception('ventilation_matrix must be non-negative')

            # assumptions about other arguments
            if not np.all(0. <= hvac_return_filtering_efficiency < 1.):
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

            elif external_ventilation_outflow is None:
                external_ventilation_outflow = 0.

            elif external_ventilation_inflow is None:
                raise Exception('must specify external_ventilation_inflow if external_ventilation_outflow is specified')

            external_ventilation_outflow = np.asarray(external_ventilation_outflow)
            internal_filtering_volume = np.asarray(internal_filtering_volume)
            internal_filtering_efficiency = np.asarray(internal_filtering_efficiency)

            if not check(external_ventilation_outflow) or \
               not check(internal_filtering_efficiency) or \
               not check(internal_filtering_volume):
                   raise Exception('external_ventilation_outflow, internal_filtering_efficiency ' \
                       'and internal_filtering_volume must be scalars or have ' \
                       'length equal to the number of boxes')

        # TODO: balance or complain?
        if external_ventilation_inflow is None:
            ventilation_matrix = balance_matrix(ventilation_matrix, check_only=False)
        else:
            # TODO: check that external_ventilation_inflow is positive.
            external_ventilation_inflow = np.asarray(external_ventilation_inflow)
            if not check(external_ventilation_inflow):
                raise Exception('external_ventilation_inflow must None, a scalar, ' \
                    'or must have length equal to the number of boxes')

            vms = ventilation_matrix.shape
            vm = np.zeros((vms[0]+1, vms[1]+1))
            vm[:vms[0], :vms[1]] = ventilation_matrix
            vm[-1, :vms[1]] = external_ventilation_inflow
            vm[:vms[0], -1] = external_ventilation_outflow
            ventilation_matrix = balance_matrix(vm, check_only=False)[:vms[0], :vms[1]]

        if not np.all(ventilation_matrix >= 0.):
            raise Exception('ventilation_matrix either unbalanced or cannot be balanced')

        self.n = n_boxes

        # we do this in two steps because we don't know the order of boxes
        self.box_volumes = np.zeros(self.n, dtype='float')
        for box in boxes: self.box_volumes[box.ventilation_system_index] = box.volume

        Q = np.copy(ventilation_matrix)

        # the matrix of flows:
        # - between rooms,
        # - to the outside world, and
        # - through (recirculating) filters.
        removal = np.sum(Q, axis=1) \
            + external_ventilation_outflow \
            + internal_filtering_efficiency * internal_filtering_volume

        # deal with the output filter on the HVAC boxes
        if hvac_box_type is not None:
            Q[[b.ventilation_system_index for b in boxes if b.use is hvac_box_type],:] *= \
                (1. - hvac_return_filtering_efficiency)

        np.fill_diagonal(Q, -removal)

        if USE_SPARSE:
            self.Qt = sparse.csr_matrix(Q.T)
            self.dotter = sparse.csr_matrix.dot
        else:
            self.Qt = Q.T
            self.dotter = np.dot

        self.reset(True)


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
        if not full and self.concentrations is not None and self._operates_full_period:
            self._initial_concentrations = [
                self.concentrations[i].at_time(END_OF_PERIOD_TIME) for i in range(self.n)]
        else:
            self._initial_concentrations = None

        self.current_soln = None
        self.concentrations = None
        self.day_start = None
        self.day_end = None
        self._start_time = False
        self._end_time = False
        self._infectivities_start_times = None
        self._infectivities_end_times = None


    @property
    def infectivities_start_times(self):
        if self._infectivities_start_times is None:
            self._infectivities_start_times = [
                np.array([i.start for i in box.infected_intervals]) \
                for box in self.boxes]
        return self._infectivities_start_times


    @property
    def infectivities_end_times(self):
        if self._infectivities_end_times is None:
            self._infectivities_end_times = [
                np.array([i.end for i in box.infected_intervals]) \
                for box in self.boxes]
        return self._infectivities_end_times


    @property
    def start_time(self):
        if self._start_time is False:
            self._start_time = min((float('inf') if len(i) == 0 else i[0]) \
                 for i in self.infectivities_start_times)
            if self._initial_concentrations is not None:
                assert self._start_time >= 0.
                self._start_time = 0.
            else:
                if self._start_time == float('inf'):
                    self._start_time = self._max_day_start_time
                elif self._max_day_start_time is not None:
                    self._start_time = min(self._start_time, self._max_day_start_time)
        return self._start_time


    @property
    def end_time(self):
        if self._end_time is False:
            self._end_time = max((float('-inf') if len(i) == 0 else i[-1]) \
                for i in self.infectivities_end_times)
            if self._end_time == float('-inf'):
                self._end_time = self._min_day_end_time
            elif self._min_day_end_time is not None:
                self._end_time = max(self._start_time, self._min_day_end_time)
        return self._end_time


    def calculate_concentrations(
            self,
            period,
            pathogen_dieoff_rate,
            day_start = None,
            day_end = None):

        # infectivities for each box through time
        infectivities = [
            np.array([i.shedding for i in box.infected_intervals]) \
                for box in self.boxes]

        if day_start is None:
            day_start = self.start_time

        if day_end is None:
            day_end = self.end_time

        if day_start is None or day_end is None:
            assert day_start is None and day_end is None
            self.current_soln = _EmptySolution(self.n, 0., END_OF_PERIOD_TIME)

        elif len(infectivities) == 0 and self._initial_concentrations is None:
            self.current_soln = _EmptySolution(self.n, 0., END_OF_PERIOD_TIME)

        else:
            # convert to hours
            if self._initial_concentrations is not None:
                day_start = 0.
                y0 = self._initial_concentrations
            else:
                y0 = np.zeros(self.n, dtype='float')

            step_funcs = [step_func(np.asarray(ts), np.asarray(te), y) \
                for ts, te, y in zip(
                    self.infectivities_start_times,
                    self.infectivities_end_times,
                    infectivities)]

            def f(t, c):
                q = self.dotter(self.Qt, c)
                s = np.array([sf(t) for sf in step_funcs])
                return (s + q) / self.box_volumes - pathogen_dieoff_rate * c

            soln = solve_ivp(
                fun      = f,
                t_span   = np.array([day_start, day_end]),
                y0       = y0,
                method   = 'LSODA',
                max_step = 1./60.)

            if not soln.success:
                raise Exception('failed to estimate concentrations in {}'.format(id(self)))

            self.current_soln = soln

        self.concentrations = [ConcentrationSeries(self.current_soln.t, y) \
            for y in self.current_soln.y]

        self.day_start = day_start
        self.day_end   = day_end

        return day_start, day_end


    def live_with_pathogen(self, person, interval):
        # accumulate risk through the period
        return self.ingestion_through_time(
            t_range = [interval.start, interval.end],
            box     = interval.box,
            f       = interval.ingestion_filtering,
            p       = interval.breathing_rate,
            D0      = 0.)


    def concentration_at_time(self, box, t):
        """How much is ingested by time.

        Assumes there is no dieoff once the pathogen has been ingested.

        :param box: The box to calculate the concentration in.

        :param t: The time in hours since the individual entered the box.
        """
        return self.concentrations[box.ventilation_system_index].at_time(t)


    def ingestion_through_time(self, t_range, box, f, p, D0):
        """How much is ingested by time.

        Assumes there is no dieoff once the pathogen has been ingested.

        :param t_range: Tuple containing the start and end times in hours to
            integrate over.

        :param box: The box to calculate the ingestion in.

        :param f: The filtration rate (e.g. of a mask).

        :param p: The breathing rate of the indivdual.

        :param D0: The amount previously ingested by the individual at the time
            they enter the box.
        """
        return D0 + (1-f) * p * self.concentrations[box.ventilation_system_index].integrate(
            np.asarray(t_range))



def random_ventilation(n):
    ventilation_matrix = np.random.rand(n, n)
    np.fill_diagonal(ventilation_matrix, 0.)
    ventilation_matrix = ventilation_matrix - np.min(ventilation_matrix) + 1.
    np.fill_diagonal(ventilation_matrix, 0.)
    return ventilation_matrix

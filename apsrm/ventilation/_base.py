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

from abc import ABC, abstractmethod

class VentilationSystem(ABC):
    @abstractmethod
    def reset(self, full):
        """Reset the state of this ventlation system.

        Varous attributes of objects in the system hold transient state, of
        which there are two kinds: those that hold the state relevant to a
        single period, and those that hold state that 'accumulate' state over
        multiple periods. The parameter *full* describes which ones to reset. If
        it is false, then only those that are relevant to a single period are
        reset, if true, then the latter are reset also.

        :param bool full: Whether to reset state that accumulates over multiple
            periods.
        """
        pass

    @property
    @abstractmethod
    def start_time(self):
        """The time the period starts."""
        pass

    @property
    @abstractmethod
    def end_time(self):
        """The time the period ends."""
        pass

    @abstractmethod
    def calculate_concentrations(
        self,
        period,
        pathogen_dieoff_rate,
        day_start = None,
        day_end = None):
        """Calculate concentrations through a period.

        :param int period: The period to calculate concentrations for.
        :param float pathogen_dieoff_rate: The rate the pathogen dies off in
            the atmosphere.
        :param typing.Optional[float] day_start: The time the day starts.
        :param typing.Optional[float] day_end: The time the day starts.
        """
        pass



class VentilationSubSystem(VentilationSystem):
    @abstractmethod
    def live_with_pathogen(self, person, interval): pass

    @abstractmethod
    def concentration_at_time(self, box, t): pass

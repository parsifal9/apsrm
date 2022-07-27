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

"""Various representations of ventilation systems.

Note that ventilation systems should only be created by calling the (factory)
function :py:func:`create_ventilation_system`.
"""

from ._core import create_ventilation_system
from ._base import VentilationSystem, VentilationSubSystem
from ._standard_ventilation import (
    random_ventilation, # for re-export
    balance_matrix)     # for re-export

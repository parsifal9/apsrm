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

"""Configuration."""

import os
from multiprocessing import cpu_count
from math import floor

#: The number of cores to use when multi-processing.
#:
#: This can be set via the environment variable *POOL_NCORES*.
POOL_NCORES = os.environ.get('POOL_NCORES', floor(.75 * cpu_count()))

#: The default pathogen die-off rate to use. The current default is based on
#: (approximately) a half life of 1.1 hours (i.e. :math:`-ln(0.5)/1.1`).
DEFAULT_PATHOGEN_DIEOFF_RATE = .63

#: The 'time' a period ends. By default this is hour of (24 hour) day.
END_OF_PERIOD_TIME = 24.

#: The default strain to use.
DEFAULT_STRAIN = 'delta'

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
import random
import numpy as np
from multiprocessing.managers import SyncManager
from numpy.random import SeedSequence

class ProcessSeeder(object):
    class _Seeder:
        def __init__(self, n_seeds=os.cpu_count(), seed=42):
            self.n_seeds = n_seeds
            self.seed = seed

        def reset(self):
            self._seeds = iter(SeedSequence(self.seed).spawn(self.n_seeds))

        def increment(self):
            return next(self._seeds)

    def __init__(self):
        SyncManager.register('ProcessSeeder', ProcessSeeder._Seeder)
        self.manager = SyncManager()
        self.manager.start()
        self.process_seeder = self.manager.ProcessSeeder()

    def __call__(self):
        seed_maker = self.process_seeder.increment()
        seed = seed_maker.generate_state(2, np.uint32)
        np.random.seed(seed[0])
        random.seed(int(seed[1]))

    def reset(self):
        self.process_seeder.reset()

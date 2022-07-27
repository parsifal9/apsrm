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
import json
import matplotlib.pyplot as plt

OUTPUT_BASE_DIR = '../outputs'
def opath(p):
    output_base_dir = OUTPUT_BASE_DIR if os.path.exists(OUTPUT_BASE_DIR) else '.'
    return os.path.join(output_base_dir, p)

with open('../apsrm/vaccineefficacy.json', 'r') as vef:
    d = json.load(vef)
    plt.figure(figsize=(10, 3))
    plt.plot([i for i in range(len(d))], d, color='black')
    plt.ylim(bottom=0.)
    plt.xlabel('Days')
    plt.savefig(opath('vaccine-efficacy.pdf'), bbox_inches='tight')
    plt.savefig('../doc/source/_static/vaccine-efficacy.png', bbox_inches='tight')

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
import re
import csv
import json

PWD         = os.path.dirname(__file__)
PARAMS_FILE = os.path.join(PWD, 'Emission.csv')
JSON_FILE   = os.path.join(PWD, '..', 'apsrm', 'emissions.json')
SPHINX_FILE = os.path.join(PWD, '..', 'doc', 'source', 'emissions.rst')

def write_sphinx(json_string):
    with open(SPHINX_FILE, 'w') as sfile:
        print(json_string)
        sfile.writelines([
            '.. _emissions-data:\n',
            '\n',
            'Shedding Data\n',
            '=============\n',
            '\n',
            'Shedding and breathing data is stored in the file *emissions.json*, which contains::\n',
            '\n',
            '    ', re.sub(r'\n', r'\n    ', json_string)])

with open(PARAMS_FILE, 'r') as infile, open(JSON_FILE, 'w') as outfile:
    reader = csv.DictReader(infile)
    data = {row['activity']: [
        float(row['virons']) * float(row['breathing_rate']) / 64000.,
        float(row['breathing_rate'])] for row in reader}
    json_data = {
        'breathing' : {k:    v[1] for k, v in data.items()},
        'variants' : {
            'wt'    : {k:    v[0] for k, v in data.items()},
            'delta' : {k: 2.*v[0] for k, v in data.items()}
        }}
    json.dump(json_data, outfile, indent=4)
    write_sphinx(json.dumps(json_data, indent=4))

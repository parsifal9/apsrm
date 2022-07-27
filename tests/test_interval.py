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

import pytest as pt
from apsrm.interval import (
    TimeInterval,
    merge_interval,
    intermediate_intervals)



def test_merge_interval():
    intervals = [TimeInterval(1., 2.), TimeInterval(3., 4.)]

    # should not modify intervals because i.length is zero
    i = TimeInterval(1., 1.)
    res = merge_interval(i, intervals)
    assert len(intervals) == 2

    # should append an i
    i = TimeInterval(.5, .6)
    res = merge_interval(i, intervals)
    assert len(res) == 3
    assert res[0] == i

    # should raise because i overlaps
    i = TimeInterval(1.5, 1.51)
    with pt.raises(AssertionError):
        res = merge_interval(i, intervals)

    # should place i in middle
    i = TimeInterval(2.5, 2.51)
    res = merge_interval(i, intervals)
    assert len(res) == 3
    assert res[1] == i

    # should place i at end
    i = TimeInterval(4.5, 4.51)
    res = merge_interval(i, intervals)
    assert len(res) == 3
    assert res[2] == i

    # should pu
    i = TimeInterval(1., 1.1)
    res = merge_interval(i, [])
    assert len(res) == 1
    assert res[0] == i



def test_interemediate_intervals():
    intervals = [
            TimeInterval(1., 2.),
            TimeInterval(3., 4.),
            TimeInterval(5., 6.),
            TimeInterval(6., 7.)]

    res = intermediate_intervals(intervals)
    assert len(res) == 2

    i = TimeInterval(2., 3.)
    r = res[0]
    assert r.start == pt.approx(i.start)
    assert r.end   == pt.approx(i.end)

    i = TimeInterval(4., 5.)
    r = res[-1]
    assert r.start == pt.approx(i.start)
    assert r.end   == pt.approx(i.end)

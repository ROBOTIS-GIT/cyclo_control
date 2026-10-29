# Copyright 2026 ROBOTIS CO., LTD.
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

"""HX5-D20 fixed endpoint positions from the ROBOTIS rev2/rev3 descriptions."""

from typing import Dict, Tuple


def endpoint_translations(
    hand_side: str, revision: str,
) -> Dict[str, Tuple[float, float, float]]:
    """Return absolute fixed-joint translations, not additive corrections.

    Values come from robotis_hand_description/urdf/hx5_d20_rev{2,3}/
    hx5_d20_{left,right}.urdf.xacro. Articulated joints are unchanged. These
    endpoints are not calibrated tactile contact points or surface normals.
    'urdf' preserves the supplied description, including custom geometry.
    """
    if hand_side not in ('left', 'right'):
        raise ValueError(f'Unsupported hand side: {hand_side}')
    if revision == 'urdf':
        return {}
    lengths = {'rev2': (0.03975, 0.02425), 'rev3': (0.04124, 0.02574)}
    if revision not in lengths:
        raise ValueError('hand_revision must be urdf, rev2, or rev3')
    thumb_length, finger_length = lengths[revision]
    side = 'r' if hand_side == 'right' else 'l'
    thumb_sign = 1 if hand_side == 'right' else -1
    translations = {
        f'finger_end_{side}_joint1': (0.0, thumb_sign * thumb_length, 0.0),
    }
    translations.update({
        f'finger_end_{side}_joint{i}': (0.0, 0.0, finger_length)
        for i in range(2, 6)
    })
    return translations

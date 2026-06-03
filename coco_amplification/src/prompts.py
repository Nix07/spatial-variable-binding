"""Spatial-prompt text for the COCO 2-object amplification experiment."""

from __future__ import annotations

import re


def clean_entity_for_prompt(s):
    s = str(s).strip()
    return re.sub(r'^(?:a|an|the)\s+', '', s, flags=re.IGNORECASE).strip()


def relative_prompt_text(subj, obj, listing, axis=None):
    subj_clean = clean_entity_for_prompt(subj)
    obj_clean = clean_entity_for_prompt(obj)
    if axis == 'horizontal':
        if listing == 'lr':
            return f'Using the {obj_clean} as the reference point, which side is the {subj_clean} on? Answer with one word: left or right.'
        if listing == 'rl':
            return f'Using the {obj_clean} as the reference point, which side is the {subj_clean} on? Answer with one word: right or left.'
    if axis == 'vertical':
        if listing == 'lr':
            return f'Using the {obj_clean} as the reference point, where is the {subj_clean} located? Answer with one word: above or below.'
        if listing == 'rl':
            return f'Using the {obj_clean} as the reference point, where is the {subj_clean} located? Answer with one word: below or above.'
    raise ValueError(f'bad listing/axis: listing={listing!r}, axis={axis!r}')


def direct_prompt_text(subj, obj, listing, axis=None):
    subj_clean = clean_entity_for_prompt(subj)
    obj_clean = clean_entity_for_prompt(obj)
    if axis == 'horizontal':
        if listing == 'lr':
            return f'Is the {subj_clean} to the left or to the right of the {obj_clean}? Answer with one word.'
        if listing == 'rl':
            return f'Is the {subj_clean} to the right or to the left of the {obj_clean}? Answer with one word.'
    if axis == 'vertical':
        if listing == 'lr':
            return f'Is the {subj_clean} above or below the {obj_clean}? Answer with one word.'
        if listing == 'rl':
            return f'Is the {subj_clean} below or above the {obj_clean}? Answer with one word.'
    raise ValueError(f'bad listing/axis: listing={listing!r}, axis={axis!r}')

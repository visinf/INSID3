"""Dataset registry: builds the evaluation dataset specified by name."""

from .coco import build as build_coco
from .lvis import build as build_lvis
from .pascal_part import build as build_pascal_part
from .isic import build as build_isic
from .lung import build as build_lung
from .paco_part import build as build_paco_part
from .suim import build as build_suim
from .isaid import build as build_isaid
from .permis import build as build_permis

_BUILDERS = {
    'coco': build_coco,
    'lvis': build_lvis,
    'pascal_part': build_pascal_part,
    'paco_part': build_paco_part,
    'isic': build_isic,
    'lung': build_lung,
    'suim': build_suim,
    'isaid': build_isaid,
    'permis': build_permis,
}


def build_dataset(dataset: str, args: object):
    if dataset not in _BUILDERS:
        raise ValueError(f'Unknown dataset: {dataset}. '
                         f'Supported: {list(_BUILDERS.keys())}')
    return _BUILDERS[dataset](args)

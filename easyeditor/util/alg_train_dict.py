from ..trainer import MEND

try:
    from ..trainer import SERAC, SERAC_MULTI
except ImportError:
    pass
try:
    from ..trainer import MALMEN
except ImportError:
    pass


ALG_TRAIN_DICT = {
    'MEND': MEND,
}

try:
    ALG_TRAIN_DICT['SERAC'] = SERAC
    ALG_TRAIN_DICT['SERAC_MULTI'] = SERAC_MULTI
except NameError:
    pass
try:
    ALG_TRAIN_DICT['MALMEN'] = MALMEN
except NameError:
    pass

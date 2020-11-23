from .dataset_loader import ImageDataset
from .ltcc_noneID import LTCC_noneID
from .ltcc_orig import LTCC_Orig
from .prcc_noneID import PRCC_noneID
from .prcc_orig import PRCC_Orig

__factory = {
    'ltcc_noneID': LTCC_noneID,
    'ltcc_orig': LTCC_Orig,
    'prcc_noneID': PRCC_noneID,
    'prcc_orig': PRCC_Orig
}


def get_names():
    return __factory.keys()


def init_dataset(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown datasets: {}".format(name))
    return __factory[name](*args, **kwargs)

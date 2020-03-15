import numpy as np

from brandear_est.evals import *


def test_calc_ndcg():
    assert ndcg_at_k([1, 2, 1], 3) == \
           (1 + 3 / np.log2(3) +  1 / np.log2(4))/ (3 + 1 / np.log2(3) + 1 / np.log2(4))

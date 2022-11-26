import glob
import pandas as pd

from typing import List
from os.path import join
from pathlib import Path
from sklearn.metrics import (accuracy_score,
                             f1_score,
                             brier_score_loss,
                             confusion_matrix)


KIND_CODES = {
    2: "believe",
    0: "factive",
    1: "contra"
}


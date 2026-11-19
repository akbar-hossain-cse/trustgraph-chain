import joblib
import pandas as pd

from src.attacker.perturb import CONTROLLABLE
from src.features.load_data import load_raw, time_split, load_config
from src.models.rf import encode

rf = joblib.load("results/rf.joblib")
cfg = load_config()
tr, va, te = time_split(load_raw(), cfg["data"]["train_ratio"], cfg["data"]["val_ratio"])
_, _, _, feats = encode(tr, va, te)
imp = pd.Series(rf.feature_importances_, index=feats).sort_values(ascending=False)
print(imp.head(12).round(4).to_string())
print("\nShare on attacker-controllable features:",
      round(imp[[c for c in feats if c in CONTROLLABLE]].sum(), 3))
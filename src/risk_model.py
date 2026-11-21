"""Interface for Person 3.  Loads the frozen calibrators and returns q_now / q_fail for scalar traces.

    from risk_model import RiskModel
    rm = RiskModel("results/person2_v3")
    out = rm.predict(trace_dict)      # trace_dict = dict loaded from a scalar-trace .npz
    out["q_now"], out["q_fail"]       # (frames, 50) probabilities
    out["valid_now"], out["valid_fail"]   # (frames, 50) bool: where the heads were fitted/evaluated
Inputs used: log_r, S, min_abs, syndrome_weight (and iteration index). info_err / cur_err / final_failure are never read.
"""
import json
from pathlib import Path
import numpy as np
import calib_lib as cl
import p2_common as pc


class RiskModel:
    def __init__(self, out_dir):
        out = Path(out_dir)
        self.card = json.load(open(out / "model_card.json", encoding="utf-8"))
        self.q_now_cal = cl.load_calibrator(out / self.card["q_now"]["file"])
        self.q_fail_cal = cl.load_calibrator(out / self.card["q_fail"]["file"])
        self.k = int(self.card["q_fail"]["k"])
        self.t_min = int(self.card["q_fail"]["t_min"])
        self.now_cond = self.card["q_now"]["fit_region"] == "syn>0"

    def predict(self, d: dict) -> dict:
        nf = len(d["frame_id"]); T = cl.T
        tab = pc.table(d, (self.k,))
        qn = self.q_now_cal.predict(tab).reshape(nf, T)
        qf = self.q_fail_cal.predict(tab).reshape(nf, T)
        syn = (d["syndrome_weight"] > 0)
        t = np.arange(1, T + 1)[None, :]
        return {"q_now": qn, "q_fail": qf, "valid_now": syn,
                "valid_fail": syn & (t >= self.t_min)}

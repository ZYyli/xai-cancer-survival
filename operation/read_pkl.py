#!/usr/bin/env python3

import pickle

pkl_path = "/home/zuoyiyi/SNN/TCGA/shap_results_2/COADREAD/COADREAD_complete_results.pkl"

with open(pkl_path, "rb") as f:
    data = pickle.load(f)

print("type:", type(data))
print("len:", len(data) if hasattr(data, "__len__") else "N/A")

# 通常是 list，每个元素是一个模型的结果 dict
if isinstance(data, list) and len(data) > 0:
    print("first element type:", type(data[0]))
    if isinstance(data[0], dict):
        print("first element keys:", list(data[0].keys())[:30])
        pf = data[0].get("prognostic_factors", None)
        print("prognostic_factors type:", type(pf))
        if isinstance(pf, list) and len(pf) > 0:
            print("prognostic_factors[0] type:", type(pf[0]))
            if isinstance(pf[0], dict):
                print("prognostic_factors[0] keys:", list(pf[0].keys()))
                print("prognostic_factors[0]:", pf[0])

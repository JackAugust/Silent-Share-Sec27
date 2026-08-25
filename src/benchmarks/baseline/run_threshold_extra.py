#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run only threshold scaling experiment (EXP-THR) for additional d values.
This script does NOT rerun other experiments. It reads the same scheme
implementations and outputs extra threshold rows to separate CSV files.

Usage:
    python run_threshold_extra.py --d-list 3,5,7,9,10
"""

import argparse
import csv
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from charm.toolbox.pairinggroup import PairingGroup, GT
import config
from silent_share import SilentShare
import fuzzy_ibme as base

CURVE = config.CURVE
DEFAULT_D = config.DEFAULT_D
ROUNDS = config.ROUNDS
WARMUP = 1

def mean_and_std(values):
    if not values:
        return float("nan"), float("nan")
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std

def attrs(prefix, n):
    return [f"{prefix}_{i}" for i in range(n)]

def build_silent_instance(policy_size, attr_size, d):
    common = attrs("common", d)
    policy = common + attrs("pol", policy_size - d)
    user_attrs = common + attrs("att", attr_size - d)
    return policy, user_attrs

def build_fuzzy_instance(policy_size, attr_size, d):
    common = attrs("common", d)
    P_B = common + attrs("pol", policy_size - d)
    S_B = common + attrs("att", attr_size - d)
    S_A = P_B[:]
    P_A = S_B[:]
    return S_A, P_B, S_B, P_A

def run_silent_threshold(policy_size, attr_size, d_list, rounds, warmup):
    group = PairingGroup(CURVE)
    rows = []
    for d in d_list:
        dec_times = []
        for _ in range(warmup + rounds):
            policy, user_attrs = build_silent_instance(policy_size, attr_size, d)
            universe = list(dict.fromkeys(policy + user_attrs))
            ss = SilentShare(group, threshold=d)
            pp, msk, _ = ss.setup(universe)
            sk, _ = ss.keygen(pp, msk, "user", user_attrs)
            ct, kappa, _ = ss.encap(pp, msk, policy)
            t0 = time.perf_counter_ns()
            rec, details = ss.decap(pp, sk, ct, optimize='precompute')
            dec_s = (time.perf_counter_ns() - t0) / 1e9
            if rec != kappa or details['trials'] != 1:
                raise RuntimeError("decryption failed")
            dec_times.append(dec_s)
        dec_mean, dec_std = mean_and_std(dec_times)
        rows.append({
            "experiment": "threshold_scaling",
            "policy_size": policy_size,
            "attr_size": attr_size,
            "d": d,
            "dec_ms_mean": dec_mean * 1000.0,
            "dec_ms_std": dec_std * 1000.0,
        })
        print(f"Silent |P|={policy_size}, |A|={attr_size}, d={d} -> dec={dec_mean*1000:.2f} ms")
    return rows

def run_fuzzy_threshold(policy_size, attr_size, d_list, rounds, warmup, faithful_H=True):
    group = PairingGroup(CURVE)
    rows = []
    for d in d_list:
        me = base.FuzzyIBME_BN254(group, d=d)
        # Setup with policy_size as n (same as original benchmark)
        t0 = time.perf_counter_ns()
        mpk, msk = me.setup(policy_size)
        setup_s = (time.perf_counter_ns() - t0) / 1e9
        if faithful_H:
            # Re-define faithful H evaluation here if needed; for simplicity, we use current implementation
            pass
        dec_times = []
        for _ in range(warmup + rounds):
            S_A, P_B, S_B, P_A = build_fuzzy_instance(policy_size, attr_size, d)
            ek, q_coeffs = me.ekgen(mpk, msk, S_A)
            dk = me.dkgen(mpk, msk, S_B, P_A, q_coeffs=q_coeffs)
            M = me.group.random(GT)
            CT = me.encrypt(mpk, ek, P_B, M, S_A)
            t0 = time.perf_counter_ns()
            rec, attempts, _ = me.decrypt(mpk, dk, CT, S_B, P_A,
                                          mode="hint", W_A_hint=P_A[:d], W_B_hint=S_B[:d],
                                          expected_message=M)
            dec_s = (time.perf_counter_ns() - t0) / 1e9
            if rec != M or attempts != 1:
                raise RuntimeError("decryption failed")
            dec_times.append(dec_s)
        dec_mean, dec_std = mean_and_std(dec_times)
        rows.append({
            "experiment": "threshold_scaling",
            "policy_size": policy_size,
            "attr_size": attr_size,
            "d": d,
            "dec_ms_mean": dec_mean * 1000.0,
            "dec_ms_std": dec_std * 1000.0,
        })
        print(f"Fuzzy |P|={policy_size}, |A|={attr_size}, d={d} -> dec={dec_mean*1000:.2f} ms")
    return rows

def write_csv(rows, filepath):
    if not rows:
        return
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def main():
    parser = argparse.ArgumentParser(description="Run extra threshold experiments only")
    parser.add_argument("--d-list", type=str, default="3,5,7,9,10")
    parser.add_argument("--policy-size", type=int, default=100)
    parser.add_argument("--attr-size", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=config.ROUNDS)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()

    d_list = [int(x) for x in args.d_list.split(",")]
    policy_size = args.policy_size
    attr_size = args.attr_size
    rounds = args.rounds
    warmup = args.warmup

    print(f"Running extra threshold experiments for d={d_list}")
    print(f"Policy size = {policy_size}, attribute size = {attr_size}")

    # Silent-Share
    silent_rows = run_silent_threshold(policy_size, attr_size, d_list, rounds, warmup)
    silent_out = os.path.join('silent_share_results', 'threshold_extra.csv')
    write_csv(silent_rows, silent_out)
    print(f"Silent extra CSV written to {silent_out}")

    # Fuzzy
    fuzzy_rows = run_fuzzy_threshold(policy_size, attr_size, d_list, rounds, warmup)
    fuzzy_out = os.path.join('fuzzy_ibme_bn254_fair_results', 'threshold_extra.csv')
    write_csv(fuzzy_rows, fuzzy_out)
    print(f"Fuzzy extra CSV written to {fuzzy_out}")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Silent-Share benchmark — redesigned fair benchmark with standard deviation.
All experiments are aligned with Fuzzy IB-ME baseline for direct comparison.

Experiments:
  EXP-ENC: sender policy size |P| varies, receiver attribute size |A| fixed.
           Measures encryption time and ciphertext size.
  EXP-DEC: receiver attribute size |A| varies, sender policy size |P| fixed.
           Measures decryption time (one correct attempt) and key size.
  EXP-THR: threshold d varies, |P| and |A| fixed.
           Measures decryption time (one correct attempt).
  EXP-SEARCH: subset-search complexity (theoretical only, identical to Fuzzy).

Outputs CSV with mean and std for repeated measurements.
"""

import argparse
import csv
import math
import os
import statistics
import sys
import time

# Python 3.6 兼容：提供 perf_counter_ns()
if not hasattr(time, 'perf_counter_ns'):
    def _perf_counter_ns():
        return int(time.perf_counter() * 1e9)
    time.perf_counter_ns = _perf_counter_ns
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from charm.toolbox.pairinggroup import PairingGroup
from silent_share import SilentShare
import config

CURVE = config.CURVE
DEFAULT_D = config.DEFAULT_D
DEFAULT_ROUNDS = config.ROUNDS
DEFAULT_WARMUP = 1
DEFAULT_OUT = "silent_share_results"

def mean_and_std(values):
    if not values:
        return float("nan"), float("nan")
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std

def binom(n, k):
    if k < 0 or k > n:
        return 0
    k = min(k, n - k)
    result = 1
    for i in range(1, k + 1):
        result = result * (n - k + i) // i
    return result

def expected_attempts_one_side(n, overlap, d):
    if overlap < d or n < d:
        return float("inf")
    num = binom(overlap, d)
    den = binom(n, d)
    if num == 0:
        return float("inf")
    return float(den) / float(num)

def attrs(prefix, n):
    return [f"{prefix}_{i}" for i in range(n)]

def build_silent_share_instance(policy_size, attr_size, d):
    """
    Build policy (size policy_size) and user attributes (size attr_size)
    with exactly d common attributes placed at the beginning.
    """
    if d > policy_size or d > attr_size:
        raise ValueError("d must not exceed both sizes")
    common = attrs("common", d)
    policy_extra = attrs("pol", policy_size - d)
    attr_extra = attrs("att", attr_size - d)
    policy = common + policy_extra
    user_attrs = common + attr_extra
    return policy, user_attrs

def serialize_element(group, elem):
    if isinstance(elem, bytes):
        return elem
    return group.serialize(elem, compression=True)

def ct_size_bytes(group, ct):
    total = len(ct['E0']) + len(ct['C'])
    for elem in ct['D1']:
        total += len(serialize_element(group, elem))
    for elem in ct['D2']:
        total += len(serialize_element(group, elem))
    return total

def sk_size_bytes(group, sk):
    total = 0
    for attr in sk.attributes:
        total += len(serialize_element(group, sk.D1[attr]))
        total += len(serialize_element(group, sk.D2[attr]))
    return total

class BenchmarkRunner:
    def __init__(self, curve=CURVE):
        self.curve = curve
        self.group = PairingGroup(curve)

    def run_encryption_scaling(self, policy_sizes, attr_size, d, rounds, warmup):
        print("\n[EXP-ENC] Encryption scaling with policy size")
        rows = []
        for policy_size in policy_sizes:
            if d > policy_size:
                continue
            enc_times = []
            ct_sizes = []
            for r in range(warmup + rounds):
                policy, user_attrs = build_silent_share_instance(policy_size, attr_size, d)
                universe = list(dict.fromkeys(policy + user_attrs))
                ss = SilentShare(self.group, threshold=d)
                pp, msk, _ = ss.setup(universe)
                sk, _ = ss.keygen(pp, msk, "user", user_attrs)
                t0 = time.perf_counter_ns()
                ct, kappa, _ = ss.encap(pp, msk, policy)
                enc_s = (time.perf_counter_ns() - t0) / 1e9
                if r >= warmup:
                    enc_times.append(enc_s)
                    ct_sizes.append(ct_size_bytes(self.group, ct))
            enc_mean, enc_std = mean_and_std(enc_times)
            ct_mean, ct_std = mean_and_std(ct_sizes)
            row = {
                "experiment": "encryption_scaling",
                "policy_size": policy_size,
                "attr_size": attr_size,
                "d": d,
                "enc_ms_mean": enc_mean * 1000.0,
                "enc_ms_std": enc_std * 1000.0,
                "ct_bytes_mean": ct_mean,
                "ct_bytes_std": ct_std,
            }
            rows.append(row)
            print(f"|P|={policy_size:4d}, |A|={attr_size:3d}, d={d:2d} | "
                  f"Enc={enc_mean*1000:9.2f}±{enc_std*1000:6.2f} ms | CT={ct_mean:8.0f} B")
        return rows

    def run_decryption_scaling(self, attr_sizes, policy_size, d, rounds, warmup):
        print("\n[EXP-DEC] Decryption scaling with receiver attribute size")
        rows = []
        for attr_size in attr_sizes:
            if d > attr_size:
                continue
            dec_times = []
            dk_sizes = []
            for r in range(warmup + rounds):
                policy, user_attrs = build_silent_share_instance(policy_size, attr_size, d)
                universe = list(dict.fromkeys(policy + user_attrs))
                ss = SilentShare(self.group, threshold=d)
                pp, msk, _ = ss.setup(universe)
                sk, _ = ss.keygen(pp, msk, "user", user_attrs)
                ct, kappa, _ = ss.encap(pp, msk, policy)
                t0 = time.perf_counter_ns()
                rec_kappa, dec_details = ss.decap(pp, sk, ct, optimize='precompute')
                dec_s = (time.perf_counter_ns() - t0) / 1e9
                if rec_kappa != kappa or dec_details['trials'] != 1:
                    raise RuntimeError("decryption failed or wrong attempts")
                if r >= warmup:
                    dec_times.append(dec_s)
                    dk_sizes.append(sk_size_bytes(self.group, sk))
            dec_mean, dec_std = mean_and_std(dec_times)
            dk_mean, dk_std = mean_and_std(dk_sizes)
            row = {
                "experiment": "decryption_scaling",
                "policy_size": policy_size,
                "attr_size": attr_size,
                "d": d,
                "dec_ms_mean": dec_mean * 1000.0,
                "dec_ms_std": dec_std * 1000.0,
                "dk_bytes_mean": dk_mean,
                "dk_bytes_std": dk_std,
            }
            rows.append(row)
            print(f"|P|={policy_size:4d}, |A|={attr_size:3d}, d={d:2d} | "
                  f"Dec={dec_mean*1000:9.2f}±{dec_std*1000:6.2f} ms | DK={dk_mean:8.0f} B")
        return rows

    def run_threshold_scaling(self, policy_size, attr_size, d_list, rounds, warmup):
        print("\n[EXP-THR] Threshold scaling")
        rows = []
        for d in d_list:
            if d > policy_size or d > attr_size:
                continue
            dec_times = []
            for r in range(warmup + rounds):
                policy, user_attrs = build_silent_share_instance(policy_size, attr_size, d)
                universe = list(dict.fromkeys(policy + user_attrs))
                ss = SilentShare(self.group, threshold=d)
                pp, msk, _ = ss.setup(universe)
                sk, _ = ss.keygen(pp, msk, "user", user_attrs)
                ct, kappa, _ = ss.encap(pp, msk, policy)
                t0 = time.perf_counter_ns()
                rec_kappa, dec_details = ss.decap(pp, sk, ct, optimize='precompute')
                dec_s = (time.perf_counter_ns() - t0) / 1e9
                if rec_kappa != kappa or dec_details['trials'] != 1:
                    raise RuntimeError("decryption failed or wrong attempts")
                if r >= warmup:
                    dec_times.append(dec_s)
            dec_mean, dec_std = mean_and_std(dec_times)
            row = {
                "experiment": "threshold_scaling",
                "policy_size": policy_size,
                "attr_size": attr_size,
                "d": d,
                "dec_ms_mean": dec_mean * 1000.0,
                "dec_ms_std": dec_std * 1000.0,
            }
            rows.append(row)
            print(f"|P|={policy_size:4d}, |A|={attr_size:3d}, d={d:2d} | "
                  f"Dec={dec_mean*1000:9.2f}±{dec_std*1000:6.2f} ms")
        return rows

    def run_search_complexity(self, policy_size, attr_size, d, overlaps):
        """
        Subset-search complexity is theoretical and identical for both schemes.
        We output the same values as Fuzzy for direct comparison.
        """
        print("\n[EXP-SEARCH] Subset-search complexity (theoretical)")
        rows = []
        for overlap in overlaps:
            if overlap < d or overlap > min(policy_size, attr_size):
                continue
            expected = expected_attempts_one_side(attr_size, overlap, d)
            row = {
                "experiment": "search_complexity",
                "policy_size": policy_size,
                "attr_size": attr_size,
                "d": d,
                "overlap": overlap,
                "expected_attempts": expected,
            }
            rows.append(row)
            print(f"overlap={overlap:3d} | E[attempts]={expected:12.2f}")
        return rows

class CSVRecorder:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write_rows(self, rows):
        if not rows:
            return
        exists = self.path.exists()
        with self.path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            if not exists:
                writer.writeheader()
            writer.writerows(rows)

def main():
    parser = argparse.ArgumentParser(description="Silent-Share redesigned fair benchmark")
    parser.add_argument("--rounds", type=int, default=config.ROUNDS)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--d", type=int, default=config.DEFAULT_D)
    parser.add_argument("--out", type=str, default=DEFAULT_OUT)
    parser.add_argument("--policy-sizes", type=str, default="10,25,50,100,200,400,800")
    parser.add_argument("--attr-sizes", type=str, default="5,10,15,20,25,30")
    parser.add_argument("--d-list", type=str, default="2,3,4,5,6,7,8,9,10")
    parser.add_argument("--overlap-list", type=str, default="5,10,15,20,25,30")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "benchmark_redesign.csv"
    if csv_path.exists():
        csv_path.unlink()
    recorder = CSVRecorder(csv_path)

    runner = BenchmarkRunner(curve=CURVE)

    policy_sizes = [int(x) for x in args.policy_sizes.split(",")]
    attr_sizes = [int(x) for x in args.attr_sizes.split(",")]
    d_list = [int(x) for x in args.d_list.split(",")]
    overlap_list = [int(x) for x in args.overlap_list.split(",")]

    print(f"[INFO] curve = {CURVE}")
    print(f"[INFO] rounds = {args.rounds}, warmup = {args.warmup}")
    print(f"[INFO] output = {out_dir.resolve()}")

    # Run experiments
    rows_enc = runner.run_encryption_scaling(policy_sizes, attr_size=30, d=args.d,
                                             rounds=args.rounds, warmup=args.warmup)
    recorder.write_rows(rows_enc)

    rows_dec = runner.run_decryption_scaling(attr_sizes, policy_size=100, d=args.d,
                                             rounds=args.rounds, warmup=args.warmup)
    recorder.write_rows(rows_dec)

    rows_thr = runner.run_threshold_scaling(policy_size=100, attr_size=30, d_list=d_list,
                                            rounds=args.rounds, warmup=args.warmup)
    recorder.write_rows(rows_thr)

    rows_search = runner.run_search_complexity(policy_size=100, attr_size=30, d=args.d,
                                               overlaps=overlap_list)
    recorder.write_rows(rows_search)

    print("\n[DONE] CSV written to", csv_path)

if __name__ == "__main__":
    main()
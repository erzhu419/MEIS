"""Validation for P4.2 structural signature extraction.

Acceptance:
    1. All 4 exp_decay domains share one fingerprint.
    2. All 3 saturation domains share one fingerprint.
    3. exp_decay fingerprint ≠ saturation fingerprint.
    4. Op-multiset contains 'sub' (Sub scalar op) for saturation but not
       exp_decay (the single op-level feature that carries the structural
       distinction).

Run:
    python -m phase4_structure.tests.test_signature
"""

from __future__ import annotations

import numpy as np

from phase4_structure.law_zoo import DOMAINS, CLASS_OF, exp_decay, saturation
from phase4_structure.signature import signature_for_domain


def _default_data(domain_name: str):
    if CLASS_OF[domain_name] == "exp_decay":
        t = exp_decay.default_t_grid(domain_name, n=10)
        y = exp_decay.DOMAIN_REGISTRY[domain_name].simulate(t, np.random.default_rng(0))
    else:
        t = saturation.default_t_grid(domain_name, n=10)
        y = saturation.DOMAIN_REGISTRY[domain_name].simulate(t, np.random.default_rng(0))
    return t, y


def test_all_exp_decay_share_fingerprint():
    names = [n for n, c in CLASS_OF.items() if c == "exp_decay"]
    fps = []
    for n in names:
        t, y = _default_data(n)
        sig = signature_for_domain(DOMAINS[n], t, y)
        fps.append((n, sig.fingerprint))
    uniq = {f for _, f in fps}
    assert len(uniq) == 1, f"exp_decay fingerprints diverge: {fps}"
    print(f"[PASS] 4 exp_decay domains share fingerprint {list(uniq)[0]}")


def test_all_saturation_share_fingerprint():
    names = [n for n, c in CLASS_OF.items() if c == "saturation"]
    fps = []
    for n in names:
        t, y = _default_data(n)
        sig = signature_for_domain(DOMAINS[n], t, y)
        fps.append((n, sig.fingerprint))
    uniq = {f for _, f in fps}
    assert len(uniq) == 1, f"saturation fingerprints diverge: {fps}"
    print(f"[PASS] 3 saturation domains share fingerprint {list(uniq)[0]}")


def test_class_fingerprints_differ():
    t_e, y_e = _default_data("rc_circuit")
    t_s, y_s = _default_data("capacitor_charging")
    sig_e = signature_for_domain(DOMAINS["rc_circuit"], t_e, y_e)
    sig_s = signature_for_domain(DOMAINS["capacitor_charging"], t_s, y_s)
    assert sig_e.fingerprint != sig_s.fingerprint, \
        f"exp_decay and saturation collapse to same fp: {sig_e.fingerprint}"
    print(f"[PASS] inter-class separation: exp_decay {sig_e.fingerprint} "
          f"≠ saturation {sig_s.fingerprint}")


def test_saturation_has_sub_op_exp_decay_does_not():
    """The op-level feature that carries the structural distinction:
    saturation's mu = ymax·(1 - exp(-kt)) has a Sub in its ancestry;
    exp_decay's mu = y0·exp(-kt) does not."""
    t_e, y_e = _default_data("rc_circuit")
    t_s, y_s = _default_data("capacitor_charging")
    sig_e = signature_for_domain(DOMAINS["rc_circuit"], t_e, y_e)
    sig_s = signature_for_domain(DOMAINS["capacitor_charging"], t_s, y_s)
    assert "Sub" in sig_s.ops, f"Sub missing from saturation ops: {sig_s.ops}"
    assert "Sub" not in sig_e.ops, f"unexpected Sub in exp_decay ops: {sig_e.ops}"
    print(f"[PASS] Sub op present in saturation ({sig_s.ops.count('Sub')}×), "
          f"absent in exp_decay")


def test_signature_stable_across_data():
    """Fingerprint should depend only on model structure, not on the specific
    t/y observations fed in (structure-only signature)."""
    name = "rc_circuit"
    rng = np.random.default_rng(0)
    t1 = np.linspace(0, 5, 8)
    y1 = DOMAINS[name].simulate(t1, rng)
    t2 = np.linspace(0, 20, 30)
    y2 = DOMAINS[name].simulate(t2, rng)
    sig1 = signature_for_domain(DOMAINS[name], t1, y1)
    sig2 = signature_for_domain(DOMAINS[name], t2, y2)
    assert sig1.fingerprint == sig2.fingerprint, \
        f"fingerprint varies with observation shape: {sig1.fingerprint} vs {sig2.fingerprint}"
    print(f"[PASS] signature invariant across two different t-grids "
          f"(8 vs 30 points)")


if __name__ == "__main__":
    print("=== P4.2 structural signature validation ===\n")
    test_all_exp_decay_share_fingerprint()
    test_all_saturation_share_fingerprint()
    test_class_fingerprints_differ()
    test_saturation_has_sub_op_exp_decay_does_not()
    test_signature_stable_across_data()
    print("\nAll P4.2 signature checks passed.")

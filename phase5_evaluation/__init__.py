"""MEIS Phase 5 — closed-loop evaluation environment.

Plan §Phase 5 success criteria (targets):
    Task 1  Hypothesis extraction → transition matrix
            Frobenius norm error < 0.1  (see task1_transition_matrix.py)
    Task 2  Equivalence-class identification
            ARI > 0.8  (delivered in phase4_structure/retrieval.py)
    Task 3  Transfer prediction
            Held-out MSE ≥ 30% reduction  (phase4_structure/transfer.py)
    Task 4  Minimum-perturbation scoring
            Expert-agreement rate ≥ 70%  (see task4_expert_agreement.py)
"""

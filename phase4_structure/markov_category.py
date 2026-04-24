"""P4.7 — Markov category primitives for structural isomorphism.

This module implements a minimal symbolic layer for Markov categories —
the missing L3 of the Plan §1 architecture. It does NOT execute
probability: it just represents belief networks as morphism trees so
that two networks can be compared by their string-diagram shape.

Why symbolic rather than computational:
  A full Markov-category library (e.g. Fritz et al. 2020) would define
  Markov kernels K: X ⇝ Y as stochastic maps and verify categorical
  laws operationally. For MEIS the value we want from the category is
  strictly structural: "do these two belief networks have isomorphic
  string diagrams?" That question reduces to checking that the
  recursive tree shape matches, which is what `shape_signature()`
  below does.

  The existing PyMC models in phase4_structure/law_zoo do the heavy
  lifting on the probabilistic side. This layer is the bridge between
  a network and its categorical abstraction.

Primitives:

  Obj(name)               an object / wire type (a measurable space)
  Atom(name, dom, cod)    an atomic morphism (prior, kernel, etc.)
  Copy(X), Discard(X)     structural morphisms in every Markov category
  Compose(f, g)           sequential composition f then g  (f.cod == g.dom)
  Tensor(f, g)            parallel composition f ⊗ g (monoidal product)

  shape_signature(m)      canonical hash of the morphism tree, modulo
                          object-name relabelling
  equivalent(f, g)        True iff shape signatures match

The construction follows the Fritz et al. 2020 definition — a
CD-category with copy (Δ) and discard (!) — without enforcing
naturality/unit/associativity axioms (they are used only structurally,
which our hash captures).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib


# ---------------------------------------------------------------------------
# Objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Obj:
    """An object in the category — a measurable-space identifier.

    Two Obj are categorically the SAME 'wire type' if they have the
    same `kind` (e.g. 'real', 'time-series', 'parameter'); the name is
    a human-readable label and does not participate in structural
    comparison.
    """
    name: str
    kind: str = "measurable"

    def __repr__(self) -> str:
        return f"{self.name}:{self.kind}"


UNIT = Obj("I", kind="unit")  # monoidal unit — objects of 0 wires


# ---------------------------------------------------------------------------
# Morphisms — abstract base
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Morph:
    """Base class for morphisms in a Markov category.

    All morphisms have a typed domain and codomain (tuples of Obj).
    Concrete morphism types are Atom, Copy, Discard, Compose, Tensor.
    """
    dom: tuple
    cod: tuple

    def __rshift__(self, other: "Morph") -> "Morph":
        return Compose(self, other)

    def __matmul__(self, other: "Morph") -> "Morph":
        return Tensor(self, other)


@dataclass(frozen=True)
class Atom(Morph):
    """An atomic morphism — a named box in the string diagram.

    `kind` categorises the box: 'prior', 'deterministic', 'observation',
    'transform', ... Two atoms are STRUCTURALLY equal if their kinds
    and (dom-kinds, cod-kinds) match — the specific name is ignored.
    """
    name: str = ""
    kind: str = "atom"


@dataclass(frozen=True)
class Copy(Morph):
    """The copy morphism Δ_X : X → X ⊗ X."""
    def __init__(self, X: Obj):
        object.__setattr__(self, "dom", (X,))
        object.__setattr__(self, "cod", (X, X))


@dataclass(frozen=True)
class Discard(Morph):
    """The discard morphism !_X : X → I."""
    def __init__(self, X: Obj):
        object.__setattr__(self, "dom", (X,))
        object.__setattr__(self, "cod", ())


@dataclass(frozen=True)
class Compose(Morph):
    """Sequential composition f ; g.  (g ∘ f in standard notation.)"""
    f: Morph = None
    g: Morph = None

    def __init__(self, f: Morph, g: Morph):
        if f.cod != g.dom:
            raise TypeError(
                f"cannot compose: f.cod={f.cod} ≠ g.dom={g.dom}"
            )
        object.__setattr__(self, "dom", f.dom)
        object.__setattr__(self, "cod", g.cod)
        object.__setattr__(self, "f", f)
        object.__setattr__(self, "g", g)


@dataclass(frozen=True)
class Tensor(Morph):
    """Parallel composition f ⊗ g."""
    f: Morph = None
    g: Morph = None

    def __init__(self, f: Morph, g: Morph):
        object.__setattr__(self, "dom", tuple(f.dom) + tuple(g.dom))
        object.__setattr__(self, "cod", tuple(f.cod) + tuple(g.cod))
        object.__setattr__(self, "f", f)
        object.__setattr__(self, "g", g)


# ---------------------------------------------------------------------------
# Structural signature
# ---------------------------------------------------------------------------


def _obj_shape(X: Obj) -> str:
    return X.kind


def _objs_shape(objs: tuple) -> str:
    return "(" + ",".join(_obj_shape(X) for X in objs) + ")"


def _shape(morph: Morph) -> str:
    """Recursive canonical string capturing the morphism tree.

    We keep:
      - Morphism TREE STRUCTURE (compose / tensor nesting)
      - Object KINDS (type labels like 'parameter', 'time-series')
      - Atom KINDS and NAMES (two deterministic kernels named
        'decay_kernel' vs 'saturation_kernel' are structurally
        DIFFERENT morphisms in the category — they have different
        underlying maps, hence different boxes)

    We drop:
      - Object NAMES (a "parameter" wire is a parameter wire regardless
        of whether it's called Θ or θ₁ — this is α-conversion in the
        categorical sense)
    """
    if isinstance(morph, Copy):
        return f"Copy{_objs_shape(morph.dom)}"
    if isinstance(morph, Discard):
        return f"Discard{_objs_shape(morph.dom)}"
    if isinstance(morph, Atom):
        # Shape carries only the atom's KIND (functional role) and
        # typed arity, NOT its human-readable name. Different prior
        # boxes across a class (e.g. prior_y0 in rc_circuit vs
        # prior_Vs in radioactive_decay) are all kind='prior' and
        # therefore categorically equivalent here.
        return (f"Atom[{morph.kind}]"
                f"{_objs_shape(morph.dom)}→{_objs_shape(morph.cod)}")
    if isinstance(morph, Compose):
        return f"({_shape(morph.f)};{_shape(morph.g)})"
    if isinstance(morph, Tensor):
        return f"({_shape(morph.f)}⊗{_shape(morph.g)})"
    raise TypeError(f"unknown morphism type {type(morph)}")


@dataclass(frozen=True)
class StringDiagramSignature:
    shape: str              # canonical tree string, name-free
    dom_kinds: tuple
    cod_kinds: tuple
    fingerprint: str        # 16-hex sha256 of shape

    def matches(self, other: "StringDiagramSignature") -> bool:
        return self.fingerprint == other.fingerprint


def shape_signature(morph: Morph) -> StringDiagramSignature:
    shape = _shape(morph)
    fp = hashlib.sha256(shape.encode()).hexdigest()[:16]
    return StringDiagramSignature(
        shape=shape,
        dom_kinds=tuple(X.kind for X in morph.dom),
        cod_kinds=tuple(X.kind for X in morph.cod),
        fingerprint=fp,
    )


def equivalent(f: Morph, g: Morph) -> bool:
    """Structural equivalence: two morphisms have the same string-diagram
    shape (modulo object-name relabelling)."""
    return shape_signature(f).matches(shape_signature(g))


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------


def prior(name: str, X: Obj) -> Atom:
    """A prior sampling morphism I → X."""
    return Atom(dom=(), cod=(X,), name=name, kind="prior")


def deterministic(name: str, dom: tuple, cod: tuple) -> Atom:
    """A deterministic morphism (delta Markov kernel)."""
    return Atom(dom=dom, cod=cod, name=name, kind="deterministic")


def observation(name: str, X: Obj, Y: Obj) -> Atom:
    """An observation / likelihood morphism X → Y."""
    return Atom(dom=(X,), cod=(Y,), name=name, kind="observation")

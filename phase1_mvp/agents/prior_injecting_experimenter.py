"""Step 3 — LLM scientist whose system message gets a cross-domain prior chunk.

Thin subclass of boxing_gym.agents.LMExperimenter. Only difference: when
set_system_message is called, we retrieve top-k priors from the library
for a given query and append them to the base message.

The rest of the scientist's behavior (generate_actions, prompt_llm, etc.)
is inherited unchanged — we do not touch boxing-gym trunk.
"""

from __future__ import annotations

from boxing_gym.agents.agent import LMExperimenter

from phase2_prior_library.retrieval import PriorLibrary


DEFAULT_PRIOR_HEADER = (
    "Cross-domain prior knowledge (from the MEIS curated library). "
    "Treat these relations as well-established background facts you can use "
    "to structure your hypotheses, choose informative observations, and "
    "explain the system to others. You do not need to rediscover them from data:"
)


class PriorInjectingExperimenter(LMExperimenter):
    """Experimenter that adds a retrieved prior block to every system message.

    Args:
        model_name, temperature, max_tokens: forwarded to LMExperimenter.
        library: a loaded PriorLibrary.
        prior_query: bag-of-words query used to retrieve priors.
        prior_k: number of priors to retrieve.
        prior_header: static header placed above the retrieved block.
        max_prior_chars: cap on the retrieved-prior text length.
    """

    def __init__(self, model_name: str, *,
                 library: PriorLibrary,
                 prior_query: str,
                 prior_k: int = 5,
                 prior_header: str = DEFAULT_PRIOR_HEADER,
                 max_prior_chars: int = 2500,
                 temperature: float = 0.0,
                 max_tokens: int = 256):
        super().__init__(model_name=model_name, temperature=temperature, max_tokens=max_tokens)
        self.library = library
        self.prior_query = prior_query
        self.prior_k = prior_k
        self.prior_header = prior_header
        self.max_prior_chars = max_prior_chars
        # Record what was actually retrieved — audit trail for the experiment JSON.
        self.last_prior_hits: list[dict] = []
        self.last_prior_block: str = ""

    def _build_prior_block(self) -> str:
        self.last_prior_hits = self.library.retrieve(self.prior_query, k=self.prior_k)
        self.last_prior_block = PriorLibrary.format_for_prompt(
            self.last_prior_hits, max_chars=self.max_prior_chars
        )
        return self.last_prior_block

    def set_system_message(self, message: str) -> None:
        prior_block = self._build_prior_block()
        if prior_block:
            augmented = f"{message}\n\n{self.prior_header}\n{prior_block}"
        else:
            augmented = message
        super().set_system_message(augmented)

"""Early decode termination that is provably accuracy-neutral.

The competition scores an answer by parsing the *generated text* with
``optiz_qwen.evaluation.answer_parsing.extract_answer``, which tries a fixed,
ordered list of regexes and returns the first pattern that matches (leftmost
match within that pattern).  The highest-precedence pattern
(``ANSWER_PATTERNS[0]``) is an explicit answer phrase -- ``Answer: X`` /
``答案是 X`` / ``final choice: X`` -- which the benchmark prompt explicitly asks
the model to emit.

The optimization
----------------
Multiple-choice answers are decided in the first handful of decode tokens, yet
greedy decode keeps going to ``max_new_tokens`` (256) emitting an explanation the
scorer never reads.  If we can commit to the answer the instant the model has
written it *and prove the final parse is unchanged*, we cut the decode tail --
which helps throughput (fewer tokens over less wall time) and end-to-end latency
at **zero accuracy cost by construction**.

The equivalence invariant
--------------------------
``committed_answer(text)`` returns a letter ``X`` only when **both** hold:

1. the highest-precedence pattern matches ``text`` (leftmost, via ``search``), and
2. that match ends strictly before ``len(text)`` -- i.e. a real boundary
   character already follows it in the decoded text.

Under these two conditions, for *any* continuation ``s`` that further decoding
could have appended, ``extract_answer(text + s) == X``:

- appending ``s`` never changes characters before ``len(text)``, so the leftmost
  highest-precedence match stays at the same position with the same captured
  group (condition 2 guarantees the match cannot grow into ``s``); and
- a higher-precedence pattern is tried *first*, so no lower-precedence pattern
  in ``s`` can override it.

Therefore stopping the moment ``committed_answer`` fires yields the identical
parsed answer the full generation would have produced.  When it never fires
(the model never emits the explicit phrase), decoding runs to completion exactly
as before -- a strict no-op.  The condition is deliberately conservative: it does
*not* commit on the weaker Chinese-verb or bare-leading-letter patterns, because
a later explicit phrase could legitimately override those.

Kill switch: ``OPTIZ_QWEN_EARLY_STOP=0`` restores full-length decoding.
"""

from __future__ import annotations

import os

from optiz_qwen.evaluation.answer_parsing import ANSWER_PATTERNS

#: The one pattern safe to commit on: the explicit, highest-precedence answer
#: phrase.  Bound to the live object so it can never drift from the scorer.
_HIGHEST_PRECEDENCE_PATTERN = ANSWER_PATTERNS[0]

EARLY_STOP_ENV = "OPTIZ_QWEN_EARLY_STOP"


def early_stop_enabled() -> bool:
    value = os.environ.get(EARLY_STOP_ENV, "").strip().lower()
    if value == "":
        return True
    return value in {"1", "true", "yes", "on"}


def committed_answer(text: str) -> str | None:
    """Return the letter safe to commit to, or ``None`` to keep decoding.

    See the module docstring for the equivalence proof.  The two conditions
    (highest-precedence match; a boundary character already follows it) are what
    make an early commit identical to the final parse.
    """

    if not text:
        return None
    match = _HIGHEST_PRECEDENCE_PATTERN.search(text)
    if match is None:
        return None
    # A trailing boundary char must already exist, so no future token can extend
    # or alter this match.  ``re.MULTILINE`` ``$`` is not relied upon here.
    if match.end() >= len(text):
        return None
    return match.group(1).upper()

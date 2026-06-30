"""Answer: vision-LLM synthesis over retrieved page images with citations."""

from facsimile.answer.synthesizer import (
    Answer,
    ClaudeSynthesizer,
    StubSynthesizer,
    Synthesizer,
    synthesize,
)

__all__ = [
    "Answer",
    "ClaudeSynthesizer",
    "StubSynthesizer",
    "Synthesizer",
    "synthesize",
]

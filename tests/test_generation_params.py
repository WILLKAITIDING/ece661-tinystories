"""GenerationParams validation - settings must fail loudly, not silently."""

from __future__ import annotations

import pytest

from tinystories_project.inference.interface import GenerationParams


def test_defaults_match_documented_values():
    p = GenerationParams()
    assert (p.max_new_tokens, p.temperature, p.top_p) == (200, 1.0, 0.95)
    assert p.do_sample is True
    assert p.num_return_sequences == 1


def test_params_are_immutable():
    p = GenerationParams()
    with pytest.raises(Exception):
        p.temperature = 2.0  # frozen dataclass


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"max_new_tokens": 0}, "max_new_tokens"),
        ({"max_new_tokens": -5}, "max_new_tokens"),
        ({"temperature": 0.0}, "temperature"),
        ({"temperature": -1.0}, "temperature"),
        ({"top_p": 0.0}, "top_p"),
        ({"top_p": 1.5}, "top_p"),
        ({"top_k": 0}, "top_k"),
        ({"repetition_penalty": 0.0}, "repetition_penalty"),
        ({"num_return_sequences": 0}, "num_return_sequences"),
    ],
)
def test_invalid_values_raise(kwargs, match):
    with pytest.raises(ValueError, match=match):
        GenerationParams(**kwargs)


def test_temperature_zero_error_suggests_greedy():
    """A common mistake deserves an actionable message."""
    with pytest.raises(ValueError, match="do_sample=False"):
        GenerationParams(temperature=0.0)


def test_roundtrip_through_dict():
    p = GenerationParams(max_new_tokens=42, temperature=0.8, top_k=50)
    assert GenerationParams.from_dict(p.to_dict()) == p


def test_from_dict_ignores_unknown_keys():
    p = GenerationParams.from_dict({"max_new_tokens": 7, "not_a_field": True})
    assert p.max_new_tokens == 7

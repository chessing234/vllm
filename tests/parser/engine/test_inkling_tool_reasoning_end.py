# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A tool-call opener ends Inkling reasoning for structured output."""

from tests.parser.engine.conftest import make_mock_tokenizer
from vllm.parser.engine.adapters import ParserEngineReasoningAdapter
from vllm.parser.engine.parser_engine import ReasoningEnd
from vllm.parser.inkling import InklingParser


class _InklingReasoningAdapter(ParserEngineReasoningAdapter):
    _parser_engine_cls = InklingParser

MSG_MODEL = "<|message_model|>"
MSG_USER = "<|message_user|>"
TEXT_START = "<|content_text|>"
THINK_START = "<|content_thinking|>"
TOOL_JSON = "<|content_invoke_tool_json|>"
TOOL_TEXT = "<|content_invoke_tool_text|>"
TOOL_ERROR = "<|content_tool_error|>"
END_MESSAGE = "<|end_message|>"
END_SAMPLING = "<|content_model_end_sampling|>"

_VOCAB = {
    "<|message_user|>": 200000,
    MSG_MODEL: 200001,
    "<|message_system|>": 200002,
    "<|message_tool|>": 200003,
    TEXT_START: 200004,
    END_SAMPLING: 200006,
    THINK_START: 200008,
    END_MESSAGE: 200010,
    TOOL_ERROR: 200022,
    TOOL_JSON: 200049,
    TOOL_TEXT: 200057,
}


def _parser(vocab: dict[str, int]):
    return InklingParser(make_mock_tokenizer(vocab))


def _after_model_turn(*tail: str) -> list[int]:
    ids = [
        _VOCAB[THINK_START],
        ord("x"),
        _VOCAB[END_MESSAGE],
        _VOCAB[MSG_MODEL],
    ]
    ids.extend(_VOCAB[token] for token in tail)
    return ids


def test_tool_opener_ends_reasoning_like_text():
    parser = _parser(_VOCAB)
    assert parser.is_reasoning_end(_after_model_turn(TEXT_START))
    for opener in (TOOL_JSON, TOOL_TEXT, TOOL_ERROR):
        assert parser.is_reasoning_end(_after_model_turn(opener))
    # The thinking block was left open and the next token is the tool opener.
    assert parser.is_reasoning_end([_VOCAB[THINK_START], ord("x"), _VOCAB[TOOL_JSON]])
    assert not parser.is_reasoning_end(
        [_VOCAB[THINK_START], ord("x"), _VOCAB[END_MESSAGE], _VOCAB[MSG_MODEL]]
    )


def test_tool_opener_is_an_implicit_reasoning_end():
    parser = _parser(_VOCAB)
    tool_id = _VOCAB[TOOL_JSON]
    text_id = _VOCAB[TEXT_START]
    end_id = _VOCAB[END_SAMPLING]
    assert tool_id in parser.implicit_reasoning_end_token_ids
    assert text_id in parser.reasoning_end_token_ids
    assert text_id not in parser.implicit_reasoning_end_token_ids
    assert end_id in parser.reasoning_end_token_ids
    assert end_id not in parser.implicit_reasoning_end_token_ids
    assert parser.find_reasoning_end([tool_id, ord("{")]) == ReasoningEnd(0, True)
    assert parser.find_reasoning_end([text_id, ord("a")]) == ReasoningEnd(0, False)
    assert parser.find_reasoning_end([ord("a")]) == ReasoningEnd(1, False)

    adapter = _InklingReasoningAdapter(make_mock_tokenizer(_VOCAB))
    assert adapter.is_reasoning_end(_after_model_turn(TOOL_JSON))
    assert adapter.find_reasoning_end([tool_id]) == ReasoningEnd(0, True)


def test_inkling2_tool_opener_ends_model_reasoning():
    vocab = {**_VOCAB, "<|content_xml|>": 200024}
    parser = _parser(vocab)
    prompt = [
        vocab["<|message_user|>"],
        vocab[TEXT_START],
        ord("q"),
        vocab[END_MESSAGE],
    ]
    assert not parser.is_reasoning_end(prompt)
    assert not parser.is_reasoning_end(prompt + [vocab[MSG_MODEL], vocab[THINK_START]])
    assert parser.is_reasoning_end(
        prompt + [vocab[MSG_MODEL], vocab[TEXT_START], ord("x")]
    )
    assert parser.is_reasoning_end(prompt + [vocab[MSG_MODEL], vocab[TOOL_JSON]])
    assert vocab[TOOL_JSON] in parser.implicit_reasoning_end_token_ids
    assert vocab[END_SAMPLING] not in parser.reasoning_end_token_ids

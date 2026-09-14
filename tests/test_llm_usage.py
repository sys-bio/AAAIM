"""Token-limit and usage-accounting tests that do not call an external API."""

from types import SimpleNamespace

from core.llm_interface import _make_api_call_with_retry, summarize_usage_records


class _FakeCompletions:
    def __init__(self, response):
        self.response = response
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def test_openai_completion_limit_and_usage_callback():
    response = SimpleNamespace(
        model="gpt-test",
        usage=SimpleNamespace(
            prompt_tokens=120,
            completion_tokens=45,
            total_tokens=165,
            prompt_tokens_details=SimpleNamespace(cached_tokens=20),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=30),
        ),
    )
    completions = _FakeCompletions(response)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    records = []

    returned = _make_api_call_with_retry(
        client,
        "gpt-test",
        [{"role": "user", "content": "test"}],
        max_completion_tokens=500,
        usage_callback=records.append,
    )

    assert returned is response
    assert completions.kwargs["max_completion_tokens"] == 500
    assert records == [{
        "input_tokens": 120,
        "output_tokens": 45,
        "total_tokens": 165,
        "reasoning_tokens": 30,
        "cached_input_tokens": 20,
        "model": "gpt-test",
    }]


def test_summarize_usage_records():
    summary = summarize_usage_records([
        {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15, "model": "gpt-test"},
        {"input_tokens": 8, "output_tokens": 4, "total_tokens": 12, "reasoning_tokens": 2, "model": "gpt-test"},
    ])
    assert summary == {
        "input_tokens": 18,
        "output_tokens": 9,
        "total_tokens": 27,
        "reasoning_tokens": 2,
        "cached_input_tokens": 0,
        "requests": 2,
        "models": ["gpt-test"],
    }

from graspgraph.planner.ollama_debug import build_request, raw_chat


class StubClient:
    """Stands in for ollama.Client: records the request and replays canned chunks."""

    def __init__(self, chunks):
        self.chunks = chunks
        self.requests = []

    def chat(self, **params):
        self.requests.append(params)
        return iter(self.chunks) if params["stream"] else self.chunks[-1]


def test_request_matches_what_the_benchmark_sends():
    _, params = build_request("tower-000", "gemma4:12b", max_tokens=300, temperature=0.0)
    assert params["model"] == "gemma4:12b" and params["think"] is False
    assert params["options"]["num_predict"] == 300 and params["options"]["temperature"] == 0.0
    assert [m["role"] for m in params["messages"]] == ["system", "user"]
    assert "submit_plan" in params["messages"][0]["content"]
    assert [t["function"]["name"] for t in params["tools"]] == ["submit_plan", "ask_clarification", "refuse"]


def test_raw_chat_stitches_streamed_chunks_and_keeps_them():
    chunks = [
        {"message": {"role": "assistant", "content": "", "thinking": "hmm"}},
        {"message": {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "submit_plan", "arguments": {}}}]}},
        {"message": {"role": "assistant", "content": ""}, "done": True, "done_reason": "stop", "eval_count": 120},
    ]
    client = StubClient(chunks)
    result = raw_chat(client, {"model": "m", "messages": []}, stream=True)
    assert result["tool_calls"] == [{"function": {"name": "submit_plan", "arguments": {}}}]
    assert (result["thinking"], result["done_reason"], result["output_tokens"]) == ("hmm", "stop", 120)
    assert len(result["chunks"]) == 3 and client.requests[0]["stream"] is True


def test_raw_chat_without_streaming_and_with_an_empty_reply():
    empty = {"message": {"role": "assistant", "content": ""}, "done": True, "done_reason": "length", "eval_count": 2048}
    result = raw_chat(StubClient([empty]), {"model": "m", "messages": []}, stream=False)
    assert (result["content"], result["tool_calls"], result["done_reason"]) == ("", [], "length")

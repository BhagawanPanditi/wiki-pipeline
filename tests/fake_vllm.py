"""A real OpenAI-compatible HTTP server for testing stages without a GPU.

Stages talk to vLLM over HTTP, so the only faithful way to exercise pooling,
balancing, retries and timeouts in a test is to serve actual HTTP. Each server
instance can be told how to answer, how many requests to fail first, and records
its own request count so tests can assert work was spread evenly.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def _default_responder(prompt: str, request_index: int):
    """Echo-style responder: a coverage score derived from the prompt's size."""
    concepts_seen = prompt.count('"title"')
    score = round(min(0.50 + 0.15 * concepts_seen, 1.0), 3)
    sufficient = score >= 0.85
    return {
        "coverage_score": score,
        "is_sufficient": sufficient,
        "missing_concepts": [] if sufficient else ["Omitted theme"],
        "missing_facts": [] if sufficient else ["Omitted metric"],
        "hallucinations": [],
        "critique": f"Fake audit; {concepts_seen} concepts observed.",
    }


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        """Silence the default per-request logging so test output stays readable."""
        pass

    def _send_json(self, code, payload):
        """Write ``payload`` as a JSON response with HTTP status ``code``.

        Args:
            code: HTTP status code.
            payload: JSON-serialisable body."""
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        """Answer ``GET .../models`` so a health probe sees a live server.

        Any other GET path gets a 404."""
        if self.path.endswith("/models"):
            self._send_json(200, {"object": "list", "data": [{"id": "test-model"}]})
            return
        self.send_error(404)

    def do_POST(self):
        """Handle one chat completion.

        Counts the request and records its prompt. The first ``fail_first`` requests fail
        with ``fail_with`` (or hang, for ``"timeout"``). Every other request is answered by
        the responder: a dict is sent as JSON, a string is sent verbatim, and a
        ``(content, finish_reason)`` tuple simulates a truncated answer."""
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length).decode("utf-8")
        state = self.server.state

        with state["lock"]:
            state["requests"] += 1
            count = state["requests"]
            if state["fail_first"] and count <= state["fail_first"]:
                fail_with = state["fail_with"]
                if fail_with == "timeout":
                    # Simulate a hung request: close without answering.
                    self.close_connection = True
                    return
                self._send_json(fail_with, {"error": {"message": "injected fault"}})
                return

        try:
            payload = json.loads(raw)
            prompt = payload["messages"][-1]["content"]
            model = payload.get("model", "test-model")
        except Exception:
            prompt, model = raw, "test-model"

        with state["lock"]:
            state["prompts"].append(prompt)

        answer = state["responder"](prompt, count)
        # A responder may return: a dict (serialised as JSON), a str (raw text),
        # or a (content, finish_reason) tuple to simulate truncation.
        if isinstance(answer, tuple):
            content, finish = answer
            if not isinstance(content, str):
                content = json.dumps(content)
        elif isinstance(answer, str):
            content, finish = answer, "stop"
        else:
            content, finish = json.dumps(answer), "stop"
        self._send_json(200, {
            "id": f"chatcmpl-{count}",
            "object": "chat.completion",
            "created": 1700000000,
            "model": model,
            "choices": [{"index": 0,
                         "message": {"role": "assistant", "content": content},
                         "finish_reason": finish}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        })


class FakeVLLMServer:
    """A throwaway vLLM-compatible endpoint bound to 127.0.0.1 on a free port."""

    def __init__(self, *, responder=None, fail_first=0, fail_with=503):
        """Create, but do not start, a fake endpoint on a free localhost port.

        Args:
            responder: Callable ``(prompt, index) -> reply``. ``index`` counts requests
                from zero. Defaults to a generic valid answer.
            fail_first: How many of the first requests fail.
            fail_with: HTTP status for those failures, or ``"timeout"`` to accept the
                connection and never answer."""
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.httpd.state = {
            "lock": threading.Lock(),
            "requests": 0,
            "prompts": [],
            "fail_first": fail_first,
            "fail_with": fail_with,
            "responder": responder or _default_responder,
        }
        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def url(self):
        """The base URL to pass to ``--servers``.

        Returns:
            ``http://127.0.0.1:<port>/v1``."""
        return f"http://127.0.0.1:{self.httpd.server_port}/v1"

    @property
    def requests(self):
        """How many requests this server has received.

        Returns:
            An int, counting failed requests as well as successful ones."""
        return self.httpd.state["requests"]

    @property
    def prompts(self):
        """Every prompt this server has received, in arrival order.

        Returns:
            A copy of the list of prompt texts."""
        return list(self.httpd.state["prompts"])

    def start(self):
        """Start serving in a background thread.

        Returns:
            This server."""
        self._thread.start()
        return self

    def stop(self):
        """Shut the server down and release its socket."""
        self.httpd.shutdown()
        self.httpd.server_close()

    def __enter__(self):
        """Start the server for use in a ``with`` block.

        Returns:
            This server."""
        return self.start()

    def __exit__(self, *exc):
        """Stop the server when leaving a ``with`` block.

        Args:
            *exc: Ignored exception context."""
        self.stop()


class FakeVLLMCluster:
    """N fake servers, with helpers for asserting even spread."""

    def __init__(self, count, **kwargs):
        """Start ``count`` independent fake servers.

        Args:
            count: How many servers to start.
            **kwargs: Passed to each :class:`FakeVLLMServer`."""
        self.servers = [FakeVLLMServer(**kwargs).start() for _ in range(count)]

    @property
    def urls(self):
        """The base URL of every server, in order.

        Returns:
            A list of URL strings."""
        return [s.url for s in self.servers]

    @property
    def counts(self):
        """The requests each server has received, in server order.

        Returns:
            A list of ints. Used to check that work was spread evenly."""
        return [s.requests for s in self.servers]

    @property
    def total_requests(self):
        """Requests received across all servers.

        Returns:
            An int."""
        return sum(self.counts)

    def assert_balanced(self, test, expected_total=None):
        """No server may hold more than one extra item versus the busiest-free peer."""
        counts = self.counts
        if expected_total is not None:
            test.assertEqual(sum(counts), expected_total, msg=f"counts={counts}")
        if sum(counts) == 0:
            return
        test.assertLessEqual(max(counts) - min(counts), 1,
                             msg=f"requests were not spread evenly: {counts}")

    def stop(self):
        """Stop every server in the cluster."""
        for s in self.servers:
            s.stop()

    def __enter__(self):
        """Return the cluster for use in a ``with`` block.

        Returns:
            This cluster."""
        return self

    def __exit__(self, *exc):
        """Stop the cluster when leaving a ``with`` block.

        Args:
            *exc: Ignored exception context."""
        self.stop()

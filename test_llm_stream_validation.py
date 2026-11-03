"""Parser tests only: synthetic fixtures are never performance measurements."""
import io
import json
import unittest
from unittest.mock import patch

from benchmark_llm_http import request


class Response(io.BytesIO):
    status = 200


class Connection:
    payload = b""

    def __init__(self, *args, **kwargs):
        pass

    def request(self, *args, **kwargs):
        pass

    def getresponse(self):
        return Response(self.payload)

    def close(self):
        pass


def stream(indices, done=True):
    data = [b'event: accepted\ndata: {"request_id":"test","index":999}\n\n']
    for i in indices:
        # Each token JSON uses multiple data lines; all events share one byte stream.
        data.append(f'event: token\ndata: {{"index":{i},\ndata: "token_id":7,"generated_ns":0}}\n\n'.encode())
    if done:
        data.append(b'event: done\ndata: {"output_tokens":256}\n\n')
    return b"".join(data)


class StreamValidation(unittest.TestCase):
    def run_fixture(self, content):
        Connection.payload = content
        callbacks = []
        with patch("benchmark_llm_http.http.client.HTTPConnection", Connection):
            result = request(0, "test", [1]*128, token_callback=lambda t: callbacks.append(t["index"]))
        return result, callbacks

    def test_auxiliary_events_are_not_tokens(self):
        result, callbacks = self.run_fixture(stream(range(1,257)))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(callbacks, list(range(1,257)))
        self.assertEqual(callbacks.count(16), 1)
        self.assertEqual(len(result["aux_events"]), 2)

    def test_duplicate_token_is_invalid(self):
        result, _ = self.run_fixture(stream(list(range(1,256))+[255]))
        self.assertEqual(result["status"], "error")

    def test_early_eof_is_invalid(self):
        result, _ = self.run_fixture(stream(range(1,257), done=False))
        self.assertEqual(result["status"], "error")

    def test_truncated_output_is_invalid(self):
        result, _ = self.run_fixture(stream(range(1,33)))
        self.assertEqual(result["status"], "error")


if __name__ == "__main__":
    unittest.main()

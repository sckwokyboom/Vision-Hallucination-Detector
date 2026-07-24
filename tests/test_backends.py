from shroom.backends.base import VLMBackend, MockBackend


def test_mock_list_script(dummy_image):
    be = MockBackend([["a", "b", "c"], ["x"]])
    assert be.generate(dummy_image, "prompt-1", n=3) == ["a", "b", "c"]
    assert be.generate(dummy_image, "prompt-2", n=1) == ["x"]
    assert be.calls[0][0] == "prompt-1"


def test_mock_callable_script(dummy_image):
    be = MockBackend(lambda text, n: ["[]"] * n)
    assert be.generate(dummy_image, "p", n=2) == ["[]", "[]"]


def test_is_subclass():
    assert issubclass(MockBackend, VLMBackend)

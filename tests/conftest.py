import os
import pytest
from PIL import Image

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "mini.train.jsonl")


@pytest.fixture
def mini_path():
    return FIXTURE


@pytest.fixture
def dummy_image():
    return Image.new("RGB", (8, 8), (127, 127, 127))

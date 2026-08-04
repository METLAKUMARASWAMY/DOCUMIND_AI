import pytest

from app.services.vision.adapters import AdapterUnavailableError
from app.services.vision.extraction_adapters import LayoutLMv3ExtractionAdapter


def test_disabled_heavy_adapter_raises_instead_of_returning_fake_output():
    adapter = LayoutLMv3ExtractionAdapter()

    with pytest.raises(AdapterUnavailableError) as exc:
        adapter.extract_fields(None)  # type: ignore[arg-type]

    assert adapter.status().status in str(exc.value)

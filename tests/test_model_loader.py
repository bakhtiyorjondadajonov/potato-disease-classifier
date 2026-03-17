from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

import model_loader
from model_loader import CLASS_NAMES, get_model, load_model


class TestLoadModel:
    def setup_method(self):
        self._original_model = model_loader._model

    def teardown_method(self):
        model_loader._model = self._original_model

    def test_load_model_success(self):
        with patch("model_loader.tf.keras.models.load_model") as mock_load:
            mock_load.return_value = MagicMock()
            load_model("fake_path.keras")
            assert model_loader._model is not None

    def test_load_model_failure_sets_none(self):
        with patch("model_loader.tf.keras.models.load_model", side_effect=Exception("fail")):
            load_model("bad_path.keras")
            assert model_loader._model is None


class TestGetModel:
    def test_get_model_when_loaded(self):
        mock = MagicMock()
        with patch.object(model_loader, "_model", mock):
            assert get_model() is mock

    def test_get_model_when_none_raises_503(self):
        with patch.object(model_loader, "_model", None):
            with pytest.raises(HTTPException) as exc_info:
                get_model()
            assert exc_info.value.status_code == 503


class TestClassNames:
    def test_class_names_values(self):
        assert CLASS_NAMES == ["Early Blight", "Late Blight", "Healthy"]

    def test_class_names_length(self):
        assert len(CLASS_NAMES) == 3

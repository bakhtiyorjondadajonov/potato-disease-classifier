from unittest.mock import patch

import numpy as np

from tests.conftest import upload_file, _create_test_image_bytes


class TestPredictSuccess:
    def test_predict_success_early_blight(self, client, jpeg_image_bytes):
        resp = client.post("/predict", files=[upload_file(jpeg_image_bytes)])
        assert resp.status_code == 200
        data = resp.json()
        assert data["class"] == "Early Blight"
        assert data["confidence"] == 0.95

    def test_predict_success_late_blight(self, client, jpeg_image_bytes):
        with patch("routes.predict.get_model") as mock_get:
            mock = mock_get.return_value
            mock.predict.return_value = np.array([[0.05, 0.90, 0.05]])
            resp = client.post("/predict", files=[upload_file(jpeg_image_bytes)])
            assert resp.status_code == 200
            assert resp.json()["class"] == "Late Blight"

    def test_predict_success_healthy(self, client, jpeg_image_bytes):
        with patch("routes.predict.get_model") as mock_get:
            mock = mock_get.return_value
            mock.predict.return_value = np.array([[0.02, 0.03, 0.95]])
            resp = client.post("/predict", files=[upload_file(jpeg_image_bytes)])
            assert resp.status_code == 200
            assert resp.json()["class"] == "Healthy"


class TestPredictErrors:
    def test_predict_unsupported_file_type(self, client):
        data = _create_test_image_bytes(fmt="JPEG")
        resp = client.post("/predict", files=[upload_file(data, content_type="image/gif")])
        assert resp.status_code == 415

    def test_predict_file_too_large(self, client, oversized_image_bytes):
        resp = client.post("/predict", files=[upload_file(oversized_image_bytes)])
        assert resp.status_code == 413

    def test_predict_corrupt_image(self, client):
        resp = client.post("/predict", files=[upload_file(b"garbage bytes")])
        assert resp.status_code == 400

    def test_predict_model_not_loaded(self, client_no_model, jpeg_image_bytes):
        resp = client_no_model.post("/predict", files=[upload_file(jpeg_image_bytes)])
        assert resp.status_code == 503

    def test_predict_model_exception(self, client, jpeg_image_bytes):
        with patch("routes.predict.get_model") as mock_get:
            mock = mock_get.return_value
            mock.predict.side_effect = Exception("TF error")
            resp = client.post("/predict", files=[upload_file(jpeg_image_bytes)])
            assert resp.status_code == 500

    def test_predict_no_file(self, client):
        resp = client.post("/predict")
        assert resp.status_code == 422

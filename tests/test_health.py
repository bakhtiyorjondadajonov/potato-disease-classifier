class TestPing:
    def test_ping(self, client):
        resp = client.get("/ping")
        assert resp.status_code == 200
        assert resp.json() == "Hello, I am alive"


class TestHealthCheck:
    def test_health_model_loaded(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_loaded"] is True
        assert data["gemini_available"] is True

    def test_health_model_not_loaded(self, client_no_model):
        resp = client_no_model.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_loaded"] is False
        assert data["gemini_available"] is True

    def test_health_gemini_not_available(self, client_no_gemini):
        resp = client_no_gemini.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_loaded"] is True
        assert data["gemini_available"] is False

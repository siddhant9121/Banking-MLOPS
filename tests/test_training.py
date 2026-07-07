import pytest
import sys
from pathlib import Path

# Make src importable when pytest runs from project root
sys.path.insert(0, str(Path(__file__).parent.parent))


# ── File existence checks ──────────────────────────────────────────────────────

def test_config_exists():
    assert Path('configs/model_config.yaml').exists(), "model_config.yaml must exist"


def test_params_exists():
    assert Path('params.yaml').exists(), "params.yaml (DVC) must exist"


def test_requirements_exists():
    assert Path('requirements.txt').exists(), "requirements.txt must exist"


def test_train_script_exists():
    assert Path('src/train.py').exists(), "src/train.py must exist"


def test_api_script_exists():
    assert Path('src/api.py').exists(), "src/api.py must exist"


def test_dockerfile_exists():
    assert Path('Dockerfile').exists(), "Dockerfile must exist"


# ── Config structure checks ────────────────────────────────────────────────────

def test_config_has_required_sections():
    import yaml
    with open('configs/model_config.yaml') as f:
        cfg = yaml.safe_load(f)
    for section in ('classifier', 'ner', 'data', 'training', 'validation'):
        assert section in cfg, f"Config missing section: {section}"


def test_config_data_splits_sum_to_one():
    import yaml
    with open('configs/model_config.yaml') as f:
        cfg = yaml.safe_load(f)
    splits = cfg['data']
    total = splits['train_split'] + splits['val_split'] + splits['test_split']
    assert abs(total - 1.0) < 1e-6, f"Data splits must sum to 1.0, got {total}"


def test_config_thresholds_are_sane():
    import yaml
    with open('configs/model_config.yaml') as f:
        cfg = yaml.safe_load(f)
    val = cfg['validation']
    assert 0 < val['min_classifier_accuracy'] <= 1.0
    assert 0 < val['min_ner_f1'] <= 1.0


# ── ModelTrainer unit tests ───────────────────────────────────────────────────

def test_model_trainer_initializes():
    from src.train import ModelTrainer
    trainer = ModelTrainer(config_path='configs/model_config.yaml')
    assert trainer.config is not None
    assert trainer.device is not None


def test_model_trainer_loads_data():
    from src.train import ModelTrainer
    trainer = ModelTrainer(config_path='configs/model_config.yaml')
    train_data, val_data = trainer.load_data()
    assert 'images' in train_data
    assert 'labels' in train_data
    assert 'annotations' in train_data


def test_model_trainer_validate_pass():
    from src.train import ModelTrainer
    trainer = ModelTrainer(config_path='configs/model_config.yaml')
    # Metrics that should comfortably clear the configured thresholds
    passed = trainer.validate_models(
        classifier_metrics={'accuracy': 0.95, 'model_path': 'models/classifier/best_model.pth'},
        ner_metrics={'f1_score': 0.93, 'model_path': 'models/ner/best_model.pth'},
    )
    assert passed is True


def test_model_trainer_validate_fail():
    from src.train import ModelTrainer
    trainer = ModelTrainer(config_path='configs/model_config.yaml')
    # Metrics that fall below threshold
    passed = trainer.validate_models(
        classifier_metrics={'accuracy': 0.50, 'model_path': ''},
        ner_metrics={'f1_score': 0.50, 'model_path': ''},
    )
    assert passed is False


# ── FastAPI smoke test ─────────────────────────────────────────────────────────

def test_api_app_importable():
    try:
        from src.api import app
        assert app is not None
    except ImportError as e:
        pytest.skip(f"API dependencies not installed: {e}")


def test_api_health_endpoint():
    try:
        from fastapi.testclient import TestClient
        from src.api import app
        client   = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    except ImportError as e:
        pytest.skip(f"FastAPI test dependencies not installed: {e}")


def test_api_rejects_unsupported_file_type():
    try:
        from fastapi.testclient import TestClient
        from src.api import app
        import io
        client   = TestClient(app)
        response = client.post(
            "/process-document",
            files={"file": ("test.txt", io.BytesIO(b"hello"), "text/plain")},
        )
        assert response.status_code == 400
    except ImportError as e:
        pytest.skip(f"FastAPI test dependencies not installed: {e}")

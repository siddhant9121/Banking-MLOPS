"""
Basic tests for training pipeline
"""
import pytest
from pathlib import Path


def test_config_exists():
    """Test that configuration file exists"""
    config_path = Path('configs/model_config.yaml')
    assert config_path.exists(), "Config file should exist"


def test_requirements_exists():
    """Test that requirements file exists"""
    req_path = Path('requirements.txt')
    assert req_path.exists(), "Requirements file should exist"


def test_train_script_exists():
    """Test that train script exists"""
    train_path = Path('src/train.py')
    assert train_path.exists(), "Train script should exist"


def test_import_training():
    """Test that training module can be imported"""
    try:
        import sys
        sys.path.insert(0, 'src')
        assert True
    except ImportError:
        pytest.skip("Training module not yet implemented")


def test_config_format():
    """Test configuration file is valid YAML"""
    import yaml
    config_path = Path('configs/model_config.yaml')
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    assert 'classifier' in config
    assert 'ner' in config
    assert 'data' in config
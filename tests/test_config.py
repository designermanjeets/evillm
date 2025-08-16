"""Tests for configuration management."""

import pytest
import os
import tempfile
from pathlib import Path
import yaml

from app.config.manager import ConfigManager, get_ocr_config, get_feature_flag
from app.config.ocr import OCRSettings


class TestConfigManager:
    """Test configuration manager functionality."""
    
    def test_find_config_path(self):
        """Test configuration file path discovery."""
        manager = ConfigManager()
        
        # Should find config in project root
        config_path = manager._find_config_path()
        assert config_path.name == "app.yaml"
        assert "config" in str(config_path)
    
    def test_load_yaml_config(self):
        """Test YAML configuration loading."""
        # Create temporary config file
        config_data = {
            "ocr": {
                "backend": "tesseract",
                "timeout_seconds": 30,
                "concurrency": 8
            },
            "storage": {
                "provider": "minio",
                "endpoint": "http://localhost:9000"
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            manager = ConfigManager(str(config_path))
            yaml_config = manager._load_yaml_config()
            
            assert yaml_config["ocr"]["backend"] == "tesseract"
            assert yaml_config["ocr"]["timeout_seconds"] == 30
            assert yaml_config["storage"]["provider"] == "minio"
        finally:
            os.unlink(config_path)
    
    def test_load_env_config(self):
        """Test environment variable configuration loading."""
        # Set environment variables
        os.environ["APP_OCR_BACKEND"] = "aws_textract"
        os.environ["APP_OCR_TIMEOUT_SECONDS"] = "45"
        os.environ["APP_STORAGE_PROVIDER"] = "s3"
        os.environ["APP_OCR_FEATURE_FLAGS_OCR_PREPROCESSING"] = "false"
        
        try:
            manager = ConfigManager()
            env_config = manager._load_env_config()
            
            assert env_config["ocr"]["backend"] == "aws_textract"
            assert env_config["ocr"]["timeout_seconds"] == 45
            assert env_config["storage"]["provider"] == "s3"
            # Feature flags are nested, so check the structure
            assert "feature_flags" in env_config["ocr"]
            assert env_config["ocr"]["feature_flags"]["ocr_preprocessing"] is False
        finally:
            # Clean up environment variables
            del os.environ["APP_OCR_BACKEND"]
            del os.environ["APP_OCR_TIMEOUT_SECONDS"]
            del os.environ["APP_STORAGE_PROVIDER"]
            del os.environ["APP_OCR_FEATURE_FLAGS_OCR_PREPROCESSING"]
    
    def test_convert_env_value(self):
        """Test environment value type conversion."""
        manager = ConfigManager()
        
        # Boolean values
        assert manager._convert_env_value("true") is True
        assert manager._convert_env_value("false") is False
        
        # Integer values
        assert manager._convert_env_value("42") == 42
        assert manager._convert_env_value("0") == 0
        
        # Float values
        assert manager._convert_env_value("3.14") == 3.14
        
        # List values
        assert manager._convert_env_value("a,b,c") == ["a", "b", "c"]
        assert manager._convert_env_value("1,2,3") == ["1", "2", "3"]
        
        # String values
        assert manager._convert_env_value("hello") == "hello"
    
    def test_merge_configs(self):
        """Test configuration merging."""
        manager = ConfigManager()
        
        yaml_config = {
            "ocr": {
                "backend": "local_stub",
                "timeout_seconds": 20
            },
            "storage": {
                "provider": "minio"
            }
        }
        
        env_config = {
            "ocr": {
                "backend": "tesseract",
                "concurrency": 8
            }
        }
        
        merged = manager._merge_configs(yaml_config, env_config)
        
        # YAML values should be preserved
        assert merged["storage"]["provider"] == "minio"
        
        # Environment should override YAML
        assert merged["ocr"]["backend"] == "tesseract"
        
        # Environment should add new values
        assert merged["ocr"]["concurrency"] == 8
        
        # YAML values should be preserved when not overridden
        assert merged["ocr"]["timeout_seconds"] == 20
    
    def test_get_ocr_config(self):
        """Test getting OCR configuration."""
        manager = ConfigManager()
        ocr_config = manager.get_ocr_config()
        
        # Should return OCRSettings instance
        assert isinstance(ocr_config, OCRSettings)
        
        # Should have default values
        assert ocr_config.backend == "local_stub"
        assert ocr_config.timeout_seconds == 20
        assert ocr_config.concurrency == 4
    
    def test_get_feature_flag(self):
        """Test feature flag retrieval."""
        manager = ConfigManager()
        
        # Test default feature flag values
        assert manager.get_feature_flag("ocr_idempotent_skip") is True
        assert manager.get_feature_flag("ocr_queue_inprocess") is True
        assert manager.get_feature_flag("nonexistent_flag") is False


class TestOCRSettings:
    """Test OCR settings validation."""
    
    def test_backend_validation(self):
        """Test backend validation."""
        # Valid backends
        valid_backends = ["local_stub", "tesseract", "aws_textract", "gcv", "azure_cv"]
        
        for backend in valid_backends:
            settings = OCRSettings(backend=backend)
            assert settings.backend == backend
        
        # Invalid backend should raise error
        with pytest.raises(ValueError, match="Invalid OCR backend"):
            OCRSettings(backend="invalid_backend")
    
    def test_concurrency_validation(self):
        """Test concurrency validation."""
        # Valid concurrency values
        valid_values = [1, 10, 20]
        
        for value in valid_values:
            settings = OCRSettings(concurrency=value)
            assert settings.concurrency == value
        
        # Invalid values should raise error
        with pytest.raises(ValueError, match="OCR concurrency must be between 1 and 20"):
            OCRSettings(concurrency=0)
        
        with pytest.raises(ValueError, match="OCR concurrency must be between 1 and 20"):
            OCRSettings(concurrency=21)
    
    def test_timeout_validation(self):
        """Test timeout validation."""
        # Valid timeout values
        valid_values = [1, 30, 300]
        
        for value in valid_values:
            settings = OCRSettings(timeout_seconds=value)
            assert settings.timeout_seconds == value
        
        # Invalid values should raise error
        with pytest.raises(ValueError, match="OCR timeout must be between 1 and 300 seconds"):
            OCRSettings(timeout_seconds=0)
        
        with pytest.raises(ValueError, match="OCR timeout must be between 1 and 300 seconds"):
            OCRSettings(timeout_seconds=301)
    
    def test_default_values(self):
        """Test default configuration values."""
        settings = OCRSettings()
        
        assert settings.backend == "local_stub"
        assert settings.enabled is True
        assert settings.timeout_seconds == 20
        assert settings.max_retries == 2
        assert settings.concurrency == 4
        assert settings.max_pages == 20
        assert settings.size_cap_mb == 15
        
        # Feature flags
        assert settings.feature_flags["ocr_idempotent_skip"] is True
        assert settings.feature_flags["ocr_queue_inprocess"] is True
        assert settings.feature_flags["ocr_redact_logs"] is True
        assert settings.feature_flags["ocr_preprocessing"] is True
        assert settings.feature_flags["ocr_tesseract_enabled"] is False
        
        # Preprocessing
        assert settings.preprocess["grayscale"] is True
        assert settings.preprocess["binarize"] is True
        assert settings.preprocess["deskew"] is True
        assert settings.preprocess["timeout_seconds"] == 5
    
    def test_mimetype_allowlist(self):
        """Test mimetype allowlist."""
        settings = OCRSettings()
        
        allowed_mimetypes = [
            "application/pdf",
            "image/png",
            "image/jpeg",
            "image/gif",
            "image/bmp",
            "image/tiff",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/msword",
            "text/plain"
        ]
        
        for mimetype in allowed_mimetypes:
            assert mimetype in settings.allow_mimetypes


class TestConfigIntegration:
    """Test configuration integration."""
    
    def test_get_ocr_config_integration(self):
        """Test OCR configuration integration."""
        ocr_config = get_ocr_config()
        
        assert isinstance(ocr_config, OCRSettings)
        assert ocr_config.backend == "local_stub"
        assert ocr_config.enabled is True
    
    def test_get_feature_flag_integration(self):
        """Test feature flag integration."""
        # Test OCR feature flags
        assert get_feature_flag("ocr_idempotent_skip") is True
        assert get_feature_flag("ocr_queue_inprocess") is True
        assert get_feature_flag("ocr_redact_logs") is True
        assert get_feature_flag("ocr_preprocessing") is True
        assert get_feature_flag("ocr_tesseract_enabled") is False
        
        # Test non-existent flags
        assert get_feature_flag("nonexistent_flag") is False


if __name__ == "__main__":
    pytest.main([__file__])

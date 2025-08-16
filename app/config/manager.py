"""Centralized configuration management."""

import os
from typing import Dict, Any, Optional
from pathlib import Path
import yaml
from pydantic import BaseModel, field_validator

from .ocr import OCRSettings


class AppConfig(BaseModel):
    """Application configuration model."""
    
    # App metadata
    app: Dict[str, Any] = {}
    
    # OCR configuration
    ocr: Optional[OCRSettings] = None
    
    # Storage configuration
    storage: Dict[str, Any] = {}
    
    # Database configuration
    database: Dict[str, Any] = {}
    
    # Models configuration
    models: Dict[str, Any] = {}
    
    # Logging configuration
    logging: Dict[str, Any] = {}
    
    # Metrics configuration
    metrics: Dict[str, Any] = {}
    
    # Security configuration
    security: Dict[str, Any] = {}
    
    # Performance configuration
    performance: Dict[str, Any] = {}


class ConfigManager:
    """Manages application configuration loading and validation."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration manager."""
        if config_path:
            self.config_path = Path(config_path)
        else:
            self.config_path = self._find_config_path()
        self._config: Optional[AppConfig] = None
    
    def _find_config_path(self) -> Path:
        """Find configuration file path."""
        # Look for config in current directory, then parent directories
        current = Path.cwd()
        while current != current.parent:
            config_file = current / "config" / "app.yaml"
            if config_file.exists():
                return config_file
            current = current.parent
        
        # Fallback to default location
        return Path(__file__).parent / "app.yaml"
    
    def load_config(self) -> AppConfig:
        """Load configuration from YAML and environment variables."""
        if self._config is not None:
            return self._config
        
        # Load YAML configuration
        yaml_config = self._load_yaml_config()
        
        # Override with environment variables
        env_config = self._load_env_config()
        
        # Merge configurations (env overrides YAML)
        merged_config = self._merge_configs(yaml_config, env_config)
        
        # Validate and create config object
        self._config = AppConfig(**merged_config)
        
        return self._config
    
    def _load_yaml_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            print(f"Warning: Configuration file not found: {self.config_path}")
            return {}
        
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
                return config or {}
        except Exception as e:
            print(f"Warning: Failed to load YAML config: {e}")
            return {}
    
    def _load_env_config(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        config = {}
        
        # Scan environment for APP_* variables
        for key, value in os.environ.items():
            if key.startswith('APP_'):
                # Convert APP_SECTION_KEY to nested dict structure
                parts = key[4:].lower().split('_', 1)
                if len(parts) == 2:
                    section, config_key = parts
                    if section not in config:
                        config[section] = {}
                    
                    # Handle nested keys like feature_flags_ocr_preprocessing
                    if '_' in config_key and config_key.startswith('feature_flags_'):
                        # Special handling for feature flags
                        feature_flag_key = config_key.replace('feature_flags_', '')
                        if 'feature_flags' not in config[section]:
                            config[section]['feature_flags'] = {}
                        config[section]['feature_flags'][feature_flag_key] = self._convert_env_value(value)
                    else:
                        # Convert value to appropriate type
                        config[section][config_key] = self._convert_env_value(value)
        
        return config
    
    def _convert_env_value(self, value: str) -> Any:
        """Convert environment variable string to appropriate type."""
        # Boolean values
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Integer values
        try:
            return int(value)
        except ValueError:
            pass
        
        # Float values
        try:
            return float(value)
        except ValueError:
            pass
        
        # List values (comma-separated)
        if ',' in value:
            return [item.strip() for item in value.split(',')]
        
        # Default to string
        return value
    
    def _merge_configs(self, yaml_config: Dict[str, Any], env_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge YAML and environment configurations."""
        merged = yaml_config.copy()
        
        # Deep merge environment overrides
        for section, values in env_config.items():
            if section not in merged:
                merged[section] = {}
            
            if isinstance(values, dict):
                for key, value in values.items():
                    merged[section][key] = value
            else:
                merged[section] = values
        
        return merged
    
    def get_ocr_config(self) -> OCRSettings:
        """Get OCR configuration."""
        config = self.load_config()
        return config.ocr or OCRSettings()
    
    def get_storage_config(self) -> Dict[str, Any]:
        """Get storage configuration."""
        config = self.load_config()
        return config.storage
    
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration."""
        config = self.load_config()
        return config.database
    
    def get_feature_flag(self, flag_name: str) -> bool:
        """Get feature flag value."""
        config = self.load_config()
        
        # Check OCR feature flags first
        if config.ocr and hasattr(config.ocr, 'feature_flags'):
            if flag_name in config.ocr.feature_flags:
                return config.ocr.feature_flags[flag_name]
        
        # Check global feature flags
        if 'feature_flags' in config.app:
            return config.app['feature_flags'].get(flag_name, False)
        
        return False
    
    def reload(self):
        """Reload configuration from disk."""
        self._config = None
        return self.load_config()


# Global configuration manager instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """Get global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def get_ocr_config() -> OCRSettings:
    """Get OCR configuration."""
    return get_config_manager().get_ocr_config()


def get_storage_config() -> Dict[str, Any]:
    """Get storage configuration."""
    return get_config_manager().get_storage_config()


def get_database_config() -> Dict[str, Any]:
    """Get database configuration."""
    return get_config_manager().get_database_config()


def get_feature_flag(flag_name: str) -> bool:
    """Get feature flag value."""
    return get_config_manager().get_feature_flag(flag_name)

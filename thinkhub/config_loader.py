"""
This module provides a configuration loader for the ThinkHub application.

It loads default and user-specific YAML configuration files, merges them, and
overrides specific settings with environment variables if available.
"""

import os

import yaml
from decouple import config as env_config


class ConfigLoader:
    """
    A utility class to load and manage application configurations.

    The loader prioritizes configurations in the following order:
    1. Default YAML configuration file.
    2. User-specific YAML configuration file (if exists).
    3. Environment variables for overriding specific settings.
    """

    def __init__(
        self, default_path="thinkhub_config.yaml", user_path="thinkhub_config.yaml"
    ):
        """
        Initialize the ConfigLoader.

        Args:
            default_path (str): Path to the default configuration YAML file.
            user_path (str): Path to the user-specific configuration YAML file.
        """
        self.default_path = default_path
        self.user_path = user_path
        self.config = self.load_config()

    def load_config(self):
        """
        Load the configuration.

        by combining default, user-specific, and environment variables

        Returns:
            dict: The final merged configuration.
        """
        # Load default configuration
        config = self._load_yaml(self.default_path)

        # Load user configuration if it exists
        if os.path.exists(self.user_path):
            user_config = self._load_yaml(self.user_path)
            config = self._merge_configs(config, user_config)

        # Override with environment variables
        self._override_with_env(config)

        return config

    @staticmethod
    def _load_yaml(file_path):
        """
        Load a YAML file.

        Args:
            file_path (str): Path to the YAML file.

        Returns:
            dict: The loaded YAML data.
        """
        with open(file_path) as file:
            return yaml.safe_load(file)

    @staticmethod
    def _merge_configs(default, override):
        """
        Merge default and user configurations recursively.

        Args:
            default (dict): The default configuration.
            override (dict): The user-specific configuration to merge.

        Returns:
            dict: The merged configuration.
        """
        for key, value in override.items():
            if isinstance(value, dict) and key in default:
                default[key] = ConfigLoader._merge_configs(default[key], value)
            else:
                default[key] = value
        return default

    def _override_with_env(self, config):
        """
        Override specific settings in the configuration with environment variables.

        Args:
            config (dict): The configuration dictionary to update.
        """
        config["openai"]["api_key"] = env_config(
            "OPENAI_API_KEY", default=config["openai"].get("api_key")
        )
        config["google"]["api_key"] = env_config(
            "GOOGLE_API_KEY", default=config["google"].get("api_key")
        )

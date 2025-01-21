import yaml
import os
from decouple import config as env_config


class ConfigLoader:
    def __init__(self, default_path="default_config.yaml", user_path="user_config.yaml"):
        self.default_path = default_path
        self.user_path = user_path
        self.config = self.load_config()

    def load_config(self):
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
        """Load a YAML file."""
        with open(file_path, "r") as file:
            return yaml.safe_load(file)

    @staticmethod
    def _merge_configs(default, override):
        """Merge default and user configurations recursively."""
        for key, value in override.items():
            if isinstance(value, dict) and key in default:
                default[key] = ConfigLoader._merge_configs(default[key], value)
            else:
                default[key] = value
        return default

    def _override_with_env(self, config):
        """Override specific settings with environment variables."""
        config["openai"]["api_key"] = env_config("OPENAI_API_KEY", default=config["openai"].get("api_key"))
        config["google"]["api_key"] = env_config("GOOGLE_API_KEY", default=config["google"].get("api_key"))

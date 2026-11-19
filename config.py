import yaml
import os
from typing import Any, Dict, Optional


class ConfigManager:
    """
    Configuration manager for loading and managing Llama model configuration
    from a YAML file.
    """

    # Default configuration values
    DEFAULT_CONFIG = {
        "model_path": "/opt/models/Qwen3.8-27B-UD-IQ3_S.gguf",
        "n_ctx": 2048,
        "n_predict": 128,
        "n_gpu_layers": 0,
        "flash_attn": False,
        "top_k": 40,
        "top_p": 0.9,
        "repeat_penalty": 1.1,
        "temperature": 0.8,
        "min_p": 0.05,
        "seed": 1337,
        "n_threads": 0,
        "n_batch": 512,
        "n_keep": 0,
        "n_gpu_batch_clamp": 0,
        "rope_scaling": None,
        "rope_freq_base": 10000.0,
        "rope_freq_scale": 1.0,
        "use_mmap": True,
        "use_mlock": False,
        "verbose": False,
        "embedding": False,
        "n_batch": 512,
        "n_parallel": 1,
        "n_threads": 0,
        "n_threads_batch": 0,
        "n_gpu_layers": 0,
        "main_gpu": 0,
        "tensor_split": None,
        "no_kv_offload": False,
        "cache_type_k": "f16",
        "cache_type_v": "f16",
        "pooling_type": None,
        "pooling_params": None,
        "logits_all": False,
        "embedding": False,
        "n_batch": 512,
        "n_parallel": 1,
        "n_threads": 0,
        "n_threads_batch": 0,
        "n_gpu_layers": 0,
        "main_gpu": 0,
        "tensor_split": None,
        "no_kv_offload": False,
        "cache_type_k": "f16",
        "cache_type_v": "f16",
        "pooling_type": None,
        "pooling_params": None,
        "logits_all": False,
    }

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the ConfigManager.

        Args:
            config_path: Path to the YAML configuration file. If None, uses defaults.
        """
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        self._load_config()

    def _load_config(self) -> None:
        """Load configuration from YAML file or use defaults."""
        if self.config_path and os.path.exists(self.config_path):
            with open(self.config_path, "r") as f:
                file_config = yaml.safe_load(f) or {}
            # Merge file config with defaults
            self.config = {**self.DEFAULT_CONFIG, **file_config}
        else:
            self.config = dict(self.DEFAULT_CONFIG)

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """
        Get a configuration value.

        Args:
            key: The configuration key.
            default: Default value if key is not found.

        Returns:
            The configuration value.
        """
        if default is None:
            default = self.DEFAULT_CONFIG.get(key)
        return self.config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value.

        Args:
            key: The configuration key.
            value: The value to set.
        """
        self.config[key] = value

    def save(self, path: Optional[str] = None) -> None:
        """
        Save the current configuration to a YAML file.

        Args:
            path: Path to save the configuration. If None, uses the original config_path.
        """
        save_path = path or self.config_path
        if save_path is None:
            raise ValueError("No path specified for saving configuration.")
        with open(save_path, "w") as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=True)

    def get_llama_params(self) -> Dict[str, Any]:
        """
        Get parameters suitable for llama_cpp Llama model loading.

        Returns:
            A dictionary of parameters for Llama model initialization.
        """
        return self.config
        return {
            "model_path": self.get("model"),
            "n_ctx": self.get("n_ctx"),
            "n_predict": self.get("n_predict"),
            "n_gpu_layers": self.get("n_gpu_layers"),
            "flash_attn": self.get("flash_attn"),
            "top_k": self.get("top_k"),
            "top_p": self.get("top_p"),
            "repeat_penalty": self.get("repeat_penalty"),
            "temperature": self.get("temperature"),
            "min_p": self.get("min_p"),
            "seed": self.get("seed"),
            "n_threads": self.get("n_threads"),
            "n_batch": self.get("n_batch"),
            "n_keep": self.get("n_keep"),
            "n_gpu_batch_clamp": self.get("n_gpu_batch_clamp"),
            "rope_scaling": self.get("rope_scaling"),
            "rope_freq_base": self.get("rope_freq_base"),
            "rope_freq_scale": self.get("rope_freq_scale"),
            "use_mmap": self.get("use_mmap"),
            "use_mlock": self.get("use_mlock"),
            "verbose": self.get("verbose"),
            "embedding": self.get("embedding"),
            "n_parallel": self.get("n_parallel"),
            "n_threads_batch": self.get("n_threads_batch"),
            "main_gpu": self.get("main_gpu"),
            "tensor_split": self.get("tensor_split"),
            "no_kv_offload": self.get("no_kv_offload"),
            "cache_type_k": self.get("cache_type_k"),
            "cache_type_v": self.get("cache_type_v"),
            "pooling_type": self.get("pooling_type"),
            "pooling_params": self.get("pooling_params"),
            "logits_all": self.get("logits_all"),
        }

    def __getitem__(self, key: str) -> Any:
        """Allow dict-like access to configuration values."""
        return self.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Allow dict-like setting of configuration values."""
        self.set(key, value)

    def __contains__(self, key: str) -> bool:
        """Check if a key exists in the configuration."""
        return key in self.config

    def __repr__(self) -> str:
        return f"ConfigManager(config={self.config})"

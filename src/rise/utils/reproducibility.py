"""
Reproducibility utilities for RISE experiments.

This module provides functions for ensuring reproducible experiments:
- Random seed management
- Environment logging
- Checkpoint management
"""

import logging
import os
import random
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import json

logger = logging.getLogger(__name__)


@dataclass
class EnvironmentInfo:
    """Information about the execution environment."""

    python_version: str
    """Python version string."""

    platform: str
    """Platform identifier."""

    timestamp: str
    """Experiment start timestamp."""

    random_seed: int
    """Random seed used."""

    cwd: str
    """Current working directory."""

    torch_version: Optional[str] = None
    """PyTorch version if available."""

    cuda_available: Optional[bool] = None
    """Whether CUDA is available."""

    cuda_version: Optional[str] = None
    """CUDA version if available."""

    gpu_name: Optional[str] = None
    """GPU name if available."""


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.

    Sets seeds for:
    - Python's random module
    - NumPy (if available)
    - PyTorch (if available)

    Args:
        seed: Random seed to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np
        np.random.seed(seed)
        logger.debug(f"NumPy seed set to {seed}")
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For reproducibility with CUDA
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.debug(f"PyTorch seed set to {seed}")
    except ImportError:
        pass

    logger.info(f"Random seed set to {seed}")


def get_environment_info(seed: int = 42) -> EnvironmentInfo:
    """
    Collect information about the execution environment.

    Args:
        seed: Random seed being used.

    Returns:
        EnvironmentInfo with environment details.
    """
    info = EnvironmentInfo(
        python_version=sys.version,
        platform=sys.platform,
        timestamp=datetime.now().isoformat(),
        random_seed=seed,
        cwd=os.getcwd(),
    )

    try:
        import torch
        info.torch_version = torch.__version__
        info.cuda_available = torch.cuda.is_available()
        if info.cuda_available:
            info.cuda_version = torch.version.cuda
            info.gpu_name = torch.cuda.get_device_name(0)
    except ImportError:
        pass

    return info


def save_environment_info(info: EnvironmentInfo, path: Path) -> None:
    """Save environment info to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(asdict(info), f, indent=2)
    logger.info(f"Environment info saved to {path}")


class ExperimentLogger:
    """
    Logger for experiment tracking and reproducibility.

    Logs experiment configuration, metrics, and artifacts to a structured
    output directory.
    """

    def __init__(
        self,
        experiment_name: str,
        output_dir: Path,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize experiment logger.

        Args:
            experiment_name: Name of the experiment.
            output_dir: Base output directory.
            config: Experiment configuration to log.
        """
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(output_dir) / experiment_name / self.timestamp
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.metrics: Dict[str, Any] = {}
        self.artifacts: Dict[str, Path] = {}

        # Save config if provided
        if config is not None:
            self.log_config(config)

        logger.info(f"Experiment logger initialized: {self.run_dir}")

    def log_config(self, config: Dict[str, Any]) -> None:
        """Log experiment configuration."""
        config_path = self.run_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        self.artifacts["config"] = config_path

    def log_metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        """
        Log a metric value.

        Args:
            name: Metric name.
            value: Metric value.
            step: Optional step number for time-series metrics.
        """
        if name not in self.metrics:
            self.metrics[name] = []

        entry = {"value": value, "timestamp": datetime.now().isoformat()}
        if step is not None:
            entry["step"] = step

        self.metrics[name].append(entry)
        logger.debug(f"Metric {name}: {value}")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log multiple metrics at once."""
        for name, value in metrics.items():
            self.log_metric(name, value, step)

    def log_artifact(self, name: str, path: Path) -> None:
        """Register an artifact file."""
        self.artifacts[name] = path

    def save(self) -> Path:
        """
        Save all logged data to disk.

        Returns:
            Path to the run directory.
        """
        # Save metrics
        metrics_path = self.run_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)

        # Save artifact registry
        artifacts_path = self.run_dir / "artifacts.json"
        artifacts_data = {k: str(v) for k, v in self.artifacts.items()}
        with open(artifacts_path, 'w') as f:
            json.dump(artifacts_data, f, indent=2)

        # Save summary
        summary = {
            "experiment_name": self.experiment_name,
            "timestamp": self.timestamp,
            "run_dir": str(self.run_dir),
            "num_metrics": len(self.metrics),
            "num_artifacts": len(self.artifacts),
        }
        summary_path = self.run_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Experiment data saved to {self.run_dir}")
        return self.run_dir


class Checkpoint:
    """
    Checkpoint management for saving and loading experiment state.
    """

    def __init__(self, checkpoint_dir: Path):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory for checkpoint files.
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        state: Dict[str, Any],
        name: str = "checkpoint",
        is_best: bool = False,
    ) -> Path:
        """
        Save a checkpoint.

        Args:
            state: State dictionary to save.
            name: Checkpoint name.
            is_best: If True, also save as 'best' checkpoint.

        Returns:
            Path to saved checkpoint.
        """
        try:
            import torch
            path = self.checkpoint_dir / f"{name}.pt"
            torch.save(state, path)

            if is_best:
                best_path = self.checkpoint_dir / "best.pt"
                torch.save(state, best_path)

            logger.info(f"Checkpoint saved: {path}")
            return path
        except ImportError:
            # Fallback to JSON for non-tensor state
            path = self.checkpoint_dir / f"{name}.json"
            with open(path, 'w') as f:
                json.dump(state, f, indent=2)
            return path

    def load(self, name: str = "checkpoint") -> Dict[str, Any]:
        """
        Load a checkpoint.

        Args:
            name: Checkpoint name to load.

        Returns:
            State dictionary.
        """
        try:
            import torch
            path = self.checkpoint_dir / f"{name}.pt"
            if path.exists():
                return torch.load(path)
        except ImportError:
            pass

        # Try JSON fallback
        path = self.checkpoint_dir / f"{name}.json"
        if path.exists():
            with open(path) as f:
                return json.load(f)

        raise FileNotFoundError(f"No checkpoint found: {name}")

    def list_checkpoints(self) -> list:
        """List available checkpoints."""
        checkpoints = []
        for ext in [".pt", ".json"]:
            checkpoints.extend(self.checkpoint_dir.glob(f"*{ext}"))
        return sorted(checkpoints)

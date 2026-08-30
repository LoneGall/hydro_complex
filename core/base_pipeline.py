"""
Base interface for all processing pipelines.
Any pipeline intended to be used with the GUI must inherit from this class.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


class PipelineType:
    """Enumeration of pipeline types."""
    PROCESSING = "processing"  # Data processing pipelines (e.g., PSD analysis)
    GENERATOR = "generator"    # Signal generation pipelines


class PipelineConfig:
    """Base class for pipeline configuration."""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__


class BasePipeline(ABC):
    """
    Abstract base class for all pipeline plugins.
    
    To create a new pipeline plugin:
    1. Inherit from this class.
    2. Implement all abstract methods.
    3. Register the class in the plugin manager.
    """

    @abstractmethod
    def get_name(self) -> str:
        """Returns the display name of the pipeline."""
        pass

    @abstractmethod
    def get_description(self) -> str:
        """Returns a short description of what the pipeline does."""
        pass

    @abstractmethod
    def get_type(self) -> str:
        """Returns the type of the pipeline (PROCESSING or GENERATOR)."""
        return PipelineType.PROCESSING

    @abstractmethod
    def get_config_schema(self) -> Dict[str, Any]:
        """
        Returns a schema describing the configuration parameters.
        Used by GUI to generate input fields dynamically.
        
        Format:
        {
            "param_name": {
                "type": "int" | "float" | "str" | "bool" | "list",
                "default": value,
                "label": "Human readable label",
                "min": optional_min_value,
                "max": optional_max_value,
                "options": ["list", "of", "options"] # for 'list' or 'str' dropdowns
            }
        }
        """
        pass

    @abstractmethod
    def run(self, data: Optional[pd.DataFrame], config: PipelineConfig) -> Dict[str, Any]:
        """
        Executes the pipeline processing.
        
        Args:
            data: Input data as a pandas DataFrame (None for generators).
            config: Configuration object with parameters.
            
        Returns:
            Dictionary containing results:
            {
                "success": bool,
                "message": str,
                "plots": List[Dict[str, Any]], # Plot data for GUI
                "files": List[str], # Paths to generated files if any
                "dataframes": Dict[str, pd.DataFrame] # Resulting dataframes
            }
        """
        pass

    @staticmethod
    def validate_data(data: pd.DataFrame) -> Tuple[bool, str]:
        """
        Optional: Validate input data before running.
        Returns (is_valid, error_message).
        """
        if data.empty:
            return False, "Input data is empty."
        return True, ""

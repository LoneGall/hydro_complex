"""
Plugin manager for discovering and loading pipeline plugins.
Automatically scans the 'pipelines' directory for available pipelines.
"""
import importlib
import inspect
import os
from typing import Dict, List, Type, Optional
from pathlib import Path

from core.base_pipeline import BasePipeline


class PipelineManager:
    """
    Manages discovery, registration, and instantiation of pipeline plugins.
    """

    def __init__(self, pipelines_dir: str = "pipelines"):
        self.pipelines_dir = Path(pipelines_dir)
        self._registry: Dict[str, Type[BasePipeline]] = {}
        self._discover_pipelines()

    def _discover_pipelines(self):
        """Scan the pipelines directory and register all valid pipeline classes."""
        if not self.pipelines_dir.exists():
            return

        for file_path in self.pipelines_dir.glob("*.py"):
            if file_path.name.startswith("_"):
                continue

            module_name = f"pipelines.{file_path.stem}"
            try:
                module = importlib.import_module(module_name)
                
                # Find all classes that inherit from BasePipeline
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if issubclass(obj, BasePipeline) and obj is not BasePipeline:
                        instance = obj()
                        self._registry[instance.get_name()] = obj
                        
            except Exception as e:
                print(f"Error loading pipeline from {file_path}: {e}")

    def get_available_pipelines(self, pipeline_type: Optional[str] = None) -> List[str]:
        """
        Returns a list of registered pipeline names.
        
        Args:
            pipeline_type: Filter by type ('processing' or 'generator'). 
                           If None, returns all pipelines.
        """
        if pipeline_type is None:
            return list(self._registry.keys())
        
        filtered = []
        for name, cls in self._registry.items():
            try:
                instance = cls()
                if instance.get_type() == pipeline_type:
                    filtered.append(name)
            except Exception:
                continue
        return filtered

    def get_pipeline(self, name: str) -> Optional[BasePipeline]:
        """Instantiates and returns a pipeline by name."""
        pipeline_class = self._registry.get(name)
        if pipeline_class:
            return pipeline_class()
        return None

    def get_pipeline_description(self, name: str) -> str:
        """Returns the description of a pipeline."""
        pipeline = self.get_pipeline(name)
        if pipeline:
            return pipeline.get_description()
        return ""

    def get_pipeline_config_schema(self, name: str) -> Dict:
        """Returns the configuration schema for a pipeline."""
        pipeline = self.get_pipeline(name)
        if pipeline:
            return pipeline.get_config_schema()
        return {}

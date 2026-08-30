"""
Signal Generator Pipeline Plugin.
Generates synthetic test signals with configurable parameters.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
import datetime

from core.base_pipeline import BasePipeline, PipelineConfig


class SignalGeneratorConfig(PipelineConfig):
    """Configuration for signal generation."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Sinusoid parameters
        self.amplitude = kwargs.get('amplitude', 1.0)
        self.frequency = kwargs.get('frequency', 10.0)  # Hz
        self.phase = kwargs.get('phase', 0.0)  # radians
        
        # Time parameters
        self.start_time = kwargs.get('start_time', 0.0)  # seconds
        self.end_time = kwargs.get('end_time', 1.0)  # seconds
        self.sampling_rate = kwargs.get('sampling_rate', 1000.0)  # Hz
        
        # Offset and noise
        self.offset = kwargs.get('offset', 0.0)
        self.add_noise = kwargs.get('add_noise', False)
        self.noise_level = kwargs.get('noise_level', 0.1)  # standard deviation
        
        # Output settings
        self.output_dir = kwargs.get('output_dir', 'input')
        self.filename = kwargs.get('filename', 'test_signal.csv')


class SignalGeneratorPipeline(BasePipeline):
    """Test Signal Generation Pipeline Plugin."""

    def get_name(self) -> str:
        return "Signal Generator"

    def get_description(self) -> str:
        return "Generates synthetic test signals (sinusoids with noise) for testing."

    def get_config_schema(self) -> Dict[str, Any]:
        return {
            "amplitude": {
                "type": "float",
                "default": 1.0,
                "label": "Amplitude",
                "min": 0.0,
                "max": 100.0
            },
            "frequency": {
                "type": "float",
                "default": 10.0,
                "label": "Frequency (Hz)",
                "min": 0.1,
                "max": 500.0
            },
            "phase": {
                "type": "float",
                "default": 0.0,
                "label": "Phase (radians)",
                "min": 0.0,
                "max": 6.28  # 2π
            },
            "start_time": {
                "type": "float",
                "default": 0.0,
                "label": "Start Time (s)",
                "min": 0.0
            },
            "end_time": {
                "type": "float",
                "default": 1.0,
                "label": "End Time (s)",
                "min": 0.0
            },
            "sampling_rate": {
                "type": "float",
                "default": 1000.0,
                "label": "Sampling Rate (Hz)",
                "min": 10.0,
                "max": 10000.0
            },
            "offset": {
                "type": "float",
                "default": 0.0,
                "label": "DC Offset",
                "min": -100.0,
                "max": 100.0
            },
            "add_noise": {
                "type": "bool",
                "default": False,
                "label": "Add Gaussian Noise"
            },
            "noise_level": {
                "type": "float",
                "default": 0.1,
                "label": "Noise Level (std dev)",
                "min": 0.0,
                "max": 10.0
            },
            "output_dir": {
                "type": "str",
                "default": "input",
                "label": "Output Directory"
            },
            "filename": {
                "type": "str",
                "default": "test_signal.csv",
                "label": "Output Filename"
            }
        }

    def run(self, data: pd.DataFrame, config: PipelineConfig) -> Dict[str, Any]:
        """
        Generates a test signal based on configuration.
        
        Args:
            data: Not used for generation (can be empty DataFrame)
            config: SignalGeneratorConfig object
            
        Returns:
            Dictionary with results
        """
        result = {
            "success": False,
            "message": "",
            "plots": [],
            "files": [],
            "dataframes": {},
            "metrics": {}
        }

        try:
            # Extract parameters
            amplitude = config.amplitude if hasattr(config, 'amplitude') else 1.0
            frequency = config.frequency if hasattr(config, 'frequency') else 10.0
            phase = config.phase if hasattr(config, 'phase') else 0.0
            start_time = config.start_time if hasattr(config, 'start_time') else 0.0
            end_time = config.end_time if hasattr(config, 'end_time') else 1.0
            sampling_rate = config.sampling_rate if hasattr(config, 'sampling_rate') else 1000.0
            offset = config.offset if hasattr(config, 'offset') else 0.0
            add_noise = config.add_noise if hasattr(config, 'add_noise') else False
            noise_level = config.noise_level if hasattr(config, 'noise_level') else 0.1
            output_dir = config.output_dir if hasattr(config, 'output_dir') else 'input'
            filename = config.filename if hasattr(config, 'filename') else 'test_signal.csv'

            # Generate time array
            num_samples = int((end_time - start_time) * sampling_rate) + 1
            time_array = np.linspace(start_time, end_time, num_samples)

            # Generate sinusoidal signal
            signal = amplitude * np.sin(2 * np.pi * frequency * time_array + phase) + offset

            # Add noise if requested
            if add_noise:
                noise = np.random.normal(0, noise_level, num_samples)
                signal += noise

            # Create DataFrame
            df = pd.DataFrame({
                'time': time_array,
                'signal_1': signal
            })

            # Calculate signal metrics
            metrics = self._calculate_metrics(time_array, signal, sampling_rate)
            result["metrics"] = metrics

            # Save to file
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            file_path = output_path / filename
            
            df.to_csv(file_path, index=False)
            result["files"].append(str(file_path))

            # Store dataframe
            result["dataframes"]["signal"] = df

            # Prepare plot data for GUI
            plot_data = {
                "x": time_array.tolist(),
                "y": [signal.tolist()],
                "labels": ["signal_1"],
                "title": f"Generated Signal: {frequency} Hz",
                "xlabel": "Time (s)",
                "ylabel": "Amplitude"
            }
            result["plots"].append(plot_data)

            result["success"] = True
            result["message"] = f"Signal generated successfully: {num_samples} samples, saved to {file_path}"

        except Exception as e:
            result["message"] = f"Error during signal generation: {str(e)}"

        return result

    def _calculate_metrics(self, time_array: np.ndarray, signal: np.ndarray, 
                          sampling_rate: float) -> Dict[str, Any]:
        """Calculate signal quality metrics."""
        metrics = {}
        
        # Basic statistics
        metrics["mean"] = float(np.mean(signal))
        metrics["std"] = float(np.std(signal))
        metrics["min"] = float(np.min(signal))
        metrics["max"] = float(np.max(signal))
        metrics["rms"] = float(np.sqrt(np.mean(signal**2)))
        
        # Stationarity test (simple variance comparison)
        n_segments = min(10, len(signal) // 100)
        if n_segments > 1:
            segment_length = len(signal) // n_segments
            segments = [signal[i:i+segment_length] for i in range(0, len(signal), segment_length)]
            variances = [np.var(seg) for seg in segments]
            metrics["stationarity_variance_ratio"] = float(max(variances) / min(variances)) if min(variances) > 0 else float('inf')
        
        # Ergodicity estimate (compare time average with ensemble average approximation)
        metrics["time_average_mean"] = float(np.mean(signal))
        metrics["time_average_std"] = float(np.std(signal))
        
        # Zero crossings rate
        zero_crossings = np.where(np.diff(np.signbit(signal)))[0]
        metrics["zero_crossings_rate"] = float(len(zero_crossings) / (time_array[-1] - time_array[0]))
        
        # Signal duration
        metrics["duration_seconds"] = float(time_array[-1] - time_array[0])
        metrics["num_samples"] = int(len(signal))
        metrics["sampling_rate_hz"] = float(sampling_rate)
        
        return metrics

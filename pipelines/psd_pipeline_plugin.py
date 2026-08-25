"""
PSD Pipeline plugin implementation.
Wraps the existing PSD_pipeline.py functionality into a plugin format.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Import existing pipeline functions
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from PSD_pipeline import (
    read_csv_input,
    test_channels,
    compute_psd_fft,
    write_results,
    parse_channels_input,
    process_psd_pipeline
)

from core.base_pipeline import BasePipeline, PipelineConfig


class PSDPipelineConfig(PipelineConfig):
    """Configuration specific to PSD Pipeline."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Set defaults if not provided
        self.output_dir = kwargs.get('output_dir', 'output')
        self.psd_method = kwargs.get('psd_method', 'welch')
        self.nperseg = kwargs.get('nperseg', 256)
        self.noverlap = kwargs.get('noverlap', None)
        self.window = kwargs.get('window', 'hann')
        self.test_channels = kwargs.get('test_channels', True)


class PSDPipeline(BasePipeline):
    """PSD Processing Pipeline Plugin."""

    def get_name(self) -> str:
        return "PSD Analysis"

    def get_description(self) -> str:
        return "Calculates Power Spectral Density (PSD) for pressure sensor data."

    def get_config_schema(self) -> Dict[str, Any]:
        return {
            "output_dir": {
                "type": "str",
                "default": "output",
                "label": "Output Directory"
            },
            "psd_method": {
                "type": "list",
                "default": "welch",
                "label": "PSD Method",
                "options": ["welch", "periodogram"]
            },
            "nperseg": {
                "type": "int",
                "default": 256,
                "label": "Segment Length",
                "min": 64,
                "max": 4096
            },
            "noverlap": {
                "type": "int",
                "default": 128,
                "label": "Overlap Length",
                "min": 0,
                "max": 256
            },
            "window": {
                "type": "list",
                "default": "hann",
                "label": "Window Function",
                "options": ["hann", "hamming", "blackman", "rectangular"]
            },
            "test_channels": {
                "type": "bool",
                "default": True,
                "label": "Test Channels Before Processing"
            }
        }

    def run(self, data: pd.DataFrame, config: PipelineConfig) -> Dict[str, Any]:
        """
        Executes the PSD pipeline.
        
        Args:
            data: Input DataFrame with columns ['time', 'p1', 'p2', ...]
            config: PSDPipelineConfig object
            
        Returns:
            Dictionary with results
        """
        result = {
            "success": False,
            "message": "",
            "plots": [],
            "files": [],
            "dataframes": {}
        }

        try:
            # Validate data
            is_valid, error_msg = self.validate_data(data)
            if not is_valid:
                result["message"] = error_msg
                return result

            # Extract time and channels
            time_col = data.columns[0]
            time_array = data[time_col].values
            fs = self._estimate_sampling_rate(time_array)

            channels = data.columns[1:].tolist()
            
            # Test channels if requested
            if hasattr(config, 'test_channels') and config.test_channels:
                test_result = test_channels(data, time_col)
                if not test_result:
                    result["message"] = "Channel testing failed."
                    return result

            # Calculate PSD for each channel
            psd_results = {}
            all_freqs = None
            all_psd_data = []

            for channel in channels:
                signal = data[channel].values
                
                freqs, psd = compute_psd_fft(
                    signal=signal,
                    fs=fs,
                    method=config.psd_method if hasattr(config, 'psd_method') else 'welch',
                    nperseg=config.nperseg if hasattr(config, 'nperseg') else 256,
                    noverlap=config.noverlap if hasattr(config, 'noverlap') else None,
                    window=config.window if hasattr(config, 'window') else 'hann'
                )
                
                psd_results[channel] = psd
                all_freqs = freqs
                all_psd_data.append(psd)

            # Create output directory using the logic from process_psd_pipeline
            import datetime
            base_name = "gui_input"
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            root_output_dir = Path(config.output_dir if hasattr(config, 'output_dir') else 'output')
            root_output_dir.mkdir(parents=True, exist_ok=True)
            output_dir_name = f"output_{base_name}_{timestamp}"
            output_dir = root_output_dir / output_dir_name
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save results
            filename_base = "input_data"
            saved_files = write_results(
                freqs=all_freqs,
                psd_data=np.array(all_psd_data),
                channel_names=channels,
                output_dir=output_dir,
                filename_base=filename_base
            )
            result["files"].extend(saved_files)

            # Store dataframe
            df_result = pd.DataFrame({'frequency': all_freqs})
            for i, ch in enumerate(channels):
                df_result[ch] = all_psd_data[i]
            result["dataframes"]["psd"] = df_result

            # Prepare plot data for GUI
            plot_data = {
                "x": all_freqs.tolist(),
                "y": [arr.tolist() for arr in all_psd_data],
                "labels": channels,
                "title": "Power Spectral Density",
                "xlabel": "Frequency (Hz)",
                "ylabel": "PSD"
            }
            result["plots"].append(plot_data)

            result["success"] = True
            result["message"] = f"PSD analysis completed for {len(channels)} channels."

        except Exception as e:
            result["message"] = f"Error during processing: {str(e)}"

        return result

    def _estimate_sampling_rate(self, time_array: np.ndarray) -> float:
        """Estimate sampling rate from time array."""
        dt = np.mean(np.diff(time_array))
        return 1.0 / dt if dt > 0 else 1000.0

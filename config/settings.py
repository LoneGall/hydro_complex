"""
Конфигурация для PSD пайплайна.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class PSDConfig:
    """Конфигурация обработки спектральной плотности мощности."""
    
    # Пути
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    input_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "input")
    output_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "output")
    
    # Файлы
    default_input_file: str = "Pres_r2_LONGER.csv"
    
    # Параметры обработки
    sampling_frequency: float = 1000.0  # Гц (примерное значение, должно переопределяться)
    window_size: int = 256
    overlap: float = 0.5
    
    # Каналы для обработки (None означает все каналы)
    channels_to_process: Optional[List[str]] = None
    
    # Параметры валидации
    min_signal_length: int = 10
    max_nan_percentage: float = 0.1  # 10%
    
    def get_input_path(self, filename: Optional[str] = None) -> Path:
        """Возвращает полный путь к входному файлу."""
        if filename is None:
            filename = self.default_input_file
        return self.input_dir / filename
    
    def ensure_output_dir(self) -> None:
        """Создает директорию вывода, если она не существует."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

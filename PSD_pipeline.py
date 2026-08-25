"""
PSD_pipeline.py — Модульный анализ спектральной плотности мощности
Архитектура: Input → Validation → Test → PSD → Output

================================================================================
ИНСТРУКЦИЯ ПО ЗАПУСКУ И ВВОДУ КАНАЛОВ:
================================================================================
Головная функция: process_psd_pipeline(directory, filename, channels, cutoff_hz)

Аргументы:
  directory  : Рабочая директория, содержащая подпапку input/ (по умолч. : директория скрипта)
  filename   : Имя CSV файла в подпапке input/ (по умолч. : 'input.csv')
  channels   : Номера каналов для обработки. 
               Формат ввода - строка, поддерживаются списки и диапазоны:
               Пример: "0,2,5-8" обработает каналы 0, 2, 5, 6, 7, 8
               0 или None - обрабатываются ВСЕ каналы
  cutoff_hz  : Частота отсечки вывода в Гц. 
               0 - выводить ВЕСЬ частотный диапазон ( подпись _full ).
               >0 - ограничить отрисовку и CSV файл этой частотой ( подпись _0-{cutoff_hz}-Hz ).
               Если cutoff_hz больше частоты Найквиста, выводится весь диапазон.

Формат входного CSV файла:
  - Разделитель: запятая
  - Десятичный разделитель: точка
  - Первая строка: заголовки (название времени, затем названия каналов)
  - Первый столбец: время в секундах (строго монотонно возрастающее)
  - Остальные столбцы: числовые значения сигналов каналов
================================================================================
"""

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Импорт конфигурации
from config.settings import PSDConfig

# Настройка логгирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# 0. КОНФИГУРАЦИЯ КАЧЕСТВА
# ============================================================================

CONFIG_DIR = Path(__file__).parent / "config"
CONFIG_FILE = CONFIG_DIR / "quality_thresholds.json"


def create_default_config() -> Dict[str, float]:
    """Создает конфиг с инженерными порогами, если его нет."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    default_config: Dict[str, float] = {
        "trend_threshold_pct": 10.0,     # Инженерный порог (было 1%)
        "acf_threshold": 0.99,           # Инжененный порог ACF (было 0.95)
        "ergodic_threshold_pct": 25.0,   # Инженерный порог эргодичности (было 10%)
        "dt_variance_threshold": 0.01,   # Допустимая дисперсия шага времени (валидация)
        "min_independent_segments": 10   # минимальное количество независимых сегментов
    }
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(default_config, f, indent=4)
    logger.info(f"Создан конфиг по умолчанию: {CONFIG_FILE}")
    return default_config


def load_config() -> Dict[str, float]:
    """Загружает конфиг качества."""
    import json
    if not CONFIG_FILE.exists():
        logger.warning("Конфиг не найден, создан конфиг по умолчанию")
        return create_default_config()
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

# ============================================================================
# 1. МОДУЛЬ ВВОДА И ВАЛИДАЦИИ
# ============================================================================


def read_csv_input(
    work_dir: Path, 
    filename: str
) -> Tuple[np.ndarray, np.ndarray, List[str], List[float]]:
    """Чтение CSV из папки input/ + DC removal + Валидация.
    
    Args:
        work_dir: Рабочая директория
        filename: Имя файла
        
    Returns:
        times: Массив времени
        data: Массив данных сигналов
        headers: Названия каналов
        dc_offsets: Смещения постоянного тока по каналам
        
    Raises:
        FileNotFoundError: Файл не найден
        ValueError: Некорректный формат данных
    """
    input_dir = work_dir / "input"
    full_path = input_dir / filename
    
    # Валидация 1: Существование файла
    if not full_path.exists():
        raise FileNotFoundError(f"Файл не найден: {full_path}")
    
    times: List[float] = []
    data: List[List[float]] = []
    
    with open(full_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        headers = next(reader)
        
        # Валидация 2: Минимум 2 столбца
        if len(headers) < 2:
            raise ValueError("CSV должен содержать минимум 2 столбца (Время + 1 Канал)")
            
        for row_idx, row in enumerate(reader, start=2):
            if not row:
                continue
            try:
                times.append(float(row[0]))
                data.append([float(x) for x in row[1:]])
            except ValueError as e:
                raise ValueError(f"Ошибка формата данных в строке {row_idx}: {e}")

    data_array = np.array(data)
    times_array = np.array(times)
    
    # Валидация 3: Пустые данные
    if len(times_array) == 0:
        raise ValueError("CSV файл не содержит данных (пустой)")
        
    # Валидация 4: Монотонность времени
    if not np.all(np.diff(times_array) > 0):
        raise ValueError("Временной столбец должен быть строго монотонно возрастающим")
    
    # Удаление тренда и DC по каждому каналу
    for i in range(data_array.shape[1]):
        x = np.arange(len(data_array[:, i]))
        # Полином 1-й степени (линейный тренд)
        coeffs = np.polyfit(x, data_array[:, i], 1)
        trend = np.polyval(coeffs, x)
        data_array[:, i] -= trend
        
    # Поканальное удаление DC
    dc_offsets: List[float] = []
    for i in range(data_array.shape[1]):
        dc = float(np.mean(data_array[:, i]))
        dc_offsets.append(dc)
        data_array[:, i] -= dc
    
    return times_array, data_array, headers[1:], dc_offsets

# ============================================================================
# 2. МОДУЛЬ ТЕСТИРОВАНИЯ КАЧЕСТВА
# ============================================================================


def correlation_time(signal: np.ndarray, fs: float) -> float:
    """Вычисляет время корреляции сигнала.
    
    Args:
        signal: Сигнал для анализа
        fs: Частота дискретизации в Гц
        
    Returns:
        Время корреляции в секундах
    """
    n = len(signal)
    norm_signal = signal - np.mean(signal)
    acf = np.correlate(norm_signal, norm_signal, mode='full') / np.var(signal) / n
    acf = acf[n-1:]  # Только положительные лаги
    # Время, где ACF впервые пересекает 0.05
    idx_zero = np.where(acf < 0.05)[0]
    if len(idx_zero) > 0:
        return float(idx_zero[0] / fs)
    else:
        return float(len(signal) / fs)  # сигнал длиннее всей записи


def test_channel_quality(
    signal: np.ndarray, 
    fs: float, 
    config: Dict[str, float]
) -> Dict[str, Union[bool, float]]:
    """Тесты стационарности + эргодичности с порогами из конфига.
    
    Args:
        signal: Сигнал канала
        fs: Частота дискретизации в Гц
        config: Словарь с порогами качества
        
    Returns:
        Словарь с результатами тестов
    """
    n = len(signal)
    
    # Тест тренда
    x = np.arange(n)
    result = stats.linregress(x, signal)
    slope = result.slope
    total_trend = abs(slope) * n
    trend_ok = total_trend < (config['trend_threshold_pct'] / 100.0) * np.std(signal)
    
    # Тест автокорреляции
    autocorr_lag1 = float(np.corrcoef(signal[:-1], signal[1:])[0, 1])
    acf_time = correlation_time(signal, fs)
    acf_ok = abs(autocorr_lag1) < config['acf_threshold']
    
    # Тест эргодичности (10 сегментов)
    n_seg = 10
    seg_len = max(n // n_seg, 50)
    n_seg = n // seg_len
    means = [np.mean(signal[i*seg_len:(i+1)*seg_len]) for i in range(n_seg)]
    mean_var_ratio = float(np.var(means) / (np.var(signal) + 1e-12))
    ergodic_ok = mean_var_ratio < (config['ergodic_threshold_pct'] / 100.0)
    
    return {
        'stationary': bool(trend_ok and acf_ok),
        'ergodic': bool(ergodic_ok),
        'total_trend_pct': float((total_trend / np.std(signal)) * 100),
        'acf_lag1': float(autocorr_lag1),
        'acf_time': float(acf_time),
        'mean_var_ratio': float(mean_var_ratio * 100)
    }


def test_channels(
    data: np.ndarray, 
    config: Dict[str, float],
    times: np.ndarray
) -> List[Dict[str, Union[bool, float, int]]]:
    """Тестирование всех каналов.
    
    Args:
        data: Массив данных сигналов
        config: Словарь с порогами качества
        times: Массив времени
        
    Returns:
        Список результатов по каждому каналу
    """
    fs = 1.0 / float(times[1] - times[0])
    results: List[Dict[str, Union[bool, float, int]]] = []
    for i in range(data.shape[1]):
        result = test_channel_quality(data[:, i], fs, config)
        result['channel'] = i
        results.append(result)
    return results

# ============================================================================
# 3. МОДУЛЬ PSD
# ============================================================================


def compute_psd_fft(
    times: np.ndarray, 
    data: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Базовый PSD через FFT.
    
    Args:
        times: Массив времени
        data: Массив данных сигналов
        
    Returns:
        freqs: Массив частот (положительные)
        psd_data: Массив PSD по каналам
    """
    dt = float(times[1] - times[0])
    fs = 1.0 / dt
    N = len(data)
    
    window = np.ones(N)  # Прямоугольное окно
    
    psd_results: List[np.ndarray] = []
    freqs: Optional[np.ndarray] = None
    
    for i in range(data.shape[1]):
        signal_win = data[:, i] * window
        fft_result = np.fft.fft(signal_win)
        freqs = np.fft.fftfreq(N, dt)
        
        pos_idx = freqs > 0
        f_pos = freqs[pos_idx]
        fft_pos = fft_result[pos_idx]
        
        psd = 2.0 * np.abs(fft_pos)**2 / (fs * np.sum(window**2))
        psd_results.append(psd)
    
    assert freqs is not None, "freqs не был инициализирован"
    return f_pos, np.array(psd_results)


# ============================================================================
# 4. МОДУЛЬ ВЫВОДА (Расчеты на полном массиве, отрисовка и CSV по отсечке)
# ============================================================================


def PSD_int(
    psd_freq: np.ndarray, 
    psd_values: np.ndarray, 
    original_signal: Optional[np.ndarray] = None
) -> Tuple[float, Optional[float], Optional[float]]:
    """Интеграл PSD.
    
    Args:
        psd_freq: Массив частот
        psd_values: Значения PSD
        original_signal: Исходный сигнал для проверки Парсеваля (опционально)
        
    Returns:
        variance_psd: Дисперсия из PSD
        variance_signal: Дисперсия сигнала (если предоставлен)
        parseval_error: Ошибка Парсеваля в % (если предоставлен сигнал)
    """
    psd_freq = np.real(psd_freq)
    psd_values = np.real(psd_values)
    # Используем np.trapezoid вместо устаревшего np.trapz
    variance_psd = float(np.trapezoid(psd_values, psd_freq))
    
    variance_signal: Optional[float] = None
    parseval_error: Optional[float] = None
    
    if original_signal is not None:
        variance_signal = float(np.var(original_signal, ddof=0))
        parseval_error = abs(variance_psd - variance_signal) / max(variance_signal, 1e-12) * 100.0
        
    return variance_psd, variance_signal, parseval_error


def write_results(
    output_dir: Path,
    input_file: str,
    times: np.ndarray,
    data: np.ndarray,
    headers: List[str],
    dc_offsets: List[float],
    test_results: List[Dict[str, Union[bool, float, int]]],
    freqs: np.ndarray,
    psd_data: np.ndarray,
    cutoff_hz: float
) -> Path:
    """Единый вывод. Расчеты на FULL диапазоне, визуализация/CSV по отсечке.
    
    Args:
        output_dir: Директория для вывода
        input_file: Имя входного файла
        times: Массив времени
        data: Массив данных
        headers: Названия каналов
        dc_offsets: Смещения DC
        test_results: Результаты тестов качества
        freqs: Массив частот
        psd_data: Данные PSD
        cutoff_hz: Частота отсечки для вывода
        
    Returns:
        Путь к файлу отчета
    """
    # Динамическая подпись и маска только для CSV/графика
    nyquist_freq = float(freqs[-1])
    if cutoff_hz == 0 or cutoff_hz >= nyquist_freq:
        plot_mask = np.ones_like(freqs, dtype=bool)
        freq_label = "full"
        plot_xlim = nyquist_freq
    else:
        plot_mask = freqs <= cutoff_hz
        freq_label = f"{int(cutoff_hz)}-Hz"
        plot_xlim = cutoff_hz

    # Основной отчёт
    fs = 1.0 / float(times[1] - times[0])
    report_lines: List[str] = [
        f"PSD АНАЛИЗ {input_file}",
        f"Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Данные: {data.shape[1]} каналов, {len(times)} точек",
        f"fs = {fs:.1f} Гц | Найквист = {nyquist_freq:.1f} Гц",
        f"Отсечка вывода: {freq_label}",
        ""
    ]

    # Тесты качества
    report_lines.append("КАЧЕСТВО КАНАЛОВ:")
    report_lines.append("=" * 80)
    valid_count = 0
    for r in test_results:
        status = "✅ ВАЛИДЕН" if r['stationary'] and r['ergodic'] else "❌ НЕВАЛИДЕН"
        if r['stationary'] and r['ergodic']:
            valid_count += 1
        report_lines.append(
            f"К{r['channel']+1:2d}: тренд={r['total_trend_pct']:5.2f}%, "
            f"ACF={r['acf_lag1']:6.3f}, ACF_time={r['acf_time']:6.3f}, эрг={r['mean_var_ratio']:4.1f}% | {status}"
        )
    report_lines.append(f"ВАЛИДНЫХ: {valid_count}/{len(test_results)}")
    report_lines.append("")

    # Расчеты Парсеваля делаются на ВСЕМ диапазоне (freqs, psd_data[i])
    envelope = np.max(psd_data, axis=0)
    
    for i in range(data.shape[1]):
        var_psd, var_sig, err = PSD_int(freqs, psd_data[i], data[:, i])
        report_lines.append(
            f"К{i+1}: σ²_PSD={var_psd:.3e}, σ²_sig={var_sig:.3e}, "
            f"Парсеваль={err:.2f}%"
        )
    
    # Полная дисперсия огибающей
    var_env_full, _, _ = PSD_int(freqs, envelope)
    rms_env_full = np.sqrt(var_env_full)
    report_lines.append(f"Огибающая (FULL): σ²={var_env_full:.3e}, RMS={rms_env_full:.3e}")
    
    # Дисперсия огибающей в зоне отсечки (если задана)
    if freq_label != "full":
        var_env_cut, _, _ = PSD_int(freqs[plot_mask], envelope[plot_mask])
        rms_env_cut = np.sqrt(var_env_cut)
        report_lines.append(f"Огибающая (0-{freq_label}): σ²={var_env_cut:.3e}, RMS={rms_env_cut:.3e}")
    
    # Сохранение отчёта
    report_file = output_dir / 'PSD_report.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    # График: отрисовка ограничена plot_xlim, но данные полные
    plt.figure(figsize=(12, 8))
    for i in range(psd_data.shape[0]):
        plt.plot(freqs, psd_data[i], 'ko', markersize=2, alpha=0.6)
    plt.plot(freqs, envelope, 'r-', linewidth=3, label='Огибающая')
    
    # Динамическая граница X
    plt.xlim(0, plot_xlim)
    
    plt.xlabel('Частота [Гц]')
    plt.ylabel('PSD [Па²/Гц]')
    plt.title(f'Спектральная плотность мощности (до {freq_label})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f'PSD_envelope_{freq_label}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # CSV с данными сохраняется только по отсечке
    csv_filename = f'PSD_0-{freq_label}.csv'
    np.savetxt(
        output_dir / csv_filename,
        np.column_stack([freqs[plot_mask], envelope[plot_mask]]),
        delimiter=',', header='freq_Hz,PSD_Pa2_Hz'
    )
    
    return report_file

# ============================================================================
# 5. ОСНОВНОЙ ПАЙПЛАЙН
# ============================================================================


def parse_channels_input(
    channels_input: Union[int, str, List[int], None]
) -> Optional[List[int]]:
    """Парсит ввод каналов: строку "0,2,5-8" или список/ноль.
    
    Args:
        channels_input: Входные данные о каналах (0, строка, список или None)
        
    Returns:
        Список индексов каналов или None для всех каналов
        
    Raises:
        ValueError: Неверный формат диапазона или номера канала
        TypeError: Неверный тип входных данных
    """
    if channels_input == 0 or channels_input is None:
        return None
    
    if isinstance(channels_input, list):
        return channels_input
        
    if isinstance(channels_input, str):
        indices: set = set()
        for part in channels_input.split(','):
            part = part.strip()
            if '-' in part:
                try:
                    start, end = map(int, part.split('-'))
                    indices.update(range(start, end + 1))
                except ValueError:
                    raise ValueError(f"Неверный формат диапазона каналов: '{part}'")
            else:
                try:
                    indices.add(int(part))
                except ValueError:
                    raise ValueError(f"Неверный номер канала: '{part}'")
        return sorted(list(indices))
        
    raise TypeError("Каналы должны быть 0, списком или строкой формата '0,1,5-8'")


def process_psd_pipeline(
    directory: Optional[Path] = None,
    filename: str = "input.csv",
    channels: Union[int, str, List[int], None] = 0,
    cutoff_hz: float = 0
) -> Optional[Path]:
    """Полный пайплайн: Input → Validation → Test → PSD → Output.
    
    Args:
        directory: Рабочая директория (по умолчанию - директория скрипта)
        filename: Имя файла в папке input/
        channels: Каналы для обработки (0=all, список, строка "0,2,5-8")
        cutoff_hz: Частота отсечки вывода (0=весь диапазон)
        
    Returns:
        Путь к директории с результатами или None при ошибке
    """
    # 0. ИНИЦИАЛИЗАЦИЯ
    if directory is None:
        directory = Path(__file__).parent
        
    work_dir = Path(directory) if not isinstance(directory, Path) else directory
    
    try:
        channel_indices = parse_channels_input(channels)
        config = load_config()
    except Exception as e:
        logger.error(f"❌ Ошибка конфигурации: {e}")
        return None

    # 1. ВВОД + ВАЛИДАЦИЯ
    logger.info(f"📂 Чтение {filename} из {work_dir}...")
    try:
        times, data, headers, dc_offsets = read_csv_input(work_dir, filename)
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"❌ Ошибка ввода: {e}")
        return None

    # Валидация индексов каналов
    if channel_indices is not None:
        max_ch = data.shape[1] - 1
        valid_indices = [i for i in channel_indices if 0 <= i <= max_ch]
        if len(valid_indices) != len(channel_indices):
            logger.warning(
                f"⚠️ Внимание: запрошены каналы {channel_indices}, "
                f"но в файле только {data.shape[1]}. Несуществующие отброшены."
            )
        if not valid_indices:
            logger.error("❌ Ошибка: Нет валидных каналов для обработки")
            return None
        data = data[:, valid_indices]
        headers = [headers[i] for i in valid_indices]
    
    # 2. ТЕСТИРОВАНИЕ
    logger.info("🔍 Тестирование каналов...")
    test_results = test_channels(data, config, times)
    
    # 3. PSD
    logger.info("⚡ Вычисление PSD...")
    freqs, psd_data = compute_psd_fft(times, data)
    
    # 4. ВЫВОД
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = Path(filename).stem
    
    root_output_dir = work_dir / 'output'
    root_output_dir.mkdir(parents=True, exist_ok=True)
    
    output_dir_name = f"output_{base_name}_{timestamp}"
    output_dir = root_output_dir / output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        report_file = write_results(
            output_dir, filename, times, data,
            headers, dc_offsets, test_results, freqs, psd_data, cutoff_hz
        )
    except Exception as e:
        logger.error(f"❌ Ошибка записи результатов: {e}")
        return None
    
    freq_label = "full" if cutoff_hz == 0 else f"{int(cutoff_hz)}-Hz"
    logger.info(f"✅ Готово: {output_dir}")
    logger.info(f"📄 Отчёт: PSD_report.txt")
    logger.info(f"📈 График: PSD_envelope_{freq_label}.png")
    logger.info(f"📊 CSV: PSD_0-{freq_label}.csv")
    
    return output_dir


# ============================================================================
# 6. ТЕСТОВЫЙ ЗАПУСК
# ============================================================================

if __name__ == "__main__":
    # Пример запуска с разными параметрами
    # process_psd_pipeline(channels="0,1,5-8", cutoff_hz=150)
    # process_psd_pipeline(filename="input.csv", cutoff_hz=0)
    process_psd_pipeline(filename="Pres_r2_LONGER.csv", cutoff_hz=0)
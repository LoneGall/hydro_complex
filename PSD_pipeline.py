"""
PSD_pipeline.py — Модульный анализ спектральной плотности мощности
Архитектура: Input → Test → PSD → Output
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
from datetime import datetime

# ============================================================================
# 1. МОДУЛЬ ВВОДА
# ============================================================================

def read_csv_input(filename):
    """Чтение CSV из папки input/ + DC removal"""
    input_dir = 'input/'
    full_path = os.path.join(input_dir, filename)
    
    times = []
    data = []
    dc_offsets = []
    
    with open(full_path, 'r') as file:
        reader = csv.reader(file)
        headers = next(reader)
        for row in reader:
            if row:
                times.append(float(row[0]))
                data.append([float(x) for x in row[1:]])
    
    data = np.array(data)
    
    # Поканальное удаление DC
    for i in range(data.shape[1]):
        dc = np.mean(data[:, i])
        dc_offsets.append(dc)
        data[:, i] -= dc
    
    return np.array(times), data, headers[1:], dc_offsets

# ============================================================================
# 2. МОДУЛЬ ТЕСТИРОВАНИЯ КАЧЕСТВА
# ============================================================================

def test_channel_quality(signal):
    """Тесты стационарности + эргодичности"""
    n = len(signal)
    
    # Тест тренда
    x = np.arange(n)
    result = stats.linregress(x, signal)
    slope = result.slope
    total_trend = abs(slope) * n
    trend_ok = total_trend < 0.01 * np.std(signal)
    
    # Тест автокорреляции
    autocorr_lag1 = np.corrcoef(signal[:-1], signal[1:])[0,1]
    acf_ok = abs(autocorr_lag1) < 0.95
    
    # Тест эргодичности (10 сегментов)
    n_seg = 10
    seg_len = max(n // n_seg, 50)
    n_seg = n // seg_len
    means = [np.mean(signal[i*seg_len:(i+1)*seg_len]) for i in range(n_seg)]
    mean_var_ratio = np.var(means) / (np.var(signal) + 1e-12)
    ergodic_ok = mean_var_ratio < 0.10
    
    return {
        'stationary': trend_ok and acf_ok,
        'ergodic': ergodic_ok,
        'total_trend_pct': (total_trend / np.std(signal)) * 100,
        'acf_lag1': autocorr_lag1,
        'mean_var_ratio': mean_var_ratio * 100
    }

def test_channels(data):
    """Тестирование всех каналов"""
    results = []
    for i in range(data.shape[1]):
        result = test_channel_quality(data[:, i])
        result['channel'] = i
        results.append(result)
    return results

# ============================================================================
# 3. МОДУЛЬ PSD (универсальный, готов под Bartlett/Welch)
# ============================================================================

def compute_psd_fft(times, data):
    """Базовый PSD через FFT (для будущих Bartlett/Welch)"""
    dt = times[1] - times[0]
    fs = 1 / dt
    N = len(data)
    
    # Прямоугольное окно (фиксированное)
    window = np.ones(N)
    
    psd_results = []
    freqs = None
    
    for i in range(data.shape[1]):
        signal_win = data[:, i] * window
        fft_result = np.fft.fft(signal_win)
        freqs = np.fft.fftfreq(N, dt)
        
        # Только положительные частоты
        pos_idx = freqs > 0
        f_pos = freqs[pos_idx]
        fft_pos = fft_result[pos_idx]
        
        # Универсальная нормировка PSD
        psd = 2.0 * np.abs(fft_pos)**2 / (fs * np.sum(window**2))
        psd_results.append(psd)
    
    return f_pos, np.array(psd_results)

# ============================================================================
# 4. МОДУЛЬ ВЫВОДА (единый файл + график)
# ============================================================================

def PSD_int(psd_freq, psd_values, original_signal=None):
    """Интеграл PSD с проверкой Парсеваля"""
    psd_freq = np.real(psd_freq)
    psd_values = np.real(psd_values)
    variance_psd = np.trapz(psd_values, psd_freq)
    
    if original_signal is not None:
        variance_signal = np.var(original_signal, ddof=0)
        parseval_error = abs(variance_psd - variance_signal) / max(variance_signal, 1e-12) * 100
        return variance_psd, variance_signal, parseval_error
    return variance_psd, None, None

def write_results(output_dir, input_file, times, data, headers, dc_offsets, 
                 test_results, freqs, psd_data):
    """ЕДИНЫЙ вывод всех результатов"""
    
    # Основной отчёт
    report_lines = [f"PSD АНАЛИЗ {input_file}",
                   f"Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                   f"Данные: {data.shape[1]} каналов, {len(times)} точек",
                   f"fs = {1/(times[1]-times[0]):.1f} Гц",
                   ""]

    # Тесты качества
    report_lines.append("КАЧЕСТВО КАНАЛОВ:")
    report_lines.append("="*80)
    valid_count = 0
    for r in test_results:
        status = "✅ ВАЛИДЕН" if r['stationary'] and r['ergodic'] else "❌ НЕВАЛИДЕН"
        if r['stationary'] and r['ergodic']:
            valid_count += 1
        report_lines.append(
            f"К{r['channel']+1:2d}: тренд={r['total_trend_pct']:5.2f}%, "
            f"ACF={r['acf_lag1']:6.3f}, эрг={r['mean_var_ratio']:4.1f}% | {status}"
        )
    report_lines.append(f"ВАЛИДНЫХ: {valid_count}/{len(test_results)}")
    report_lines.append("")

    # PSD результаты
    envelope = np.max(psd_data, axis=0)
    mask_30hz = freqs <= 30
    
    for i in range(data.shape[1]):
        var_psd, var_sig, err = PSD_int(freqs, psd_data[i], data[:,i])
        report_lines.append(f"К{i+1}: σ²_PSD={var_psd:.3e}, σ²_sig={var_sig:.3e}, "
                          f"Парсеваль={err:.2f}%")
    
    var_30hz, _, _ = PSD_int(freqs[mask_30hz], envelope[mask_30hz])
    rms_30hz = np.sqrt(var_30hz)
    report_lines.append(f"Огибающая 0-30Гц: σ²={var_30hz:.3e}, RMS={rms_30hz:.3e}")
    
    # Сохранение отчёта
    report_file = f'{output_dir}PSD_report.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    # График
    plt.figure(figsize=(12, 8))
    for i in range(psd_data.shape[0]):
        plt.plot(freqs, psd_data[i], 'ko', markersize=2, alpha=0.6)
    plt.plot(freqs, envelope, 'r-', linewidth=3, label='Огибающая')
    plt.xlim(0, 30)
    plt.xlabel('Частота [Гц]')
    plt.ylabel('PSD [Па²/Гц]')
    plt.title('Спектральная плотность мощности')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(f'{output_dir}PSD_envelope.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # CSV с данными
    np.savetxt(f'{output_dir}PSD_0-30Hz.csv',
              np.column_stack([freqs[mask_30hz], envelope[mask_30hz]]),
              delimiter=',', header='freq_Gts,PSD_Pa2_Gts')
    
    return report_file

# ============================================================================
# 5. ОСНОВНОЙ ПАЙПЛАЙН
# ============================================================================

def process_psd_pipeline(input_filename, channel_indices=None):
    """
    Полный пайплайн: Input → Test → PSD → Output
    
    Args:
        input_filename: str, имя CSV файла в input/
        channel_indices: list[int], каналы для анализа (None=все)
    
    Returns:
        output_dir: str, папка с результатами
    """
    
    # 1. ВВОД
    print(f"📂 Чтение {input_filename}...")
    times, data, headers, dc_offsets = read_csv_input(input_filename)
    
    # 2. ВЫБОР КАНАЛОВ
    if channel_indices is not None:
        data = data[:, channel_indices]
        headers = [headers[i] for i in channel_indices]
    
    # 3. ТЕСТИРОВАНИЕ
    print("🔍 Тестирование каналов...")
    test_results = test_channels(data)
    
    # 4. PSD
    print("⚡ Вычисление PSD...")
    freqs, psd_data = compute_psd_fft(times, data)[:2]
    
    # 5. ВЫВОД
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(os.path.basename(input_filename))[0]
    
    # Корневая папка для всех результатов
    root_output_dir = 'output'
    os.makedirs(root_output_dir, exist_ok=True)
    
    # Папка конкретного запуска
    output_dir_name = f"output_{base_name}_{timestamp}"
    output_dir = os.path.join(root_output_dir, output_dir_name)
    os.makedirs(output_dir, exist_ok=True)
    
    report_file = write_results(
        output_dir, input_filename, times, data,
        headers, dc_offsets, test_results, freqs, psd_data
    )
    
    print(f"✅ Готово: {output_dir}")
    print(f"📄 Отчёт: {os.path.basename(report_file)}")
    print(f"📈 График: PSD_envelope.png")
    print(f"📊 CSV: PSD_0-30Hz.csv")
    
    return output_dir

# ============================================================================
# 6. ТЕСТОВЫЙ ЗАПУСК
# ============================================================================

if __name__ == "__main__":
    # Пример запуска
    #output_dir = process_psd_pipeline('Pres_r1.csv', channel_indices=[0,1,2,3])
    output_dir = process_psd_pipeline('Pres_r1_LONGER.csv',)

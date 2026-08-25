import os
import logging
from pathlib import Path
from typing import Optional, List, Tuple, Any

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# Подключаем наш пайплайн
import PSD_pipeline

# Настройка логгирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PSDApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Вычисление PSD сигналов")
        self.root.geometry("1200x700")

        # Переменные для хранения состояния
        self.work_dir: str = ""
        self.current_file: Optional[str] = None
        self.times: Optional[NDArray[np.float64]] = None          # массив времени
        self.data: Optional[NDArray[np.float64]] = None           # numpy массив данных
        self.headers: List[str] = []          # список названий каналов
        self.current_channel: Optional[str] = None
        self.psd_method_var: tk.StringVar = tk.StringVar(value="whole")  # whole, bartlett, welch
        self.segment_length_var: tk.IntVar = tk.IntVar(value=256)
        self.overlap_var: tk.DoubleVar = tk.DoubleVar(value=0.5)
        self.cutoff_hz_var: tk.DoubleVar = tk.DoubleVar(value=0.0)

        # Результаты вычислений PSD
        self.freqs: Optional[NDArray[np.float64]] = None
        self.psd_data: Optional[NDArray[np.float64]] = None       # numpy массив PSD всех каналов
        
        # Последние вычисленные значения для сохранения
        self.last_psd_freqs: Optional[NDArray[np.float64]] = None
        self.last_psd_values: Optional[NDArray[np.float64]] = None
        
        # Объекты matplotlib для оптимизации отрисовки
        self.time_fig: Figure
        self.time_ax: Axes
        self.time_canvas: FigureCanvasTkAgg
        self.psd_fig: Figure
        self.psd_ax: Axes
        self.psd_canvas: FigureCanvasTkAgg
        
        # Ссылки на линии графиков для обновления без перерисовки
        self.time_line: Optional[Any] = None
        self.psd_line: Optional[Any] = None
        self.psd_envelope_line: Optional[Any] = None
        self.psd_bg_lines: List[Any] = []

        # Создание интерфейса
        self.create_widgets()

    def create_widgets(self):
        # Главный контейнер с изменяемыми панелями
        main_panel = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_panel.pack(fill=tk.BOTH, expand=True)

        # Левая панель (1/3 ширины)
        left_frame = ttk.Frame(main_panel, width=400, relief=tk.SUNKEN)
        main_panel.add(left_frame, weight=1)

        # Правая панель (2/3 ширины)
        right_frame = ttk.Frame(main_panel, width=800, relief=tk.SUNKEN)
        main_panel.add(right_frame, weight=2)

        # ---------- Левая панель ----------
        # Выбор рабочей директории
        dir_frame = ttk.LabelFrame(left_frame, text="Рабочая директория", padding=5)
        dir_frame.pack(fill=tk.X, padx=5, pady=5)

        self.dir_label = ttk.Label(dir_frame, text="Не выбрана", relief=tk.SUNKEN, anchor=tk.W)
        self.dir_label.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(dir_frame, text="Обзор...", command=self.browse_directory).pack(pady=5)

        # Список файлов в input
        file_frame = ttk.LabelFrame(left_frame, text="Файлы в папке input", padding=5)
        file_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.file_listbox = tk.Listbox(file_frame, height=6)
        self.file_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.file_listbox.bind('<<ListboxSelect>>', self.on_file_select)

        # Список каналов
        channel_frame = ttk.LabelFrame(left_frame, text="Каналы", padding=5)
        channel_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.channel_listbox = tk.Listbox(channel_frame, height=8, selectmode=tk.SINGLE)
        self.channel_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.channel_listbox.bind('<<ListboxSelect>>', self.on_channel_select)

        # Настройки PSD
        psd_frame = ttk.LabelFrame(left_frame, text="Настройки PSD и вывода", padding=5)
        psd_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Radiobutton(psd_frame, text="Whole record (FFT)", variable=self.psd_method, value="whole").pack(anchor=tk.W)
        # Заглушки для будущих методов
        ttk.Radiobutton(psd_frame, text="Бартлетт (в разработке)", variable=self.psd_method, value="bartlett", state=tk.DISABLED).pack(anchor=tk.W)
        ttk.Radiobutton(psd_frame, text="Уэлч (в разработке)", variable=self.psd_method, value="welch", state=tk.DISABLED).pack(anchor=tk.W)

        # Параметры методов
        param_frame = ttk.Frame(psd_frame)
        param_frame.pack(fill=tk.X, pady=5)
        ttk.Label(param_frame, text="Длина сегмента:").grid(row=0, column=0, sticky=tk.W)
        seg_entry = ttk.Entry(param_frame, textvariable=self.segment_length, width=8, state=tk.DISABLED)
        seg_entry.grid(row=0, column=1, padx=5)
        ttk.Label(param_frame, text="Перекрытие (0-1):").grid(row=1, column=0, sticky=tk.W)
        ov_entry = ttk.Entry(param_frame, textvariable=self.overlap, width=8, state=tk.DISABLED)
        ov_entry.grid(row=1, column=1, padx=5)

        # Частота отсечки вывода
        cutoff_frame = ttk.Frame(psd_frame)
        cutoff_frame.pack(fill=tk.X, pady=5)
        ttk.Label(cutoff_frame, text="Отсечка (Гц, 0=всё):").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(cutoff_frame, textvariable=self.cutoff_hz, width=10).grid(row=0, column=1, padx=5)

        # Кнопки действий
        btn_frame = ttk.Frame(left_frame, padding=5)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(btn_frame, text="Запись PSD текущего канала", command=self.save_psd).pack(fill=tk.X, pady=2)
        ttk.Button(btn_frame, text="▶ Запустить полный пайплайн", command=self.run_full_pipeline).pack(fill=tk.X, pady=2)

        # ---------- Правая панель ----------
        # График временного ряда (сверху)
        self.time_fig, self.time_ax = plt.subplots(figsize=(6, 3))
        self.time_canvas = FigureCanvasTkAgg(self.time_fig, master=right_frame)
        self.time_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.time_ax.set_title("Временной ряд")
        self.time_ax.set_xlabel("Время, с")
        self.time_ax.set_ylabel("Давление")

        # График PSD (снизу)
        self.psd_fig, self.psd_ax = plt.subplots(figsize=(6, 3))
        self.psd_canvas = FigureCanvasTkAgg(self.psd_fig, master=right_frame)
        self.psd_canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        self.psd_ax.set_title("Спектральная плотность мощности (PSD)")
        self.psd_ax.set_xlabel("Частота, Гц")
        self.psd_ax.set_ylabel("PSD")

        plt.tight_layout()

    def browse_directory(self):
        directory = filedialog.askdirectory(title="Выберите рабочую директорию")
        if directory:
            self.work_dir = directory
            self.dir_label.config(text=directory)
            self.scan_input_folder()

    def scan_input_folder(self):
        """Сканирует подпапку input и обновляет список файлов."""
        input_path = os.path.join(self.work_dir, "input")
        if not os.path.isdir(input_path):
            messagebox.showerror("Ошибка", "В выбранной директории нет папки 'input'.")
            return
        files = [f for f in os.listdir(input_path) if f.lower().endswith('.csv')]
        if not files:
            messagebox.showinfo("Информация", "В папке input нет CSV-файлов.")
        self.file_listbox.delete(0, tk.END)
        for f in files:
            self.file_listbox.insert(tk.END, f)

    def on_file_select(self, event):
        """Обработчик выбора файла из списка."""
        selection = self.file_listbox.curselection()
        if not selection:
            return
        filename = self.file_listbox.get(selection[0])
        
        try:
            # Используем функцию чтения из пайплайна!
            self.times, self.data, self.headers, _ = PSD_pipeline.read_csv_input(self.work_dir, filename)
            self.current_file = filename
            
            # Заполняем список каналов
            self.channel_listbox.delete(0, tk.END)
            for ch in self.headers:
                self.channel_listbox.insert(tk.END, ch)
            self.channel_listbox.insert(tk.END, "Все каналы (огибающая)")
            
            # Рассчитываем PSD для всех каналов сразу при загрузке
            self.compute_all_psd()
            
            # Очищаем графики
            self.time_ax.clear()
            self.psd_ax.clear()
            self.time_canvas.draw()
            self.psd_canvas.draw()
            
        except (FileNotFoundError, ValueError) as e:
            messagebox.showerror("Ошибка загрузки файла", str(e))
            self.current_file = None

    def compute_all_psd(self):
        """Вычисляет PSD для всех каналов текущего файла"""
        if self.times is None or self.data is None:
            return
        
        try:
            # Вызов расчета из пайплайна (пока только whole FFT)
            self.freqs, self.psd_data = PSD_pipeline.compute_psd_fft(self.times, self.data)
        except Exception as e:
            messagebox.showerror("Ошибка расчета PSD", str(e))
            self.freqs = None
            self.psd_data = None

    def on_channel_select(self, event):
        """Обработчик выбора канала из списка."""
        selection = self.channel_listbox.curselection()
        if not selection or self.data is None:
            return
        idx = selection[0]
        
        # Определяем, выбран ли пункт "Все каналы"
        if idx == len(self.headers):  # последний пункт
            self.current_channel = "all"
            self.update_plots_for_all()
        else:
            self.current_channel = self.headers[idx]
            self.update_plots_for_channel(idx)

    def get_cutoff_xlim(self):
        """Возвращает предел по X для графиков на основе отсечки"""
        try:
            cutoff = self.cutoff_hz.get()
        except tk.TclError:
            cutoff = 0
            
        if cutoff <= 0 or (self.freqs is not None and cutoff >= self.freqs[-1]):
            return self.freqs[-1] if self.freqs is not None else 30 # Весь диапазон (Найквист)
        return cutoff

    def update_plots_for_channel(self, channel_idx):
        """Обновляет графики для конкретного канала."""
        if self.data is None or self.psd_data is None:
            return

        channel_name = self.headers[channel_idx]

        # Временной ряд
        self.time_ax.clear()
        self.time_ax.plot(self.times, self.data[:, channel_idx], color='blue', linewidth=0.8)
        self.time_ax.set_title(f"Временной ряд: {channel_name}")
        self.time_ax.set_xlabel("Время, с")
        self.time_ax.set_ylabel("Давление")
        self.time_canvas.draw()

        # PSD
        self.psd_ax.clear()
        self.psd_ax.plot(self.freqs, self.psd_data[channel_idx], color='green', linewidth=0.8)
        self.psd_ax.set_title(f"PSD (метод: {self.psd_method.get()}) - {channel_name}")
        self.psd_ax.set_xlabel("Частота, Гц")
        self.psd_ax.set_ylabel("PSD")
        self.psd_ax.set_xlim(0, self.get_cutoff_xlim())
        self.psd_canvas.draw()

        # Сохраняем последние вычисленные значения для возможной записи
        self.last_psd_freqs = self.freqs
        self.last_psd_values = self.psd_data[channel_idx]

    def update_plots_for_all(self):
        """Обновляет графики для режима 'все каналы' (PSD огибающая)."""
        if self.data is None or self.psd_data is None:
            return

        self.time_ax.clear()
        self.time_ax.text(0.5, 0.5, "Временной ряд не отображается\nв режиме 'Все каналы'",
                          horizontalalignment='center', verticalalignment='center',
                          transform=self.time_ax.transAxes, fontsize=12, color='gray')
        self.time_ax.set_title("Режим: все каналы")
        self.time_canvas.draw()

        # Огибающая - поэлементный максимум по всем PSD
        envelope = np.max(self.psd_data, axis=0)

        self.psd_ax.clear()
        self.psd_ax.plot(self.freqs, envelope, color='red', linewidth=2, label='Огибающая (максимум)')
        for i, psd in enumerate(self.psd_data):
            self.psd_ax.plot(self.freqs, psd, color='gray', alpha=0.3, linewidth=0.5)
            
        self.psd_ax.set_title("PSD: огибающая по всем каналам")
        self.psd_ax.set_xlabel("Частота, Гц")
        self.psd_ax.set_ylabel("PSD")
        self.psd_ax.set_xlim(0, self.get_cutoff_xlim())
        self.psd_ax.legend()
        self.psd_canvas.draw()

    def save_psd(self):
        """Сохраняет PSD текущего канала в папку output (одиночный файл)."""
        if self.current_channel is None or self.current_channel == "all":
            messagebox.showwarning("Предупреждение", "Выберите конкретный канал для сохранения одиночного PSD.")
            return
        if not hasattr(self, 'last_psd_freqs') or not hasattr(self, 'last_psd_values'):
            messagebox.showerror("Ошибка", "Сначала вычислите PSD (выберите канал).")
            return

        output_dir = os.path.join(self.work_dir, "output")
        os.makedirs(output_dir, exist_ok=True)

        base = os.path.splitext(self.current_file)[0]
        channel_clean = "".join(c for c in self.current_channel if c.isalnum() or c in (' ', '-', '_')).strip().replace(' ', '_')
        
        # Формируем подпись отсечки
        try:
            cutoff = self.cutoff_hz.get()
        except tk.TclError:
            cutoff = 0
            
        cutoff_label = "full" if cutoff <= 0 else f"{int(cutoff)}-Hz"
        out_filename = f"{base}_PSD_{channel_clean}_0-{cutoff_label}.csv"
        out_path = os.path.join(output_dir, out_filename)

        # Сохраняем с применением маски отсечки
        if cutoff <= 0 or cutoff >= self.last_psd_freqs[-1]:
            mask = np.ones_like(self.last_psd_freqs, dtype=bool)
        else:
            mask = self.last_psd_freqs <= cutoff
            
        data_to_save = np.column_stack([self.last_psd_freqs[mask], self.last_psd_values[mask]])
        
        try:
            np.savetxt(out_path, data_to_save, delimiter=',', header='freq_Hz,PSD_Pa2_Hz')
            messagebox.showinfo("Успех", f"PSD сохранён в:\n{out_path}")
        except Exception as e:
            messagebox.showerror("Ошибка сохранения", str(e))

    def run_full_pipeline(self):
        """Запускает полный пайплайн PSD_pipeline для текущего файла"""
        if not self.current_file or not self.work_dir:
            messagebox.showwarning("Предупреждение", "Сначала выберите рабочую директорию и файл.")
            return
            
        try:
            cutoff = self.cutoff_hz.get()
        except tk.TclError:
            cutoff = 0
            
        # Формируем аргумент каналов (пока передаем 0 - все каналы, 
        # так как GUI не поддерживает множественный выбор)
        channels_arg = 0
        
        try:
            output_dir = PSD_pipeline.process_psd_pipeline(
                directory=self.work_dir,
                filename=self.current_file,
                channels=channels_arg,
                cutoff_hz=cutoff
            )
            if output_dir:
                messagebox.showinfo("Успех", f"Полный пайплайн выполнен!\nРезультаты в:\n{output_dir}")
        except Exception as e:
            messagebox.showerror("Ошибка пайплайна", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = PSDApp(root)
    root.mainloop()
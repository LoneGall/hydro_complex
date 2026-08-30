import os
import logging
from pathlib import Path
from typing import Optional, List, Tuple, Any, Dict

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# Подключаем систему плагинов
from core.plugin_manager import PipelineManager
from core.base_pipeline import PipelineConfig
import pandas as pd

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
        self.root.geometry("1400x800")
        
        # Инициализация логгера для экземпляра класса
        self.logger = logging.getLogger(__name__)

        # Менеджер плагинов
        self.plugin_manager = PipelineManager()
        self.current_pipeline: Optional[Any] = None
        self.active_pipeline_name: str = ""

        # Переменные для хранения состояния
        self.work_dir: Path = Path.cwd()  # Используем Path вместо str
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
        
        # Динамические виджеты конфигурации пайплайна
        self.pipeline_config_widgets: Dict[str, Any] = {}

        # Создание интерфейса
        self.create_widgets()

    def create_widgets(self):
        # Создаем вкладки
        tab_control = ttk.Notebook(self.root)
        tab_control.pack(fill=tk.BOTH, expand=True)
        
        # Вкладка 1: PSD Анализ
        self.psd_tab = ttk.Frame(tab_control)
        tab_control.add(self.psd_tab, text="PSD Анализ")
        
        # Вкладка 2: Генератор сигналов
        self.generator_tab = ttk.Frame(tab_control)
        tab_control.add(self.generator_tab, text="Генератор сигналов")
        
        # Создаем интерфейс для каждой вкладки
        self.create_psd_tab_widgets()
        self.create_generator_tab_widgets()

    def create_psd_tab_widgets(self):
        """Создает виджеты вкладки PSD анализа."""
        # Главный контейнер с изменяемыми панелями
        main_panel = ttk.PanedWindow(self.psd_tab, orient=tk.HORIZONTAL)
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

        # Выбор пайплайна
        pipeline_frame = ttk.LabelFrame(left_frame, text="Выбор пайплайна", padding=5)
        pipeline_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.pipeline_var = tk.StringVar(value="PSD Analysis")
        available_pipelines = self.plugin_manager.get_available_pipelines()
        if not available_pipelines:
            available_pipelines = ["Нет доступных плагинов"]
        
        self.pipeline_combo = ttk.Combobox(
            pipeline_frame, 
            textvariable=self.pipeline_var,
            values=available_pipelines,
            state="readonly"
        )
        self.pipeline_combo.pack(fill=tk.X, padx=5, pady=5)
        self.pipeline_combo.bind('<<ComboboxSelected>>', self.on_pipeline_change)
        
        # Динамическая конфигурация пайплайна
        self.config_frame = ttk.LabelFrame(left_frame, text="Настройки пайплайна", padding=5)
        self.config_frame.pack(fill=tk.X, padx=5, pady=5)
        self.build_pipeline_config_ui("PSD Analysis")

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
    
    def create_generator_tab_widgets(self):
        """Создает виджеты вкладки генератора сигналов."""
        # Контейнер с панелями
        gen_panel = ttk.PanedWindow(self.generator_tab, orient=tk.HORIZONTAL)
        gen_panel.pack(fill=tk.BOTH, expand=True)
        
        # Левая панель - настройки
        left_gen = ttk.Frame(gen_panel, width=450, relief=tk.SUNKEN)
        gen_panel.add(left_gen, weight=1)
        
        # Правая панель - предпросмотр
        right_gen = ttk.Frame(gen_panel, width=800, relief=tk.SUNKEN)
        gen_panel.add(right_gen, weight=2)
        
        # ---------- Левая панель генератора ----------
        # Выбор пайплайна генерации
        gen_pipeline_frame = ttk.LabelFrame(left_gen, text="Пайплайн генерации", padding=5)
        gen_pipeline_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.gen_pipeline_var = tk.StringVar(value="Signal Generator")
        available_gen_pipelines = [p for p in self.plugin_manager.get_available_pipelines() if "Generator" in p or "generator" in p.lower()]
        if not available_gen_pipelines:
            available_gen_pipelines = self.plugin_manager.get_available_pipelines()
        
        self.gen_pipeline_combo = ttk.Combobox(
            gen_pipeline_frame,
            textvariable=self.gen_pipeline_var,
            values=available_gen_pipelines,
            state="readonly"
        )
        self.gen_pipeline_combo.pack(fill=tk.X, padx=5, pady=5)
        self.gen_pipeline_combo.bind('<<ComboboxSelected>>', self.on_gen_pipeline_change)
        
        # Конфигурация генератора
        self.gen_config_frame = ttk.LabelFrame(left_gen, text="Параметры сигнала", padding=5)
        self.gen_config_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.build_generator_config_ui("Signal Generator")
        
        # Кнопка генерации
        gen_btn_frame = ttk.Frame(left_gen, padding=5)
        gen_btn_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(gen_btn_frame, text="🔄 Сгенерировать сигнал", command=self.run_signal_generator).pack(fill=tk.X, pady=5)
        
        # ---------- Правая панель генератора ----------
        # График предпросмотра
        self.gen_fig, self.gen_ax = plt.subplots(figsize=(6, 4))
        self.gen_canvas = FigureCanvasTkAgg(self.gen_fig, master=right_gen)
        self.gen_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.gen_ax.set_title("Предпросмотр сигнала")
        self.gen_ax.set_xlabel("Время, с")
        self.gen_ax.set_ylabel("Амплитуда")
        
        # Метрики
        self.metrics_frame = ttk.LabelFrame(right_gen, text="Метрики сигнала", padding=5)
        self.metrics_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.metrics_text = tk.Text(self.metrics_frame, height=10, width=50)
        self.metrics_text.pack(fill=tk.BOTH, expand=True)
        
        plt.tight_layout()
    
    def build_pipeline_config_ui(self, pipeline_name: str):
        """Строит UI для конфигурации выбранного пайплайна."""
        # Очищаем старые виджеты
        for widget in self.config_frame.winfo_children():
            widget.destroy()
        self.pipeline_config_widgets.clear()
        
        pipeline = self.plugin_manager.get_pipeline(pipeline_name)
        if not pipeline:
            ttk.Label(self.config_frame, text="Пайплайн не найден").pack()
            return
        
        schema = pipeline.get_config_schema()
        
        row = 0
        for param_name, param_info in schema.items():
            label_text = param_info.get("label", param_name)
            param_type = param_info.get("type", "str")
            default_value = param_info.get("default", "")
            
            ttk.Label(self.config_frame, text=f"{label_text}:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
            
            if param_type == "bool":
                var = tk.BooleanVar(value=default_value)
                widget = ttk.Checkbutton(self.config_frame, variable=var)
                widget.grid(row=row, column=1, sticky=tk.W, padx=5, pady=2)
                self.pipeline_config_widgets[param_name] = {"type": "bool", "var": var}
            elif param_type == "list" and "options" in param_info:
                var = tk.StringVar(value=default_value)
                widget = ttk.Combobox(self.config_frame, textvariable=var, values=param_info["options"], width=20)
                widget.grid(row=row, column=1, sticky=tk.W, padx=5, pady=2)
                self.pipeline_config_widgets[param_name] = {"type": "list", "var": var}
            elif param_type == "int":
                var = tk.IntVar(value=default_value)
                widget = ttk.Entry(self.config_frame, textvariable=var, width=10)
                widget.grid(row=row, column=1, sticky=tk.W, padx=5, pady=2)
                self.pipeline_config_widgets[param_name] = {"type": "int", "var": var}
            elif param_type == "float":
                var = tk.DoubleVar(value=default_value)
                widget = ttk.Entry(self.config_frame, textvariable=var, width=10)
                widget.grid(row=row, column=1, sticky=tk.W, padx=5, pady=2)
                self.pipeline_config_widgets[param_name] = {"type": "float", "var": var}
            else:  # str
                var = tk.StringVar(value=default_value)
                widget = ttk.Entry(self.config_frame, textvariable=var, width=20)
                widget.grid(row=row, column=1, sticky=tk.W, padx=5, pady=2)
                self.pipeline_config_widgets[param_name] = {"type": "str", "var": var}
            
            row += 1
    
    def build_generator_config_ui(self, pipeline_name: str):
        """Строит UI для конфигурации генератора сигналов."""
        # Очищаем старые виджеты
        for widget in self.gen_config_frame.winfo_children():
            widget.destroy()
        
        pipeline = self.plugin_manager.get_pipeline(pipeline_name)
        if not pipeline:
            ttk.Label(self.gen_config_frame, text="Пайплайн не найден").pack()
            return
        
        schema = pipeline.get_config_schema()
        
        # Используем grid для более компактного расположения
        row = 0
        for param_name, param_info in schema.items():
            label_text = param_info.get("label", param_name)
            param_type = param_info.get("type", "str")
            default_value = param_info.get("default", "")
            
            ttk.Label(self.gen_config_frame, text=f"{label_text}:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
            
            if param_type == "bool":
                var = tk.BooleanVar(value=default_value)
                widget = ttk.Checkbutton(self.gen_config_frame, variable=var)
                widget.grid(row=row, column=1, sticky=tk.W, padx=5, pady=2)
                setattr(self, f"gen_{param_name}", var)
            elif param_type == "list" and "options" in param_info:
                var = tk.StringVar(value=default_value)
                widget = ttk.Combobox(self.gen_config_frame, textvariable=var, values=param_info["options"], width=20)
                widget.grid(row=row, column=1, sticky=tk.W, padx=5, pady=2)
                setattr(self, f"gen_{param_name}", var)
            elif param_type == "int":
                var = tk.IntVar(value=default_value)
                widget = ttk.Entry(self.gen_config_frame, textvariable=var, width=10)
                widget.grid(row=row, column=1, sticky=tk.W, padx=5, pady=2)
                setattr(self, f"gen_{param_name}", var)
            elif param_type == "float":
                var = tk.DoubleVar(value=default_value)
                widget = ttk.Entry(self.gen_config_frame, textvariable=var, width=10)
                widget.grid(row=row, column=1, sticky=tk.W, padx=5, pady=2)
                setattr(self, f"gen_{param_name}", var)
            else:  # str
                var = tk.StringVar(value=default_value)
                widget = ttk.Entry(self.gen_config_frame, textvariable=var, width=20)
                widget.grid(row=row, column=1, sticky=tk.W, padx=5, pady=2)
                setattr(self, f"gen_{param_name}", var)
            
            row += 1
    
    def on_pipeline_change(self, event=None):
        """Обработчик смены пайплайна."""
        selected = self.pipeline_var.get()
        self.active_pipeline_name = selected
        self.build_pipeline_config_ui(selected)
        logger.info(f"Выбран пайплайн: {selected}")
    
    def on_gen_pipeline_change(self, event=None):
        """Обработчик смены пайплайна генератора."""
        selected = self.gen_pipeline_var.get()
        self.build_generator_config_ui(selected)
        logger.info(f"Выбран пайплайн генератора: {selected}")

    def browse_directory(self):
        directory = filedialog.askdirectory(title="Выберите рабочую директорию")
        if directory:
            self.work_dir = Path(directory)
            self.dir_label.config(text=str(directory))
            self.scan_input_folder()

    def scan_input_folder(self):
        """Сканирует подпапку input и обновляет список файлов."""
        input_path = self.work_dir / "input"
        if not input_path.is_dir():
            messagebox.showerror("Ошибка", "В выбранной директории нет папки 'input'.")
            return
        files = [f for f in input_path.iterdir() if f.suffix.lower() == '.csv']
        if not files:
            messagebox.showinfo("Информация", "В папке input нет CSV-файлов.")
        self.file_listbox.delete(0, tk.END)
        for f in files:
            self.file_listbox.insert(tk.END, f.name)

    def on_file_select(self, event: Any) -> None:
        """Обработчик выбора файла из списка."""
        selection = self.file_listbox.curselection()
        if not selection:
            return
            
        filename = self.file_listbox.get(selection[0])
        self.logger.info(f"Выбран файл: {filename}")
        
        try:
            # Получаем активный пайплайн PSD
            psd_pipeline = self.plugin_manager.get_pipeline("PSD Analysis")
            if not psd_pipeline:
                raise RuntimeError("Пайплайн 'PSD Analysis' не найден")
            
            # Вызываем функцию чтения из пайплайна
            # read_csv_input возвращает: times, data, headers, dc_offsets
            result = psd_pipeline.read_csv_input(self.work_dir, filename)
            
            # Распаковываем результат
            self.times = result[0]
            self.data = result[1]
            self.headers = result[2]
            # result[3] - dc_offsets (не используем пока)
            
            self.current_file = filename
            self.logger.info(f"Файл загружен успешно. Каналов: {len(self.headers)}")
            
            # Заполняем список каналов
            self.channel_listbox.delete(0, tk.END)
            for ch in self.headers:
                self.channel_listbox.insert(tk.END, ch)
            self.channel_listbox.insert(tk.END, "Все каналы (огибающая)")
            
            # Рассчитываем PSD для всех каналов сразу при загрузке
            self.compute_all_psd()
            
            # Очищаем графики перед первым отображением
            self.time_ax.clear()
            self.psd_ax.clear()
            self.time_canvas.draw()
            self.psd_canvas.draw()
            
        except FileNotFoundError as e:
            error_msg = f"Файл не найден: {filename}"
            self.logger.error(error_msg)
            messagebox.showerror("Ошибка загрузки файла", error_msg)
            self.current_file = None
        except ValueError as e:
            error_msg = f"Некорректный формат файла: {str(e)}"
            self.logger.error(error_msg)
            messagebox.showerror("Ошибка формата", error_msg)
            self.current_file = None
        except Exception as e:
            error_msg = f"Неожиданная ошибка: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            messagebox.showerror("Ошибка", error_msg)
            self.current_file = None

    def compute_all_psd(self):
        """Вычисляет PSD для всех каналов текущего файла"""
        if self.times is None or self.data is None:
            return
        
        try:
            # Получаем активный пайплайн PSD
            psd_pipeline = self.plugin_manager.get_pipeline("PSD Analysis")
            if not psd_pipeline:
                raise RuntimeError("Пайплайн 'PSD Analysis' не найден")
            
            # Вызов расчета из пайплайна через экземпляр
            self.freqs, self.psd_data = psd_pipeline.compute_psd_fft(self.times, self.data)
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

    def update_plots_for_channel(self, channel_idx: int) -> None:
        """Обновляет графики для конкретного канала с оптимизацией отрисовки."""
        if self.data is None or self.psd_data is None or self.freqs is None:
            return

        channel_name = self.headers[channel_idx]

        # Временной ряд - используем set_data вместо пересоздания
        if self.time_line is None:
            self.time_ax.clear()
            self.time_line, = self.time_ax.plot(
                self.times, self.data[:, channel_idx], 
                color='blue', linewidth=0.8
            )
            self.time_ax.set_title(f"Временной ряд: {channel_name}")
            self.time_ax.set_xlabel("Время, с")
            self.time_ax.set_ylabel("Давление")
            self.time_canvas.draw_idle()
        else:
            self.time_line.set_data(self.times, self.data[:, channel_idx])
            self.time_ax.set_title(f"Временной ряд: {channel_name}")
            self.time_canvas.draw_idle()

        # PSD - используем set_data вместо пересоздания
        if self.psd_line is None:
            self.psd_ax.clear()
            self.psd_line, = self.psd_ax.plot(
                self.freqs, self.psd_data[channel_idx], 
                color='green', linewidth=0.8
            )
            self.psd_ax.set_title(f"PSD (метод: {self.psd_method_var.get()}) - {channel_name}")
            self.psd_ax.set_xlabel("Частота, Гц")
            self.psd_ax.set_ylabel("PSD")
            self.psd_canvas.draw_idle()
        else:
            self.psd_line.set_data(self.freqs, self.psd_data[channel_idx])
            self.psd_ax.set_title(f"PSD (метод: {self.psd_method_var.get()}) - {channel_name}")
            self.psd_canvas.draw_idle()
        
        self.psd_ax.set_xlim(0, self.get_cutoff_xlim())
        self.psd_canvas.draw_idle()

        # Сохраняем последние вычисленные значения для возможной записи (полные данные)
        self.last_psd_freqs = self.freqs
        self.last_psd_values = self.psd_data[channel_idx]

    def update_plots_for_all(self) -> None:
        """Обновляет графики для режима 'все каналы' (PSD огибающая) с оптимизацией."""
        if self.data is None or self.psd_data is None:
            return

        # Очищаем только при первом создании или если нужно пересоздать фон
        if not self.psd_bg_lines:
            self.time_ax.text(0.5, 0.5, "Временной ряд не отображается\nв режиме 'Все каналы'",
                              horizontalalignment='center', verticalalignment='center',
                              transform=self.time_ax.transAxes, fontsize=12, color='gray')
            self.time_ax.set_title("Режим: все каналы")
            self.time_canvas.draw_idle()

            # Создаем фоновые линии для всех каналов (серые, полупрозрачные)
            self.psd_bg_lines = []
            for i, psd in enumerate(self.psd_data):
                line, = self.psd_ax.plot(
                    self.freqs, psd, 
                    color='gray', alpha=0.3, linewidth=0.5
                )
                self.psd_bg_lines.append(line)
            
            # Создаем линию огибающей
            self.psd_envelope_line, = self.psd_ax.plot([], [], color='red', linewidth=2, label='Огибающая (максимум)')
            self.psd_ax.set_title("PSD: огибающая по всем каналам")
            self.psd_ax.set_xlabel("Частота, Гц")
            self.psd_ax.set_ylabel("PSD")
            self.psd_ax.legend()
        
        # Обновляем данные огибающей
        envelope = np.max(self.psd_data, axis=0)
        if self.psd_envelope_line is not None:
            self.psd_envelope_line.set_data(self.freqs, envelope)
        
        # Обновляем данные фоновых линий
        for i, line in enumerate(self.psd_bg_lines):
            if i < len(self.psd_data):
                line.set_data(self.freqs, self.psd_data[i])
        
        self.psd_ax.set_xlim(0, self.get_cutoff_xlim())
        self.psd_canvas.draw_idle()

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

    def run_full_pipeline(self) -> None:
        """Запускает полный пайплайн через систему плагинов."""
        if not self.current_file or not self.work_dir:
            logger.warning("Попытка запуска пайплайна без выбора файла или директории")
            messagebox.showwarning("Предупреждение", "Сначала выберите рабочую директорию и файл.")
            return
        
        # Получаем выбранный пайплайн
        pipeline_name = self.pipeline_var.get()
        pipeline = self.plugin_manager.get_pipeline(pipeline_name)
        
        if not pipeline:
            messagebox.showerror("Ошибка", f"Пайплайн '{pipeline_name}' не найден.")
            return
        
        try:
            # Собираем конфигурацию из UI
            config_dict = {}
            for param_name, widget_info in self.pipeline_config_widgets.items():
                var = widget_info["var"]
                param_type = widget_info["type"]
                
                try:
                    if param_type == "bool":
                        config_dict[param_name] = var.get()
                    elif param_type == "int":
                        config_dict[param_name] = var.get()
                    elif param_type == "float":
                        config_dict[param_name] = float(var.get())
                    elif param_type == "list" or param_type == "str":
                        config_dict[param_name] = var.get()
                except (tk.TclError, ValueError) as e:
                    logger.warning(f"Ошибка получения значения {param_name}: {e}")
                    config_dict[param_name] = None
            
            # Добавляем путь к рабочей директории
            config_dict["output_dir"] = os.path.join(self.work_dir, "output")
            
            # Создаем объект конфигурации
            config = PipelineConfig(**config_dict)
            
            # Читаем данные из файла
            input_path = Path(self.work_dir) / "input" / self.current_file
            df = pd.read_csv(input_path)
            
            # Запускаем пайплайн
            result = pipeline.run(df, config)
            
            if result.get("success"):
                logger.info(f"Пайплайн выполнен успешно: {result.get('message')}")
                messagebox.showinfo("Успех", f"{result.get('message')}\n\nФайлы:\n{chr(10).join(result.get('files', []))}")
                
                # Если есть графики - отображаем
                if result.get("plots"):
                    self._display_pipeline_plots(result["plots"])
            else:
                messagebox.showerror("Ошибка пайплайна", result.get("message", "Неизвестная ошибка"))
                
        except Exception as e:
            logger.error(f"Ошибка выполнения пайплайна: {e}")
            messagebox.showerror("Ошибка пайплайна", str(e))
    
    def _display_pipeline_plots(self, plots: List[Dict[str, Any]]) -> None:
        """Отображает результаты пайплайна на графиках."""
        if not plots:
            return
        
        # Очищаем текущие графики
        self.time_ax.clear()
        self.psd_ax.clear()
        
        plot_data = plots[0]  # Берем первый график для отображения
        
        x_data = np.array(plot_data.get("x", []))
        y_data_list = plot_data.get("y", [])
        labels = plot_data.get("labels", [])
        
        if len(y_data_list) > 0 and len(x_data) > 0:
            # Отображаем первый канал на временном графике (если это временной ряд)
            if "time" in plot_data.get("xlabel", "").lower():
                self.time_line, = self.time_ax.plot(x_data, y_data_list[0], label=labels[0] if labels else "Signal")
                self.time_ax.set_title(plot_data.get("title", "Signal"))
                self.time_ax.set_xlabel(plot_data.get("xlabel", "X"))
                self.time_ax.set_ylabel(plot_data.get("ylabel", "Y"))
                self.time_canvas.draw_idle()
            
            # Отображаем PSD или другой частотный график
            if "freq" in plot_data.get("xlabel", "").lower() or "psd" in plot_data.get("title", "").lower():
                for i, y_data in enumerate(y_data_list):
                    label = labels[i] if i < len(labels) else f"Channel {i}"
                    self.psd_ax.plot(x_data, y_data, label=label, alpha=0.7)
                
                self.psd_ax.set_title(plot_data.get("title", "PSD"))
                self.psd_ax.set_xlabel(plot_data.get("xlabel", "Frequency (Hz)"))
                self.psd_ax.set_ylabel(plot_data.get("ylabel", "PSD"))
                self.psd_ax.legend()
                self.psd_canvas.draw_idle()
    
    def run_signal_generator(self) -> None:
        """Запускает генератор сигналов."""
        pipeline_name = self.gen_pipeline_var.get()
        pipeline = self.plugin_manager.get_pipeline(pipeline_name)
        
        if not pipeline:
            messagebox.showerror("Ошибка", f"Пайплайн '{pipeline_name}' не найден.")
            return
        
        try:
            # Собираем параметры из атрибутов объекта
            config_dict = {}
            schema = pipeline.get_config_schema()
            
            for param_name, param_info in schema.items():
                attr_name = f"gen_{param_name}"
                if hasattr(self, attr_name):
                    var = getattr(self, attr_name)
                    param_type = param_info.get("type", "str")
                    
                    try:
                        if param_type == "bool":
                            config_dict[param_name] = var.get()
                        elif param_type == "int":
                            config_dict[param_name] = var.get()
                        elif param_type == "float":
                            config_dict[param_name] = float(var.get())
                        else:  # str или list
                            config_dict[param_name] = var.get()
                    except (tk.TclError, ValueError) as e:
                        logger.warning(f"Ошибка получения значения {param_name}: {e}")
                        config_dict[param_name] = param_info.get("default")
            
            # Устанавливаем output_dir в input по умолчанию
            if "output_dir" not in config_dict:
                config_dict["output_dir"] = os.path.join(self.work_dir, "input") if self.work_dir else "input"
            
            config = PipelineConfig(**config_dict)
            
            # Пустой DataFrame для генератора (он не использует входные данные)
            df = pd.DataFrame()
            
            # Запускаем генератор
            result = pipeline.run(df, config)
            
            if result.get("success"):
                logger.info(f"Генерация успешна: {result.get('message')}")
                messagebox.showinfo("Успех", f"{result.get('message')}")
                
                # Обновляем график предпросмотра
                if result.get("plots"):
                    self._display_generator_plots(result["plots"])
                
                # Отображаем метрики
                if result.get("metrics"):
                    self.metrics_text.delete(1.0, tk.END)
                    metrics = result["metrics"]
                    for key, value in metrics.items():
                        formatted_value = f"{value:.4f}" if isinstance(value, float) else str(value)
                        self.metrics_text.insert(tk.END, f"{key}: {formatted_value}\n")
                
                # Обновляем список файлов во вкладке PSD
                if self.work_dir:
                    self.scan_input_folder()
            else:
                messagebox.showerror("Ошибка генерации", result.get("message", "Неизвестная ошибка"))
                
        except Exception as e:
            logger.error(f"Ошибка генерации сигнала: {e}")
            messagebox.showerror("Ошибка генерации", str(e))
    
    def _display_generator_plots(self, plots: List[Dict[str, Any]]) -> None:
        """Отображает сгенерированный сигнал на графике предпросмотра."""
        if not plots:
            return
        
        self.gen_ax.clear()
        plot_data = plots[0]
        
        x_data = np.array(plot_data.get("x", []))
        y_data_list = plot_data.get("y", [])
        labels = plot_data.get("labels", [])
        
        if len(y_data_list) > 0 and len(x_data) > 0:
            for i, y_data in enumerate(y_data_list):
                label = labels[i] if i < len(labels) else f"Signal {i}"
                self.gen_ax.plot(x_data, y_data, label=label, linewidth=0.8)
            
            self.gen_ax.set_title(plot_data.get("title", "Generated Signal"))
            self.gen_ax.set_xlabel(plot_data.get("xlabel", "Time (s)"))
            self.gen_ax.set_ylabel(plot_data.get("ylabel", "Amplitude"))
            self.gen_ax.legend()
            self.gen_ax.grid(True, alpha=0.3)
            self.gen_canvas.draw_idle()

if __name__ == "__main__":
    # Устанавливаем рабочую директорию на уровень выше от текущей, если мы в input/
    current_dir = Path.cwd()
    if current_dir.name == "input" and current_dir.parent.name:
        # Если запущены из папки input, поднимаемся на уровень выше
        work_dir = current_dir.parent
        os.chdir(work_dir)
        logger.info(f"Рабочая директория установлена в: {work_dir}")
    
    root = tk.Tk()
    app = PSDApp(root)
    
    # Автоматически выбираем текущую директорию как рабочую, если она содержит input/
    if (Path.cwd() / "input").is_dir():
        app.work_dir = Path.cwd()
        app.dir_label.config(text=str(app.work_dir))
        app.scan_input_folder()
        logger.info(f"Автоматически выбрана рабочая директория: {app.work_dir}")
    
    root.mainloop()
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import numpy as np

class PSDApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Вычисление PSD сигналов")
        self.root.geometry("1200x700")

        # Переменные для хранения состояния
        self.work_dir = ""
        self.current_file = None
        self.data = None          # DataFrame с загруженным файлом
        self.time = None          # столбец времени
        self.channels = []         # список названий каналов (без времени)
        self.current_channel = None
        self.psd_method = tk.StringVar(value="whole")  # whole, bartlett, welch
        self.segment_length = tk.IntVar(value=256)
        self.overlap = tk.DoubleVar(value=0.5)

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
        psd_frame = ttk.LabelFrame(left_frame, text="Метод вычисления PSD", padding=5)
        psd_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Radiobutton(psd_frame, text="Whole record", variable=self.psd_method, value="whole").pack(anchor=tk.W)
        ttk.Radiobutton(psd_frame, text="Бартлетт (без перекрытия)", variable=self.psd_method, value="bartlett").pack(anchor=tk.W)
        ttk.Radiobutton(psd_frame, text="Уэлч (с перекрытием)", variable=self.psd_method, value="welch").pack(anchor=tk.W)

        # Параметры (пока заглушки, можно будет активировать)
        param_frame = ttk.Frame(psd_frame)
        param_frame.pack(fill=tk.X, pady=5)
        ttk.Label(param_frame, text="Длина сегмента:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(param_frame, textvariable=self.segment_length, width=8).grid(row=0, column=1, padx=5)
        ttk.Label(param_frame, text="Перекрытие (0-1):").grid(row=1, column=0, sticky=tk.W)
        ttk.Entry(param_frame, textvariable=self.overlap, width=8).grid(row=1, column=1, padx=5)

        # Кнопка записи
        ttk.Button(psd_frame, text="Запись PSD текущего канала", command=self.save_psd).pack(pady=10)

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

        # Небольшая настройка отступов
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
        filepath = os.path.join(self.work_dir, "input", filename)
        try:
            # Загружаем CSV: предполагаем разделитель запятая, десятичная точка
            self.data = pd.read_csv(filepath, sep=',', decimal='.')
            # Проверяем наличие хотя бы двух столбцов (время + хотя бы один сигнал)
            if self.data.shape[1] < 2:
                messagebox.showerror("Ошибка", "Файл должен содержать как минимум столбец времени и один сигнал.")
                return
            # Первый столбец - время
            self.time = self.data.iloc[:, 0]
            # Остальные - каналы
            self.channels = list(self.data.columns[1:])
            # Заполняем список каналов
            self.channel_listbox.delete(0, tk.END)
            for ch in self.channels:
                self.channel_listbox.insert(tk.END, ch)
            # Добавляем специальный пункт "Все каналы (огибающая)"
            self.channel_listbox.insert(tk.END, "Все каналы (огибающая)")
            self.current_file = filename
            # Очищаем графики
            self.time_ax.clear()
            self.psd_ax.clear()
            self.time_canvas.draw()
            self.psd_canvas.draw()
        except Exception as e:
            messagebox.showerror("Ошибка загрузки файла", str(e))

    def on_channel_select(self, event):
        """Обработчик выбора канала из списка."""
        selection = self.channel_listbox.curselection()
        if not selection or self.data is None:
            return
        idx = selection[0]
        # Определяем, выбран ли пункт "Все каналы"
        if idx == len(self.channels):  # последний пункт
            self.current_channel = "all"
            self.update_plots_for_all()
        else:
            self.current_channel = self.channels[idx]
            self.update_plots_for_channel(self.current_channel)

    def update_plots_for_channel(self, channel_name):
        """Обновляет графики для конкретного канала."""
        if self.data is None:
            return

        # Временной ряд
        self.time_ax.clear()
        self.time_ax.plot(self.time, self.data[channel_name], color='blue')
        self.time_ax.set_title(f"Временной ряд: {channel_name}")
        self.time_ax.set_xlabel("Время, с")
        self.time_ax.set_ylabel("Давление")
        self.time_canvas.draw()

        # PSD (пока заглушка - случайный спектр)
        self.compute_and_plot_psd(self.data[channel_name])

    def update_plots_for_all(self):
        """Обновляет графики для режима 'все каналы' (только PSD огибающая)."""
        if self.data is None:
            return

        # Очищаем верхний график и пишем, что временной ряд не отображается
        self.time_ax.clear()
        self.time_ax.text(0.5, 0.5, "Временной ряд не отображается\nв режиме 'Все каналы'",
                          horizontalalignment='center', verticalalignment='center',
                          transform=self.time_ax.transAxes, fontsize=12, color='gray')
        self.time_ax.set_title("Режим: все каналы")
        self.time_canvas.draw()

        # Вычисляем PSD для каждого канала и строим огибающую (максимум)
        all_psd = []
        freqs = None
        for ch in self.channels:
            # Заглушка: генерируем случайный спектр для демонстрации
            # В реальности здесь должен быть вызов compute_psd()
            signal = self.data[ch]
            psd = self._dummy_psd(signal)
            all_psd.append(psd)
            if freqs is None:
                freqs = np.linspace(0, 100, len(psd))  # заглушка для частот

        # Огибающая - поэлементный максимум
        envelope = np.max(all_psd, axis=0)

        # Рисуем огибающую на нижнем графике
        self.psd_ax.clear()
        self.psd_ax.plot(freqs, envelope, color='red', linewidth=2, label='Огибающая (максимум)')
        # Для наглядности можно также показать полупрозрачные спектры всех каналов
        for i, psd in enumerate(all_psd):
            self.psd_ax.plot(freqs, psd, color='gray', alpha=0.3, linewidth=0.5)
        self.psd_ax.set_title("PSD: огибающая по всем каналам")
        self.psd_ax.set_xlabel("Частота, Гц")
        self.psd_ax.set_ylabel("PSD")
        self.psd_ax.legend()
        self.psd_canvas.draw()

    def compute_and_plot_psd(self, signal):
        """Вычисляет PSD выбранным методом и обновляет нижний график."""
        # Здесь должна быть реальная реализация методов
        # Пока используем заглушку
        psd = self._dummy_psd(signal)
        freqs = np.linspace(0, 100, len(psd))  # заглушка для частот

        self.psd_ax.clear()
        self.psd_ax.plot(freqs, psd, color='green')
        self.psd_ax.set_title(f"PSD (метод: {self.psd_method.get()})")
        self.psd_ax.set_xlabel("Частота, Гц")
        self.psd_ax.set_ylabel("PSD")
        self.psd_canvas.draw()

        # Сохраняем последние вычисленные значения для возможной записи
        self.last_psd_freqs = freqs
        self.last_psd_values = psd

    def _dummy_psd(self, signal):
        """Заглушка для генерации случайного спектра."""
        # Просто для демонстрации: берём БПФ от сигнала, но сигнал может быть длинным,
        # поэтому для скорости возьмём первые 1024 точки, если возможно.
        n = min(1024, len(signal))
        sig = signal.values[:n]
        # Добавим окно Ханна, чтобы спектр был более гладким
        windowed = sig * np.hanning(n)
        fft_vals = np.fft.rfft(windowed)
        psd = np.abs(fft_vals)**2 / (n * 1.0)  # простая оценка
        return psd

    def save_psd(self):
        """Сохраняет PSD текущего канала в папку output."""
        if self.current_channel is None:
            messagebox.showwarning("Предупреждение", "Не выбран канал для сохранения.")
            return
        if self.current_channel == "all":
            messagebox.showinfo("Информация", "В режиме 'Все каналы' сохранение PSD не производится (выберите конкретный канал).")
            return
        if not hasattr(self, 'last_psd_freqs') or not hasattr(self, 'last_psd_values'):
            messagebox.showerror("Ошибка", "Сначала вычислите PSD (выберите канал).")
            return

        # Создаём папку output, если её нет
        output_dir = os.path.join(self.work_dir, "output")
        os.makedirs(output_dir, exist_ok=True)

        # Формируем имя файла
        base = os.path.splitext(self.current_file)[0]
        # Очищаем имя канала от возможных пробелов/спецсимволов для имени файла
        channel_clean = "".join(c for c in self.current_channel if c.isalnum() or c in (' ', '-', '_')).strip()
        channel_clean = channel_clean.replace(' ', '_')
        out_filename = f"{base}_PSD_{channel_clean}.csv"
        out_path = os.path.join(output_dir, out_filename)

        # Сохраняем два столбца: частота, PSD
        df_out = pd.DataFrame({
            'Frequency': self.last_psd_freqs,
            'PSD': self.last_psd_values
        })
        try:
            df_out.to_csv(out_path, sep=',', decimal='.', index=False)
            messagebox.showinfo("Успех", f"PSD сохранён в:\n{out_path}")
        except Exception as e:
            messagebox.showerror("Ошибка сохранения", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = PSDApp(root)
    root.mainloop()
"""
Главное приложение с улучшенным интерфейсом и обработкой ошибок
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import logging
import threading
import time
import csv
import os

logger = logging.getLogger(__name__)

from point_cloud_processor import PointCloud, AxesLoader, WidthMeasurer
from visualizer import PointCloudVisualizer

class TkLogHandler(logging.Handler):
    """Кастомный обработчик логов для Tkinter"""
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget
        self.formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s", 
            datefmt="%H:%M:%S"
        )
    
    def emit(self, record):
        msg = self.format(record)
        self.text_widget.configure(state='normal')
        self.text_widget.insert(tk.END, msg + "\n")
        self.text_widget.configure(state='disabled')
        self.text_widget.see(tk.END)

class ProgressWindow(tk.Toplevel):
    """Окно прогресса для длительных операций"""
    def __init__(self, parent, title="Выполнение"):
        super().__init__(parent)
        self.title(title)
        self.geometry("300x100")
        self.resizable(False, False)
        
        self.label = ttk.Label(self, text="Пожалуйста, подождите...")
        self.label.pack(pady=10)
        
        self.progress = ttk.Progressbar(
            self, orient="horizontal", length=250, mode="indeterminate")
        self.progress.pack(pady=10)
        self.progress.start()
        
        self.grab_set()

class PointCloudApp:
    """Главное приложение для обработки облака точек"""
    def __init__(self, root):
        self.root = root
        self.root.title("Измерение ширины по облаку точек")
        self.root.geometry("1600x800")
        
        self.point_cloud = None
        self.axes = []
        self.measurements = []
        self.current_radius = 0.025
        
        self.create_widgets()
        
        self.setup_logging()
        
        logger.info("Приложение инициализировано")

    def create_widgets(self):
        """Создание элементов интерфейса"""
        self.viz_frame = ttk.Frame(self.root)
        self.viz_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.visualizer = PointCloudVisualizer(self.viz_frame)
        
        control_frame = ttk.LabelFrame(self.root, text="Управление", width=300)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        control_frame.pack_propagate(False)
        
        ttk.Label(control_frame, text="Радиус измерения:").grid(
            row=0, column=0, padx=5, pady=2, sticky=tk.W)
        self.radius_var = tk.DoubleVar(value=self.current_radius)
        ttk.Entry(control_frame, textvariable=self.radius_var).grid(
            row=0, column=1, padx=5, pady=2, sticky=tk.EW)
        
        buttons = [
            ("Загрузить облако точек", self.load_point_cloud),
            ("Загрузить оси", self.load_axes),
            ("Выполнить измерения", self.run_measurements),
            ("Сохранить результаты", self.save_results),
            ("Очистить все", self.clear_all)
        ]
        
        for i, (text, command) in enumerate(buttons, start=1):
            ttk.Button(control_frame, text=text, command=command).grid(
                row=i, column=0, columnspan=2, padx=5, pady=5, sticky=tk.EW)
        
        log_frame = ttk.LabelFrame(control_frame, text="Логи")
        log_frame.grid(row=6, column=0, columnspan=2, 
                    padx=5, pady=5, sticky=tk.NSEW)
        
        self.log_text = tk.Text(log_frame, height=10, width=40, state='disabled')
        scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.status_var = tk.StringVar(value="Готово")
        status_bar = ttk.Label(control_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=7, column=0, columnspan=2, sticky=tk.EW)
        
        control_frame.rowconfigure(6, weight=1)
        control_frame.columnconfigure(1, weight=1)

    def setup_logging(self):
        """Настройка системы логирования"""
        handler = TkLogHandler(self.log_text)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s", 
            datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.INFO)

    def load_point_cloud(self):
        """Загрузка облака точек"""
        path = filedialog.askopenfilename(
            filetypes=[("XYZ files", "*.xyz"), ("All files", "*.*")])
        if not path:
            return
        
        try:
            self.status_var.set("Загрузка облака точек...")
            self.point_cloud = PointCloud.from_file(path)
            self.point_cloud.project_to_xy_plane()
            self.point_cloud.build_kd_tree()
            self.status_var.set(f"Загружено {len(self.point_cloud.points)} точек")
            logger.info(f"Облако точек загружено из {os.path.basename(path)}")
        except Exception as e:
            self.status_var.set("Ошибка загрузки")
            logger.error(f"Ошибка загрузки облака точек: {str(e)}")
            messagebox.showerror("Ошибка", f"Не удалось загрузить облако точек: {str(e)}")

    def load_axes(self):
        """Загрузка осей из DXF"""
        path = filedialog.askopenfilename(
            filetypes=[("DXF files", "*.dxf"), ("All files", "*.*")])
        if not path:
            return
        
        try:
            self.status_var.set("Загрузка осей...")
            self.axes = AxesLoader.load_from_dxf(path)
            self.status_var.set(f"Загружено {len(self.axes)} осей")
            logger.info(f"Оси загружены из {os.path.basename(path)}")
        except Exception as e:
            self.status_var.set("Ошибка загрузки")
            logger.error(f"Ошибка загрузки осей: {str(e)}")
            messagebox.showerror("Ошибка", f"Не удалось загрузить оси: {str(e)}")

    def run_measurements(self):
        """Выполнение измерений"""
        if self.point_cloud is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите облако точек")
            return
        if not self.axes:
            messagebox.showwarning("Предупреждение", "Сначала загрузите оси")
            return
        
        try:
            self.current_radius = self.radius_var.get()
            if self.current_radius <= 0:
                raise ValueError("Радиус должен быть положительным")
            
            progress = ProgressWindow(self.root, "Выполнение измерений")
            
            def worker():
                try:
                    start_time = time.time()
                    
                    measurer = WidthMeasurer(
                        self.point_cloud, 
                        self.axes, 
                        self.current_radius
                    )
                    
                    self.measurements = measurer.measure_all_widths()
                    
                    self.visualizer.plot_results(
                        self.point_cloud.points,
                        self.axes,
                        self.measurements,
                        show_all_points=True
                    )
                    
                    elapsed = time.time() - start_time
                    logger.info(f"Измерения завершены за {elapsed:.2f} сек")
                    self.status_var.set(
                        f"Готово. Измерено {len(self.measurements)} осей")
                except Exception as e:
                    logger.error(f"Ошибка измерений: {str(e)}")
                    self.status_var.set("Ошибка измерений")
                finally:
                    progress.destroy()
            
            threading.Thread(target=worker, daemon=True).start()
            
        except Exception as e:
            self.status_var.set("Ошибка параметров")
            logger.error(f"Ошибка запуска измерений: {str(e)}")
            messagebox.showerror("Ошибка", f"Не удалось запустить измерения: {str(e)}")

    def save_results(self):
        """Сохранение результатов в CSV"""
        if not self.measurements:
            messagebox.showwarning("Предупреждение", "Нет данных для сохранения")
            return
        
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not path:
            return
        
        try:
            with open(path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "ID оси", "Начало X", "Начало Y", 
                    "Конец X", "Конец Y", "Ширина", 
                    "Кол-во точек"
                ])
                
                for m in self.measurements:
                    writer.writerow([
                        m.axis_id,
                        m.start_point[0], m.start_point[1],
                        m.end_point[0], m.end_point[1],
                        m.width,
                        len(m.points_used)
                    ])
            
            logger.info(f"Результаты сохранены в {path}")
            self.status_var.set(f"Результаты сохранены в {os.path.basename(path)}")
        except Exception as e:
            logger.error(f"Ошибка сохранения: {str(e)}")
            messagebox.showerror("Ошибка", f"Не удалось сохранить результаты: {str(e)}")

    def clear_all(self):
        """Очистка всех данных"""
        self.point_cloud = None
        self.axes = []
        self.measurements = []
        self.visualizer.ax.clear()
        self.visualizer.canvas.draw()
        self.status_var.set("Данные очищены")
        logger.info("Все данные очищены")

if __name__ == "__main__":
    root = tk.Tk()
    app = PointCloudApp(root)
    root.mainloop()

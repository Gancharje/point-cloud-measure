"""
Модуль визуализации с улучшенной обработкой данных
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from typing import List
from point_cloud_processor import MeasurementResult, Axis

class PointCloudVisualizer:
    """Класс для визуализации результатов"""
    def __init__(self, master=None, figsize=(12, 8)):
        if master:
            self.figure, self.ax = plt.subplots(figsize=figsize)
            self.canvas = FigureCanvasTkAgg(self.figure, master=master)
            self.canvas.get_tk_widget().pack(fill='both', expand=True)
        else:
            self.figure, self.ax = plt.subplots(figsize=figsize)
            self.canvas = None
    
    def plot_results(self, 
                    points: np.ndarray, 
                    axes: List[Axis], 
                    measurements: List[MeasurementResult],
                    show_legend: bool = True,
                    show_all_points: bool = False):
        """Визуализация результатов измерений"""
        self.ax.clear()
        
        if show_all_points and len(points) > 0:
            self.ax.scatter(
                points[:, 0], points[:, 1], 
                s=1, c='lightgray', alpha=0.5, label='Облако точек'
            )
        
        for axis in axes:
            self.ax.plot(
                [axis.start[0], axis.end[0]],
                [axis.start[1], axis.end[1]],
                'b--', linewidth=0.7, alpha=0.5
            )
        
        for i, measurement in enumerate(measurements):
            if measurement.points_used:
                used_points = np.array(measurement.points_used)
                self.ax.scatter(
                    used_points[:, 0], used_points[:, 1], 
                    s=15, alpha=0.6, label=f'Точки замера {i}'
                )
            
            self.ax.plot(
                [measurement.start_point[0], measurement.end_point[0]],
                [measurement.start_point[1], measurement.end_point[1]],
                'r-', linewidth=2, 
                label=f'Ширина: {measurement.width:.4f}'
            )
            
            self.ax.scatter(
                [measurement.start_point[0], measurement.end_point[0]],
                [measurement.start_point[1], measurement.end_point[1]],
                s=50, c='red', edgecolor='black', zorder=5
            )
            
            mid_x = (measurement.start_point[0] + measurement.end_point[0]) / 2
            mid_y = (measurement.start_point[1] + measurement.end_point[1]) / 2
            self.ax.text(
                mid_x, mid_y, f'{measurement.width:.4f}', 
                fontsize=9, bbox=dict(facecolor='white', alpha=0.8)
            )
        
        self.ax.set_aspect('equal')
        self.ax.grid(True, linestyle='--', alpha=0.7)
        self.ax.set_xlabel('X координата')
        self.ax.set_ylabel('Y координата')
        self.ax.set_title('Измерение ширины объекта')
        
        if show_legend:
            self.ax.legend(loc='upper right', fontsize=8)
        
        if self.canvas:
            self.canvas.draw()
    
    def plot_local_coordinates(self, measurement: MeasurementResult):
        """Визуализация точек в локальной системе координат"""
        if not measurement.local_coords.size:
            return
        
        fig, ax = plt.subplots(figsize=(8, 6))
        local_points = measurement.local_coords
        
        ax.scatter(local_points[:, 0], local_points[:, 1], s=30)
        
        min_idx = np.argmin(local_points[:, 0])
        max_idx = np.argmax(local_points[:, 0])
        ax.scatter(
            [local_points[min_idx, 0], local_points[max_idx, 0]],
            [local_points[min_idx, 1], local_points[max_idx, 1]],
            s=100, c='red', edgecolor='black', zorder=5
        )
        
        ax.plot(
            [local_points[min_idx, 0], local_points[max_idx, 0]],
            [local_points[min_idx, 1], local_points[max_idx, 1]],
            'r-', linewidth=2
        )
        
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel('Продольная координата')
        ax.set_ylabel('Поперечная координата')
        ax.set_title(f'Локальная система координат (Ось {measurement.axis_id})')
        ax.grid(True)
        ax.set_aspect('equal')
        
        width = local_points[max_idx, 0] - local_points[min_idx, 0]
        ax.text(
            0.05, 0.95, f'Измеренная ширина: {width:.4f}',
            transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8)
        )
        
        plt.show()

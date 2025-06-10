"""
Модуль обработки облака точек с улучшенным алгоритмом измерения ширины
"""
import logging
import math
import numpy as np
from scipy.spatial import cKDTree
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

@dataclass
class MeasurementResult:
    axis_id: int
    start_point: Tuple[float, float]
    end_point: Tuple[float, float]
    width: float
    points_used: List[Tuple[float, float]]
    local_coords: np.ndarray

class PointCloud:
    """Класс для работы с облаком точек"""
    def __init__(self, points: np.ndarray):
        self.points = points
        self.tree = None
        
    @classmethod
    def from_file(cls, path: str) -> 'PointCloud':
        """Загрузка облака точек из файла"""
        logger.info(f"Загрузка облака точек из {path}")
        points = np.loadtxt(path, usecols=(0, 1, 2))
        return cls(points)
    
    def project_to_xy_plane(self) -> None:
        """Проекция точек на плоскость XY"""
        self.points[:, 2] = 0
        logger.info("Точки спроецированы на плоскость XY")
        
    def build_kd_tree(self) -> None:
        """Построение KD-дерева для быстрого поиска точек"""
        self.tree = cKDTree(self.points[:, :2])
        logger.info("KD-дерево построено")

class Axis:
    """Класс для представления оси"""
    def __init__(self, start: Tuple[float, float, float], end: Tuple[float, float, float], axis_id: int):
        self.start = np.array([start[0], start[1]])
        self.end = np.array([end[0], end[1]])
        self.id = axis_id
        
    @property
    def direction(self) -> np.ndarray:
        """Вектор направления оси"""
        vec = self.end - self.start
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec
    
    @property
    def normal(self) -> np.ndarray:
        """Нормальный вектор к оси"""
        direction = self.direction
        return np.array([-direction[1], direction[0]])
    
    @property
    def length(self) -> float:
        """Длина оси"""
        return np.linalg.norm(self.end - self.start)
    
    @property
    def center(self) -> np.ndarray:
        """Центр оси"""
        return (self.start + self.end) / 2
    
    def transform_to_local(self, points: np.ndarray) -> np.ndarray:
        """Преобразование точек в локальную систему координат оси"""
        rot_matrix = np.array([
            [self.direction[0], self.normal[0]],
            [self.direction[1], self.normal[1]]
        ])
        
        translated = points - self.center
        return np.dot(translated, rot_matrix.T)

class AxesLoader:
    """Класс для загрузки осей из DXF"""
    @staticmethod
    def load_from_dxf(path: str) -> List[Axis]:
        """Загрузка осей из DXF-файла"""
        import ezdxf
        logger.info(f"Загрузка осей из DXF: {path}")
        
        try:
            doc = ezdxf.readfile(path)
            modelspace = doc.modelspace()
            
            axes = []
            for i, entity in enumerate(modelspace.query('LINE')):
                start = (entity.dxf.start.x, entity.dxf.start.y, 0)
                end = (entity.dxf.end.x, entity.dxf.end.y, 0)
                axes.append(Axis(start, end, i))
            
            logger.info(f"Загружено {len(axes)} осей")
            return axes
        except Exception as e:
            logger.error(f"Ошибка загрузки DXF: {str(e)}")
            raise

class WidthMeasurer:
    """Класс для измерения ширины вдоль осей"""
    def __init__(self, point_cloud: PointCloud, axes: List[Axis], radius: float = 0.025):
        self.point_cloud = point_cloud
        self.axes = axes
        self.radius = radius
        self._validate_inputs()
        
    def _validate_inputs(self):
        if self.radius <= 0:
            raise ValueError("Радиус должен быть положительным")
        if not self.axes:
            raise ValueError("Список осей пуст")
        if not hasattr(self.point_cloud, 'tree') or self.point_cloud.tree is None:
            raise RuntimeError("KD-дерево не построено")
    
    def _get_points_near_axis(self, axis: Axis) -> np.ndarray:
        """Получение точек вблизи оси с использованием KD-дерева"""
        axis_length = axis.length
        half_length = axis_length / 2 + self.radius
        
        indices = self.point_cloud.tree.query_ball_point(
            axis.center, 
            half_length
        )
        return self.point_cloud.points[indices, :2]
    
    def measure_width_along_axis(self, axis: Axis) -> Optional[MeasurementResult]:
        """Измерение ширины вдоль конкретной оси"""
        try:
            points = self._get_points_near_axis(axis)
            if len(points) < 2:
                logger.warning(f"Для оси {axis.id} найдено недостаточно точек")
                return None
            
            local_points = axis.transform_to_local(points)
            
            in_band = local_points[np.abs(local_points[:, 1]) <= self.radius]
            if len(in_band) < 2:
                logger.warning(f"Для оси {axis.id} недостаточно точек в полосе")
                return None
            
            min_idx = np.argmin(in_band[:, 0])
            max_idx = np.argmax(in_band[:, 0])
            
            width = in_band[max_idx, 0] - in_band[min_idx, 0]
            
            def to_global(point):
                rot_matrix = np.array([
                    [axis.direction[0], axis.normal[0]],
                    [axis.direction[1], axis.normal[1]]
                ])
                return np.dot(point, rot_matrix) + axis.center
            
            min_point_global = to_global(in_band[min_idx])
            max_point_global = to_global(in_band[max_idx])
            
            return MeasurementResult(
                axis_id=axis.id,
                start_point=tuple(min_point_global),
                end_point=tuple(max_point_global),
                width=width,
                points_used=[tuple(p) for p in points],
                local_coords=in_band
            )
        except Exception as e:
            logger.error(f"Ошибка измерения для оси {axis.id}: {str(e)}")
            return None

    def measure_all_widths(self) -> List[MeasurementResult]:
        """Измерение ширины для всех осей"""
        results = []
        for axis in self.axes:
            result = self.measure_width_along_axis(axis)
            if result:
                results.append(result)
                logger.info(f"Ось {axis.id}: ширина = {result.width:.4f}")
        return results

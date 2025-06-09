"""
Модуль обработки облака точек и вычисления ширины по осям.
"""
import logging
import math
from concurrent.futures import ThreadPoolExecutor

import ezdxf
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")

class PointCloudProcessor:
    """
    Обрабатывает облако точек: загрузка, проекция, измерение ширины.
    """
    def __init__(self, radius: float = 0.025):
        self._radius = None
        self.radius = radius
        self._pcd = None
        self._axes = []

    @property
    def radius(self) -> float:
        return self._radius

    @radius.setter
    def radius(self, value: float):
        if value <= 0:
            raise ValueError("Радиус должен быть положительным")
        self._radius = value

    @property
    def point_cloud(self):
        return self._pcd

    @property
    def axes(self):
        return self._axes

    def load_point_cloud(self, path: str) -> None:
        logging.info(f"Загрузка облака точек из '{path}'")
        pts = np.loadtxt(path, usecols=(0,1,2))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        self._pcd = pcd
        logging.info(f"Загружено {len(pts)} точек")

    def load_axes(self, path: str) -> None:
        logging.info(f"Загрузка осей из '{path}'")
        doc = ezdxf.readfile(path)
        self._axes = list(doc.modelspace().query('LINE'))
        logging.info(f"Найдено {len(self._axes)} осей")

    def project_to_plane(self) -> None:
        pts = np.asarray(self._pcd.points)
        pts[:,2] = 0.0
        self._pcd.points = o3d.utility.Vector3dVector(pts)
        for ax in self._axes:
            ax.dxf.start = ax.dxf.start.replace(z=0.0)
            ax.dxf.end = ax.dxf.end.replace(z=0.0)
        logging.info("Проекция на плоскость выполнена")

    def measure_width(self) -> list:
        pts2d = np.asarray(self._pcd.points)[:,:2]
        tree = cKDTree(pts2d)
        def _measure(ax):
            start = np.array([ax.dxf.start.x, ax.dxf.start.y])
            end = np.array([ax.dxf.end.x, ax.dxf.end.y])
            vec = end - start
            length = np.linalg.norm(vec)
            if length == 0: return None
            unit = vec/length
            perp = np.array([-unit[1], unit[0]])
            idxs = tree.query_ball_point((start+end)/2, length/2 + self._radius)
            sel = pts2d[idxs]
            if sel.shape[0]<2: return None
            d = (sel - start) @ perp
            i0, i1 = np.argmin(d), np.argmax(d)
            p0, p1 = sel[i0], sel[i1]
            return {'p0':p0, 'p1':p1, 'w':float(d[i1]-d[i0])}
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(_measure, self._axes))
        measurements = [r for r in results if r]
        for m in measurements:
            logging.info(f"От {m['p0']} до {m['p1']}: ширина={m['w']:.3f}")
        return measurements
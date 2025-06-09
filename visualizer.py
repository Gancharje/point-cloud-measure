"""
Модуль визуализации облака точек и результатов.
"""
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class Visualizer:
    def __init__(self, parent):
        self.figure, self.ax = plt.subplots(figsize=(12,8))
        self.canvas = FigureCanvasTkAgg(self.figure, master=parent)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

    def plot(self, pcd, axes: list, measurements: list) -> None:
        self.ax.clear()
        pts = np.asarray(pcd.points)[:,:2]
        self.ax.scatter(pts[:,0], pts[:,1], s=1, c='lightgray', label='Облако')
        for ax_line in axes:
            s = np.array([ax_line.dxf.start.x, ax_line.dxf.start.y])
            e = np.array([ax_line.dxf.end.x,   ax_line.dxf.end.y])
            self.ax.plot([s[0],e[0]],[s[1],e[1]], c='blue', lw=1, alpha=0.3)
        cmap = itertools.cycle(plt.cm.tab10.colors)
        legends=[]
        for m in measurements:
            c = next(cmap)
            p0,p1,w = m['p0'], m['p1'], m['w']
            self.ax.scatter([p0[0],p1[0]],[p0[1],p1[1]], c=[c,c], s=30)
            line, = self.ax.plot([p0[0],p1[0]],[p0[1],p1[1]], c=c, lw=2)
            legends.append((line, f"({p0[0]:.2f},{p0[1]:.2f})→({p1[0]:.2f},{p1[1]:.2f}), w={w:.3f}"))
        if legends:
            h,l = zip(*legends)
            self.ax.legend(h,l,loc='upper right', fontsize=8)
        self.ax.set_aspect('equal')
        self.ax.set_xlabel('X, м')
        self.ax.set_ylabel('Y, м')
        self.ax.relim(); self.ax.autoscale_view()
        self.canvas.draw()
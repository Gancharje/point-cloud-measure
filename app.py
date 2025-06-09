"""
Главное приложение: интерфейс и логирование.
"""
import logging
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox
import csv

from point_cloud_processor import PointCloudProcessor
from visualizer import Visualizer

class TkLogHandler(logging.Handler):
    def __init__(self, log_func):
        super().__init__()
        self.log_func = log_func
    def emit(self, record):
        msg = self.format(record)
        self.log_func(msg)

class WidthApp:
    def __init__(self, master: tk.Tk):
        self.master = master
        master.title('Измерение ширины облака точек')
        left=tk.Frame(master); left.pack(side='left', fill='both', expand=True)
        right=tk.Frame(master); right.pack(side='right', fill='y', padx=5, pady=5)
        self.proc = PointCloudProcessor()
        self.vis = Visualizer(left)
        tk.Label(right, text='Радиус:').pack(pady=(0,2))
        self.radius_var = tk.StringVar(value=str(self.proc.radius))
        tk.Entry(right,textvariable=self.radius_var).pack(fill='x')
        for text,cmd in [('Загрузить XYZ',self.load_xyz),('Загрузить DXF',self.load_dxf),
                        ('Запустить',self.run),('Сохранить CSV',self.save_csv)]:
            tk.Button(right,text=text,command=cmd).pack(fill='x', pady=2)
        self.log=tk.Text(right,width=30,height=12); self.log.pack(pady=5)
        handler=TkLogHandler(self._log)
        handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S'))
        logging.getLogger().addHandler(handler)

    def _log(self,msg: str):
        self.log.insert('end',msg+'\n'); self.log.see('end')

    def _get_radius(self):
        try:
            r=float(self.radius_var.get());
            if r<=0: raise ValueError
            return r
        except:
            messagebox.showerror('Ошибка','Введите положительный радиус')
            return None

    def load_xyz(self):
        p=filedialog.askopenfilename(filetypes=[('XYZ','*.xyz')]);
        if not p: return
        t0=time.time(); self.proc.load_point_cloud(p);
        self._log(f"XYZ загружен за {time.time()-t0:.2f} с")

    def load_dxf(self):
        p=filedialog.askopenfilename(filetypes=[('DXF','*.dxf')]);
        if not p: return
        t0=time.time(); self.proc.load_axes(p);
        self._log(f"DXF загружен за {time.time()-t0:.2f} с")

    def run(self):
        if not self.proc.point_cloud or not self.proc.axes:
            messagebox.showwarning('Ошибка','Загрузите XYZ и DXF')
            return
        r=self._get_radius();
        if r is None: return
        self.proc.radius=r
        def worker():
            t0=time.time(); self.proc.project_to_plane(); self._log(f"Проекция {time.time()-t0:.2f} с")
            t0=time.time(); m=self.proc.measure_width(); self._log(f"Замеры {time.time()-t0:.2f} с")
            self.vis.plot(self.proc.point_cloud, self.proc.axes, m)
        threading.Thread(target=worker,daemon=True).start()

    def save_csv(self):
        if not self.proc.point_cloud or not self.proc.axes:
            messagebox.showwarning('Ошибка','Загрузите XYZ и DXF')
            return
        r=self._get_radius();
        if r is None: return
        self.proc.radius=r
        path=filedialog.asksaveasfilename(defaultextension='.csv',filetypes=[('CSV','*.csv')])
        if not path: return
        t0=time.time(); m=self.proc.measure_width()
        with open(path,'w',newline='',encoding='utf-8') as f:
            w=csv.writer(f); w.writerow(['x0','y0','x1','y1','w'])
            for mm in m: p0,p1,wid=mm['p0'],mm['p1'],mm['w']; w.writerow([*p0,*p1,f"{wid:.3f}"])
        self._log(f"CSV сохранён за {time.time()-t0:.2f} с -> {path}")

if __name__=='__main__':
    import tkinter
    root=tkinter.Tk()
    app=WidthApp(root)
    root.mainloop()
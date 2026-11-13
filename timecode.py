"""
LIVE lap counter for the F1Tenth competition.
 
Test 1: 10 consecutive laps without a collision.
Test 2: Fastest lap time (among the 10 valid laps).
 
Displays a single line that updates 10 times per second:
  Lap 3/10 | Current lap: 45.3 s | Total: 325.2 s | Best: 139.85 s
 
Upon completing each lap, the time is printed, and the counter continues.
If a collision occurs, the attempt is voided. Reposition the car using
"2D Pose Estimate" in RViz, and the counter will reset automatically.


import csv
import math
import os
import shutil
import sys
 
import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseWithCovarianceStamped
 
 
class LapCounter(Node):
    def __init__(self):
        super().__init__('lap_counter')
 
        self.declare_parameter('odom_topic', '/ego_racecar/odom')
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('target_laps', 10)
        self.declare_parameter('min_lap_dist', 20.0)     # m recorridos minimo por vuelta
        self.declare_parameter('gate_half_width', 3.0)   # m, ancho de la meta
        self.declare_parameter('car_half_length', 0.29)  # m
        self.declare_parameter('car_half_width', 0.155)  # m
        self.declare_parameter('collision_margin', 0.03) # m
        self.declare_parameter('stuck_time', 2.0)        # s detenido = colision
        self.declare_parameter('csv_path', os.path.expanduser('~/resultados_vueltas.csv'))
 
        self.target = int(self.get_parameter('target_laps').value)
 
        # Estado
        self.gate = None
        self.need_init = True
        self.prev_xy = None
        self.prev_s = None
        self.path = 0.0
        self.lap_start = None     # tiempo (sim) de inicio de la vuelta actual
        self.now = 0.0            # ultimo tiempo (sim) recibido
        self.v = 0.0
        self.laps = []
        self.attempt = 1
        self.crashes = 0          # total de choques
        self.crashed = False
        self.finished = False
        self.stuck_since = None
        self.footprint = None
        self.odom_count = 0
        self.scan_count = 0
 
        self.create_subscription(Odometry, self.get_parameter('odom_topic').value,
                                 self.odom_cb, 10)
        self.create_subscription(LaserScan, self.get_parameter('scan_topic').value,
                                 self.scan_cb, 10)
        self.create_subscription(PoseWithCovarianceStamped, '/initialpose',
                                 self.reset_cb, 10)
        self.create_timer(0.1, self.live_cb)
 
        self.out(f'Contador listo. Objetivo: {self.target} vueltas sin colision. '
                 'La meta es la posicion inicial del auto.')
 
    # ------------------------------------------------------------------
    # Salida en pantalla
    def out(self, text):
        """Imprime una linea permanente (borra antes la linea en vivo)."""
        ancho = shutil.get_terminal_size((100, 20)).columns - 1
        sys.stdout.write('\r' + ' ' * ancho + '\r' + text + '\n')
        sys.stdout.flush()
 
    def live_cb(self):
        """Linea en vivo que se actualiza sola."""
        if self.finished:
            return
        if self.odom_count == 0:
            line = 'Esperando odometria... (ros2 topic list | grep odom)'
        elif self.crashed:
            line = 'COLISION | Reposiciona con 2D Pose Estimate'
        elif self.lap_start is None:
            line = 'Esperando que el auto cruce la meta para iniciar el cronometro...'
        else:
            actual = self.now - self.lap_start
            total = sum(self.laps) + actual
            mejor = f'{min(self.laps):.2f} s' if self.laps else '--'
            line = (f'Vuelta {len(self.laps) + 1}/{self.target} | '
                    f'Actual: {actual:6.1f} s | '
                    f'Total: {total:7.1f} s | '
                    f'Mejor: {mejor} | v={self.v:.1f}')
        # Ajustar al ancho de la terminal para que no se parta en varias lineas
        ancho = shutil.get_terminal_size((100, 20)).columns - 1
        sys.stdout.write('\r' + line[:ancho].ljust(ancho))
        sys.stdout.flush()
 
    # ------------------------------------------------------------------
    def reset_cb(self, _msg):
        if self.finished:
            return
        self.crashed = False
        self.laps = []
        self.need_init = True
        self.out(f'--- Intento {self.attempt} iniciado ---')
 
    def register_collision(self, why):
        if self.crashed or self.finished:
            return
        self.crashed = True
        self.crashes += 1
        self.out(f'COLISION ({why}). Intento {self.attempt} anulado con '
                 f'{len(self.laps)} vuelta(s) valida(s).')
        self.laps = []
        self.lap_start = None
        self.attempt += 1
 
    # ------------------------------------------------------------------
    def scan_cb(self, msg: LaserScan):
        self.scan_count += 1
        if self.crashed or self.finished or self.lap_start is None:
            return
        n = len(msg.ranges)
        if self.footprint is None or len(self.footprint) != n:
            ang = msg.angle_min + np.arange(n) * msg.angle_increment
            hx = float(self.get_parameter('car_half_length').value)
            hy = float(self.get_parameter('car_half_width').value)
            with np.errstate(divide='ignore'):
                tx = hx / np.abs(np.cos(ang))
                ty = hy / np.abs(np.sin(ang))
            self.footprint = np.minimum(tx, ty) + float(
                self.get_parameter('collision_margin').value)
        r = np.array(msg.ranges, dtype=np.float32)
        valid = np.isfinite(r) & (r > msg.range_min)
        if np.any(r[valid] < self.footprint[valid]):
            self.register_collision('LIDAR dentro del contorno del auto')
 
    # ------------------------------------------------------------------
    def odom_cb(self, msg: Odometry):
        if self.finished:
            return
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        v = msg.twist.twist.linear.x
        self.odom_count += 1
        self.now = t
        self.v = v
 
        if self.gate is None:
            self.gate = (x, y, yaw)
            self.out(f'Meta fijada en x={x:.2f}, y={y:.2f}, yaw={yaw:.2f} rad')
 
        gx, gy, gyaw = self.gate
        dx, dy = x - gx, y - gy
        s = dx * math.cos(gyaw) + dy * math.sin(gyaw)
        lat = -dx * math.sin(gyaw) + dy * math.cos(gyaw)
 
        if self.need_init:
            self.need_init = False
            self.prev_xy = (x, y)
            self.prev_s = None
            self.path = 0.0
            self.stuck_since = None
            near = math.hypot(dx, dy) < 1.5
            self.lap_start = t if near else None
            return
 
        if self.crashed:
            self.prev_xy = (x, y)
            self.prev_s = s
            return
 
        self.path += math.hypot(x - self.prev_xy[0], y - self.prev_xy[1])
        self.prev_xy = (x, y)
 
        # Auto detenido despues de haberse movido
        if self.lap_start is not None and self.path > 1.0 and abs(v) < 0.05:
            if self.stuck_since is None:
                self.stuck_since = t
            elif t - self.stuck_since > float(self.get_parameter('stuck_time').value):
                self.register_collision('auto detenido')
                return
        else:
            self.stuck_since = None
 
        # Cruce de meta
        crossed = (self.prev_s is not None and self.prev_s < 0.0 <= s and
                   abs(lat) < float(self.get_parameter('gate_half_width').value))
        self.prev_s = s
        if not crossed:
            return
 
        if self.lap_start is None:
            self.lap_start = t
            self.path = 0.0
            self.out('Cruzo la meta: cronometro iniciado.')
            return
 
        if self.path < float(self.get_parameter('min_lap_dist').value):
            return
 
        lap_time = t - self.lap_start
        self.laps.append(lap_time)
        self.lap_start = t
        self.path = 0.0
        self.out(f'>> Vuelta {len(self.laps)}/{self.target} completada: '
                 f'{lap_time:.2f} s  (mejor: {min(self.laps):.2f} s)')
 
        if len(self.laps) >= self.target:
            self.finish()
 
    # ------------------------------------------------------------------
    def finish(self):
        self.finished = True
        best = min(self.laps)
        best_idx = self.laps.index(best) + 1
        lines = [
            '',
            '=' * 50,
            f'  PRUEBA 1 SUPERADA: {self.target} vueltas sin colision',
            f'  (intento numero {self.attempt})',
            '-' * 50,
        ]
        for i, lt in enumerate(self.laps, 1):
            mark = '  <-- MEJOR' if i == best_idx else ''
            lines.append(f'  Vuelta {i:2d}: {lt:8.2f} s{mark}')
        lines += [
            '-' * 50,
            f'  PRUEBA 2 - Menor tiempo en una vuelta: {best:.2f} s (vuelta {best_idx})',
            f'  Tiempo total: {sum(self.laps):.2f} s | '
            f'Promedio: {sum(self.laps) / len(self.laps):.2f} s',
            '=' * 50,
        ]
        for ln in lines:
            self.out(ln)
 
        path = self.get_parameter('csv_path').value
        try:
            with open(path, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['vuelta', 'tiempo_s'])
                for i, lt in enumerate(self.laps, 1):
                    w.writerow([i, f'{lt:.3f}'])
                w.writerow(['mejor_vuelta', f'{best:.3f}'])
            self.out(f'Resultados guardados en {path}')
        except OSError as e:
            self.out(f'No se pudo guardar el CSV: {e}')
 
 
def main(args=None):
    rclpy.init(args=args)
    node = LapCounter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()
 
 
if __name__ == '__main__':
    main()

# proyecto- speed reating 

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
 
 
class FollowTheGap(Node):
    def __init__(self):
        super().__init__('follow_the_gap')
 
        # ---------- Parametros ajustables ----------
        self.declare_parameter('max_range', 15.0)        # m, recorte de lecturas
        self.declare_parameter('smooth_window', 5)      # beams para suavizar (entero)
        self.declare_parameter('fov_deg', 180.0)        # campo de vision usado
        self.declare_parameter('bubble_radius', 0.35)   # m, burbuja de seguridad
        self.declare_parameter('gap_threshold', 1.0)    # m, minimo para ser "libre"
        self.declare_parameter('v_max', 15.0)            # m/s con la via libre
        self.declare_parameter('v_min', 4.0)            # m/s minimo (curvas)
        self.declare_parameter('front_deg', 10.0)       # +- grados para medir al frente
        self.declare_parameter('d_brake', 0.8)          # m, distancia donde va a v_min
        self.declare_parameter('steer_gain', 1.5)       # cuanto frena por giro
        self.declare_parameter('steer_floor', 0.45)     # factor minimo por giro
        self.declare_parameter('steer_limit', 0.4189)   # rad (~24 deg)
 
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, '/drive', 10)
 
    # ------------------------------------------------------------------
    def preprocess(self, ranges):
        p = self.get_parameter
        r = np.array(ranges, dtype=np.float32)
        r[~np.isfinite(r)] = p('max_range').value
        r = np.clip(r, 0.0, p('max_range').value)
        w = int(p('smooth_window').value)
        return np.convolve(r, np.ones(w) / w, mode='same')
 
    @staticmethod
    def find_max_gap(free):
        """Devuelve (inicio, fin) del tramo contiguo mas largo de True."""
        best_len, best_start, cur = 0, 0, None
        for i, v in enumerate(free):
            if v and cur is None:
                cur = i
            if (not v or i == len(free) - 1) and cur is not None:
                end = i if v else i - 1
                if end - cur + 1 > best_len:
                    best_len, best_start = end - cur + 1, cur
                cur = None
        return best_start, best_start + best_len - 1
 
    # ------------------------------------------------------------------
    def scan_callback(self, msg: LaserScan):
        p = self.get_parameter
        ranges = self.preprocess(msg.ranges)
        n = len(ranges)
        angles = msg.angle_min + np.arange(n) * msg.angle_increment
 
        # 1) Recortar al campo de vision frontal
        half_fov = np.deg2rad(p('fov_deg').value) / 2.0
        idx = np.where(np.abs(angles) <= half_fov)[0]
        lo, hi = idx[0], idx[-1]
        r = ranges[lo:hi + 1].copy()
        a = angles[lo:hi + 1]
 
        # 2) Burbuja de seguridad alrededor del obstaculo mas cercano
        i_min = int(np.argmin(r))
        d_min = max(r[i_min], 0.05)
        bubble = int(np.arctan2(p('bubble_radius').value, d_min) / msg.angle_increment)
        r[max(0, i_min - bubble): i_min + bubble + 1] = 0.0
 
        # 3) Mayor hueco libre
        free = r > p('gap_threshold').value
        if not free.any():
            target = int(np.argmax(r))
        else:
            s, e = self.find_max_gap(free)
            # 4) Mejor punto: mezcla entre el punto mas lejano y el centro del hueco
            far = s + int(np.argmax(r[s:e + 1]))
            center = (s + e) // 2
            target = int(0.5 * far + 0.5 * center)
 
        lim = p('steer_limit').value
        steer = float(np.clip(a[target], -lim, lim))
 
        # 5) Velocidad adaptativa
        # Distancia libre al frente (+-front_deg grados), sin la burbuja
        front = np.abs(a) < np.deg2rad(p('front_deg').value)
        d_front = float(np.min(ranges[lo:hi + 1][front])) if front.any() else 0.0
        abs_s = abs(steer)
 
        # Mas lejos libre -> mas rapido; mas giro -> mas lento
        d_brake = p('d_brake').value
        k = np.clip((d_front - d_brake) / (p('max_range').value - d_brake), 0.0, 1.0)
        speed = p('v_min').value + k * (p('v_max').value - p('v_min').value)
        speed *= max(p('steer_floor').value, 1.0 - p('steer_gain').value * abs_s)
        speed = max(speed, p('v_min').value)
 
        cmd = AckermannDriveStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.drive.steering_angle = steer
        cmd.drive.speed = float(speed)
        self.drive_pub.publish(cmd)
 
 
def main(args=None):
    rclpy.init(args=args)
    node = FollowTheGap()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()
 
 
if __name__ == '__main__':
    main()

"""
"Follow the Gap" for TWO cars in the F1Tenth simulator (ROS 2 Humble).

 - Main car (ego):  /scan     -> /drive      (odom: /ego_racecar/odom)
 - Second car (opp):    /opp_scan -> /opp_drive  (odom: /opp_racecar/odom)

A single node controls both cars (the simulator requires both to receive commands simultaneously when two cars are present).

PREVENTING COLLISIONS AND GOING OFF-TRACK
  When both cars are SIDE-BY-SIDE (at the start or while one is overtaking the other),
  each car follows its respective OUTER WALL at a set distance (similar to the
  "Wall Follower" lab). This keeps them in parallel lanes: neither car moves
  toward the other or toward the center, and the LIDAR data from the wall
  keeps them on the track, even if the starting straight isn't perfectly linear.
  If an obstacle or a sharp turn appears ahead, or if one car moves clear
  (one ahead of the other), the system switches to "Follow the Gap."

  Additionally, there are three safety mechanisms that apply in ALL modes:
   - Lane maintenance: When an opponent is alongside, the system prevents the
     steering from turning toward them and creates space if the cars get too close.
     This logic is applied AFTER the anti-wall-collision check to ensure it
     isn't overridden.
   - Inner wall: The wall-following logic also monitors the inner side,
     pushing the car toward the outer wall if it gets too close to the inner boundary.
   - Opponent in the gap: If an opponent is directly ahead, their angular
     range is excluded so that the chosen "gap" does not point toward them.
  The car behind reduces its speed.

STARTUP
  1) The node verifies that the cars are positioned side-by-side and facing
     the same direction. If they are not, it prevents them from starting and
     provides instructions on what to do. 2) Parallel departure: same speed (`salida_speed`) until a distance of `salida_dist` meters has been covered.
3) Overtaking: the second car passes the lead car on the side with more available space.

Parameters use the prefix `ego_` or `opp_`, for example:
  python3 follow_the_gap_2autos.py --ros-args -p ego_v_max:=18.4 -p opp_v_max:=23.0
Output: `salida_dist` (m), `salida_speed` (m/s).
(use a decimal point for non-integers: 8.0, not 8)
"""

import math
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from ackermann_msgs.msg import AckermannDriveStamped

# ---------- Parametros (los de tu codigo, con velocidades mas seguras) ----------
DEFAULTS = {
    'max_range': 15.0,
    'smooth_window': 5,
    'fov_deg': 180.0,
    'bubble_radius': 0.35,
    'gap_threshold': 1.0,
    'v_max': 18.4,
    'v_min': 1.0,
    'a_long': 9.0,
    'front_deg': 10.0,
    'd_brake': 0.8,
    'steer_gain': 0.8,
    'steer_floor': 0.45,
    'steer_limit': 0.4189,
    'steer_alpha': 0.65,
    # Adelantamiento (solo lo usa el segundo auto)
    'v_boost': 2.0,
    'overtake_dist': 12.0,
    'overtake_lane': 1.2,
    'rival_margin': 0.5,
    'min_space': 0.9,
    'straight_steer': 0.2,
}
OVERRIDES = {'opp': {'v_max': 23.0, 'a_long': 11.0}}


def wrap(ang):
    return math.atan2(math.sin(ang), math.cos(ang))


class Piloto:
    """Logica de un auto. 'es_opp' activa el adelantamiento."""

    def __init__(self, node, prefijo, es_opp):
        self.node = node
        self.prefijo = prefijo
        self.es_opp = es_opp
        self.reiniciar()

    def reiniciar(self):
        self.prev_steer = 0.0
        self.wall_side = None
        self.mode = 'seguir'
        self.side = 0
        self.d0 = None
        self.junto_activo = False

    def p(self, k):
        """Parametro con prefijo. Si no esta declarado con prefijo, usa el global."""
        nombre = f'{self.prefijo}_{k}'
        if self.node.has_parameter(nombre):
            return self.node.get_parameter(nombre).value
        return self.node.get_parameter(k).value

    def n(self, k):
        return self.node.get_parameter(k).value

    # ------------------------------------------------------------------
    def preprocess(self, ranges):
        r = np.array(ranges, dtype=np.float32)
        r[np.isposinf(r)] = self.p('max_range')
        r[~np.isfinite(r)] = 0.0
        r = np.clip(r, 0.0, self.p('max_range'))
        if not r.size:
            return r
        w = min(r.size, max(1, int(self.p('smooth_window'))))
        w = max(1, w if w % 2 else w - 1)
        pad = w // 2
        return np.minimum(r, np.convolve(np.pad(r, pad, mode='edge'), np.ones(w) / w, mode='valid'))

    @staticmethod
    def find_max_gap(free):
        """Devuelve (inicio, fin) del hueco mas largo, o None si no hay."""
        best, cur = None, None
        for i, v in enumerate(free):
            if v and cur is None:
                cur = i
            if cur is not None and (not v or i == len(free) - 1):
                end = i if v else i - 1
                if end >= cur and (best is None or end - cur > best[1] - best[0]):
                    best = (cur, end)
                cur = None
        return best

    @staticmethod
    def dist_lateral(rng, a, grados):
        w = np.abs(a - np.deg2rad(grados)) < np.deg2rad(5.0)
        return float(np.min(rng[w])) if w.any() else 0.0

    @staticmethod
    def rango_en(rng, a, grados):
        """Distancia minima en el angulo pedido (ventana de 6 grados).
        None si el LiDAR no cubre ese angulo; nunca lanza excepcion."""
        obj = np.deg2rad(grados)
        i = int(np.argmin(np.abs(a - obj)))
        if abs(a[i] - obj) > np.deg2rad(12.0):
            return None
        w = np.abs(a - obj) <= np.deg2rad(6.0)
        return float(np.min(rng[w])) if w.any() else float(rng[i])

    def mascara(self, r, a, rel, lado):
        """Prohibe los rayos del lado del rival. lado<0: pasar por la derecha.
        El rumbo usa un adelanto minimo de 0.6 m para que, con los autos lado a
        lado, no se anule el scan completo."""
        dx_f, dy_l = rel[0], rel[1]
        th_r = math.atan2(dy_l, max(dx_f, 0.6))
        mitad = min(math.atan2(self.n('car_half_width') + self.p('rival_margin'),
                              max(dx_f, 0.3)),
                    np.deg2rad(80.0))
        m = r.copy()
        if lado < 0:
            m[a > th_r - mitad] = 0.0
        else:
            m[a < th_r + mitad] = 0.0
        return m

    # ------------------------------------------------------------------
    def junto(self, rel):
        """True si el otro auto va a mi lado (con histeresis)."""
        if rel is None:
            self.junto_activo = False
            return False
        dx, dy = abs(rel[0]), abs(rel[1])
        if self.junto_activo:
            self.junto_activo = dx < 1.8 and dy < 2.6
        else:
            self.junto_activo = dx < 1.2 and dy < 2.0
        return self.junto_activo

    def pared_control(self, rng, a, rel):
        """Sigue la pared EXTERIOR a la distancia inicial (wall follower) y
        empuja lejos de la pared INTERIOR si se acerca.
        Devuelve None si el LiDAR no cubre el costado elegido."""
        th = math.radians(45.0)
        if self.wall_side is None:
            self.wall_side = -1 if (rel is None or rel[1] > 0) else +1
        lado = self.wall_side
        b = self.rango_en(rng, a, lado * 80.0)
        aa = self.rango_en(rng, a, lado * 35.0)
        if b is None or aa is None:
            return None
        psi = math.atan2(aa * math.cos(th) - b, aa * math.sin(th))
        dist = b * math.cos(psi)
        if self.d0 is None:
            self.d0 = min(dist, 4.0)
        d_deseada = max(self.d0, self.n('pared_min'))
        dist_proy = dist + self.n('salida_L') * math.sin(psi)
        err = d_deseada - dist_proy
        kp = self.n('salida_kp')
        steer = kp * err if lado < 0 else -kp * err
        # Pared interior: si esta mas cerca que pared_min, empujar a la exterior.
        bi = self.rango_en(rng, a, -lado * 80.0)
        if bi is not None and bi < self.n('pared_min'):
            steer += lado * self.n('pared_kp') * (self.n('pared_min') - bi)
        lim = self.p('steer_limit')
        return float(np.clip(steer, -lim, lim))

    def carril(self, steer, rel, abrir=True):
        """Guardia lateral entre los autos. Nunca encajona el volante hacia el
        rival y, si el hueco es menor que lane_min, gira para abrirlo (acotado
        a lane_tope). Con abrir=False solo aplica el tope."""
        if rel is None:
            return steer
        dx, dy = rel[0], rel[1]
        if abs(dx) > self.n('lane_dist') or abs(dy) > self.p('overtake_lane') + 0.3:
            return steer
        lado = 1 if dy > 0 else -1
        lim = self.p('steer_limit')
        # 1) Tope de giro hacia el rival cuando ya estamos cerca.
        if steer * lado > 0 and abs(dy) < self.n('sep_lateral'):
            tope = self.n('sep_max_giro')
            if abs(steer) > tope:
                steer = math.copysign(tope, steer)
        if not abrir:
            return steer
        # 2) Abrir hueco si estamos mas cerca que el carril minimo.
        err = self.n('lane_min') - abs(dy)
        if err > 0.0:
            corr = min(self.n('lane_kp') * err, self.n('lane_tope'))
            steer = float(np.clip(steer - lado * corr, -lim, lim))
        return steer

    def s_libre(self, xs, ys, steer, d_max):
        """Distancia de avance comprobada, en marco del eje trasero.
        Contorno rectangular barrido; nunca extrapola un arco no evaluado.
        """
        k = math.tan(steer) / self.n('wheelbase')
        horizon = min(d_max, 1.5 / abs(k)) if abs(k) > 1e-5 else d_max
        # Paso adaptativo: no gastar 250 muestras por candidato en recta.
        ds = max(0.10, horizon / 80.0)
        front = self.n('car_front')
        rear = self.n('car_rear')
        half = self.n('car_half_width')
        margin = self.n('guardia_margen')
        for d in np.linspace(0.0, horizon, max(2, int(math.ceil(horizon / ds)) + 1)):
            th = k * d
            x = math.sin(th) / k if abs(k) > 1e-5 else d
            y = (1 - math.cos(th)) / k if abs(k) > 1e-5 else 0.0
            c, z = math.cos(th), math.sin(th)
            xx, yy = xs - x, ys - y
            u, v = c * xx + z * yy, -z * xx + c * yy
            m = 0.025 + margin * min(d / 0.7, 1.0)
            if np.any((u >= -rear - m) & (u <= front + m) & (np.abs(v) <= half + m)):
                return max(0.0, float(d) - ds)
        return float(horizon)

    def guardia(self, msg, steer, speed, boost=False):
        if not self.n('guardia'):
            return steer, speed
        pose = self.node.poses[self.prefijo]
        if pose is None:
            return steer, 0.0
        r = np.asarray(msg.ranges, dtype=float)
        ang = msg.angle_min + np.arange(r.size) * msg.angle_increment
        valid = np.isfinite(r) & (r >= msg.range_min) & (r < msg.range_max)
        known = valid | np.isposinf(r) | (np.isfinite(r) & (r >= msg.range_max))
        if not r.size or np.mean(known) < 0.95:
            return steer, 0.0
        xs = r[valid] * np.cos(ang[valid]) + self.n('lidar_x')
        ys = r[valid] * np.sin(ang[valid])
        horizon = min(self.n('guardia_max'), float(msg.range_max))
        if not math.isfinite(horizon) or horizon <= 0.8:
            return steer, 0.0
        gf = self.n('guardia_frenada')
        reaction = self.n('reaccion')
        actual = abs(pose[3])
        lim = self.p('steer_limit')
        top = self.p('v_max') * self.node.factor(self.prefijo)
        if boost:
            top += self.p('v_boost')
        if self.node.salida:
            top = min(top, self.n('salida_speed'))
        top = min(top, max(0.0, speed))
        paso = max(0.01, float(self.n('guardia_paso')))
        cuantos = int(np.clip(round(2 * lim / paso) + 1, 9, 61))
        candidates = np.unique(np.r_[steer, np.linspace(-lim, lim, cuantos)])
        best = (-math.inf, steer, 0.0)
        for c in candidates:
            d = self.s_libre(xs, ys, float(c), horizon)
            available = max(0.0, d - self.n('d_seguro'))
            brake = math.sqrt((gf * reaction) ** 2 + 2 * gf * available) - gf * reaction
            lat = math.sqrt(self.n('a_lat') * self.n('wheelbase') /
                            max(abs(math.tan(c)), abs(math.tan(steer)), 1e-4))
            v = min(top, brake, lat)
            score = min(v, 5.0) - 8.0 * abs(c - steer) - 0.4 * abs(c - self.prev_steer)
            if v <= 0.0:
                score -= 20.0
            if score > best[0]:
                best = (score, float(c), float(v))
        _, chosen, v = best
        ca, sa = np.cos(ang), np.sin(ang)
        with np.errstate(divide='ignore', invalid='ignore'):
            longitudinal = np.where(ca > 0, (self.n('car_front') - self.n('lidar_x')) / ca,
                                    (-self.n('car_rear') - self.n('lidar_x')) / ca)
            lateral = self.n('car_half_width') / np.abs(sa)
        edge = np.minimum(longitudinal, lateral)
        mask = valid & (ca > 0.15)
        if mask.any():
            gap = np.maximum(r[mask] - edge[mask] - 0.025, 0.0)
            v = min(v, float(np.min(gap / (self.n('ttc_min') * ca[mask]))))
        v = min(v, actual + self.p('a_long') * self.n('periodo_control'))
        self.prev_steer = chosen
        return chosen, max(0.0, v)

    def separar(self, speed, rel):
        """Ceder el paso: si el rival va delante y casi en mi carril, no
        embestirlo. Si ademas el hueco lateral es estrecho, quedarse atras."""
        if rel is None:
            return speed
        dx, dy = rel[0], rel[1]
        half = self.n('car_half_width')
        ext = self.n('car_front') + self.n('car_rear')
        if math.hypot(dx, dy) > self.n('sep_dist') or abs(dx) > ext + 1.0:
            return speed
        if dx > 0.35 and abs(dy) < 2 * half + 0.7:
            factor = self.n('sep_freno') if abs(dy) < self.n('sep_lateral') else 1.0
            speed = min(speed, rel[2] * factor)
        return max(speed, 0.0)

    def velocidad(self, rng, a, steer, boost):
        """Formula de velocidad adaptativa + limites de seguridad.
        'boost' aplica v_boost solo cuando este auto esta rebasando.
        """
        v_top = self.p('v_max') * self.node.factor(self.prefijo)
        if boost:
            v_top += self.p('v_boost')
        speed = v_top * max(self.p('steer_floor'), 1.0 - self.p('steer_gain') * abs(steer))
        speed = max(speed, min(self.p('v_min'), v_top))
        tan_s = max(abs(math.tan(steer)), 1e-3)
        v_lat = math.sqrt(self.n('a_lat') * self.n('wheelbase') / tan_s)
        if not self.n('guardia'):
            front = np.abs(a) < np.deg2rad(self.p('front_deg'))
            d = float(np.min(rng[front])) if front.any() else 0.0
            speed = min(speed, math.sqrt(2 * self.n('a_frenado') *
                         max(d - self.n('d_seguro'), 0.0)))
            if d < self.p('d_brake'):
                speed = 0.0
        speed = min(speed, v_lat)
        speed = max(speed, min(self.n('v_floor'), v_lat))
        return float(max(0.0, speed))

    # ------------------------------------------------------------------
    def actualizar_modo(self, rel, rng, a):
        if rel is None or self.node.fase != 2:
            self.mode = 'seguir'
            return
        dx_f, dy_l = rel[0], rel[1]
        if self.mode == 'seguir':
            if (0.2 < dx_f < self.p('overtake_dist')
                    and abs(rel[3]) < 0.35
                    and abs(dy_l) < self.p('overtake_lane')
                    and abs(self.prev_steer) < self.p('straight_steer')):
                izq = self.dist_lateral(rng, a, +90.0)
                der = self.dist_lateral(rng, a, -90.0)
                half = self.n('car_half_width')
                esp_izq = izq - (dy_l + half)
                esp_der = der + (dy_l - half)
                lado, espacio = (+1, esp_izq) if esp_izq > esp_der else (-1, esp_der)
                if espacio >= self.p('min_space'):
                    self.mode, self.side = 'adelantar', lado
                    self.node.get_logger().info(
                        'Segundo auto adelantando por la '
                        + ('izquierda' if lado > 0 else 'derecha'))
        else:
            if dx_f < -1.5 or dx_f > self.p('overtake_dist') + 2.0 or abs(rel[3]) > 0.8:
                self.mode = 'seguir'
                self.node.get_logger().info(
                    'Adelanto completado' if dx_f < -1.5 else 'Adelanto cancelado')

    def burbuja_rival(self, r, a, rel, inc):
        """Anula la franja angular del rival para que el hueco elegido no
        apunte al otro auto. Solo si el rival esta DELANTE: si va detras, su
        rumbo cae cerca de 90 grados y anularia la pared lateral."""
        if rel is None or not (0.0 < rel[0] < 3.0):
            return
        dist = max(math.hypot(rel[0], rel[1]), 0.3)
        th_r = math.atan2(rel[1], rel[0])
        j = int(np.argmin(np.abs(a - th_r)))
        if abs(a[j] - th_r) > np.deg2rad(15.0):
            return
        rad = max(0.35, 0.9 - 0.3 * rel[0])
        k = int(np.arctan2(rad, dist) / inc)
        r[max(0, j - k): j + k + 1] = 0.0

    # ------------------------------------------------------------------
    def paso(self, msg, rel, salida):
        """Devuelve (steer, speed).
        rel = (adelante, izquierda, v_rival, dyaw) o None."""
        if (len(msg.ranges) == 0 or not math.isfinite(msg.angle_increment)
                or msg.angle_increment <= 0 or not math.isfinite(msg.angle_min)):
            return 0.0, 0.0
        pose = self.node.poses[self.prefijo]
        if pose is None:
            return 0.0, 0.0
        ranges = self.preprocess(msg.ranges)
        n = len(ranges)
        angles = msg.angle_min + np.arange(n) * msg.angle_increment

        # 1) Campo de vision frontal
        half_fov = np.deg2rad(self.p('fov_deg')) / 2.0
        idx = np.where(np.abs(angles) <= half_fov)[0]
        if idx.size == 0:
            return 0.0, 0.0
        lo, hi = idx[0], idx[-1]
        rng = ranges[lo:hi + 1]
        a = angles[lo:hi + 1]

        # 2) Carriles paralelos: en la salida y mientras el otro va a mi lado
        junto = self.junto(rel)
        paralelo = salida or junto
        if paralelo:
            frente = np.abs(a) < np.deg2rad(12.0)
            if frente.any() and float(np.min(rng[frente])) < self.n('salida_frente'):
                if salida:
                    # Se aplica al final del ciclo, no aqui (ver control_cb).
                    self.node.motivo_fin_salida = 'hay un obstaculo o curva cerca'
                paralelo = False
        if paralelo:
            if self.es_opp:
                self.actualizar_modo(rel, ranges, angles)
            steer = self.pared_control(ranges, angles, rel)
            if steer is None:
                # Sin lectura de costado: este auto cae a Follow the Gap, que ya
                # lleva la guardia lateral activa.
                self.d0 = None
                self.wall_side = None
                paralelo = False
            else:
                speed = (float(self.n('salida_speed')) if salida
                         else self.velocidad(rng, a, steer, False))
                speed = self.separar(speed, rel)
                steer = self.carril(steer, rel)
                steer, speed = self.guardia(msg, steer, speed, False)
                steer = self.carril(steer, rel, abrir=False)
                self.prev_steer = steer
                return steer, speed
        self.d0 = None
        self.wall_side = None

        # 3) Follow the Gap: burbuja de seguridad en el obstaculo mas cercano
        r = rng.copy()
        i_min = int(np.argmin(np.where(r > 0.0, r, np.inf)))
        d_min = max(r[i_min], 0.05)
        bubble = int(np.arctan2(self.p('bubble_radius'), d_min) / msg.angle_increment)
        r[max(0, i_min - bubble): i_min + bubble + 1] = 0.0
        # Burbuja extra sobre el rival (solo si va delante).
        self.burbuja_rival(r, a, rel, msg.angle_increment)

        # 4) Adelantamiento (solo el segundo auto)
        r_gap = r
        if self.es_opp:
            self.actualizar_modo(rel, ranges, angles)
            if self.mode == 'adelantar' and rel is not None:
                r_gap = self.mascara(r, a, rel, self.side)

        # 5) Mayor hueco libre
        free = r_gap > self.p('gap_threshold')
        hueco = self.find_max_gap(free) if free.any() else None
        if hueco is None and self.mode == 'adelantar':
            self.mode = 'seguir'
            r_gap = r
            free = r_gap > self.p('gap_threshold')
            hueco = self.find_max_gap(free) if free.any() else None
        if hueco is None:
            self.mode = 'seguir'
            if float(np.max(rng)) <= 0.0:
                return 0.0, 0.0
            target = int(np.argmax(rng))
        else:
            s, e = hueco
            center = (s + e) // 2
            if self.es_opp:
                seg = r_gap[s:e + 1]
                best = np.where(seg >= seg.max() - 0.3)[0]
                far = s + int(np.median(best))
            else:
                far = s + int(np.argmax(r_gap[s:e + 1]))
            target = int(0.5 * far + 0.5 * center)

        lim = self.p('steer_limit')
        look = max(0.8, min(float(rng[target]), 1.2 + 0.25 * abs(pose[3])))
        steer = float(np.clip(math.atan2(2 * self.n('wheelbase') * math.sin(a[target]), look),
                              -lim, lim))
        alpha = self.p('steer_alpha')
        steer = alpha * steer + (1.0 - alpha) * self.prev_steer

        # 6) Velocidad adaptativa y guardias (el tope lateral va al final).
        boost = self.mode == 'adelantar'
        speed = self.velocidad(rng, a, steer, boost)
        speed = self.separar(speed, rel)
        steer = self.carril(steer, rel)
        steer, speed = self.guardia(msg, steer, speed, boost)
        steer = self.carril(steer, rel, abrir=False)
        self.prev_steer = steer
        return steer, speed


class FollowTheGap2Autos(Node):
    def __init__(self):
        super().__init__('follow_the_gap_2autos')
        for pref in ('ego', 'opp'):
            for k, v in DEFAULTS.items():
                v = OVERRIDES.get(pref, {}).get(k, v)
                self.declare_parameter(f'{pref}_{k}', v)
        self.declare_parameter('salida_dist', 5.0)
        self.declare_parameter('salida_speed', 6.0)
        self.declare_parameter('salida_frente', 3.5)
        self.declare_parameter('salida_kp', 1.0)
        self.declare_parameter('salida_L', 1.0)
        self.declare_parameter('exigir_alineacion', True)
        self.declare_parameter('alinear', False)
        self.declare_parameter('separacion', 0.8)
        # Seguridad para no salirse de la pista
        self.declare_parameter('a_lat', 7.5)
        self.declare_parameter('wheelbase', 0.33)
        self.declare_parameter('v_floor', 0.0)
        self.declare_parameter('a_frenado', 8.0)
        self.declare_parameter('d_seguro', 0.8)
        self.declare_parameter('parar_al_chocar', True)
        self.declare_parameter('choque_liberacion', 10)   # ciclos sin contacto para reanudar
        self.declare_parameter('choque_confirma', 2)      # ciclos seguidos para dar el golpe por real
        # Estrategia de carrera
        self.declare_parameter('fase1_dist', 12.0)
        self.declare_parameter('opp_f1', 0.75)
        self.declare_parameter('ego_f2', 1.0)
        # Guardia anti-pared
        self.declare_parameter('lidar_x', 0.275)
        self.declare_parameter('car_front', 0.455)
        self.declare_parameter('car_rear', 0.125)
        self.declare_parameter('car_half_width', 0.155)
        self.declare_parameter('reaccion', 0.30)
        self.declare_parameter('ttc_min', 0.30)
        self.declare_parameter('guardia', True)
        self.declare_parameter('guardia_margen', 0.12)
        self.declare_parameter('guardia_frenada', 6.0)
        self.declare_parameter('guardia_max', 25.0)
        self.declare_parameter('pared_min', 0.5)
        self.declare_parameter('pared_kp', 2.0)     # empuje lejos de la pared interior
        self.declare_parameter('guardia_paso', 0.04)
        # Regla anti-choque entre los dos autos
        self.declare_parameter('sep_dist', 2.0)
        self.declare_parameter('sep_freno', 0.85)
        self.declare_parameter('sep_lateral', 0.75)
        self.declare_parameter('sep_max_giro', 0.12)
        self.declare_parameter('lane_min', 0.7)     # m de hueco lateral minimo
        self.declare_parameter('lane_kp', 1.2)      # ganancia para abrir hueco
        self.declare_parameter('lane_tope', 0.25)   # rad maximos de empuje lateral
        self.declare_parameter('lane_dist', 1.6)    # m de desfase en que aplica

        self.pilotos = {'ego': Piloto(self, 'ego', False),
                        'opp': Piloto(self, 'opp', True)}
        self.poses = {'ego': None, 'opp': None}
        self.dist = {'ego': 0.0, 'opp': 0.0}
        self.salida = True
        self.listo = False
        self.intentos = 0
        self.avisos = 0
        self.detenido = {'ego': False, 'opp': False}
        self.libre_ciclos = {'ego': 0, 'opp': 0}
        self.golpes = {'ego': 0, 'opp': 0}
        self.motivo_fin_salida = None
        self.fase = 1
        self.scans = {'ego': None, 'opp': None}
        self.scan_times = {'ego': -math.inf, 'opp': -math.inf}
        self.pose_times = self.scan_times.copy()
        self.declare_parameter('timeout_datos', 0.5)
        self.declare_parameter('periodo_control', 0.05)

        self.pubs = {
            'ego': self.create_publisher(AckermannDriveStamped, '/drive', 10),
            'opp': self.create_publisher(AckermannDriveStamped, '/opp_drive', 10)}
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)

        self.create_subscription(Odometry, '/ego_racecar/odom',
                                 lambda m: self.guardar_pose('ego', m), qos_profile_sensor_data)
        self.create_subscription(Odometry, '/opp_racecar/odom',
                                 lambda m: self.guardar_pose('opp', m), qos_profile_sensor_data)
        self.create_subscription(LaserScan, '/scan',
                                 lambda m: self.scan_cb('ego', m), qos_profile_sensor_data)
        self.create_subscription(LaserScan, '/opp_scan',
                                 lambda m: self.scan_cb('opp', m), qos_profile_sensor_data)
        self.create_subscription(PoseWithCovarianceStamped, '/initialpose',
                                 self.reiniciar_arranque, 10)
        for k in ('wheelbase', 'a_lat', 'a_frenado', 'guardia_frenada', 'guardia_paso',
                  'timeout_datos', 'periodo_control', 'choque_liberacion',
                  'choque_confirma', 'lane_dist', 'lane_min', 'lane_kp', 'lane_tope',
                  'pared_kp'):
            if not math.isfinite(self.get_parameter(k).value) or self.get_parameter(k).value <= 0:
                raise ValueError(f'Parametro {k} debe ser positivo y finito')
        if not (0 <= self.get_parameter('lidar_x').value < self.get_parameter('car_front').value):
            raise ValueError('lidar_x debe estar dentro del contorno longitudinal')
        for k in ('car_front', 'car_rear', 'car_half_width', 'reaccion', 'ttc_min'):
            if not math.isfinite(self.get_parameter(k).value) or self.get_parameter(k).value <= 0:
                raise ValueError(f'Parametro {k} invalido')
        for pref in self.pilotos:
            pil = self.pilotos[pref]
            if not (0 <= pil.p('d_brake') < pil.p('max_range')
                    and 0 < pil.p('steer_limit') < math.pi / 2
                    and 0 <= pil.p('steer_alpha') <= 1
                    and 0 < pil.p('fov_deg') <= 360
                    and 0 <= pil.p('v_min') <= pil.p('v_max')
                    and math.isfinite(pil.p('a_long')) and pil.p('a_long') > 0):
                raise ValueError(f'Parametros invalidos para {pref}')
        self.create_timer(1.0, self.verificar_cb)
        self.create_timer(self.get_parameter('periodo_control').value, self.control_cb)
        self.get_logger().info('Esperando la odometria de los dos autos...')

    # ------------------------------------------------------------------
    def reiniciar_arranque(self, _msg):
        self.scans = dict.fromkeys(self.pilotos)
        self.scan_times = dict.fromkeys(self.pilotos, -math.inf)
        self.pose_times = self.scan_times.copy()
        self.poses = {'ego': None, 'opp': None}
        self.dist = {'ego': 0.0, 'opp': 0.0}
        self.salida = True
        self.listo = False
        self.intentos = 0
        self.avisos = 0
        self.detenido = {'ego': False, 'opp': False}
        self.libre_ciclos = {'ego': 0, 'opp': 0}
        self.golpes = {'ego': 0, 'opp': 0}
        self.motivo_fin_salida = None
        self.fase = 1
        for pil in self.pilotos.values():
            pil.reiniciar()
        self.get_logger().info('Reposicionado: se repite la verificacion y la salida.')

    def terminar_salida(self, motivo):
        if self.salida:
            self.salida = False
            self.motivo_fin_salida = None
            self.get_logger().info(f'Salida terminada ({motivo}): carrera libre.')

    def verificar_cb(self):
        if self.listo:
            return
        now = time.monotonic()
        timeout = self.get_parameter('timeout_datos').value
        if any(self.poses[k] is None or now - self.pose_times[k] > timeout for k in self.pilotos):
            return
        adelante, izquierda, _, dyaw = self.relativo('ego')
        bien = (abs(dyaw) < 0.35 and abs(adelante) < 0.8
                and 0.55 < abs(izquierda) < 1.8)
        if bien:
            self.listo = True
            self.get_logger().info"Follow the Gap" for TWO cars in the F1Tenth simulator (ROS 2 Humble).

 - Main car (ego):  /scan     -> /drive      (odom: /ego_racecar/odom)
 - Second car (opp):    /opp_scan -> /opp_drive  (odom: /opp_racecar/odom)

A single node controls both cars (the simulator requires both to receive commands simultaneously when two cars are present).

PREVENTING COLLISIONS AND GOING OFF-TRACK
  When both cars are SIDE-BY-SIDE (at the start or while one is overtaking the other),
  each car follows its respective OUTER WALL at a set distance (similar to the
  "Wall Follower" lab). This keeps them in parallel lanes: neither car moves
  toward the other or toward the center, and the LIDAR data from the wall
  keeps them on the track, even if the starting straight isn't perfectly linear.
  If an obstacle or a sharp turn appears ahead, or if one car moves clear
  (one ahead of the other), the system switches to "Follow the Gap."

  Additionally, there are three safety mechanisms that apply in ALL modes:
   - Lane maintenance: When an opponent is alongside, the system prevents the
     steering from turning toward them and creates space if the cars get too close.
     This logic is applied AFTER the anti-wall-collision check to ensure it
     isn't overridden.
   - Inner wall: The wall-following logic also monitors the inner side,
     pushing the car toward the outer wall if it gets too close to the inner boundary.
   - Opponent in the gap: If an opponent is directly ahead, their angular
     range is excluded so that the chosen "gap" does not point toward them.
  The car behind reduces its speed.

STARTUP
  1) The node verifies that the cars are positioned side-by-side and facing
     the same direction. If they are not, it prevents them from starting and
     provides instructions on what to do. 2) Parallel departure: same speed (`salida_speed`) until a distance of `salida_dist` meters has been covered.
3) Overtaking: the second car passes the lead car on the side with more available space.

Parameters use the prefix `ego_` or `opp_`, for example:
  python3 follow_the_gap_2autos.py --ros-args -p ego_v_max:=18.4 -p opp_v_max:=23.0
Output: `salida_dist` (m), `salida_speed` (m/s).
(use a decimal point for non-integers: 8.0, not 8)('Autos alineados: salen juntos y en paralelo.')
            return

        mensaje = (f'Los autos NO estan bien puestos: giro relativo {math.degrees(dyaw):.0f} '
                   f'grados, adelante {adelante:.1f} m, a la izquierda {izquierda:.1f} m. '
                   'Deben estar lado a lado (0.55 a 1.8 m) y mirando al mismo lado.')

        if self.get_parameter('alinear').value and self.intentos < 3:
            sep = float(self.get_parameter('separacion').value)
            x, y, yaw, _ = self.poses['ego']
            msg = PoseStamped()
            msg.header.frame_id = 'map'
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.pose.position.x = x - sep * math.sin(yaw)
            msg.pose.position.y = y + sep * math.cos(yaw)
            msg.pose.orientation.z = math.sin(yaw / 2.0)
            msg.pose.orientation.w = math.cos(yaw / 2.0)
            self.goal_pub.publish(msg)
            self.intentos += 1
            self.get_logger().info(f'Recolocando al oponente (intento {self.intentos}/3)...')
            self.poses = {'ego': None, 'opp': None}
            return

        if self.get_parameter('exigir_alineacion').value:
            if self.avisos % 5 == 0:
                self.get_logger().warn(
                    mensaje + ' NO arrancan. Soluciona con: '
                    'python3 ~/guardar_poses.py lado aplicar  y reinicia el simulador.')
            self.avisos += 1
            return

        self.get_logger().warn(mensaje + ' Arrancan de todos modos.')
        self.listo = True
        self.terminar_salida('arranque sin alineacion')

    # ------------------------------------------------------------------
    def guardar_pose(self, quien, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        if not all(math.isfinite(v) for v in (p.x, p.y, yaw, msg.twist.twist.linear.x)):
            self.pose_times[quien] = -math.inf
            return
        self.pose_times[quien] = time.monotonic()
        prev = self.poses[quien]
        if prev is not None and self.listo:
            self.dist[quien] += math.hypot(p.x - prev[0], p.y - prev[1])
        self.poses[quien] = (p.x, p.y, yaw, msg.twist.twist.linear.x)
        if self.listo and self.salida and \
                min(self.dist.values()) >= self.get_parameter('salida_dist').value:
            self.terminar_salida('recorrio la distancia de salida')
        if (self.listo and not self.salida and self.fase == 1 and self.dist['ego']
                >= self.get_parameter('salida_dist').value
                + self.get_parameter('fase1_dist').value):
            self.fase = 2
            self.get_logger().info(
                'Fase 2: el oponente acelera para rebasar al auto principal.')

    def relativo(self, quien):
        """Rival en el marco del auto 'quien':
        (adelante, izquierda, v_rival, giro relativo del rival)."""
        otro = 'opp' if quien == 'ego' else 'ego'
        mi, rv = self.poses[quien], self.poses[otro]
        if mi is None or rv is None:
            return None
        dx, dy = rv[0] - mi[0], rv[1] - mi[1]
        adelante = math.cos(mi[2]) * dx + math.sin(mi[2]) * dy
        izquierda = -math.sin(mi[2]) * dx + math.cos(mi[2]) * dy
        return adelante, izquierda, rv[3], wrap(rv[2] - mi[2])

    def factor(self, quien):
        if self.fase == 1:
            return 1.0 if quien == 'ego' else float(self.get_parameter('opp_f1').value)
        return float(self.get_parameter('ego_f2').value) if quien == 'ego' else 1.0

    def choque(self, msg):
        """True si el LIDAR ve algo DENTRO del contorno del auto (pared tocada)."""
        r = np.asarray(msg.ranges, dtype=float)
        ang = msg.angle_min + np.arange(r.size) * msg.angle_increment
        valid = np.isfinite(r) & (r >= msg.range_min) & (r < msg.range_max)
        x = r[valid] * np.cos(ang[valid]) + self.get_parameter('lidar_x').value
        y = r[valid] * np.sin(ang[valid])
        inside = ((x > -self.get_parameter('car_rear').value)
                  & (x < self.get_parameter('car_front').value)
                  & (np.abs(y) < self.get_parameter('car_half_width').value))
        if not inside.any():
            return None
        i = np.flatnonzero(valid)[np.flatnonzero(inside)[0]]
        return float(ang[i]), float(r[i])

    def publicar(self, quien, steer, speed):
        cmd = AckermannDriveStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.drive.steering_angle = float(steer)
        cmd.drive.speed = float(speed)
        pose = self.poses[quien]
        actual = abs(pose[3]) if pose is not None else 0.0
        cmd.drive.acceleration = float(self.pilotos[quien].p('a_long')
                                       if speed > actual else self.get_parameter('a_frenado').value)
        self.pubs[quien].publish(cmd)

    def scan_cb(self, quien, msg):
        self.scans[quien] = msg
        self.scan_times[quien] = time.monotonic()

    def control_cb(self):
        now = time.monotonic()
        timeout = self.get_parameter('timeout_datos').value
        fresh = all(self.scans[k] is not None and self.poses[k] is not None
                    and now - self.scan_times[k] <= timeout
                    and now - self.pose_times[k] <= timeout for k in self.pilotos)
        if not self.listo or not fresh:
            self.get_logger().warn(
                f'Espera: alineacion={self.listo}, sensores_frescos={fresh}',
                throttle_duration_sec=3.0)
            for k in self.pilotos:
                self.publicar(k, 0.0, 0.0)
            return
        # Foto unica de la fase de salida: los dos autos calculan con el MISMO valor.
        salida = self.salida
        try:
            commands = {k: self.calcular(k, self.scans[k], salida) for k in self.pilotos}
        except (ValueError, IndexError, ArithmeticError, TypeError) as exc:
            self.get_logger().error(f'Control invalido: {exc}')
            commands = dict.fromkeys(self.pilotos, (0.0, 0.0))
        # El fin de salida pedido por un carril se aplica ya calculados ambos.
        if self.motivo_fin_salida:
            self.terminar_salida(self.motivo_fin_salida)
        elif salida and min(v for _, v in commands.values()) <= 1e-6:
            self.terminar_salida('un carril requiere frenado')
        for k, (steer, speed) in commands.items():
            self.publicar(k, steer, speed)

    def calcular(self, quien, msg, salida):
        if (len(msg.ranges) == 0 or not math.isfinite(msg.angle_min)
                or not math.isfinite(msg.angle_increment) or msg.angle_increment <= 0):
            return 0.0, 0.0
        if self.poses[quien] is None:
            return 0.0, 0.0
        parar = self.get_parameter('parar_al_chocar').value
        golpe = self.choque(msg) if parar else None
        # Una lectura suelta no basta: exigir choque_confirma ciclos seguidos.
        self.golpes[quien] = self.golpes[quien] + 1 if golpe is not None else 0
        if self.detenido[quien]:
            if golpe is None:
                self.libre_ciclos[quien] += 1
                if self.libre_ciclos[quien] >= self.get_parameter('choque_liberacion').value:
                    self.detenido[quien] = False
            else:
                self.libre_ciclos[quien] = 0
            return 0.0, 0.0
        if self.golpes[quien] >= self.get_parameter('choque_confirma').value:
            self.detenido[quien] = True
            self.libre_ciclos[quien] = 0
            ang, d = golpe
            lado = ('el frente' if abs(ang) < math.radians(30)
                    else ('la izquierda' if ang > 0 else 'la derecha'))
            x, y, _, v = self.poses[quien]
            self.get_logger().warn(
                f'Auto {quien} toco algo por {lado} (a {d:.2f} m del centro), yendo a '
                f'{v:.1f} m/s, en x={x:.1f} y={y:.1f}, fase {self.fase}. Frenado preventivo. '
                'Verifica si el simulador registro contacto.')
            return 0.0, 0.0
        return self.pilotos[quien].paso(msg, self.relativo(quien), salida)


def main(args=None):
    rclpy.init(args=args)
    node = FollowTheGap2Autos()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            for k in node.pilotos:
                node.publicar(k, 0.0, 0.0)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
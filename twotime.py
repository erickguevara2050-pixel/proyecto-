#!/usr/bin/env python3
"""Read-only F1Tenth dual-car race monitor (ROS 2 Humble)."""
import csv
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


@dataclass
class Lap:
    seconds: float
    distance: float
    speed_integral: float
    peak: float

    @property
    def average(self):
        return self.speed_integral / self.seconds


@dataclass
class Car:
    name: str
    gate: tuple = None
    previous: tuple = None
    start: float = None
    now: float = 0.0
    speed: float = 0.0
    distance: float = 0.0
    integral: float = 0.0
    peak: float = 0.0
    laps: list = field(default_factory=list)
    attempt: int = 1
    collisions: int = 0
    invalid: str = ''
    finished: bool = False
    contacts: int = 0
    odom_rx: float = -math.inf
    scan_rx: float = -math.inf
    scan_stamp: tuple = None
    history: list = field(default_factory=list)
    stopped_since: float = None

    def clear_lap(self, start=None):
        self.start = start
        self.distance = self.integral = self.peak = 0.0

    def add_segment(self, dt, ds, v0, v1):
        self.distance += ds
        self.integral += dt * (v0 + v1) / 2
        self.peak = max(self.peak, v0, v1)

    def update(self, x, y, yaw, t, v, min_dist, gate_width, max_speed, max_gap):
        """Return event and optional detail. Velocities are odometry magnitudes."""
        self.now, self.speed = t, v
        if self.invalid or self.finished:
            return '', None
        if self.gate is None:
            self.gate = (x, y, yaw)
        gx, gy, heading = self.gate
        dx, dy = x-gx, y-gy
        s = dx*math.cos(heading)+dy*math.sin(heading)
        lat = -dx*math.sin(heading)+dy*math.cos(heading)
        old = self.previous
        self.previous = (x, y, s, lat, t, v)
        if old is None:
            if math.hypot(dx,dy)<0.5 and abs(wrap(yaw-heading))<math.pi/3:
                self.clear_lap(t)
                return 'start', None
            return '', None
        dt = t-old[4]
        ds = math.hypot(x-old[0], y-old[1])
        if dt<0 or dt>max_gap or ds>max_speed*max(dt,0)+0.5:
            return 'invalid', 'Odometry discontinuity or data gap'
        if dt==0:
            if ds>0.01:
                return 'invalid', 'Pose changed without advancing timestamp'
            return '', None
        crossed = old[2]<0<=s and abs(wrap(yaw-heading))<math.pi/3
        f = -old[2]/(s-old[2]) if crossed else 1.0
        crossed = crossed and abs(old[3]+f*(lat-old[3]))<=gate_width
        if not crossed:
            if self.start is not None:
                self.add_segment(dt,ds,old[5],v)
            return '', None
        tc = old[4]+f*dt
        vc = old[5]+f*(v-old[5])
        if self.start is None:
            self.clear_lap(tc)
            self.add_segment(dt*(1-f),ds*(1-f),vc,v)
            return 'start', None
        if self.distance+ds*f<min_dist or tc<=self.start:
            self.add_segment(dt,ds,old[5],v)
            return '', None
        self.add_segment(dt*f,ds*f,old[5],vc)
        lap = Lap(tc-self.start,self.distance,self.integral,self.peak)
        self.laps.append(lap)
        self.clear_lap(tc)
        self.add_segment(dt*(1-f),ds*(1-f),vc,v)
        return 'lap', lap

    def summary(self):
        total = sum(l.seconds for l in self.laps)
        avg = sum(l.speed_integral for l in self.laps)/total if total else 0.0
        peak = max((l.peak for l in self.laps),default=0.0)
        return total, avg, peak

    def reset(self):
        self.history.append((self.attempt,list(self.laps),self.invalid or ('completed' if self.finished else 'reset'),self.collisions))
        self.attempt += 1
        self.previous = None
        self.clear_lap()
        self.laps.clear()
        self.invalid = ''
        self.finished = False
        self.contacts = 0
        self.stopped_since = None
        self.scan_stamp = None
        self.odom_rx = self.scan_rx = -math.inf


def lap_report(car, target):
    total, avg, peak = car.summary()
    status = 'COMPLETED' if car.finished else (car.invalid or 'INCOMPLETE')
    lines = [f'\n===== {car.name} | Attempt {car.attempt} | {status} =====',
             'Lap  Time (s)  Avg (m/s)  Avg (km/h)  Max (m/s)  Max (km/h)  Total (s)']
    elapsed = 0.0
    for i, lap in enumerate(car.laps, 1):
        elapsed += lap.seconds
        lines.append(f'{i:3d}  {lap.seconds:8.2f}  {lap.average:9.2f}  {lap.average*3.6:10.2f}  {lap.peak:9.2f}  {lap.peak*3.6:10.2f}  {elapsed:9.2f}')
    if not car.laps:
        lines.append('No completed laps.')
    lines.extend([f'Completed laps: {len(car.laps)}/{target}',
                  f'Total completed-lap time: {total:.2f} s',
                  f'Overall average speed: {avg:.2f} m/s ({avg*3.6:.2f} km/h)',
                  f'Overall maximum speed: {peak:.2f} m/s ({peak*3.6:.2f} km/h)',
                  f'Collision events detected (all attempts): {car.collisions}'])
    if car.laps:
        best = min(range(len(car.laps)), key=lambda i: car.laps[i].seconds)
        lines.append(f'Fastest lap: {best+1} ({car.laps[best].seconds:.2f} s)')
    return '\n'.join(lines)


def self_test():
    c=Car('EGO')
    def u(x,y,t,v=2): return c.update(x,y,0,t,v,3,1,30,2)
    assert u(0,0,0)[0]=='start'
    u(2,0,1); u(2,2,2); u(-2,2,3); u(-1,0,4)
    event,lap=u(1,0,5)
    assert event=='lap' and abs(lap.seconds-4.5)<1e-9 and abs(lap.average-2)<1e-9
    assert abs(c.distance-1)<1e-9
    assert u(100,0,6)[0]=='invalid'
    c.reset(); assert c.attempt==2 and len(c.history[0][1])==1
    d=Car('OPP'); assert not d.laps and d.collisions==0
    d.laps = [Lap(10, 50, 50, 8), Lap(20, 120, 120, 9)]
    report = lap_report(d, 10)
    assert '18.00' in report and '21.60' in report and '30.00 s' in report
    assert 'No completed laps.' in lap_report(Car('EGO'), 10)
    print('Self-test passed: lap crossing, speed integration, split segment, pose jump, reset, independent state.')


def run():
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import LaserScan
    from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
    from std_msgs.msg import Bool

    class Monitor(Node):
        def __init__(self):
            super().__init__('lap_counter_dual_english')
            params=dict(target_laps=10,min_lap_dist=20.0,gate_half_width=3.0,
                        car_half_length=0.29,car_half_width=0.155,lidar_x=0.275,
                        footprint_center_x=0.0,lidar_y=0.0,contact_scans=3,
                        sensor_timeout=1.0,max_speed=30.0,max_odom_gap=1.0,
                        csv_path='~/race_results_dual.csv',
                        ego_odom_topic='/ego_racecar/odom',opp_odom_topic='/opp_racecar/odom',
                        ego_scan_topic='/scan',opp_scan_topic='/opp_scan',
                        ego_reset_topic='/initialpose',opp_reset_topic='/goal_pose',
                        ego_collision_topic='',opp_collision_topic='')
            for k,v in params.items(): self.declare_parameter(k,v)
            self.target=self.p('target_laps')
            if self.target<1 or self.p('contact_scans')<1: raise ValueError('Invalid count parameter')
            for k in ('min_lap_dist','gate_half_width','car_half_length','car_half_width','sensor_timeout','max_speed','max_odom_gap'):
                if not math.isfinite(self.p(k)) or self.p(k)<=0: raise ValueError('Invalid parameter: '+k)
            self.cars={k:Car(k.upper()) for k in ('ego','opp')}
            self.subs=[]
            for k in self.cars:
                self.subs.append(self.create_subscription(Odometry,self.p(k+'_odom_topic'),lambda m,k=k:self.odom(k,m),qos_profile_sensor_data))
                self.subs.append(self.create_subscription(LaserScan,self.p(k+'_scan_topic'),lambda m,k=k:self.scan(k,m),qos_profile_sensor_data))
                topic=self.p(k+'_collision_topic')
                if topic:
                    self.subs.append(self.create_subscription(Bool,topic,lambda m,k=k:self.official(k,m),qos_profile_sensor_data))
            self.subs.append(self.create_subscription(PoseWithCovarianceStamped,self.p('ego_reset_topic'),lambda m:self.reset('ego'),10))
            self.subs.append(self.create_subscription(PoseStamped,self.p('opp_reset_topic'),lambda m:self.reset('opp'),10))
            self.create_timer(1.0,self.live)
            print(f'Race monitor ready: {self.target} consecutive laps per car. Start before driving.',flush=True)
            print('Collision source: configured simulator Bool topics, otherwise estimated LiDAR contact.',flush=True)
            self.save()

        def p(self,k): return self.get_parameter(k).value

        def reset(self,k):
            c=self.cars[k]; c.reset(); print(f'[{c.name}] Attempt {c.attempt} reset; original finish line retained.',flush=True); self.save()

        def invalidate(self,c,reason,collision=False):
            if c.invalid or c.finished: return
            c.invalid=reason
            if collision: c.collisions+=1
            print(f'[{c.name}] Attempt {c.attempt} INVALID: {reason} | Collision count: {c.collisions}',flush=True)
            self.save()

        def official(self,k,msg):
            if msg.data: self.invalidate(self.cars[k],'Confirmed collision (simulator topic)',True)

        def scan(self,k,msg):
            c=self.cars[k]; r=np.asarray(msg.ranges,dtype=float)
            if not r.size or not all(map(math.isfinite,(msg.angle_min,msg.angle_increment,msg.range_min,msg.range_max))) or msg.angle_increment==0: return
            valid=np.isfinite(r)&(r>=msg.range_min)&(r<msg.range_max)&(r>0)
            known=valid|np.isposinf(r)|(np.isfinite(r)&(r>=msg.range_max))
            if not np.any(known): return
            c.scan_rx=time.monotonic()
            if c.finished or c.invalid or c.start is None or self.p(k+'_collision_topic'): return
            stamp=(msg.header.stamp.sec,msg.header.stamp.nanosec)
            if stamp!=(0,0) and stamp==c.scan_stamp: return
            c.scan_stamp=stamp
            a=msg.angle_min+np.arange(r.size)*msg.angle_increment
            x=r[valid]*np.cos(a[valid])+self.p('lidar_x')-self.p('footprint_center_x')
            y=r[valid]*np.sin(a[valid])+self.p('lidar_y')
            hit=np.any((np.abs(x)<self.p('car_half_length'))&(np.abs(y)<self.p('car_half_width')))
            c.contacts=c.contacts+1 if hit else 0
            if c.contacts>=self.p('contact_scans'): self.invalidate(c,'Estimated collision (LiDAR footprint contact)',True)

        def odom(self,k,msg):
            c=self.cars[k]; p=msg.pose.pose.position; q=msg.pose.pose.orientation
            yaw=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
            t=msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9
            v=math.hypot(msg.twist.twist.linear.x,msg.twist.twist.linear.y)
            if not all(map(math.isfinite,(p.x,p.y,yaw,t,v))): return
            c.odom_rx=time.monotonic()
            if c.start is not None and t-c.start>self.p('sensor_timeout') and time.monotonic()-c.scan_rx>self.p('sensor_timeout'):
                self.invalidate(c,'LiDAR data missing'); return
            event,detail=c.update(p.x,p.y,yaw,t,v,self.p('min_lap_dist'),self.p('gate_half_width'),self.p('max_speed'),self.p('max_odom_gap'))
            if event=='invalid': self.invalidate(c,detail)
            elif event=='start': print(f'[{c.name}] Timer started at finish line.',flush=True)
            elif event=='lap':
                total,avg,peak=c.summary()
                print(f'[{c.name}] Lap {len(c.laps)}/{self.target} | Time: {detail.seconds:.2f} s | Avg speed: {detail.average:.2f} m/s ({detail.average*3.6:.2f} km/h) | Top speed: {detail.peak:.2f} m/s | Total: {total:.2f} s | Collisions: {c.collisions}',flush=True)
                if len(c.laps)>=self.target:
                    c.finished=True
                    print(f'[{c.name}] FINISHED | Total: {total:.2f} s | Best lap: {min(l.seconds for l in c.laps):.2f} s | Overall avg: {avg:.2f} m/s | Top: {peak:.2f} m/s | No collision detected in this attempt | All attempts: {c.collisions} collision events',flush=True)
                self.save()
                if all(z.finished for z in self.cars.values()):
                    winner=min(self.cars.values(),key=lambda z:z.summary()[0]); fastest=min(self.cars.values(),key=lambda z:min(l.seconds for l in z.laps))
                    self.print_reports()
                    print(f'RACE SUMMARY | Lowest total time: {winner.name} | Fastest lap: {fastest.name}',flush=True)

        def print_reports(self):
            for c in self.cars.values():
                print(lap_report(c, self.target), flush=True)

        def live(self):
            if all(c.finished for c in self.cars.values()): return
            now=time.monotonic()
            for c in self.cars.values():
                if c.start is not None and not c.invalid and not c.finished:
                    if now-c.odom_rx>self.p('sensor_timeout'): self.invalidate(c,'Odometry timeout')
                    elif now-c.scan_rx>self.p('sensor_timeout') and c.now-c.start>self.p('sensor_timeout'): self.invalidate(c,'LiDAR timeout')
                total,avg,peak=c.summary()
                status=c.invalid or ('FINISHED' if c.finished else 'WAITING' if c.start is None else 'RUNNING')
                elapsed=max(0,c.now-c.start) if c.start is not None and not c.finished else 0
                print(f'[{c.name}] {status} | Laps: {len(c.laps)}/{self.target} | Speed: {c.speed:.2f} m/s | Current lap: {elapsed:.1f} s | Total: {total+elapsed:.1f} s | Collisions: {c.collisions}',flush=True)

        def save(self):
            path=Path(self.p('csv_path')).expanduser(); tmp=path.with_name(path.name+'.tmp')
            try:
                path.parent.mkdir(parents=True,exist_ok=True)
                with tmp.open('w',newline='',encoding='utf-8') as f:
                    w=csv.writer(f); w.writerow(['vehicle','attempt','lap','lap_time_s','lap_distance_m','average_speed_mps','average_speed_kmh','peak_speed_mps','peak_speed_kmh','total_time_s','overall_average_speed_mps','overall_peak_speed_mps','best_lap_s','collision_count_cumulative','status'])
                    for c in self.cars.values():
                        records=c.history+[(c.attempt,c.laps,c.invalid or ('completed' if c.finished else 'running'),c.collisions)]
                        for attempt,laps,status,count in records:
                            total=sum(l.seconds for l in laps); avg=sum(l.speed_integral for l in laps)/total if total else 0; peak=max((l.peak for l in laps),default=0); best=min((l.seconds for l in laps),default=0)
                            for i,l in enumerate(laps,1): w.writerow([c.name,attempt,i,l.seconds,l.distance,l.average,l.average*3.6,l.peak,l.peak*3.6,total,avg,peak,best,count,status])
                            if not laps: w.writerow([c.name,attempt]+['']*11+[count,status])
                os.replace(tmp,path)
            except OSError as e: print(f'CSV error: {e}',flush=True)

    rclpy.init(); node=None
    try:
        node=Monitor(); rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        if node is not None:
            if not all(c.finished for c in node.cars.values()): node.print_reports()
            node.save(); node.destroy_node()
        rclpy.try_shutdown()


if __name__=='__main__':
    import sys
    if '--self-test' in sys.argv: self_test()
    else: run()

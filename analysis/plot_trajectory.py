#!/usr/bin/env python3
from rosbags.highlevel import AnyReader
from pathlib import Path
import matplotlib.pyplot as plt

bag_path = Path('square_test_1')

xs, ys, ts = [], [], []

with AnyReader([bag_path]) as reader:
    connections = [c for c in reader.connections if c.topic == '/diff_drive_controller/odom']
    for connection, timestamp, rawdata in reader.messages(connections=connections):
        msg = reader.deserialize(rawdata, connection.msgtype)
        xs.append(msg.pose.pose.position.x)
        ys.append(msg.pose.pose.position.y)
        ts.append(timestamp)

print(f"Total odom messages: {len(xs)}")
print(f"Duration: {(ts[-1]-ts[0])/1e9:.1f} s")
print(f"X range: {min(xs):.2f} to {max(xs):.2f}")
print(f"Y range: {min(ys):.2f} to {max(ys):.2f}")

plt.figure(figsize=(7,7))
plt.plot(xs, ys, '-', linewidth=1, color='steelblue', label='Actual trajectory')
plt.plot(xs[0], ys[0], 'go', markersize=10, label='Start')
plt.plot(xs[-1], ys[-1], 'ro', markersize=10, label='End')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('Robot Trajectory — Event-Triggered Control')
plt.axis('equal')
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig('trajectory_plot.png', dpi=150)
print("Saved trajectory_plot.png")
plt.show()

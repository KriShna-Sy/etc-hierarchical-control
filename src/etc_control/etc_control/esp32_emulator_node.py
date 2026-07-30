import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class ESP32Emulator(Node):
    """
    L1/L2-compatible low-level actuation emulator.

    hold_mode='zoh'   : original behavior, unchanged. Holds the last full
                         combined command (v, omega) between events. This
                         reproduces your existing L1 results exactly.

    hold_mode='smart' : holds only the feedback CORRECTION (delta_v,
                         delta_omega) from the last event, and continuously
                         adds it to the *live* feedforward (v_d, omega_d)
                         from the trajectory generator. Between events, the
                         robot still follows the nominal trajectory shape
                         (including anticipating corners) instead of coasting
                         on a stale absolute velocity.

    NOTE / hardware TODO: in this sim prototype, the feedforward is read
    directly from /trajectory_desired/twist (cheap in sim, but this would
    be a continuous stream in real life). For the real RPi5->ESP32 port,
    replace this subscription with a small locally-replicated deterministic
    trajectory clock on the ESP32 (same v_d/omega_turn/side parameters,
    sent once at startup), so smart hold keeps the same communication
    budget as ZOH instead of streaming feedforward continuously. Tracked
    as a follow-up once the algorithm is validated in sim.
    """
    def __init__(self):
        super().__init__('esp32_emulator_node')
        self.declare_parameter('tau_max_ms', 5.0)
        self.declare_parameter('hold_mode', 'zoh')  # 'zoh' or 'smart'
        self.hold_mode = self.get_parameter('hold_mode').value

        # ZOH state
        self.held_cmd = Twist()

        # Smart-hold state
        self.held_correction = Twist()   # (delta_v, delta_omega) from last event
        self.v_d = 0.0
        self.omega_d = 0.0

        if self.hold_mode == 'zoh':
            self.create_subscription(Twist, '/cmd_vel_triggered', self.on_cmd_zoh, 10)
        elif self.hold_mode == 'smart':
            self.create_subscription(Twist, '/cmd_vel_correction_triggered', self.on_correction, 10)
            self.create_subscription(Twist, '/trajectory_desired/twist', self.on_feedforward, 10)
        else:
            raise ValueError(f"Unknown hold_mode '{self.hold_mode}', expected 'zoh' or 'smart'")

        self.pub = self.create_publisher(Twist, '/diff_drive_controller/cmd_vel', 10)
        self.create_timer(1.0/200.0, self.publish_held)  # 200Hz, ESP32-like rate

        self.get_logger().info(f'esp32_emulator_node started in hold_mode={self.hold_mode}')

    # ---- ZOH mode ----
    def on_cmd_zoh(self, msg):
        self.held_cmd = msg

    # ---- Smart-hold mode ----
    def on_correction(self, msg):
        self.held_correction = msg

    def on_feedforward(self, msg):
        self.v_d = msg.linear.x
        self.omega_d = msg.angular.z

    def publish_held(self):
        if self.hold_mode == 'zoh':
            self.pub.publish(self.held_cmd)
        else:
            out = Twist()
            out.linear.x = self.v_d + self.held_correction.linear.x
            out.angular.z = self.omega_d + self.held_correction.angular.z
            self.pub.publish(out)

def main():
    rclpy.init()
    rclpy.spin(ESP32Emulator())

if __name__ == '__main__':
    main()

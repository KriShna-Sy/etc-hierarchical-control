import rclpy, math
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from tf_transformations import euler_from_quaternion

def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))

class EventTrigger(Node):
    """
    L1/L2/L3-compatible event trigger node.

    L1/L2 unchanged: trigger condition structure, controller law, combined
    /cmd_vel_triggered output (ZOH mode), and /cmd_vel_correction_triggered
    (smart-hold mode).

    L3 addition: curvature-adaptive sigma. When adaptive_sigma=True, the
    effective threshold is scaled down during turns (high |omega_d|) and
    relaxed on straights (omega_d ~ 0), instead of using a single fixed
    sigma everywhere. This tightens tracking exactly where ZOH/smart-hold
    error was shown to be largest (corners) without paying the bandwidth
    cost of a permanently tighter threshold on straights.

        sigma_eff = sigma * (1 - (1-sigma_min_factor) * min(|omega_d|/omega_max, 1))

    sigma_min_factor=1.0 recovers the original fixed-sigma behavior exactly
    (sigma_eff == sigma always), so L1/L2 experiments are unaffected unless
    adaptive_sigma is explicitly turned on.
    """
    def __init__(self):
        super().__init__('event_trigger_node')
        self.declare_parameter('sigma', 0.1)
        self.declare_parameter('c', 0.03)
        self.declare_parameter('k_x', 0.8)
        self.declare_parameter('k', 1.0)      # k_y = k, per our Lyapunov proof
        self.declare_parameter('k_theta', 1.5)

        # L3: curvature-adaptive sigma parameters
        self.declare_parameter('adaptive_sigma', False)
        self.declare_parameter('sigma_min_factor', 0.5)  # floor multiplier at full turn rate
        self.declare_parameter('omega_max', 0.5)          # matches trajectory_generator's omega_turn

        self.sigma = self.get_parameter('sigma').value
        self.c = self.get_parameter('c').value
        self.k_x = self.get_parameter('k_x').value
        self.k = self.get_parameter('k').value
        self.k_theta = self.get_parameter('k_theta').value

        self.adaptive_sigma = self.get_parameter('adaptive_sigma').value
        self.sigma_min_factor = self.get_parameter('sigma_min_factor').value
        self.omega_max = self.get_parameter('omega_max').value

        self.x = self.y = self.theta = 0.0
        self.x_d = self.y_d = self.theta_d = 0.0
        self.v_d = self.omega_d = 0.0

        self.create_subscription(Odometry, '/diff_drive_controller/odom', self.on_odom, 10)
        self.create_subscription(PoseStamped, '/trajectory_desired/pose', self.on_desired_pose, 10)
        self.create_subscription(Twist, '/trajectory_desired/twist', self.on_desired_twist, 10)

        # Original combined output — kept for ZOH-mode experiments (L1)
        self.pub = self.create_publisher(Twist, '/cmd_vel_triggered', 10)
        # Feedback-only correction — used by smart-hold experiments (L2)
        self.corr_pub = self.create_publisher(Twist, '/cmd_vel_correction_triggered', 10)

        self.trigger_count = 0
        self.total_ticks = 0
        self.create_timer(1.0/50.0, self.check_trigger)

        self.get_logger().info(
            f'event_trigger_node started: sigma={self.sigma} adaptive_sigma={self.adaptive_sigma} '
            f'sigma_min_factor={self.sigma_min_factor} omega_max={self.omega_max}'
        )

    def on_odom(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        _,_,self.theta = euler_from_quaternion([q.x,q.y,q.z,q.w])

    def on_desired_pose(self, msg):
        self.x_d = msg.pose.position.x
        self.y_d = msg.pose.position.y
        q = msg.pose.orientation
        _,_,self.theta_d = euler_from_quaternion([q.x,q.y,q.z,q.w])

    def on_desired_twist(self, msg):
        self.v_d = msg.linear.x
        self.omega_d = msg.angular.z

    def effective_sigma(self):
        """L3: scale sigma down during turns, relax it on straights."""
        if not self.adaptive_sigma:
            return self.sigma
        curvature_frac = min(abs(self.omega_d) / self.omega_max, 1.0) if self.omega_max > 0 else 0.0
        scale = 1.0 - (1.0 - self.sigma_min_factor) * curvature_frac
        return self.sigma * scale

    def check_trigger(self):
        self.total_ticks += 1
        dx, dy = self.x_d - self.x, self.y_d - self.y
        e_x =  math.cos(self.theta)*dx + math.sin(self.theta)*dy
        e_y = -math.sin(self.theta)*dx + math.cos(self.theta)*dy
        e_theta = wrap(self.theta_d - self.theta)

        e_norm = math.sqrt(e_x**2 + e_y**2 + e_theta**2)
        x_norm = math.sqrt(self.v_d**2 + self.omega_d**2)  # nominal control effort scale
        sigma_eff = self.effective_sigma()
        threshold = sigma_eff*x_norm + self.c

        if e_norm > threshold:
            v = self.v_d*math.cos(e_theta) + self.k_x*e_x
            omega = self.omega_d + self.k*self.v_d*e_y + self.k_theta*e_theta

            cmd = Twist()
            cmd.linear.x = v
            cmd.angular.z = omega
            self.pub.publish(cmd)

            corr = Twist()
            corr.linear.x = v - self.v_d
            corr.angular.z = omega - self.omega_d
            self.corr_pub.publish(corr)

            self.trigger_count += 1
            self.get_logger().info(
                f'EVENT {self.trigger_count}/{self.total_ticks} e={e_norm:.3f} thr={threshold:.3f} sigma_eff={sigma_eff:.3f}'
            )

def main():
    rclpy.init()
    rclpy.spin(EventTrigger())

if __name__ == '__main__':
    main()

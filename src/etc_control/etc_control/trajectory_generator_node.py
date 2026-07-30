import rclpy, math, time
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from tf_transformations import quaternion_from_euler

class TrajectoryGenerator(Node):
    """
    L1/L2/L3 trajectory generator, unchanged for shape='square' (the default
    -- existing experiments are completely unaffected).

    shape='circle' (new): constant-curvature path, no corners. Added to test
    whether the ZOH/smart-hold/adaptive-sigma ordering established on the
    square generalizes to a trajectory with sustained turning instead of
    brief, discrete corners.

    NOTE on the adaptive-sigma mechanism and this new shape: effective_sigma()
    in event_trigger_node.py scales sigma based on how large |omega_d| is
    relative to omega_max. On the circle, omega_d is CONSTANT throughout the
    run, so sigma_eff will also be constant -- there is no curvature change
    within a run for the adaptive mechanism to react to. This is expected,
    not a bug: it just means the circle run tests the ZOH vs smart-hold vs
    adaptive-sigma ORDERING under sustained turning, not the adaptive
    mechanism's within-run reactivity (that needs a trajectory whose
    curvature changes sign/magnitude during the run, e.g. a figure-eight).
    """
    def __init__(self):
        super().__init__('trajectory_generator')

        self.declare_parameter('shape', 'square')  # 'square' (default) or 'circle'
        self.shape = self.get_parameter('shape').value

        self.pose_pub = self.create_publisher(PoseStamped, '/trajectory_desired/pose', 10)
        self.twist_pub = self.create_publisher(Twist, '/trajectory_desired/twist', 10)

        self.v_d = 0.2      # m/s -- same forward speed for both shapes, for a fair comparison

        if self.shape == 'square':
            self.omega_turn = 0.5   # rad/s during 90 deg turns
            self.side = 2.0         # metres per side
            self.x_d, self.y_d, self.theta_d = 0.0, 0.0, 0.0
            self.state = 'MOVE'
            self.dist_done = 0.0
            self.turn_done = 0.0
        elif self.shape == 'circle':
            self.radius = 1.0                  # m
            self.omega_d_circle = self.v_d / self.radius  # rad/s, constant curvature
            self.t_start = None
            self.x_d, self.y_d, self.theta_d = 0.0, 0.0, 0.0
        else:
            raise ValueError(f"Unknown shape '{self.shape}', expected 'square' or 'circle'")

        self.dt = 1.0/50.0
        self.timer = self.create_timer(self.dt, self.step)

        self.get_logger().info(f'trajectory_generator started: shape={self.shape} v_d={self.v_d}')

    def step(self):
        if self.shape == 'square':
            v_cmd, w_cmd = self._step_square()
        else:
            v_cmd, w_cmd = self._step_circle()

        ps = PoseStamped()
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.header.frame_id = 'odom'
        ps.pose.position.x = self.x_d
        ps.pose.position.y = self.y_d
        q = quaternion_from_euler(0, 0, self.theta_d)
        ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w = q
        self.pose_pub.publish(ps)

        tw = Twist()
        tw.linear.x = v_cmd
        tw.angular.z = w_cmd
        self.twist_pub.publish(tw)

    def _step_square(self):
        """Identical to the original implementation -- untouched."""
        v_cmd, w_cmd = 0.0, 0.0
        if self.state == 'MOVE':
            v_cmd = self.v_d
            self.x_d += self.v_d*math.cos(self.theta_d)*self.dt
            self.y_d += self.v_d*math.sin(self.theta_d)*self.dt
            self.dist_done += self.v_d*self.dt
            if self.dist_done >= self.side:
                self.dist_done = 0.0
                self.state = 'TURN'
        elif self.state == 'TURN':
            w_cmd = self.omega_turn
            self.theta_d += self.omega_turn*self.dt
            self.turn_done += self.omega_turn*self.dt
            if self.turn_done >= math.pi/2:
                self.turn_done = 0.0
                self.theta_d = round(self.theta_d/(math.pi/2))*(math.pi/2)
                self.state = 'MOVE'
        return v_cmd, w_cmd

    def _step_circle(self):
        """Constant curvature: v_d constant, omega_d constant. No states,
        no corners -- this is the whole point of this trajectory shape."""
        if self.t_start is None:
            self.t_start = self.get_clock().now()
        t = (self.get_clock().now() - self.t_start).nanoseconds * 1e-9

        theta = self.omega_d_circle * t
        self.x_d = self.radius * math.sin(theta)
        self.y_d = self.radius * (1.0 - math.cos(theta))
        self.theta_d = theta

        return self.v_d, self.omega_d_circle

def main():
    rclpy.init()
    rclpy.spin(TrajectoryGenerator())

if __name__ == '__main__':
    main()

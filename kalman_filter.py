import numpy as np


class KalmanFilter2D:
    def __init__(self, dt):
        # ---------------------------------
        # basic timing
        # ---------------------------------
        self.dt = dt

        # ---------------------------------
        # state vector
        # x = [x, y, vx, vy]^T
        # ---------------------------------
        self.x = np.zeros((4, 1), dtype=np.float64)

        # ---------------------------------
        # state covariance
        # ---------------------------------
        self.P = np.eye(4, dtype=np.float64)

        # ---------------------------------
        # state transition matrix
        # ---------------------------------
        self.F = np.array([
            [1.0, 0.0, dt,  0.0],
            [0.0, 1.0, 0.0, dt ],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float64)

        # ---------------------------------
        # control matrix
        # u = [ax, ay]^T
        # ---------------------------------
        self.B = np.array([
            [0.5 * dt * dt, 0.0],
            [0.0, 0.5 * dt * dt],
            [dt, 0.0],
            [0.0, dt]
        ], dtype=np.float64)

        # ---------------------------------
        # measurement matrix
        # GPS measures x and y only
        # z = [x_gps, y_gps]^T
        # ---------------------------------
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ], dtype=np.float64)

        # ---------------------------------
        # process noise covariance Q
        # based on accel noise
        # ---------------------------------
        sigma_a = 0.8

        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2

        q1d = np.array([
            [0.25 * dt4, 0.5 * dt3],
            [0.5 * dt3,  dt2]
        ], dtype=np.float64) * (sigma_a ** 2)

        self.Q = np.array([
            [q1d[0, 0], 0.0,       q1d[0, 1], 0.0],
            [0.0,       q1d[0, 0], 0.0,       q1d[0, 1]],
            [q1d[1, 0], 0.0,       q1d[1, 1], 0.0],
            [0.0,       q1d[1, 0], 0.0,       q1d[1, 1]]
        ], dtype=np.float64)

        # ---------------------------------
        # measurement noise covariance R
        # based on GPS noise
        # ---------------------------------
        sigma_gps = 3.0

        self.R = np.array([
            [sigma_gps ** 2, 0.0],
            [0.0, sigma_gps ** 2]
        ], dtype=np.float64)

    def set_state(self, x0, y0, vx0=0.0, vy0=0.0):
        # ---------------------------------
        # initialize state estimate
        # ---------------------------------
        self.x = np.array([
            [x0],
            [y0],
            [vx0],
            [vy0]
        ], dtype=np.float64)

        # ---------------------------------
        # initialize covariance
        # position: somewhat certain
        # velocity: less certain
        # ---------------------------------
        self.P = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, 0.0],
            [0.0, 0.0, 0.0, 5.0]
        ], dtype=np.float64)

    def predict(self, ax, ay):
        # ---------------------------------
        # control input from IMU acceleration
        # ---------------------------------
        u = np.array([
            [ax],
            [ay]
        ], dtype=np.float64)

        # ---------------------------------
        # state prediction
        # x_k|k-1 = F x_k-1|k-1 + B u_k
        # ---------------------------------
        self.x = self.F @ self.x + self.B @ u

        # ---------------------------------
        # covariance prediction
        # P_k|k-1 = F P_k-1|k-1 F^T + Q
        # ---------------------------------
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, gps_x, gps_y):
        # ---------------------------------
        # measurement vector
        # ---------------------------------
        z = np.array([
            [gps_x],
            [gps_y]
        ], dtype=np.float64)

        # ---------------------------------
        # innovation / residual
        # y = z - Hx
        # ---------------------------------
        y = z - (self.H @ self.x)

        # ---------------------------------
        # innovation covariance
        # S = HPH^T + R
        # ---------------------------------
        S = self.H @ self.P @ self.H.T + self.R

        # ---------------------------------
        # Kalman gain
        # K = PH^T S^-1
        # ---------------------------------
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # ---------------------------------
        # state update
        # x_k|k = x_k|k-1 + K y
        # ---------------------------------
        self.x = self.x + K @ y

        # ---------------------------------
        # covariance update
        # P_k|k = (I - KH) P_k|k-1
        # ---------------------------------
        I = np.eye(4, dtype=np.float64)
        self.P = (I - K @ self.H) @ self.P

    def get_state(self):
        # ---------------------------------
        # return state as scalars
        # ---------------------------------
        return (
            float(self.x[0, 0]),
            float(self.x[1, 0]),
            float(self.x[2, 0]),
            float(self.x[3, 0])
        )

    def set_process_noise(self, sigma_a):
        # ---------------------------------
        # update process noise Q
        # ---------------------------------
        dt = self.dt
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2

        q1d = np.array([
            [0.25 * dt4, 0.5 * dt3],
            [0.5 * dt3,  dt2]
        ], dtype=np.float64) * (sigma_a ** 2)

        self.Q = np.array([
            [q1d[0, 0], 0.0,       q1d[0, 1], 0.0],
            [0.0,       q1d[0, 0], 0.0,       q1d[0, 1]],
            [q1d[1, 0], 0.0,       q1d[1, 1], 0.0],
            [0.0,       q1d[1, 0], 0.0,       q1d[1, 1]]
        ], dtype=np.float64)

    def set_measurement_noise(self, sigma_gps):
        # ---------------------------------
        # update GPS measurement noise R
        # ---------------------------------
        self.R = np.array([
            [sigma_gps ** 2, 0.0],
            [0.0, sigma_gps ** 2]
        ], dtype=np.float64)
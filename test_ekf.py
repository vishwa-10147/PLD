from modules.adaptive_ekf import AdaptiveEKF

weights = {

    "camera":0.3386,

    "gps":0.3346,

    "imu":0.3268

}

ekf = AdaptiveEKF()

result = ekf.compute(weights)

print("\nMeasurement Covariance (R)\n")

print(result["R"])

print("\nProcess Covariance (Q)\n")

print(result["Q"])
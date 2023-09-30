# Just disables the warning, doesn't take advantage of AVX/FMA to run faster
import os
import tensorflow as tf
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def check_tensorflow_features():
    # Check TensorFlow version
    print("TensorFlow Version:", tf.__version__)

    # Check if GPU support is available
    if tf.test.is_gpu_available():
        print("GPU support is available")
        gpu_time = run_example_on_device("GPU")
    else:
        print("GPU support is not available")
        gpu_time = None

    # Check if AVX2 and FMA instructions are enabled
    if tf.config.experimental.list_physical_devices("CPU"):
        devices = tf.config.experimental.list_physical_devices("CPU")
        for device in devices:
            print("Device:", device.name)
            if "avx2" in device.name.lower():
                print("AVX2 support is available")
            if "fma" in device.name.lower():
                print("FMA support is available")
        cpu_time = run_example_on_device("CPU")
    else:
        print("No CPU devices found")
        cpu_time = None

    # Compare CPU and GPU performance
    if gpu_time and cpu_time:
        print("\nPerformance Comparison:")
        print("CPU Time:", cpu_time, "seconds")
        print("GPU Time:", gpu_time, "seconds")
        print(
            "GPU is approximately {} times faster than CPU.".format(cpu_time / gpu_time)
        )


def run_example_on_device(device_name):
    if device_name == "GPU":
        device = "/device:GPU:0"
    elif device_name == "CPU":
        device = "/device:CPU:0"
    else:
        raise ValueError("Invalid device name")

    with tf.device(device):
        # Example TensorFlow operation
        start_time = time.time()
        a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name="a")
        b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name="b")
        c = tf.matmul(a, b)
        tf.print(c)
        end_time = time.time()
        execution_time = end_time - start_time
        print(device_name, "Execution Time:", execution_time, "seconds")

    return execution_time


if __name__ == "__main__":
    check_tensorflow_features()

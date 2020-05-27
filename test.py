# ----------------------
# - read the input data:

import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)

# ---------------------
# - network.py example:
"""
import clasNet.py


net = network.Network([784, 30, 10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
"""

# ----------------------
# - network2.py example:
import natenet


net = natenet.Network([784, 30, 10], cost=natenet.CrossEntropyCost)

net.SGD(training_data, 30, 10, 0.1, lmbda = 5.0,evaluation_data=validation_data,
    monitor_evaluation_accuracy=True)

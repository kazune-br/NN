import numpy as np
from mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True)
print("x_train")
print(len(x_train)) # => 60000
print(x_train[0])

print("t_train")
print(len(t_train)) # => 60000
print(t_train[0]) # => 5




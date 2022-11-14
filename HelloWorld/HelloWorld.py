import tensorflow as tf
import numpy as np
from keras import Sequential
from keras.layers import Dense

l0 = Dense(units=1, input_shape=[1])
model = Sequential([l0])
model.compile(optimizer='sgd', loss='mean_squared_error')

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

model.fit(xs, ys, epochs=500)

print(model.predict([25.0]))
print(f"Here is what i learned: {l0.get_weights()}")

# 2022-11-14 12:27:00.678858: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could 
# not load dynamic library 'cudart64_110.dll'; dlerror: cudart64_110.dll not found
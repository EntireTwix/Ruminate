![Ruminate Logo](https://github.com/EntireTwix/Ruminate/blob/main/Banner.png){:height="50%" width="50%"}
# Ruminate
an ML library that aims to be lightweight, fast, and generic. Written in and for Cpp20

# Features
Implementation:
* :sparkles: polymorphically based NeuralNetwork with constrained template argument for future LayerTypes
* :racehorse: fast initialization of weights and biases using pcg32 implementation from https://github.com/wjakob/pcg32

Essential:
* FC layers
* Activation functions: Relu,Sigmoid,Tanh,Swish
* saving/loading functionality

Other:
* dropout variants for input&hidden layers

# :construction:In Progress
* :sparkles: convolutions and pooling
* helper functions for weight/bias init

# Future Features
* batch normalization
* :sparkles: optimizer argument for network (like Adam for example)
* :racehorse: GPU or SIMD integration (to be tested)

# Dependencies
* mat.h    from EntireTwix/OptimizedHeaders
* pcg32.h  from wjakob/pcg32

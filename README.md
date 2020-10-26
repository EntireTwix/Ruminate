# Ruminate
an ML library that aims to be lightweight, fast, and generic. Written in and for Cpp20

# Features
* :sparkles: polymorphically based NeuralNetwork with constrained template argument for future LayerTypes
* Activation functions: Relu,Sigmoid,Tanh,Swish
* :racehorse: fast initialization of weights and biases using pcg32 implementation from https://github.com/wjakob/pcg32

# :construction:In Progress
* FC layers

# Future Features
* dropout layer
* helper functions for weight/bias init
* network saving functionality
* :sparkles: convalutions and pooling
* :sparkles: optimizer argument for network
* :racehorse: GPU integration

# Dependencies
* mat.h    from EntireTwix/OptimizedHeaders
* pcg32.h  from wjakob/pcg32

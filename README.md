![Ruminate Logo](https://github.com/EntireTwix/Ruminate/blob/main/Banner.png)

# Ruminate v2 (TEMPORARY DISCONTINUED)

a Header-Only ML library that aims to be lightweight, fast, and generic. Written in and for **C++20**

# Usage

### Network Initilization Example:

```cpp
NeuralNetwork<MLMat> net{
     new Input<MLMat>(2),                      //2 node input
     new Weight(2, 3, RngInit()),              //2x1 weights
     new DropOut(3, 0.25),                     //dropout of 25% of the input
     new Hidden(3, ReluLeaky, ReluLeakyPrime), //3 node hidden layer
     new Weight(3, 1, RngInit()),              //3x1 weights
     new Output(1, ReluLeaky, ReluLeakyPrime), //1 output node
};
```

### Learning Loop Example:

```cpp
//setting inputs and anwsers
data.FastAt(0) = gen.nextUInt(100); //A
data.FastAt(1) = gen.nextUInt(100); //B
anw.FastAt(0) = data.FastAt(0) + data.FastAt(1); //A+B

//forward prop and storing as variable res
auto res = net.ForwardProp(data);

//backprop with forwardprop as input
auto corrections = net.BackwordProp(res, net.GetCostPrime(res.back(), anw), 0.002);

//applying corrections
net.Learn(corrections);
```

#### check out the examples folder for more examples

# Features

#### Implementation:

- :sparkles: **polymorphically** based
- :racehorse: **fast random generator** using pcg32 implementation from https://github.com/wjakob/pcg32
- :racehorse: Forward&Back propogation are **thread safe**

#### Essential:

- ANN functionality
- Activation functions: Relu, ReluLeaky, Sigmoid, Tanh, Swish

#### Other:

- variable batch size support / layer
- dropout layer
- LOG_LAYERS_FLAG for debugging (inside layers.hpp)

# Use in your Project

in your CMake simply add

```
target_include_directories(${PROJECT_NAME} PUBLIC {path to ruminate}/include)
```

# Contributing

if you want to contribute layer modules or just to use on your own project all you have to do is inheret from the Layers abstract and define forwardprop(), backprop(), and internal(), usually learn() is the same for most layer types so you rarely need to re-define it.

```cpp
virtual internal() = 0;
virtual M ForwardProp(const M &);
virtual M BackwardProp(M&, const std::vector<M>&, Layer **const, size_t) const;
virtual void Learn(const M &);
```

and if your layer implements activation functions then inheret the IActivationFuncs interface aswell

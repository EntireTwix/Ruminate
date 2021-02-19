![Ruminate Logo](https://github.com/EntireTwix/Ruminate/blob/main/Banner.png)

# Ruminate v2

an Header-Only ML library that aims to be lightweight, fast, and generic. Written in and for **C++20**

# Usage

### Network Initilization Example:

```cpp
NeuralNetwork<ANN> net{
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
data.At(0, 0) = gen.nextUInt(100);
data.At(0, 1) = gen.nextUInt(100);
anw.At(0, 0) = data.At(0, 0) + data.At(0, 1);

auto res = net.ForwardProp(data); //forward propogating
auto corrections = net.BackwordProp(res, net.GetCostPrime(res.back(), anw), 0.002);
net.Learn(corrections); //applying corrections
```

and this could be easily **multi-threaded** with something like **tpool.h** in my OptimizedHeaders repo or any other thread pool

```cpp
//Start() thread pool
for (int i = 0; i < 10; ++i)
{
    pool.AddTask([i, &corrections]() {
        data.At(0, 0) = gen.nextUInt(100);
        data.At(0, 1) = gen.nextUInt(100);
        anw.At(0, 0) = data.At(0, 0) + data.At(0, 1);

        res = net.ForwardProp(data);
        corrections[i] = net.BackwordProp(res, net.GetCostPrime(res.back(), anw), 0.002);
    });
}
while(pool.Jobs()) {} //complete jobs
//...
//Learn() after averaging corrections into one vec
```

and this is assuming your batch size is 1, as this library supports variable batch size you could multi thread each batch.

#### check out Examples folder for more examples

# Features

#### Implementation:

- :sparkles: **polymorphically** based
- :racehorse: **fast random generator** using pcg32 implementation from https://github.com/wjakob/pcg32
- :racehorse: Forward&Back propogation are **thread safe**

#### Essential:

- ANN functionality
- Activation functions: Relu, ReluLeaky, Sigmoid, Tanh, Swish
- saving/loading functionality
- variable batch size

#### Other:

- batch normalization layer
- dropout layer
- LOG_LAYERS_FLAG for debugging

# Build in Project

in your CMake simply add

```
target_include_directories(${PROJECT_NAME} PUBLIC third_party/Ruminate/include)
```

# Contributing

if you want to contribute layer modules or just to use on your own project all you have to do is inheret from the Layers abstract and define forwardprop(), backprop(), and internal(), usually learn() is the same for most layer types.

```
virtual internal() = 0;
virtual M ForwardProp(const M &input)
virtual M BackwardProp(M &cost, const std::vector<M> &forwardRes, Layer **const layers, size_t index) const;
virtual void Learn(const M &correction)
```

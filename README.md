# [THIS PROJECT IS CURRENTLY ON HOLD AND DOES NOT WORK]

![Ruminate Logo](https://github.com/EntireTwix/Ruminate/blob/main/Banner.png)

# Ruminate v1.1.4

an ML library that aims to be lightweight, fast, and generic. Written in and for **C++20**

# Usage

### Network Initilization Example:

```cpp
NeuralNetwork<ANN> net{
     new Input<MLMat>(2),                      //2 node input
     new Weight(2, 3, new RngInit()),          //2x1 weights
     new DropOut(3, 0.25),                     //dropout of 25% of the input
     new Hidden(3, ReluLeaky, ReluLeakyPrime), //3 node hidden layer
     new Weight(3, 1, new RngInit()),          //3x1 weights
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
- :racehorse: **CUDA** integration

#### Essential:

- ANN functionality
- Activation functions: Relu, ReluLeaky, Sigmoid, Tanh, Swish
- saving/loading functionality
- variable batch size

#### Other:

- batch normalization layer
- dropout layer

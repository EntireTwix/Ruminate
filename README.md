![Ruminate Logo](https://github.com/EntireTwix/Ruminate/blob/main/Banner.png)
# Ruminate v1.0.3
an ML library that aims to be lightweight, fast, and generic. Written in and for Cpp20

### Network Initilization Example:
```cpp
#include "../Ruminate-main/Ruminate/net.hpp"
#include "../Ruminate-main/Ruminate/LayerModules/ann_layers.hpp"
#include "../Ruminate-main/Ruminate/HelperFiles/a_funcs.hpp"

using namespace rum;

int main()
{
     NeuralNetwork<ANN> net{
        new Input(2),                             //2 node input
        new Weight(2, 3, new RngInit()),          //2x1 weights
        new DropOut(3, 0.25),                     //dropout of 25% of the input
        new Hidden(3, ReluLeaky, ReluLeakyPrime), //3 node hidden layer
        new Weight(3, 1, new RngInit()),          //3x1 weights
        new Output(1, ReluLeaky, ReluLeakyPrime), //1 output node
    };
 }
 ```
### Learning Loop Example:
```cpp
//setting inputs and anwsers
data.At(0, 0) = gen.nextUInt(100); 
data.At(0, 1) = gen.nextUInt(100);
anw.At(0, 0) = data.At(0, 0) + data.At(0, 1);

auto res = net.ForwardProp(data); //forward propogating 
auto corrections = net.BackwordProp(res, res.back(), anw, 0.00001); //generating corrections
net.Learn(corrections); //applying corrections
```
and this could be easily multi-threaded with something like tpool.h in my OptimizedHeaders repo or any other thread pool
```cpp
for (int i = 0; i < 10; ++i)
{
    pool.AddTask([i, &corrections]() {
        data.At(0, 0) = gen.nextUInt(100);
        data.At(0, 1) = gen.nextUInt(100);
        anw.At(0, 0) = data.At(0, 0) + data.At(0, 1);

        res = net.ForwardProp(data);
        corrections[i] = net.BackwordProp(res, res.back(), anw, 0.00001);
    });
    while(pool.JobsLeft()) {} //complete jobs
    //...
    //Learn() after averaging corrections into one vec
}
```
and this is assuming your batch size is 1, as this library supports variable batch size you could multi thread each batch

# Features
#### Implementation:
* :sparkles: polymorphically based NeuralNetwork with constrained template argument for future LayerTypes
* :racehorse: fast initialization of weights and biases using pcg32 implementation from https://github.com/wjakob/pcg32
* :sparkles: Forward&Back propogation are thread safe
#### Essential:
* FC layers
* Activation functions: Relu, ReluLeaky, Sigmoid, Tanh, Swish
* saving/loading functionality
* variable batch size
#### Other:
* dropout layer

# :construction:In Progress
* He/Xavier init
* :sparkles: convolutions and pooling

# Future Features
* batch normalization
* Softmax
* :sparkles: optimizer argument for network (like Adam for example)
* :racehorse: GPU or SIMD integration (to be tested)

# Dependencies
* mat.h    from EntireTwix/OptimizedHeaders
* pcg32.h  from wjakob/pcg32

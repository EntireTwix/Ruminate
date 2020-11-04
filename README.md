![Ruminate Logo](https://github.com/EntireTwix/Ruminate/blob/main/Banner.png)
# Ruminate
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
        new Input(2),                                             //2 node input
        new Weight(2, 3, new RngInit()),                          //2x3 weights
        new HiddenDrop(3, ReluLeaky, ReluLeakyPrime, 0, 1, 0.33), //3 node hidden layer with dropout of 33%
        new Weight(3, 1,new RngInit()),                           //3x1 weights
        new Output(1, ReluLeaky, ReluLeakyPrime),                 //1 output node
    };
 }
 ```

# Features
#### Implementation:
* :sparkles: polymorphically based NeuralNetwork with constrained template argument for future LayerTypes
* :racehorse: fast initialization of weights and biases using pcg32 implementation from https://github.com/wjakob/pcg32
* :sparkles: ForwardProp thread safe
#### Essential:
* FC layers
* Activation functions: Relu, ReluLeaky, Sigmoid, Tanh, Swish
* saving/loading functionality
#### Other:
* dropout variants for input&hidden layers

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

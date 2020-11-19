![Ruminate Logo](https://github.com/EntireTwix/Ruminate/blob/main/Banner.png)
# Ruminate v1.1.3
an ML library that aims to be lightweight, fast, and generic. Written in and for **C++20**

# Usage
### Network Initilization Example:
```cpp
NeuralNetwork<ANN> net{
     new Input(2),                             //2 node input
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
auto corrections = net.BackwordProp(res, res.back(), anw, 0.002); //generating corrections
net.Learn(corrections); //applying corrections
```
and this could be easily **multi-threaded** with something like **tpool.h** in my OptimizedHeaders repo or any other thread pool
```cpp
for (int i = 0; i < 10; ++i)
{
    pool.AddTask([i, &corrections]() {
        data.At(0, 0) = gen.nextUInt(100);
        data.At(0, 1) = gen.nextUInt(100);
        anw.At(0, 0) = data.At(0, 0) + data.At(0, 1);

        res = net.ForwardProp(data);
        corrections[i] = net.BackwordProp(res, res.back(), anw, 0.002);
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
* :sparkles: polymorphically based NeuralNetwork with constrained template argument for future LayerTypes
* :racehorse: fast initialization of weights and biases using pcg32 implementation from https://github.com/wjakob/pcg32
* :sparkles: Forward&Back propogation are thread safe
* :racehorse: Cuda integration
#### Essential:
* FC layers
* Activation functions: Relu, ReluLeaky, Sigmoid, Tanh, Swish
* saving/loading functionality
* variable batch size
#### Other:
* dropout layer

# CUDA
### Usage:
to use simply compile with nvcc and will be compiled for cuda
### Performance:
Speedup can range widely depending on gpu and cpu but generally gpu does better for large matrices and cpu for small
### Compilation:
here is a compilation example
```nvcc -ccbin 'path/cl.exe' --std c++17 $fileName -o $fileNameWithoutExt -IC:/Ruminate-main/ -O3 && $dir$fileNameWithoutExt"```
### Optimization:
* you can use CUDA Occupancy Calculator found here: https://docs.nvidia.com/cuda/cuda-occupancy-calculator/index.html
to modify cuda_mat.cu's call to gpu_mat_mult to fit your gpu better
* use the program in OptimzationHeaders/CUDA called main.cu to fine tune the macro ```cpu_threshold``` in that same folder to better match the point at which your systems gpu out performs your cpu
* compiler args found here https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html

# :construction:In Progress
* batch normalization
* Softmax
* He/Xavier init
* :sparkles: CNN functionality

# Future Features
* :sparkles: optimizer argument for network (like Adam for example)

# Dependencies
* mat.hpp    from EntireTwix/OptimizedHeaders
* generics.h from EntireTwix/MiscHeaderFiles
* pcg32.h  from wjakob/pcg32
* \*mat.hpp & cuda_mat.hpp from EntireTwix/OptimzedHeaders/CUDA

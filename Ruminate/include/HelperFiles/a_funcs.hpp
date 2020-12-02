#pragma once
#include <cmath>

constexpr auto A = 0.0001;

//relu
float Relu(float x);
float ReluPrime(float x);

//leaky relu
float ReluLeaky(float x);
float ReluLeakyPrime(float x);

//tanh
float Tanh(float x);
float TanhPrime(float x);

//sigmoid
float Sigmoid(float x);
float SigmoidPrime(float x);

//swish
float Swish(float x);
float SwishPrime(float x);
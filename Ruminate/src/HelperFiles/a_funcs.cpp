#include "HelperFiles/a_funcs.hpp"

//relu

float Relu(float x)
{
    return (x > 0) * x;
}
float ReluPrime(float x)
{
    return x > 0;
}

//leaky relu

float ReluLeaky(float x)
{
    if (x < 0)
    {
        return x * A;
    }
    else
    {
        return x;
    }
}

float ReluLeakyPrime(float x)
{
    if (x < 0)
    {
        return A;
    }
    else
    {
        return 1;
    }
}

//tanh

float Tanh(float x)
{
    return tanh(x);
}
float TanhPrime(float x)
{
    return sinh(x) / cosh(x);
}

//sigmoid

float Sigmoid(float x)
{
    return 1 / (1 + exp(-x));
}

float SigmoidPrime(float x)
{
    return exp(-x) / std::pow(1 + exp(-x), 2);
}

//swish

float Swish(float x)
{
    return x * Sigmoid(x);
}

float SwishPrime(float x)
{
    return (exp(x) * (exp(x) + x + 1)) / std::pow(exp(x) + 1, 2);
}

#include "HelperFiles/a_funcs.hpp"

//relu

inline float Relu(float x)
{
    return (x > 0) * x;
}
inline float ReluPrime(float x)
{
    return x > 0;
}

//leaky relu

inline float ReluLeaky(float x)
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

inline float ReluLeakyPrime(float x)
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

inline float Tanh(float x)
{
    return tanh(x);
}
inline float TanhPrime(float x)
{
    return sinh(x) / cosh(x);
}

//sigmoid

inline float Sigmoid(float x)
{
    return 1 / (1 + exp(-x));
}

inline float SigmoidPrime(float x)
{
    return exp(-x) / std::pow(1 + exp(-x), 2);
}

//swish

inline float Swish(float x)
{
    return x * Sigmoid(x);
}

inline float SwishPrime(float x)
{
    return (exp(x) * (exp(x) + x + 1)) / std::pow(exp(x) + 1, 2);
}

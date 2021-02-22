#pragma once
#include "../third_party/half/half.hpp"

constexpr auto A = 0.0001;

namespace rum
{
    //relu

    FLOAT16 Relu(FLOAT16 x)
    {
        return (x > 0) * x;
    }
    FLOAT16 ReluPrime(FLOAT16 x)
    {
        return x > 0;
    }

    //leaky relu

    FLOAT16 ReluLeaky(FLOAT16 x)
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

    FLOAT16 ReluLeakyPrime(FLOAT16 x)
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

    FLOAT16 Tanh(FLOAT16 x)
    {
        return tanh(x);
    }
    FLOAT16 TanhPrime(FLOAT16 x)
    {
        return sinh(x) / cosh(x);
    }

    //sigmoid

    FLOAT16 Sigmoid(FLOAT16 x)
    {
        return 1 / (1 + exp(-x));
    }

    FLOAT16 SigmoidPrime(FLOAT16 x)
    {
        return exp(-x) / std::pow(1 + exp(-x), 2);
    }

    //swish

    FLOAT16 Swish(FLOAT16 x)
    {
        return x * Sigmoid(x);
    }

    FLOAT16 SwishPrime(FLOAT16 x)
    {
        return (exp(x) * (exp(x) + x + 1)) / std::pow(exp(x) + 1, 2);
    }
} // namespace rum
#pragma once
#include "../layers.hpp"
#include "../../MiscHeaderFiles-master/mat.h"
#include <random>
#include <ctime>

using FC = Layer<fMat>;

class Weight : public FC
{
private:
    fMat values;

public:
    Weight(size_t prev, size_t next, double min, double max) : values(next, prev)
    {
        std::default_random_engine generator(time(NULL));
        std::uniform_real_distribution<double> distribution(min, max);
        for (size_t i = 0; i < values.Area(); ++i)
        {
            values.FastAt(i) = distribution(generator);
        }
    }
    virtual fMat ForwardProp(const fMat &input) const override
    {
        return input.Dot(values);
    }
};

class Hidden : public FC
{
private:
    fMat bias;
    float (*Activation)(float);
    float (*ActivationPrime)(float);

public:
    Hidden(size_t nodes, float (*a)(float), float (*ap)(float)) : bias(nodes, 1), Activation(a), ActivationPrime(ap) {}
    virtual fMat ForwardProp(const fMat &input) const override
    {
        return input.Transform<true>([this](float v, size_t x, size_t y) { return Activation(v + bias.At(x, 0)); }); //to be replaced
    }
};

class InputWeight : public Weight
{
private:
    size_t expected_input;

public:
    InputWeight(size_t input_sz, size_t next, double min, double max) : Weight(1, next, min, max), expected_input(input_sz) {}
    virtual fMat ForwardProp(const fMat &input) const override
    {
        if (expected_input != input.SizeY())
            throw std::invalid_argument("input dimensions must match the expected input size");
        return Weight::ForwardProp(input);
    }
};

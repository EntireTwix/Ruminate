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
protected:
    fMat bias;
    float (*Activation)(float);
    float (*ActivationPrime)(float);

public:
    Hidden(size_t nodes, float (*a)(float), float (*ap)(float)) : bias(nodes, 1), Activation(a), ActivationPrime(ap) {}
    virtual fMat ForwardProp(const fMat &input) const override
    {
        fMat res(input.SizeX(), input.SizeY());
        for (size_t i = 0; i < input.SizeY(); ++i)
        {
            for (size_t j = 0; j < input.SizeX(); ++j)
            {
                res.At(j, i) = Activation(input.At(j, i) + bias.At(j, 0));
            }
        }
        return res;
    }
};

class Input : public Weight
{
private:
    size_t expected_input;

public:
    Input(size_t input_sz, size_t next, size_t n_inputs, double min, double max) : Weight(n_inputs, next, min, max), expected_input(input_sz) {}
    virtual fMat ForwardProp(const fMat &input) const override
    {
        if (expected_input != input.SizeY())
            throw std::invalid_argument("input dimensions must match the expected input size");
        return Weight::ForwardProp(input);
    }
};

class Output : public Hidden
{
public:
    Output(size_t nodes, float (*a)(float), float (*ap)(float)) : Hidden(nodes, a, ap) {}
    virtual fMat ForwardProp(const fMat &input) const override
    {
        fMat res(input.SizeX(), 1);
        for (size_t i = 0; i < input.SizeY(); ++i)
        {
            for (size_t j = 0; j < input.SizeX(); ++j)
            {
                res.At(j, 0) += Activation(input.At(j, i) + bias.At(j, 0));
            }
            res /= input.SizeY();
        }
        return res;
    }
};
//to do:
//improved weight and bias initilization

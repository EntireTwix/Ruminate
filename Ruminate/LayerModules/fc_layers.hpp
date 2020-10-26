#pragma once
#include <random>
#include <ctime>
#include "../layers.hpp"
#include "../../MiscHeaderFiles-master/mat.h"
#include "../pcg32.hpp"

using FC = Layer<MLMat>;

class Weight : public FC
{
private:
    MLMat values;

public:
    Weight(uint16_t prev, uint16_t next, double min, double max, pcg32 &p) : values(next, prev)
    {
        for (uint16_t i = 0; i < values.Area(); ++i)
        {
            values.FastAt(i) = p.nextFloat();
        }
    }
    virtual MLMat ForwardProp(const MLMat &input) const override
    {
        return input.Dot(values);
    }
};

class Hidden : public FC, public virtual IActivationFuncs<float>
{
protected:
    MLMat bias;

public:
    Hidden(uint16_t nodes, float (*a)(float), float (*ap)(float), pcg32 &p) : IActivationFuncs(a, ap), bias(nodes, 1)
    {
        for (uint16_t i = 0; i < bias.Area(); ++i)
        {
            bias.FastAt(i) = p.nextFloat();
        }
    }
    virtual MLMat ForwardProp(const MLMat &input) const override
    {
        MLMat res(input.SizeX(), input.SizeY());
        for (uint16_t i = 0; i < input.SizeY(); ++i)
        {
            for (uint16_t j = 0; j < input.SizeX(); ++j)
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
    uint16_t expected_input;

public:
    Input(uint16_t input_sz, uint16_t next, uint16_t n_inputs, double min, double max, pcg32 &p) : Weight(n_inputs, next, min, max, p), expected_input(input_sz) {}
    virtual MLMat ForwardProp(const MLMat &input) const override
    {
        if (expected_input != input.SizeY())
            throw std::invalid_argument("input dimensions must match the expected input size");
        return Weight::ForwardProp(input);
    }
};

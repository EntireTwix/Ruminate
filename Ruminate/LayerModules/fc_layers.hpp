#pragma once
#include <random>
#include <ctime>
#include "../layers.hpp"
#include "../../../OptimizedHeaders-main/mat.hpp"
#include "../../../pcg32-master/pcg32.h"

namespace rum
{
    using FC = Layer<MLMat>;

    class Weight : public FC
    {
    private:
        MLMat values;

    public:
        Weight(uint16_t prev, uint16_t next, float min, float max, pcg32 &p) : values(next, prev)
        {
            for (uint16_t i = 0; i < values.Area(); ++i)
            {
                values.FastAt(i) = min + (p.nextFloat() * (max - min));
            }
        }
        virtual MLMat &internal() { return values; }
        virtual MLMat ForwardProp(const MLMat &input) const override { return input.Dot(values); }
        virtual MLMat BackwardProp(MLMat &cost, const std::vector<MLMat> &forwardRes, Layer **layers, size_t index) const override
        {
            return forwardRes[index - 1].VecMult(cost);
        }
    };

    class Hidden : public FC, public IActivationFuncs<float>
    {
    protected:
        MLMat bias;

    public:
        Hidden(uint16_t nodes, float (*a)(float), float (*ap)(float), float min, float max, pcg32 &p) : IActivationFuncs(a, ap), bias(nodes, 1)
        {
            for (uint16_t i = 0; i < bias.Area(); ++i)
            {
                bias.FastAt(i) = min + (p.nextFloat() * (max - min));
            }
        }
        virtual MLMat &internal() override { return bias; }
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
    public:
        Input(uint16_t n_inputs, uint16_t input_sz, uint16_t next, float min, float max, pcg32 &p) : Weight(input_sz, next, min, max, p) {}
    };

    class Output : public Hidden
    {
    public:
        Output(uint16_t nodes, float (*a)(float), float (*ap)(float), float min, float max, pcg32 &p) : Hidden(nodes, a, ap, min, max, p) {}
        virtual MLMat BackwardProp(MLMat &cost, const std::vector<MLMat> &forwardRes, Layer **layers, size_t index) const override
        {
            return cost *= forwardRes[index].Transform(ActivationPrime); //will be optimized later
        }
    };
} // namespace rum

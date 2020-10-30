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
    protected:
        MLMat weights;

    public:
        Weight(uint16_t prev, uint16_t next, float w_min, float w_max, pcg32 &rng) : weights(prev, next)
        {
            for (uint16_t i = 0; i < weights.Area(); ++i)
            {
                weights.FastAt(i) = (rng.nextFloat() * w_max) + w_min;
            }
        }

        MLMat &internal() override
        {
            return weights;
        }
        virtual MLMat ForwardProp(const MLMat &input) override
        {
            std::cout << "W\n";
            return weights.Dot(input);
        }
        virtual MLMat BackwardProp(MLMat &cost, const std::vector<MLMat> &forwardRes, FC **layers, size_t index) const override
        {
            std::cout << "W\n";
            return cost.Dot(forwardRes[index - 1]);
        }
    };

    class Hidden : public FC, public IActivationFuncs<float>
    {
    protected:
        MLMat biases;

    public:
        Hidden(uint16_t hidden_nodes, float (*a)(float), float (*ap)(float), float b_min, float b_max, pcg32 &rng) : IActivationFuncs(a, ap), biases(hidden_nodes, 1)
        {
            for (uint16_t i = 0; i < biases.Area(); ++i)
            {
                biases.FastAt(i) = (rng.nextFloat() * b_max) + b_min;
            }
        }

        MLMat &internal() override
        {
            return biases;
        }
        virtual MLMat ForwardProp(const MLMat &input) override
        {
            std::cout << "H\n";
            MLMat res(input.SizeX(), input.SizeY());
            for (uint16_t i = 0; i < input.SizeY(); ++i)
            {
                for (uint16_t j = 0; j < input.SizeX(); ++j)
                {
                    res.At(j, i) = this->Activation(input.At(j, i) + biases.FastAt(j));
                }
            }
            return res;
        }
        virtual MLMat BackwardProp(MLMat &cost, const std::vector<MLMat> &forwardRes, FC **layers, size_t index) const override
        {
            std::cout << "H\n";
            return cost = cost.Dot(layers[index + 1]->internal()) * forwardRes[index - 1].Transform(ActivationPrime);
        }
    };

    class Input : public FC
    {
    protected:
        MLMat inp;

    public:
        virtual MLMat &internal()
        {
            return inp;
        }
        virtual MLMat ForwardProp(const MLMat &input) override
        {
            std::cout << "I\n";
            return inp = input;
        }
    };

    class Output : public Hidden
    {
    public:
        Output(uint16_t hidden_nodes, float (*a)(float), float (*ap)(float), float b_min, float b_max, pcg32 &rng) : Hidden(hidden_nodes, a, ap, b_min, b_max, rng) {}
        virtual MLMat BackwardProp(MLMat &cost, const std::vector<MLMat> &forwardRes, FC **layers, size_t index) const override
        {
            std::cout << "O\n";
            return cost = forwardRes[index].Transform(ActivationPrime) * cost;
        }
    };
} // namespace rum

#pragma once
#include <random>
#include "../layers.hpp"
#include "../../../OptimizedHeaders-main/mat.hpp"
#include "../../../pcg32-master/pcg32.h"

//weight
//hidden
//input
//output

namespace rum
{
    using ANN = Layer<MLMat>;

    class Weight : public ANN
    {
    protected:
        MLMat weights;

    public:
        Weight(uint16_t prev, uint16_t next, float w_min, float w_max, pcg32 &rng, auto &&... saved_params) : weights(prev, next, saved_params...)
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
            //std::cout << "W\n";
            return weights.Dot(input);
        }
        virtual MLMat BackwardProp(MLMat &cost, const std::vector<MLMat> &forwardRes, ANN **layers, size_t index) const override
        {
            //std::cout << "W\n";
            return cost.Dot(forwardRes[index - 1]);
        }
    };

    class Hidden : public ANN, public IActivationFuncs<float>
    {
    private:
        MLMat
            biases,
            t_vals;
        float dropThres;
        pcg32 generator;

    public:
        Hidden(uint16_t hidden_nodes, float (*a)(float), float (*ap)(float), float b_min, float b_max, float dropThres, pcg32 &rng, auto &&... saved_params) : IActivationFuncs(a, ap), biases(hidden_nodes, 1, saved_params...), t_vals(hidden_nodes, 1), dropThres(dropThres), generator(rng)
        {
            //only random init if saved params is empty
            for (uint16_t i = 0; i < biases.Area(); ++i)
            {
                if constexpr (!sizeof...(saved_params))
                    biases.FastAt(i) = (rng.nextFloat() * b_max) + b_min;
            }
        }

        MLMat &internal() override
        {
            return biases;
        }
        virtual MLMat ForwardProp(const MLMat &input) override
        {
            //std::cout << "H\n";
            MLMat res(input.SizeX(), input.SizeY());
            for (uint16_t i = 0; i < input.SizeY(); ++i)
            {
                for (uint16_t j = 0; j < input.SizeX(); ++j)
                {
                    res.At(j, i) = this->Activation(input.At(j, i) + biases.FastAt(j)) * (t_vals.FastAt(i) = (generator.nextFloat() > dropThres));
                }
            }
            return res;
        }
        virtual MLMat BackwardProp(MLMat &cost, const std::vector<MLMat> &forwardRes, ANN **layers, size_t index) const override
        {
            //std::cout << "H\n";
            return cost = cost.Dot(layers[index + 1]->internal()) * forwardRes[index - 1].Transform(ActivationPrime) * t_vals; //to be optimized
        }
    }; // namespace rum

    class Input : public ANN
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
            //std::cout << "I\n";
            return inp = input;
        }
    };

    class Output : public ANN, public IActivationFuncs<float>
    {
    private:
        MLMat biases;

    public:
        Output(uint16_t hidden_nodes, float (*a)(float), float (*ap)(float), float b_min, float b_max, pcg32 &rng, auto &&... saved_params) : IActivationFuncs(a, ap), biases(hidden_nodes, 1, saved_params...)
        {
            //only random init if saved params is empty
            for (uint16_t i = 0; i < biases.Area(); ++i)
            {
                if constexpr (!sizeof...(saved_params))
                    biases.FastAt(i) = (rng.nextFloat() * b_max) + b_min;
            }
        }
        virtual MLMat ForwardProp(const MLMat &input) override
        {
            //std::cout << "H\n";
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
        virtual MLMat BackwardProp(MLMat &cost, const std::vector<MLMat> &forwardRes, ANN **layers, size_t index) const override
        {
            //std::cout << "O\n";
            return cost = forwardRes[index].Transform(ActivationPrime) * cost; //to be optimized
        }
        virtual MLMat &internal()
        {
            return biases;
        }
    };
} // namespace rum

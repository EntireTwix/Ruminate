#pragma once
#include "layers.hpp"
#include "rng_helpers.hpp"

namespace rum
{
    using ANN = Layer<MLMat>;

    class Weight : public ANN
    {
    protected:
        MLMat weights;

    public:
        template <RngGen T>
        Weight(uint16_t prev, uint16_t next, T &&rng) : weights(prev, next)
        {
            rng.Generator(weights);
        }
        template <typename... Params>
        Weight(uint16_t prev, uint16_t next, Params &&...saved_params) : weights(prev, next, saved_params...) {}

        MLMat &internal() override
        {
            return weights;
        }

        virtual MLMat ForwardProp(const MLMat &input) override
        {
            if constexpr (LOG_LAYERS_FLAG)
            {
                std::cout << "W F\n";
            }
            return weights.Dot(input);
        }

        virtual MLMat BackwardProp(MLMat &cost, const std::vector<MLMat> &forwardRes, ANN **const layers, size_t index) const override
        {
            if constexpr (LOG_LAYERS_FLAG)
            {
                std::cout << "W B\n";
            }
            return cost.Dot(forwardRes[index - 1]);
        }
    };

    class Hidden : public ANN, public IActivationFuncs<float>
    {
    protected:
        MLMat biases;

    public:
        template <typename... Params>
        Hidden(uint16_t hidden_nodes, float (*a)(float), float (*ap)(float), Params &&...saved_params) : IActivationFuncs(a, ap), biases(hidden_nodes, 1, saved_params...) {}

        MLMat &internal() override
        {
            return biases;
        }

        virtual MLMat ForwardProp(const MLMat &input) override
        {
            if constexpr (LOG_LAYERS_FLAG)
            {
                std::cout << "H F\n";
            }
            MLMat res(input.SizeX(), input.SizeY());
            for (uint16_t i = 0; i < input.SizeY(); ++i)
            {
                for (uint16_t j = 0; j < input.SizeX(); ++j)
                {
                    res.At(j, i) = this->Activation(input.At(j, i) + biases.FastAt(i));
                }
            }

            return res;
        }

        virtual MLMat BackwardProp(MLMat &cost, const std::vector<MLMat> &forwardRes, ANN **const layers, size_t index) const override
        {
            if constexpr (LOG_LAYERS_FLAG)
            {
                std::cout << "H B\n";
            }
            cost = (cost.Dot(layers[index + 1]->inside()) * forwardRes[index - 1]); //TODO: to be optimized
            std::transform(cost.begin(), cost.end(), cost.begin(), ActivationPrime);
            return cost;
        }
    };

    class Output : public Hidden
    {
    public:
        Output(uint16_t hidden_nodes, float (*a)(float), float (*ap)(float)) : Hidden(hidden_nodes, a, ap) {}
        virtual MLMat BackwardProp(MLMat &cost, const std::vector<MLMat> &forwardRes, ANN **layers, size_t index) const override
        {
            if constexpr (LOG_LAYERS_FLAG)
            {
                std::cout << "O B\n";
            }
            for (uint32_t i = 0; i < cost.SizeX() * cost.SizeY(); ++i)
            {
                cost.FastAt(i) *= ActivationPrime(forwardRes[index].FastAt(i));
            }
            return cost;
        }
    };
} // namespace rum

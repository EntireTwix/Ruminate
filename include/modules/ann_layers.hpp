#pragma once
#include "layers.hpp"
#include "rng_helpers.hpp"

namespace rum
{
    template <Matrix M = MLMat>
    class Weight : public Layer<M>
    {
    protected:
        M weights;

    public:
        template <RngGen T>
        Weight(uint_fast16_t prev, uint_fast16_t next, T &&rng) : weights(prev, next)
        {
            rng.Generator(weights);
        }

        template <typename... Params>
        Weight(uint_fast16_t prev, uint_fast16_t next, Params &&...saved_params) : weights(prev, next, saved_params...) {}

        M &internal() override
        {
            return weights;
        }

        virtual M ForwardProp(const M &input) override
        {
            if constexpr (LOG_LAYERS_FLAG)
            {
                std::cout << "W F\n";
            }
            return weights.Dot(input);
        }

        virtual M BackwardProp(M &cost, const std::vector<M> &forwardRes, Layer<M> **const layers, size_t index) const override
        {
            if constexpr (LOG_LAYERS_FLAG)
            {
                std::cout << "W B\n";
            }
            return cost.Dot(forwardRes[index - 1]);
        }
    };

    template <Matrix M = MLMat>
    class Hidden : public Layer<M>, public IActivationFuncs<typename M::type>
    {
    protected:
        M biases;

    public:
        template <typename... Params>
        Hidden(uint_fast16_t hidden_nodes, typename M::type (*a)(typename M::type), typename M::type (*ap)(typename M::type), Params &&...saved_params) : IActivationFuncs<typename M::type>(a, ap), biases(hidden_nodes, 1, saved_params...) {}

        M &internal() override
        {
            return biases;
        }

        virtual M ForwardProp(const M &input) override
        {
            if constexpr (LOG_LAYERS_FLAG)
            {
                std::cout << "H F\n";
            }
            M res(input.SizeX(), input.SizeY());
            for (uint_fast16_t i = 0; i < input.SizeY(); ++i)
            {
                for (uint_fast16_t j = 0; j < input.SizeX(); ++j)
                {
                    res.At(j, i) = this->Activation(input.At(j, i) + biases.FastAt(i));
                }
            }

            return res;
        }

        virtual M BackwardProp(M &cost, const std::vector<M> &forwardRes, Layer<M> **const layers, size_t index) const override
        {
            if constexpr (LOG_LAYERS_FLAG)
            {
                std::cout << "H B\n";
            }
            cost = (cost.Dot(layers[index + 1]->inside()) * forwardRes[index - 1]); //TODO: to be optimized
            std::transform(cost.begin(), cost.end(), cost.begin(), this->ActivationPrime);
            return cost;
        }
    };

    template <Matrix M = MLMat>
    class Output : public Hidden<M>
    {
    public:
        Output(uint_fast16_t hidden_nodes, typename M::type (*a)(typename M::type), typename M::type (*ap)(typename M::type)) : Hidden<M>(hidden_nodes, a, ap) {}
        virtual M BackwardProp(M &cost, const std::vector<M> &forwardRes, Layer<M> **layers, size_t index) const override
        {
            if constexpr (LOG_LAYERS_FLAG)
            {
                std::cout << "O B\n";
            }
            for (uint32_t i = 0; i < cost.SizeX() * cost.SizeY(); ++i)
            {
                cost.FastAt(i) *= this->ActivationPrime(forwardRes[index].FastAt(i));
            }
            return cost;
        }
    };
} // namespace rum

#pragma once
#include <numeric>
#include "layers.h"

namespace rum
{
    template <LayerType T>
    class NeuralNetwork
    {
    private:
        T **layers = nullptr;
        uint8_t sz;
        using RT = typename T::type; //representation type, for example: fMat

    public:
        NeuralNetwork() = delete;

        template <typename... Params>
        NeuralNetwork(Params &&...args);

        //thread safe
        std::vector<RT> ForwardProp(const RT &input) const;

        //thread safe
        std::vector<RT> BackwordProp(const std::vector<RT> &forwardRes, RT &&cost_prime, float lr);

        //not thread safe
        void Learn(const std::vector<RT> &backRes);
        std::string Save() const;
        RT Cost(const RT &guess, const RT &anwser) const;
        RT CostPrime(const RT &guess, const RT &anwser) const;
        ~NeuralNetwork();
    };
}; // namespace rum

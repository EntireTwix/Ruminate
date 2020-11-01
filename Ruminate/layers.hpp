#pragma once
#include <concepts>
#include <type_traits>
#include <vector>
#include "../../OptimizedHeaders-main/mat.hpp"

namespace rum
{
    //an abstract type that is used polymorphically in net.h
    template <Matrix M>
    class Layer
    {
    public:
        using type = M;

        virtual M ForwardProp(const M &input) { return input; } //maybe could be made reference
        virtual M BackwardProp(M &cost, const std::vector<M> &forwardRes, Layer **layers, size_t index) const { return cost; }
        virtual M &internal() = 0;
    };

    template <typename T>
    class IActivationFuncs
    {
    protected:
        T(*Activation)
        (T);
        T(*ActivationPrime)
        (T);

    public:
        IActivationFuncs(T (*a)(T), T (*ap)(T)) : Activation(a), ActivationPrime(ap) {}
    };

    template <typename T>
    concept LayerType = std::is_base_of<Layer<typename T::type>, T>::value;
}; // namespace rum

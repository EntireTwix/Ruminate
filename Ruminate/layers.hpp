#pragma once
#include <concepts>
#include <type_traits>
#include <vector>

namespace rum
{

    //an abstract type that is used polymorphically in net.h
    template <typename T>
    class Layer
    {
    public:
        using type = T;

        virtual T ForwardProp(const T &) const = 0;
        virtual T BackwardProp(T &cost, const std::vector<T> &forwardRes, Layer **layers, size_t index) const { return cost; }
        virtual T &internal() = 0;
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

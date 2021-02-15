#include "layers.h"

namespace rum
{
    //an abstract type that is used polymorphically in net.h
    template <Matrix M>
    M Layer<M>::ForwardProp(const M &input)
    {
        return input;
    }

    template <Matrix M>
    M Layer<M>::BackwardProp(M &cost, const std::vector<M> &forwardRes, Layer **layers, size_t index) const
    {
        return cost;
    }

    template <Matrix M>
    void Layer<M>::Learn(const M &correction)
    {
        this->internal() -= correction;
    }

    template <typename T>
    IActivationFuncs<T>::IActivationFuncs(T (*a)(T), T (*ap)(T)) : Activation(a), ActivationPrime(ap) {}
}; // namespace rum
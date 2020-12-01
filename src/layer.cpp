#include "layers.hpp"

namespace rum
{
    //an abstract type that is used polymorphically in net.h
#ifdef __NVCC__
    template <typename M>
#else
    template <Matrix M>
#endif
    virtual M Layer::ForwardProp(const M &input)
    {
        return input;
    }

#ifdef __NVCC__
    template <typename M>
#else
    template <Matrix M>
#endif
    virtual M Layer::BackwardProp(M &cost, const std::vector<M> &forwardRes, Layer **layers, size_t index) const
    {
        return cost;
    }

#ifdef __NVCC__
    template <typename M>
#else
    template <Matrix M>
#endif
    virtual void Layer::Learn(const M &correction)
    {
        this->internal() -= correction;
    }

    template <typename T>
    IActivationFuncs::IActivationFuncs(T (*a)(T), T (*ap)(T)) : Activation(a), ActivationPrime(ap) {}
}; // namespace rum
#pragma once
#include <vector>
#include "../dependencies/CUDA/mat.hpp"

namespace rum
{
    //an abstract type that is used polymorphically in net.h
    template <typename M>
    class Layer
    {
    public:
        using type = M;

        virtual M ForwardProp(const M &input);
        virtual M BackwardProp(M &cost, const std::vector<M> &forwardRes, Layer **layers, size_t index) const;
        virtual void Learn(const M &correction);
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
        IActivationFuncs(T (*a)(T), T (*ap)(T));
    };
}; // namespace rum

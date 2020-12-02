#pragma once
#include <vector>

#ifdef __NVCC__
#include "../dependencies/CUDA/mat.hpp"
#else
#include "../dependencies/mat.hpp"
#endif

namespace rum
{
    //an abstract type that is used polymorphically in net.h
#ifdef __NVCC__
    template <typename M>
#else
    template <Matrix M>
#endif
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

    template <typename T>
    concept LayerType = std::is_base_of<Layer<typename T::type>, T>::value;
}; // namespace rum

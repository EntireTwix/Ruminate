#pragma once
#include <vector>
#include "../third_party/OptimizedHeaders/mat.hpp"

namespace rum
{
    using MLMat = Mat<float>;

    constexpr bool LOG_LAYERS_FLAG = false; //when toggled program will be compiled with outputs for forward and backprops for debugging

    //an abstract typecd  that is used polymorphically in net.h
    template <Matrix M>
    class Layer
    {
    private:
        //to be used for exclusively Learn()
        virtual M &internal() = 0;

    public:
        using type = M;

        virtual M ForwardProp(const M &input)
        {
            return input;
        }

        virtual M BackwardProp(M &cost, const std::vector<M> &forwardRes, Layer **const layers, size_t index) const
        {
            return cost;
        }

        virtual void Learn(const M &correction)
        {
            this->internal() -= correction;
        }

        //to be used in backprop
        const M &inside() { return internal(); }

        //( ⚆ _ ⚆ ) to avoid UB
        virtual ~Layer() {}
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
} // namespace rum
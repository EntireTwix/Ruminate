#pragma once
#include "layers.hpp"

namespace rum
{
    template <uint_fast8_t channels>
    struct Pixel : public std::array<uint_fast8_t, channels>
    {
        Pixel operator-=(Pixel p)
        {
            Pixel res;
            for (uint_fast8_t i = 0; i < channels; ++i)
            {
                res[i] -= p[i];
            }
            return res;
        }
        friend std::ostream &operator<<(std::ostream &os, Pixel p)
        {
            os << '(';
            for (uint_fast8_t i = 0; i < channels; ++i)
            {
                os << (int)p[i];
                if (i + 1 < channels)
                {
                    os << ',';
                }
            }
            os << ')';
            return os;
        }
    };

    using RGB = Mat<Pixel<3>>;
    using RGBA = Mat<Pixel<4>>;

    // template <Matrix M = RGB>
    // class KernelWeights
    // {
    // protected:
    //     using SizeT = typename M::storage_type;
    //     M weights;

    // public:
    //     Kernel(SizeT width, SizeT height) : weights(width, height) {}
    // };

    template <Matrix M = RGB>
    class Flatten : public Layer<M>
    {
    public:
        Flatten() {}
        virtual M *internal()
        {
            return nullptr;
        }

        virtual M ForwardProp(const M &input) override
        {
            if constexpr (LOG_LAYERS_FLAG)
            {
                std::cout << "F F\n";
            }

            M temp = input;
            temp.Flatten();
            return temp;
        }
        virtual void Learn(const M &correction) override {} //doesnt correct
    };
};
#pragma once
#include <array>
#include "layers.hpp"

namespace rum
{
    template <Matrix M>
    using CNN = Layer<M>;

    template <uint_fast8_t channels>
    struct Pixel : public std::array<uint_fast8_t, channels>
    {
        Pixel operator-=(const Pixel &p)
        {
            Pixel res;
            for (uint_fast8_t i = 0; i < channels; ++i)
            {
                res[i] -= p[i];
            }
            return res;
        }
        friend std::ostream &operator<<(std::ostream &os, const Pixel &p)
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
};
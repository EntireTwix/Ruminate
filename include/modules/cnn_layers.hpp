#pragma once
#include <array>
#include "layers.hpp"

namespace rum
{
    template <Matrix M>
    using CNN = Layer<M>;

    template <uint_fast8_t channels>
    using IMG = Mat<uint_fast8_t, channels>;

    using RGB = std::array<uint_fast8_t, 3>;
    using RGBA = std::array<uint_fast8_t, 4>;
};
#pragma once
#include <array>
#include "layers.hpp"

namespace rum
{
    template <Matrix M>
    using CNN = Layer<M>;

    using RGB = Mat<std::array<uint_fast8_t, 3>>;
    using RGBA = Mat<std::array<uint_fast8_t, 4>>;
};
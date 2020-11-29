#pragma once
#include "../layers.hpp"
#include "../Dependencies/mat.hpp"
#include "../Dependencies/pcg32.h"

//convolution
//pooling

namespace rum
{
    using CNN = Layer<ImgMat>;

    class Convolution : public CNN
    {
    };
}; // namespace rum

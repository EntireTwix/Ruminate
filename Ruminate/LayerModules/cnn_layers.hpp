#pragma once
#include "../layers.hpp"
#include "../../../OptimizedHeaders-main/mat.hpp"
#include "../../../pcg32-master/pcg32.h"

//convolution
//pooling

namespace rum
{
    using CNN = Layer<ImgMat>;

    class Convolution : public CNN
    {
    };
}; // namespace rum

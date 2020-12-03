#pragma once
#include <ctime>
#include "../../vendor/pcg32/pcg32.h"

#ifdef __NVCC__
#include "../../vendor/OptimizedHeaders/CUDA/mat.hpp"
#else
#include "../../vendor/OptimizedHeaders/mat.hpp"
#endif


namespace rum
{
    static pcg32 gen(time(NULL) << 8, time(NULL) >> 8);

    class RngInit
    {
    protected:
        float lowest = 0, highest = 1;

    public:
        RngInit() = default;
        RngInit(float lowest, float highest);
        virtual void Generator(MLMat &mat) const;
    };

}; // namespace rum

#pragma once
#include <ctime>
#include "pcg32/pcg32.h"
#include "OptimizedHeaders/mat.hpp"

namespace rum
{
    static pcg32 gen(time(NULL) << 8, time(NULL) >> 8); //generator for all rng

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

#pragma once
#include <ctime>
#include "../third_party/pcg32/pcg32.h"
#include "../third_party/OptimizedHeaders/mat.hpp"

namespace rum
{
    static pcg32 gen(time(NULL) << 8, time(NULL) >> 8); //generator for all rng

    class RngInit
    {
    protected:
        float lowest = 0, highest = 1;

    public:
        RngInit() = default;
        RngInit(float lowest, float highest) : lowest(lowest), highest(highest) {}
        virtual void Generator(MLMat &mat) const
        {
            for (auto &i : mat)
            {
                i = (gen.nextFloat() * this->highest) + this->lowest;
            }
        }
    };

    template <typename T>
    concept RngGen = std::is_base_of_v<T, RngInit>;
}; // namespace rum

#include "HelperFiles/rng_helpers.hpp"

namespace rum
{
    RngInit::RngInit(float lowest, float highest) : lowest(lowest), highest(highest) {}
    void RngInit::Generator(MLMat &mat) const
    {
        for (uint32_t i = 0; i < mat.Area(); ++i)
        {
            mat.FastAt(i) = (gen.nextFloat() * this->highest) + this->lowest;
        }
    }

}; // namespace rum

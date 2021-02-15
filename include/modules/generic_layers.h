#pragma once
#include "layers.h"
#include "rng_helpers.h"

namespace rum
{
    template <Matrix M>
    class Input : public Layer<M>
    {
    protected:
        M inp;

    public:
        Input(uint16_t input_sz);
        virtual M &internal();
        virtual M ForwardProp(const M &input) override;
        virtual void Learn(const M &correction) override;
    };

    template <Matrix M>
    class DropOut : public Layer<M>
    {
    private:
        M t_vals;
        float thres;

    public:
        DropOut(uint16_t sz, float thres);
        virtual M ForwardProp(const M &input) override;
        virtual M BackwardProp(M &cost, const std::vector<M> &forwardRes, Layer<M> **layers, size_t index) const override;
        virtual M &internal() override;
        virtual void Learn(const M &correction) override;
    };

    template <Matrix M>
    class Batch : public Input<M>
    {
    private:
        uint8_t batch_sz;

    public:
        Batch(uint8_t batch_sz, uint16_t input_sz);
        virtual M ForwardProp(const M &input) override;
    };
} // namespace rum

#pragma once
#include "layers.hpp"

namespace rum
{
#ifdef __NVCC__
    template <typename M>
#else
    template <Matrix M>
#endif
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

#ifdef __NVCC__
    template <typename M>
#else
    template <Matrix M>
#endif
    class DropOut : public Layer<M>
    {
    private:
        M t_vals;
        float thres;

    public:
        DropOut(uint16_t sz, float thres);
        virtual M ForwardProp(const M &input) override;
        virtual M BackwardProp(M &cost, const std::vector<M> &forwardRes, ANN **layers, size_t index) const override;
        virtual M &internal() override;
        virtual void Learn(const M &correction) override;
    };

#ifdef __NVCC__
    template <typename M>
#else
    template <Matrix M>
#endif
    class Batch : public Input<M>
    {
    private:
        uint8_t batch_sz;

    public:
        Batch(uint8_t batch_sz, uint16_t input_sz);
        virtual M ForwardProp(const M &input) override;
    };
} // namespace rum

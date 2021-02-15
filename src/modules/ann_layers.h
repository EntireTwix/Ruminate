#pragma once
#include "layers.h"
#include "rng_helpers.h"

namespace rum
{
    using ANN = Layer<MLMat>;

    class Weight : public ANN
    {
    protected:
        MLMat weights;

    public:
        template <typename... Params>
        Weight(uint16_t prev, uint16_t next, RngInit *rng, Params &&...saved_params);

        MLMat &internal() override;
        virtual MLMat ForwardProp(const MLMat &input) override;
        virtual MLMat BackwardProp(MLMat &cost, const std::vector<MLMat> &forwardRes, ANN **layers, size_t index) const override;
    };

    class Hidden : public ANN, public IActivationFuncs<float>
    {
    protected:
        MLMat biases;

    public:
        template <typename... Params>
        Hidden(uint16_t hidden_nodes, float (*a)(float), float (*ap)(float), Params &&...saved_params);

        MLMat &internal() override;
        virtual MLMat ForwardProp(const MLMat &input) override;
        virtual MLMat BackwardProp(MLMat &cost, const std::vector<MLMat> &forwardRes, ANN **layers, size_t index) const override;
    };

    class Output : public Hidden
    {
    public:
        Output(uint16_t hidden_nodes, float (*a)(float), float (*ap)(float));
        virtual MLMat BackwardProp(MLMat &cost, const std::vector<MLMat> &forwardRes, ANN **layers, size_t index) const override;
    };
} // namespace rum
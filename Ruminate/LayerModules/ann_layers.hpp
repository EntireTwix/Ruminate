#pragma once
#include <random>
#include <ctime>
#include "../layers.hpp"
#include "../HelperFiles/rng_helpers.hpp"

namespace rum
{
    using ANN = Layer<MLMat>;

    class Input : public ANN
    {
    protected:
        MLMat inp;

    public:
        Input(uint8_t input_sz) : inp(1, input_sz) {}
        virtual MLMat &internal()
        {
            return inp;
        }
        virtual MLMat ForwardProp(const MLMat &input) override
        {
            //std::cout << "I\n";
            return inp = input;
        }
    };

    class Weight : public ANN
    {
    protected:
        MLMat weights;

    public:
        Weight(uint16_t prev, uint16_t next, RngInit *rng, auto &&... saved_params) : weights(prev, next, saved_params...)
        {
            rng->Generator(weights);
            delete rng;
        }

        MLMat &internal() override
        {
            return weights;
        }
        virtual MLMat ForwardProp(const MLMat &input) override
        {
            //std::cout << "W\n";
            return weights.Dot(input);
        }
        virtual MLMat BackwardProp(MLMat &cost, const std::vector<MLMat> &forwardRes, ANN **layers, size_t index) const override
        {
            //std::cout << "W\n";
            return cost.Dot(forwardRes[index - 1]);
        }
    };

    class Hidden : public ANN, public IActivationFuncs<float>
    {
    protected:
        MLMat biases;

    public:
        Hidden(uint16_t hidden_nodes, float (*a)(float), float (*ap)(float), auto &&... saved_params) : IActivationFuncs(a, ap), biases(hidden_nodes, 1, saved_params...) {}

        MLMat &internal() override
        {
            return biases;
        }
        virtual MLMat ForwardProp(const MLMat &input) override
        {
            //std::cout << "H\n";
            MLMat res(input.SizeX(), input.SizeY());
            for (uint16_t i = 0; i < input.SizeY(); ++i)
            {
                for (uint16_t j = 0; j < input.SizeX(); ++j)
                {
                    res.At(j, i) = this->Activation(input.At(j, i) + biases.FastAt(i));
                }
            }
            return res;
        }
        virtual MLMat BackwardProp(MLMat &cost, const std::vector<MLMat> &forwardRes, ANN **layers, size_t index) const override
        {
            //std::cout << "H\n";
            return cost = cost.Dot(layers[index + 1]->internal()) * forwardRes[index - 1].Transform(ActivationPrime); //to be optimized
        }
    };

    class Output : public Hidden
    {
    public:
        Output(uint16_t hidden_nodes, float (*a)(float), float (*ap)(float)) : Hidden(hidden_nodes, a, ap) {}
        virtual MLMat BackwardProp(MLMat &cost, const std::vector<MLMat> &forwardRes, ANN **layers, size_t index) const override
        {
            //std::cout << "O\n";
            for (uint32_t i = 0; i < cost.Area(); ++i)
            {
                cost.FastAt(i) *= ActivationPrime(forwardRes[index].FastAt(i));
            }
            return cost;
        }
    };

    class IDrop //interface for dropout layers
    {
    protected:
        MLMat t_vals;
        float threshold;

    public:
        IDrop(uint16_t hidden_nodes, float thres) : t_vals(hidden_nodes, 1), threshold(thres) {}
    };

    class InputDrop : public Input, public IDrop
    {
    public:
        InputDrop(uint8_t input_sz, float thres) : Input(input_sz), IDrop(input_sz, thres) {}
        virtual MLMat &internal()
        {
            return inp;
        }
        virtual MLMat ForwardProp(const MLMat &input) override
        {
            //std::cout << "Id\n";
            for (uint32_t i = 0; i < input.Area(); ++i)
            {
                inp.FastAt(i) = input.FastAt(i) * (t_vals.FastAt(i) = gen.nextFloat() > threshold);
            }
            return inp;
        }
    };

    class HiddenDrop : public Hidden, public IDrop
    {
    public:
        HiddenDrop(uint16_t hidden_nodes, float (*a)(float), float (*ap)(float), float thres, auto &&... saved_params) : Hidden(hidden_nodes, a, ap, saved_params...), IDrop(hidden_nodes, thres) {}
        virtual MLMat ForwardProp(const MLMat &input) override
        {
            //std::cout << "Hd\n";
            MLMat res(input.SizeX(), input.SizeY());
            for (uint16_t i = 0; i < input.SizeY(); ++i)
            {
                for (uint16_t j = 0; j < input.SizeX(); ++j)
                {
                    res.At(j, i) = this->Activation(input.At(j, i) + biases.FastAt(i)) * (t_vals.FastAt(i) = gen.nextFloat() > threshold);
                }
            }
            return res;
        }
        virtual MLMat BackwardProp(MLMat &cost, const std::vector<MLMat> &forwardRes, ANN **layers, size_t index) const override
        {
            //std::cout << "Hd\n";
            return cost = cost.Dot(layers[index + 1]->internal()) * forwardRes[index - 1].Transform(ActivationPrime) * t_vals; //to be optimized
        }
    };
} // namespace rum

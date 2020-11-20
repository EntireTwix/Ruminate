#pragma once
#include <random>
#include <ctime>
#include "../layers.hpp"
#include "../HelperFiles/rng_helpers.hpp"

namespace rum
{
    using ANN = Layer<MLMat>;

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
        Input(uint16_t input_sz) : inp(1, input_sz) {}
        virtual M &internal()
        {
            return inp;
        }
        virtual M ForwardProp(const M &input) override
        {
            //std::cout << "I\n";
            return inp = input; //unfortunately a copy call
        }
        virtual void Learn(const M &correction) override {} //doesnt correct
    };

    class Weight : public ANN
    {
    protected:
        MLMat weights;

    public:
        template <typename... Params>
        Weight(uint16_t prev, uint16_t next, RngInit *rng, Params &&... saved_params) : weights(prev, next, saved_params...)
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
        template <typename... Params>
        Hidden(uint16_t hidden_nodes, float (*a)(float), float (*ap)(float), Params &&... saved_params) : IActivationFuncs(a, ap), biases(hidden_nodes, 1, saved_params...) {}

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
            return cost = cost.Dot(layers[index + 1]->internal()) * forwardRes[index - 1].Transform(ActivationPrime); //TODO: to be optimized
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
        DropOut(uint16_t sz, float thres) : t_vals(sz, 1), thres(thres) {}
        virtual M ForwardProp(const M &input) override
        {
            M res(input.SizeX(), input.SizeY());
            for (size_t i = 0; i < input.Area(); ++i)
            {
                res.FastAt(i) = input.FastAt(i) * (t_vals.FastAt(i) = gen.nextFloat() > thres);
            }
            return res;
        }
        virtual M BackwardProp(M &cost, const std::vector<M> &forwardRes, ANN **layers, size_t index) const override
        {
            cost *= t_vals;
            return M(); //as MLMat doesnt utilize/isnt utilized by other layers
        }
        virtual M &internal() override { return t_vals; }
        virtual void Learn(const M &correction) override {} //doesnt correct
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
        Batch(uint8_t batch_sz, uint16_t input_sz) : batch_sz(batch_sz), Input<M>(input_sz) {}
        virtual M ForwardProp(const M &input) override
        {
            for (typename M::storage_type i = 0; i < input.SizeY(); ++i)
            {
                for (typename M::storage_type j = 0; j < input.SizeX(); ++j)
                {
                    this->inp.At(0, i) += input.At(j, i);
                }
                this->inp.At(0, i) /= input.SizeX();
            }
            return this->inp;
        }
    };
} // namespace rum

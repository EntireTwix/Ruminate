#pragma once
#include "layers.hpp"
#include "rng_helpers.hpp"

namespace rum
{
    template <Matrix M>
    class Input : public Layer<M>
    {
    protected:
        M inp;

    public:
        virtual M &internal()
        {
            return inp;
        }

        virtual M ForwardProp(const M &input) override
        {
            //std::cout << "I\n";
            //copies it as long as input matches inp's area, not nessasirly dimensions
            inp = input;
            return inp;
        }

        virtual void Learn(const M &correction) override {} //doesnt correct
    };

    template <Matrix M>
    class DropOut : public Layer<M>
    {
    private:
        M t_vals;
        float thres;

    public:
        DropOut(uint16_t sz, float thres) {}
        virtual M ForwardProp(const M &input) override
        {
            M res(input.SizeX(), input.SizeY());
            for (size_t i = 0; i < input.Area(); ++i)
            {
                res.FastAt(i) = input.FastAt(i) * (t_vals.FastAt(i) = gen.nextFloat() > thres);
            }
            return res;
        }

        virtual M BackwardProp(M &cost, const std::vector<M> &forwardRes, Layer<M> **layers, size_t index) const override
        {
            cost *= t_vals;
            return M(); //as MLMat doesnt utilize/isnt utilized by other layers
        }

        virtual M &internal() override
        {
            return t_vals;
        }

        virtual void Learn(const M &correction) override
        {
        } //doesnt correct
    };

    template <Matrix M>
    class Batch : public Input<M>
    {
    private:
        uint8_t batch_sz;

    public:
        Batch(uint8_t batch_sz, uint16_t input_sz) : batch_sz(batch_sz), Input<M>(input_sz){};
        virtual M ForwardProp(const M &input) override
        {
            typename M::type sum;
            for (typename M::storage_type i = 0; i < input.SizeY(); ++i)
            {
                sum = 0;
                for (typename M::storage_type j = 0; j < input.SizeX(); ++j)
                {
                    sum += input.At(j, i);
                }
                this->inp.At(0, i) = sum / input.SizeX();
            }
            return this->inp;
        }
    };
} // namespace rum

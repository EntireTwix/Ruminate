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
        using SizeT = typename M::storage_type;

    public:
        Input(SizeT size) : inp(1, size){};
        virtual M *internal()
        {
            return nullptr;
        }

        virtual M ForwardProp(const M &input) override
        {
            if constexpr (LOG_LAYERS_FLAG)
            {
                std::cout << "I F\n";
            }

            //throws if too small, concatenates if too big
            //flattens regardless
            for (size_t i = 0; i < (inp.SizeX() * inp.SizeY()); ++i)
            {
                inp.FastAt(i) = input.FastAt(i);
            }
            return inp;
        }
        virtual void Learn(const M &correction) override {} //doesnt correct
    };

    template <Matrix M>
    class DropOut : public Layer<M>
    {
    private:
        M t_vals;
        const float thres;
        using SizeT = typename M::storage_type;

    public:
        DropOut(SizeT sz, float thres) : t_vals(sz, 1), thres(thres) {}
        virtual M ForwardProp(const M &input) override
        {
            if constexpr (LOG_LAYERS_FLAG)
            {
                std::cout << "D F\n";
            }
            M res(input.SizeX(), input.SizeY());
            for (size_t i = 0; i < (input.SizeX() * input.SizeY()); ++i)
            {
                res.FastAt(i) = input.FastAt(i) * (t_vals.FastAt(i) = gen.nextFloat() > thres);
            }

            return res;
        }

        virtual M BackwardProp(M &cost, const std::vector<M> &forwardRes, Layer<M> **layers, size_t index) const override
        {
            if constexpr (LOG_LAYERS_FLAG)
            {
                std::cout << "D B\n";
            }
            cost *= t_vals;
            return M(); //as MLMat doesnt utilize/isnt utilized by other layers
        }

        virtual M *internal() override
        {
            return t_vals;
        }

        virtual void Learn(const M &correction) override {} //doesnt correct
    };

    template <Matrix M>
    class Batch : public Input<M>
    {
    private:
        using SizeT = typename M::storage_type;
        const SizeT batch_sz;

    public:
        Batch(SizeT batch_sz, typename M::storage_type input_sz) : batch_sz(batch_sz), Input<M>(input_sz){};
        virtual M ForwardProp(const M &input) override
        {
            if constexpr (LOG_LAYERS_FLAG)
            {
                std::cout << "B F\n";
            }
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

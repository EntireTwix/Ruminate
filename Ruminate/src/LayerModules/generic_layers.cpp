#include "LayerModules/generic_layers.hpp"

namespace rum
{
#ifdef __NVCC__
    template <typename M>
#else
    template <Matrix M>
#endif
    Input<M>::Input(uint16_t input_sz) : inp(1, input_sz) = default;

#ifdef __NVCC__
    template <typename M>
#else
    template <Matrix M>
#endif
    M &Input<M>::internal()
    {
        return inp;
    }

#ifdef __NVCC__
    template <typename M>
#else
    template <Matrix M>
#endif
    M Input<M>::ForwardProp(const M &input)
    {
        //std::cout << "I\n";
        //copies it as long as input matches inp's area, not nessasirly dimensions
        for (size_t i = 0; i < inp.Area(); ++i)
        {
            inp.FastAt(i) = input.FastAt(i);
        }
        return inp;
    }

#ifdef __NVCC__
    template <typename M>
#else
    template <Matrix M>
#endif
    void Input<M>::Learn(const M &correction)
    {
    } //doesnt correct

#ifdef __NVCC__
    template <typename M>
#else
    template <Matrix M>
#endif
    DropOut<M>::DropOut(uint16_t sz, float thres) : t_vals(sz, 1), thres(thres)
    {
    }

#ifdef __NVCC__
    template <typename M>
#else
    template <Matrix M>
#endif
    M DropOut<M>::ForwardProp(const M &input)
    {
        M res(input.SizeX(), input.SizeY());
        for (size_t i = 0; i < input.Area(); ++i)
        {
            res.FastAt(i) = input.FastAt(i) * (t_vals.FastAt(i) = gen.nextFloat() > thres);
        }
        return res;
    }

#ifdef __NVCC__
    template <typename M>
#else
    template <Matrix M>
#endif
    M DropOut<M>::BackwardProp(M &cost, const std::vector<M> &forwardRes, Layer<M> **layers, size_t index) const
    {
        cost *= t_vals;
        return M(); //as MLMat doesnt utilize/isnt utilized by other layers
    }

#ifdef __NVCC__
    template <typename M>
#else
    template <Matrix M>
#endif
    M &DropOut<M>::internal()
    {
        return t_vals;
    }

#ifdef __NVCC__
    template <typename M>
#else
    template <Matrix M>
#endif
    void DropOut<M>::Learn(const M &correction)
    {
    } //doesnt correct

#ifdef __NVCC__
    template <typename M>
#else
    template <Matrix M>
#endif
    Batch<M>::Batch(uint8_t batch_sz, uint16_t input_sz) : batch_sz(batch_sz), Input<M>(input_sz) = default;

#ifdef __NVCC__
    template <typename M>
#else
    template <Matrix M>
#endif
    M Batch<M>::ForwardProp(const M &input)
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
} // namespace rum

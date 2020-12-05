#include "net.hpp"
#include "layers.hpp"

namespace rum
{
#ifdef __NVCC__
    template <typename T>
#else
    template <LayerType T>
#endif
    template <typename... Params>
    NeuralNetwork<T>::NeuralNetwork(Params &&... args) : sz(sizeof...(args))
    {
        layers = new T *[sz] { args... };
    }

    //thread safe
#ifdef __NVCC__
    template <typename T>
#else
    template <LayerType T>
#endif
    std::vector<typename T::type> NeuralNetwork<T>::ForwardProp(const typename T::type &input) const
    {
        std::vector<typename T::type> res(sz);
        res[0] = layers[0]->ForwardProp(input);
        //std::cout << res[0] << '\n'; //debugging
        for (size_t i = 1; i < res.size(); ++i)
        {
            res[i] = layers[i]->ForwardProp(res[i - 1]); //result 1 = 1th layer propogated with 0th result
            //std::cout << res[i] << '\n';                 //debugging
        }
        return res;
    }

    //thread safe
#ifdef __NVCC__
    template <typename T>
#else
    template <LayerType T>
#endif
    std::vector<typename T::type> NeuralNetwork<T>::BackwordProp(const std::vector<typename T::type> &forwardRes, typename T::type &&cost_prime, float lr)
    {
        std::vector<typename T::type> res(sz);
        typename T::type cost = cost_prime *= lr; //not optimal

        //std::cout << "\nBackProp:\n" << cost << '\n';
        for (uint8_t i = sz - 1; i > 0; --i)
        {
            res[i] = layers[i]->BackwardProp(cost, forwardRes, layers, i);
            // std::cout << "{\nCorrection:\n"
            //           << res[i] << "\nOriginal:\n"
            //           << layers[i]->internal() << "}\n\n";
        }
        return res;
    }

    //not thread safe
#ifdef __NVCC__
    template <typename T>
#else
    template <LayerType T>
#endif
    inline void NeuralNetwork<T>::Learn(const std::vector<typename T::type> &backRes)
    {
        for (uint8_t i = 0; i < sz; ++i)
        {
            layers[i]->Learn(backRes[i]);
        }
    }

#ifdef __NVCC__
    template <typename T>
#else
    template <LayerType T>
#endif
    inline std::string NeuralNetwork<T>::Save() const
    {
        std::string res;
        for (uint8_t i = 1; i < sz; ++i)
        {
            res += layers[i]->internal().Save() + '\n';
        }
        return res;
    }

#ifdef __NVCC__
    template <typename T>
#else
    template <LayerType T>
#endif
    inline typename T::type NeuralNetwork<T>::Cost(const typename T::type &guess, const typename T::type &anwser) const
    {
        typename T::type res(guess.SizeX(), guess.SizeY());
        for (size_t i = 0; i < guess.Area(); ++i)
        {
            res.FastAt(i) = 0.5 * ((guess.FastAt(i) - anwser.FastAt(i)) * (guess.FastAt(i) - anwser.FastAt(i)));
        }
        return res;
    }

#ifdef __NVCC__
    template <typename T>
#else
    template <LayerType T>
#endif
    inline typename T::type NeuralNetwork<T>::CostPrime(const typename T::type &guess, const typename T::type &anwser) const
    {
        return guess - anwser;
    }

#ifdef __NVCC__
    template <typename T>
#else
    template <LayerType T>
#endif
    inline NeuralNetwork<T>::~NeuralNetwork()
    {
        for (size_t i = 0; i < sz; ++i)
        {
            delete layers[i];
        }
        delete[] layers;
    }
}; // namespace rum

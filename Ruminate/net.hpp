#pragma once
#include "layers.hpp"

namespace rum
{
    template <LayerType T>
    class NeuralNetwork
    {
    private:
        T **layers = nullptr;
        uint8_t sz;
        using RT = typename T::type; //representation type, for example: fMat

    public:
        NeuralNetwork() = delete;
        NeuralNetwork(auto &&... args) : sz(sizeof...(args))
        {
            layers = new T *[sz] { args... };
        }

        std::vector<RT> ForwardProp(const RT &input) const
        {
            std::vector<RT> res(sz);
            res[0] = layers[0]->ForwardProp(input);
            //std::cout << res[0] << '\n'; //debugging
            for (size_t i = 1; i < res.size(); ++i)
            {
                res[i] = layers[i]->ForwardProp(res[i - 1]); //result 1 = 1th layer propogated with 0th result
                //std::cout << res[i] << '\n';                 //debugging
            }
            return res;
        }
        std::vector<RT> BackwordProp(const std::vector<RT> &forwardRes, const RT &guess, const RT &anwser, float lr)
        {
            std::vector<RT> res(sz);
            RT cost = CostPrime(guess, anwser) * lr; //not optimal

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
        void Learn(const std::vector<RT> &backRes)
        {
            for (uint8_t i = 1; i < sz; ++i)
            {
                layers[i]->internal() -= backRes[i];
            }
        }
        void Save()
        {
            for (uint8_t i = 1; i < sz; ++i)
            {
                std::cout << layers[i]->internal() << '\n';
            }
        }
        RT Cost(const RT &guess, const RT &anwser) const
        {
            RT res(guess.SizeX(), guess.SizeY());
            for (size_t i = 0; i < guess.Area(); ++i)
            {
                res.FastAt(i) = 0.5 * ((guess.FastAt(i) - anwser.FastAt(i)) * (guess.FastAt(i) - anwser.FastAt(i)));
            }
            return res;
        }
        RT CostPrime(const RT &guess, const RT &anwser) const
        {
            return guess - anwser;
        }

        ~NeuralNetwork()
        {
            for (size_t i = 0; i < sz; ++i)
            {
                delete layers[i];
            }
            delete[] layers;
        }
    };
}; // namespace rum

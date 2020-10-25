#pragma once
#include "layers.hpp"

template <LayerType T>
class NeuralNetwork
{
private:
    T **layers = nullptr;
    size_t sz;
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
        for (size_t i = 1; i < res.size(); ++i)
        {
            res[i] = layers[i]->ForwardProp(res[i - 1]); //result 1 = 1th layer propogated with 0th result
        }
        return res;
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
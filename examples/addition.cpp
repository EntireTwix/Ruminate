#include "iostream"
#include "core.h"
#include "modules/ann_layers.hpp"
#include "a_funcs.hpp"

using namespace rum;

int main()
{
    NeuralNetwork<Layer<MLMat>> net{
        new Input<MLMat>(2),                      //2 node input
        new Weight(2, 1, RngInit()),              //2x1 weights
        new Output(1, ReluLeaky, ReluLeakyPrime), //1 output node
    };

    MLMat data(1, 2);
    MLMat anw(1, 1);

    while (1)
    {
        system("clear");
        std::cout << net.Save() << '\n';

        data.At(0, 0) = gen.nextUInt(100);
        data.At(0, 1) = gen.nextUInt(100);
        anw.At(0, 0) = data.At(0, 0) + data.At(0, 1);
        std::cout << "Input:\n"
                  << data << '\n'
                  << "Anwser:\n"
                  << anw << '\n';
        auto res = net.ForwardProp(data);
        std::cout << res.back() << "\nCost:\n"
                  << net.Cost(res.back(), anw) << '\n';
        auto corrections = net.BackwordProp(res, net.CostPrime(res.back(), anw), 0.0001);
        net.Learn(corrections);

        //hold enter to see training progress
        std::cin.get();
    }
    return 0;
}
#include <iostream>
#include "core.h"
#include "modules/ann_layers.hpp"
#include "a_funcs.hpp"
#include "rng_helpers.hpp"

using namespace rum;

int main()
{

    NeuralNetwork<ANN> net{
        new Batch<MLMat>(10, 2),                  //batch input of 10
        new Weight(2, 3, RngInit()),              //2x1 weights
        new Hidden(3, ReluLeaky, ReluLeakyPrime), //3 hidden nodes
        new Weight(3, 1, RngInit()),              //3x1 weights
        new Output(1, ReluLeaky, ReluLeakyPrime), //1 output node
    };

    MLMat data(10, 2);
    MLMat anw(1, 1);

    while (1)
    {
        system("CLS");
        std::cout << net.Save() << '\n';

        for (int i = 0; i < 10; ++i)
        {
            data.At(i, 0) = gen.nextUInt(100);
            data.At(i, 1) = gen.nextUInt(100);
            anw.At(0, 0) += data.At(i, 0) + data.At(i, 1);
        }
        anw.At(0, 0) /= 10;

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
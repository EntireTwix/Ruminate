#include <iostream>
#include "../Ruminate-main/Ruminate/net.hpp"
#include "../Ruminate-main/Ruminate/LayerModules/ann_layers.hpp"
#include "../Ruminate-main/Ruminate/HelperFiles/a_funcs.hpp"

using namespace rum;

int main()
{
    pcg32 rng(time(NULL) << 8, time(NULL) >> 8);
    NeuralNetwork<ANN> net{
        new Input(2),                                        //2 node input
        new Weight(2, 1, 0, 1, rng),                         //2x1 weights
        new Output(1, Relu, ReluPrime, 0, 1, rng),           //1 output node
    };

    MLMat data(1, 2);
    MLMat anw(1, 1);

    while (1)
    {
        system("CLS");
        std::cout << net.Save() << '\n';

        data.At(0, 0) = rng.nextUInt(100);
        data.At(0, 1) = rng.nextUInt(100);
        anw.At(0, 0) = data.At(0, 0) + data.At(0, 1);
        std::cout << "Input:\n"
                  << data << '\n'
                  << "Anwser:\n"
                  << anw << '\n';
        auto res = net.ForwardProp(data);
        std::cout << res.back() << "\nCost:\n"
                  << net.Cost(res.back(), anw) << '\n';
        auto corrections = net.BackwordProp(res, res.back(), anw, 0.00001);
        net.Learn(corrections);

        //hold enter to see training progress
        std::cin.get();
    }
    return 0;
}

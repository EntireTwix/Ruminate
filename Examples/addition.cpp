#include <iostream>
#include "../Ruminate-main/Ruminate/net.hpp"
#include "../Ruminate-main/Ruminate/LayerModules/fc_layers.hpp"
#include "../Ruminate-main/Ruminate/HelperFiles/a_funcs.hpp"

using namespace rum;

int main()
{
    pcg32 rng(time(NULL) << 8, time(NULL) >> 8);
    NeuralNetwork<FC> net{
        new Input(),
        new Weight(2, 3, 0, 1, rng),
        new Hidden(3, Relu, ReluPrime, 0, 1, rng),
        new Weight(3, 1, 0, 1, rng),
        new Output(1, Relu, ReluPrime, 0, 1, rng),
    };

    MLMat data(1, 2);
    MLMat anw(1, 1);

    while (1)
    {
        system("CLS");

        data.At(0, 0) = rng.nextUInt(100);
        data.At(0, 1) = rng.nextUInt(100);
        anw.At(0, 0) = data.At(0, 0) + data.At(0, 1);
        std::cout << "Input:\n"
            << data << '\n'
            << "Anwser:\n"
            << anw << '\n';
        auto res = net.ForwardProp(data);
        std::cout << res.back() << "Cost:\n"
            << net.Cost(res.back(), anw);
        auto corrections = net.BackwordProp(res, res.back(), anw, 0.0001);
        net.Learn(corrections);
        std::cin.get();
    }

    return 0;
}

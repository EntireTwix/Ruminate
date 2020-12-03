#include <iostream>
#include <net.hpp>
#include <LayerModules/ann_layers.hpp>
#include <LayerModules/generic_layers.hpp>
#include <HelperFiles/a_funcs.hpp>
#include <HelperFiles/rng_helpers.hpp>

using namespace rum;

int main()
{

    NeuralNetwork<ANN> net{
        new Input<MLMat>(2),
        new Weight(2, 3, new RngInit()),          //2x1 weights
        new DropOut<MLMat>(3, 0.25),              //25% dropout
        new Hidden(3, ReluLeaky, ReluLeakyPrime), //3 hidden nodes
        new Weight(3, 1, new RngInit()),          //3x1 weights
        new Output(1, ReluLeaky, ReluLeakyPrime), //1 output node
    };

    MLMat data(1, 2);
    MLMat anw(1, 1);

    while (1)
    {
        system("CLS");
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

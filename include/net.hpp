#pragma once
#include <string>
#include <numeric>
#include "layers.hpp"

namespace rum
{
    template <LayerType T>
    class NeuralNetwork
    {
    private:
        T **layers = nullptr;
        const uint_fast8_t sz;
        using RT = typename T::type; //representation type, for example: fMat

    public:
        NeuralNetwork() noexcept = delete;

        template <typename... Params>
        NeuralNetwork(Params *&&...args) noexcept : sz(sizeof...(args))
        {
            layers = new T *[sz] { args... };
        }

        /**
         * @brief 
         * Forward Propogates the given Input matrix across the network 
         * then returns a vector of matrices containing each step
         * (THREAD SAFE)
         * @param input
         * @return std::vector<Mat<T>> of steps
         */
        std::vector<RT> ForwardProp(const RT &input) const
        {
            std::vector<typename T::type> res(sz);
            res[0] = layers[0]->ForwardProp(input);
            if constexpr (LOG_LAYERS_FLAG)
            {
                std::cout << res[0] << '\n'; //debugging
            }
            for (size_t i = 1; i < res.size(); ++i)
            {
                res[i] = layers[i]->ForwardProp(res[i - 1]); //result 1 = 1th layer propogated with 0th result
                if constexpr (LOG_LAYERS_FLAG)
                {
                    std::cout << res[i] << '\n'; //debugging
                }
            }
            return res;
        }

        /**
         * @brief Backpropogates the cost of a given ForwardProp() starting from the last layer
         * to the first layer of the network. Each layer may modify the cost and is given the context
         * of the forwardprops result, current layers state, and current cost.
         * (THREAD SAFE)
         * @param forwardRes, the result of a forward prop
         * @param cost_prime, the error of the last anwser, caclulated with CostPrime()
         * @param lr, the learning rate of the network, you could say how sensitive it is to change
         * @return std::vector<RT> of corrections to be applied with Learn()
         */
        std::vector<RT> BackwordProp(const std::vector<RT> &forwardRes, RT &&cost_prime, float lr) const
        {
            std::vector<typename T::type> res(sz);
            typename T::type cost = std::move(cost_prime *= lr); //not optimal

            //std::cout << "\nBackProp:\n" << cost << '\n';
            for (uint8_t i = sz - 1; i > 0; --i)
            {
                res[i] = layers[i]->BackwardProp(cost, forwardRes, layers, i);
                if (LOG_LAYERS_FLAG)
                {
                    std::cout << "{\nCorrection:\n"
                              << res[i] << "\nOriginal:\n"
                              << layers[i]->inside() << "}\n\n";
                }
            }
            return res;
        }

        /**
         * @brief calls each layers 
         * Learn() function passing the corresponding correction as an arg,
         * applying the corrections.
         * (NOT THREAD SAFE)
         * 
         * @param backRes, backpropogation correction results to be applied
         */
        void Learn(const std::vector<RT> &backRes)
        {
            for (uint_fast8_t i = 0; i < sz; ++i)
            {
                layers[i]->Learn(backRes[i]);
            }
        }

        /**
         * @brief a janky solution to problem of saving networks, when called this function
         * will return a string that if copied and pasted would be the constructor for the layer
         * it corresponds to.
         * Ex Output: (2,1,0.999917,0.999930)
         * Ex Paste:  new Weight(2,1,0.999917,0.999930);
         * (TO BE REFRACTORED LATER)
         * 
         * @return std::string, each save is on a newline
         */
        std::string Save() const noexcept
        {
            std::string res;
            for (uint8_t i = 1; i < sz; ++i) //skipping input layer
            {
                res += layers[i]->inside().Save() + '\n';
            }
            return res;
        }

        /**
         * @brief returns the cost of the 0.5(guess - anwser)^2
         * 
         * @param guess
         * @param anwser 
         * @return matrix of cost 
         */
        RT Cost(const RT &guess, const RT &anwser) const
        {
            typename T::type res(guess.SizeX(), guess.SizeY());
            for (size_t i = 0; i < guess.SizeX() * guess.SizeY(); ++i) //kinda bad to keep calling Area()
            {
                res.FastAt(i) = 0.5 * ((guess.FastAt(i) - anwser.FastAt(i)) * (guess.FastAt(i) - anwser.FastAt(i)));
            }
            return res;
        }

        /**
         * @brief passed to BackProp() to calculate raw cost of guess-anwser
         * 
         * @param guess 
         * @param anwser 
         * @return matrix of cost
         */
        RT CostPrime(const RT &guess, const RT &anwser) const noexcept
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

//TODO: make forward&back prop std::array
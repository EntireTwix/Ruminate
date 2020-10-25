#pragma once
#include <concepts>
#include <type_traits>
#include <vector>

//an abstract type that is used polymorphically in net.h
template <typename T>
class Layer
{
public:
    using type = T;

    //these vectors passing as ref is for performance reasons (safety issues come with it)
    virtual T ForwardProp(const T &) const = 0;
    //virtual T BackProp(T &cost, std::vector<T> &backprops, std::vector<T> &forwardprops, size_t current_index) const = 0;
    //virtual void Learn(std::vector<T> &backprops, size_t current_index) = 0;
};

template <typename T>
concept LayerType = std::is_base_of<Layer<typename T::type>, T>::value;
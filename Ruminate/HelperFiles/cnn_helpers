#pragma once
#include "../../../OptimizedHeaders-main/mat.hpp"

MLMat Img2MLMat(const ImgMat &mat)
{
    MLMat res(mat.SizeX(), mat.SizeY());
    for (uint16_t i = 0; i < mat.Area(); ++i)
    {
        res.FastAt(i) = mat.FastAt(i) / 255;
    }
    return res;
}

ImgMat MLMat2ImgMat(const MLMat &mat)
{
    ImgMat res(mat.SizeX(), mat.SizeY());
    for (uint16_t i = 0; i < mat.Area(); ++i)
    {
        res.FastAt(i) = mat.FastAt(i) * 255;
    }
    return res;
}

template <typename T, typename T2>
MLMat Flatten2MLMat(const Mat<T, T2> &mat)
{
    MLMat res(1, mat.Area());
    for (T2 i = 0; i < mat.Area(); ++i)
    {
        res.FastAt(i) = mat.FastAt(i);
    }
    return res;
}

template <>
MLMat Flatten2MLMat(const ImgMat &mat)
{
    MLMat res(1, mat.Area());
    for (uint16_t i = 0; i < mat.Area(); ++i)
    {
        res.FastAt(i) = mat.FastAt(i) / 255;
    }
    return res;
}

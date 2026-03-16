#pragma once
#include <vector>
#include "layer.h"


class MLP {
    private:
        size_t m_nin;
        std::vector<size_t> m_sizes;
        Manager* m_vm;
        OpType m_op;
        std::vector<Layer> layers;

    public:
        MLP(size_t nin, std::vector<size_t> sizes, Manager* vm, OpType activation);

        std::vector<Value*> operator()(const std::vector<double>& inputs);

        std::vector<Value*> parameters() const;

    private:
        std::vector<Value*> forward(std::vector<Value*> inputs);
};
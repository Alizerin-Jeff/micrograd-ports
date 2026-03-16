#include "mlp.h"
#include <ranges>
#include <iostream>

MLP::MLP(size_t nin, std::vector<size_t> sizes, Manager* vm, OpType op) 
    : m_nin(nin), m_sizes(sizes), m_vm(vm), m_op(op)
    {
    std::vector<size_t> layout = {m_nin};
    layout.insert(layout.end(), sizes.begin(), sizes.end());

    auto pairs = std::views::zip(layout, layout | std::views::drop(1));
    for(auto const& [in_size, out_size] : pairs) {
        layers.push_back(Layer(in_size, out_size, m_vm, m_op));
    }
}

std::vector<Value*> MLP::operator()(const std::vector<double>& inputs) {
    // wrap inputs into the Value class
    std::vector<Value*> wrapped;
    for(const auto& input : inputs) {
        wrapped.push_back(&m_vm->create(input, "input"));
    }
    auto output = forward(wrapped);
    return output;
}

std::vector<Value*> MLP::forward(std::vector<Value*> x) {
    for(auto& layer : layers) {
        x = layer(x);
    }
    return x;
}

std::vector<Value*> MLP::parameters() const {
    std::vector<Value*> params;
    for(auto& layer : layers) {
        auto layer_params = layer.parameters();
        params.insert(params.end(), layer_params.begin(), layer_params.end());
    }
    return params;
}
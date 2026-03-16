#include "neuron.h"
#include <random>
#include <iostream>

static std::mt19937 gen{42u};
static std::uniform_real_distribution<double> dist{-1.0,1.0}; // matches python example, but bad for Relu

Neuron::Neuron(size_t nin, Manager* vm, OpType op)
    : m_vm(vm), m_op(op)
    {
        if(m_op != OpType::TANH &&
            m_op != OpType::RELU)
        {
            throw std::invalid_argument("Invalid activation function");
        }
        for(size_t i=0; i < nin; ++i) {
            double val = dist(gen);
            Value& wi = vm->create(val, "w" + std::to_string(i));
            w.push_back(&wi);
        }
        double b_val = dist(gen);
        Value& bias = m_vm->create(b_val, "b");
        b = &bias;
    }

 Value& Neuron::operator()(const std::vector<Value*>& x) {
    if(x.size() != w.size()) {
        throw std::invalid_argument("Neuron: input size does not match weights");
    }
    Value* z = b;
    //dot product
    for(size_t i = 0; i < w.size(); ++i) {
        z = &(*z + *x[i] * *w[i]);
    }
    // activate
    if(m_op == OpType::TANH) {
        return z->tanh();
    } else if(m_op == OpType::RELU) {
        return z->relu();
    }
    throw std::logic_error("Unsupported activation function\n");
}
std::vector<Value*> Neuron::parameters() const {
    std::vector<Value*> params;
    params.push_back(b);
    params.insert(params.end(), w.begin(), w.end());
    return params;
}

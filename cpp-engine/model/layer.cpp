#include "layer.h"

//------------------------------------------------------------------------
// Constructor and methods for Layer class
Layer::Layer(size_t nin, size_t nout, Manager* vm, OpType op) 
    : m_nin(nin), m_nout(nout), m_vm(vm), m_op(op)
    {
        neurons.reserve(m_nout);
        for(size_t i=0; i < m_nout; ++i) {
            neurons.emplace_back(m_nin, m_vm, m_op);
        }
    }

std::vector<Value*> Layer::operator()(std::vector<Value*>& inputs){
    std::vector<Value*> outs;
    outs.reserve(neurons.size());
    for(auto& neuron : neurons) {
        outs.push_back(&neuron(inputs));
    }
    return outs;
}

std::vector<Value*> Layer::parameters() const {
    std::vector<Value*> params;
    for(auto& neuron : neurons) {
        auto neuron_params = neuron.parameters();
        params.insert(params.end(), neuron_params.begin(), neuron_params.end());
    }
    return params;
}
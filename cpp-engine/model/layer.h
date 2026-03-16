#pragma once
#include <vector>
#include "neuron.h"

class Layer {
    private:
        size_t m_nin;
        size_t m_nout;
        Manager* m_vm;
        OpType m_op;
        std::vector<Neuron> neurons;
    
    public:
        Layer(size_t nin, size_t nout, Manager* vm, OpType op);

        std::vector<Value*> operator()(std::vector<Value*>& inputs);
        
        std::vector<Value*> parameters() const;
        
};
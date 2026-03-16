#pragma once
#include <vector>
#include "../engine.h"

class Neuron {
    private:
        Manager* m_vm;
        OpType m_op;
        std::vector<Value*> w;
        Value* b;
        

    public:
        Neuron(size_t nin, Manager* vm, OpType op);

        Value& operator()(const std::vector<Value*>& x);

        std::vector<Value*> parameters() const;
};
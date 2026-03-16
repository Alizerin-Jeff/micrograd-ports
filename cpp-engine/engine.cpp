#include <cmath>
#include "engine.h"
#include <iostream>
#include <unordered_set>


//---------------------------------------------------------------------------------------------
// Constructor and methods for Manager class
Value& Manager::create(double data, std::string label, OpType op) {
    auto node = std::make_unique<Value>(data, this, label, op);
    Value& ref = *node;
    m_all_nodes.push_back(std::move(node));

    return ref;
}

void Manager::backward(Value& loss) {
    std::vector<Value*> topo;
    std::unordered_set<Value*> visited;
    build_topo(&loss, topo, visited);

    for(auto& node : topo) {
        node->m_grad = 0.0;
    }

    loss.m_grad = 1.0;
    for(auto it = topo.rbegin(); it != topo.rend(); ++it){
        Value* node = *it;
        if(!node || !node->m_prev[0]){
            continue;
        }

        if(node->m_op == OpType::NONE) {
            continue; // no children
        }else if( node->m_op == OpType::MUL){
            if(node->m_prev[1]) {
                node->m_prev[0]->m_grad += node->m_prev[1]->m_data * node->m_grad;
                node->m_prev[1]->m_grad += node->m_prev[0]->m_data * node->m_grad;
            } else {
                node->m_prev[0]->m_grad += node->scalar * node->m_grad;
            }
        } else if(node->m_op == OpType::ADD) {
            if(node->m_prev[1]) {
                node->m_prev[0]->m_grad += node->m_grad;
                node->m_prev[1]->m_grad += node->m_grad;
            } else {
                node->m_prev[0]->m_grad += node->m_grad;
            }
        } else if(node->m_op == OpType::POW) {
            double data = node->m_prev[0]->m_data;
            double exp = node->scalar;
            node->m_prev[0]->m_grad += exp * std::pow(data, (exp -1)) * node->m_grad;
        } else if(node->m_op == OpType::TANH) {
            double t = node->m_data;
            double local_derivative = 1.0 - (t*t);
            node->m_prev[0]->m_grad += local_derivative * node->m_grad;
        } else if(node->m_op == OpType::RELU) {
            if(node->m_prev[0]){
                node->m_prev[0]->m_grad += (node->m_data > 0 ? 1.0 : 0.0) * node->m_grad;
            }
        }
    }
}

void Manager::clear_ephemeral_nodes(const std::vector<Value*>& parameters){
    std::unordered_set<Value*> keep_set(parameters.begin(), parameters.end());
    std::erase_if(m_all_nodes, [&](const std::unique_ptr<Value>& node) {
        return keep_set.find(node.get()) == keep_set.end();
    });
}

void Manager::reserve(size_t size){
    m_all_nodes.reserve(size);
}

void Manager::build_topo(Value* v, std::vector<Value*>& topo, std::unordered_set<Value*>& visited) {
    if(v == nullptr || visited.count(v)) return;
    visited.insert(v);

    for(Value* prev : v->m_prev){
        if(prev != nullptr) {
            build_topo(prev, topo, visited);
        }
    }
    topo.push_back(v);
}

//-------------------------------------------------------------------------------------------------
// Constructor and methods for Value class
Value::Value(double data, Manager* vm, std::string label, OpType op)
            : m_data(data),
              m_vm(vm),
              m_label(label),
              m_op(op)
    {}

// Power Function
Value& Value::pow(double scalar_b) {
    double newData = std::pow(this->m_data, scalar_b);
    Value& out = this->m_vm->create(newData, "", OpType::POW);
    out.scalar = scalar_b;
    out.m_prev[0] = this;
    out.m_prev[1] = nullptr;
    return out;
}

// Tangent Hyperbolic Function
Value& Value::tanh() {
    double newData = std::tanh(this->m_data);
    Value& out = this->m_vm->create(newData, "", OpType::TANH);
    out.m_prev[0] = this;
    out.m_prev[1] = nullptr;
    return out;
}

// Rectified Linear Unit
Value& Value::relu() {
    double newData = std::max(0.0, this->m_data);
    Value& out = this->m_vm->create(newData, "", OpType::RELU);
    out.m_prev[0] = this;
    out.m_prev[1] = nullptr;
    return out;
}

//-----------------------------------------------------------------------------
// Multiplication Operators

// Multiply two Value objects
Value& operator*(Value& a, Value& b) {
    double newData = a.m_data * b.m_data;
    Value& out = a.m_vm->create(newData, "", OpType::MUL);
    out.m_prev[0] = &a;
    out.m_prev[1] = &b;
    return out;
}

// Multiply a Value object times a scalar
Value& operator*(Value& a, double scalar_b) {
    double newData = a.m_data * scalar_b;
    Value& out = a.m_vm->create(newData, "", OpType::MUL);
    out.scalar = scalar_b;
    out.m_prev[0] = &a;
    out.m_prev[1] = nullptr;
    return out;
}

// Multiply a scalar times a Value object
Value& operator*(double scalar_b, Value& a) {
    return a * scalar_b;
}    

//------------------------------------------------------------------------------
// Addition Operators

// Add two Value objects
Value& operator+(Value& a, Value& b) {
    double newData = a.m_data + b.m_data;
    Value& out = a.m_vm->create(newData, "", OpType::ADD);
    out.m_prev[0] = &a;
    out.m_prev[1] = &b;
    return out;
}

// Add a Value object plus a scalar
Value& operator+(Value& a, double scalar_b) {
    double newData = a.m_data + scalar_b;
    Value& out = a.m_vm->create(newData, "", OpType::ADD);
    out.m_prev[0] = &a;
    out.m_prev[1] = nullptr;
    return out;
}

// Add a scalar plus a Value object
Value& operator+(double scalar_b, Value& a) {
    return a + scalar_b;
}   

//--------------------------------------------------------------------------------------
// Subtraction operators

// Subtract a Value object from another Value object
Value& operator-(Value& a, Value& b) {
    return a + (-1.0*b);
}

// Subtract a scalar from a Value object 
Value& operator-(Value& a, double scalar_b) {
    return a + (-1.0*scalar_b);
}

// Subtract a Value object from a scalar 
Value& operator-(double scalar_b, Value& a) {
    return (-1.0)*a + scalar_b;
}   
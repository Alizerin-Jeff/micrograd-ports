#pragma once
#include <string>
#include <vector>
#include <memory>
#include <unordered_set>

class Value;

enum class OpType { NONE, ADD, MUL, POW, TANH, RELU};

class Manager {
    private:
        std::vector<std::unique_ptr<Value>> m_all_nodes;
    public:
        Value& create(double data, std::string label = "", OpType op = OpType::NONE);

        void backward(Value& loss); 
        void clear_ephemeral_nodes(const std::vector<Value*>& parameters);
        void reserve(size_t size);
        void build_topo(Value* loss, std::vector<Value*>& topo, std::unordered_set<Value*>& visited);
};

class Value {
    public:
        double m_data;
        double m_grad = 0.0;
    private:
        Manager* m_vm;
        std::string m_label;
        OpType m_op;
        Value* m_prev[2] = {nullptr, nullptr};
        double scalar = 1.0;

    public:
        friend class Manager;

        friend Value& operator*(Value& a, Value& b);
        friend Value& operator*(Value& a, double scalar_b);
        friend Value& operator*(double scalar_b, Value& a);

        friend Value& operator+(Value& a, Value& b);
        friend Value& operator+(Value& a, double scalar_b);
        friend Value& operator+(double scalar_b, Value& a);

        friend Value& operator-(Value& a, Value& b);
        friend Value& operator-(Value& a, double scalar_b);
        friend Value& operator-(double scalar_b, Value& a);

        Value(double data, Manager* vm, std::string label = "", OpType op = OpType::NONE);

        // Power Function
        Value& pow(const double scalar_b);

        // Tangent Hyperbolic Function
        Value& tanh();

        // Rectified Linear Unit
        Value& relu();
};

//--------------------------------------------------------------
// Operator declarations
Value& operator*(Value& a, Value& b);
Value& operator*(Value& a, double scalar_b);
Value& operator*(double scalar_b, Value& a);

Value& operator+(Value& a, Value& b);
Value& operator+(Value& a, double scalar_b);
Value& operator+(double scalar_b, Value& a);

Value& operator-(Value& a, Value& b);
Value& operator-(Value& a, double scalar_b);
Value& operator-(double scalar_b, Value& a);



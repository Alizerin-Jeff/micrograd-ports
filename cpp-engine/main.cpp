#include "engine.h"
#include <vector>
#include "model/mlp.h"
#include <ranges>
#include <numeric>
#include <iostream>
#include <iomanip>
#include <chrono>

int main(int argc, char* argv[]) {
    double h = 0.25;
    size_t epochs = 10;
    size_t nit = 200;
    OpType op = OpType::TANH;
    Manager vm;
    vm.reserve(400);

    if(argc < 4){
        std::cout << "Standard Usage: micrograd [epochs] [iters] [lr]\n";
        std::cout << "Defaulting to : micrograd 20 200 0.25\n\n";
    } else {
        try {
            epochs = std::stoi(argv[1]);
            nit = std::stoi(argv[2]);
            h = std::stod(argv[3]);
        } catch (...){
            std::cerr << "Error: Invalid argument format.  Using defaults.\n\n";
        }
    }

    std::vector<std::vector<double>> inputs = {
        {2.0,3.0,-1.0},
        {3.0,-1.0,0.5},
        {0.5,1.0,1.0},
        {1.0,1.0,-1.0}
    };

    std::vector<double> targets = {1.0,-1.0,1.0,-1.0};

    size_t nin = 3;
    std::vector<size_t> sizes = {4,4,1};
    MLP mlp = MLP(nin, sizes, &vm, op);
    std::vector<Value*> params = mlp.parameters();
    
    
    
    auto start = std::chrono::high_resolution_clock::now();
    for(size_t epoch = 0; epoch < epochs; ++epoch) {
        for(size_t it=0; it < nit; ++it) {
            std::vector<Value*> preds = {};
            Value* loss = &vm.create(0.0);

            for(auto& input : inputs){
                auto out = mlp(input)[0];
                preds.push_back(out);
            }
            auto zipped = std::views::zip(preds, targets);
            for(const auto& [pred, target] : zipped) {
                loss = &(*loss + (*pred - target).pow(2.0));
            }
            vm.backward(*loss);
            for(auto param : params){
                param->m_data -= param->m_grad * h;
            }
            if(it == nit -1 && epoch == epochs -1) {
                std::cout << std::fixed << std::setprecision(4);
                std::cout << "--- Training Results ---\n";
                std::cout << "Pred\t | Target\n";
                std::cout << "------------------------------------\n";
                for(auto const& [pred, target] : std::views::zip(preds, targets)){
                std::cout << pred->m_data << "\t " << target << "\n";
                }
                std::cout << "------------------------------------\n";
                std::cout << "Total Loss: " << std::scientific << std::setprecision(8) << loss->m_data << "\n\n";
            }
            vm.clear_ephemeral_nodes(params);

        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end-start);
    std::cout << "Total time: " << std::setprecision(2) << duration.count() / 1000.0 << " seconds\n\n";
    return 0;
}
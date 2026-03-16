# micrograd ports

Multi-language ports of [Andrej Karpathy's micrograd](https://github.com/karpathy/micrograd) — a minimal autograd engine and neural network library. The original Python implementation is included as a Jupyter notebook, with ports to C++ and Rust (in progress).

## Motivation

This project explores how the same tiny neural network engine looks across different languages, comparing:

- **Implementation difficulty** — Python's expressiveness vs. manual memory management in C++ vs. Rust's ownership model
- **Design patterns** — how each language handles operator overloading, polymorphism, and graph construction
- **Performance** — raw training speed across languages

## Project structure

```
micrograd_ports/
├── micrograd.ipynb     # Original Python implementation (Jupyter notebook)
├── cpp-engine/         # C++ port
│   ├── CMakeLists.txt
│   ├── main.cpp        # Training loop & benchmarks
│   ├── engine.h/.cpp   # Value node & Manager (computation graph + memory)
│   └── model/
│       ├── neuron.h/.cpp
│       ├── layer.h/.cpp
│       └── mlp.h/.cpp
└── rust-engine/        # Rust port (not yet started)
```

## Performance comparison

Training a 3→4→4→1 MLP on 4 samples with MSE loss:

| Language | Iterations | Time     | Speedup |
|----------|------------|----------|---------|
| Python   | 2,000      | ~2.6s    | 1×      |
| C++      | 2,000      | ~0.494s  | ~5.3×   |
| Rust     | TBD        | TBD      | TBD     |

## Implementation notes

- **Python** — clean and concise. Operator overloading via dunder methods, backprop closures captured per-node, topological sort in `backward()`. ~100 lines for the full engine + network.
- **C++** — requires a `Manager` class that owns all `Value` nodes via `std::unique_ptr`, acting as an arena for the computation graph. Ephemeral nodes are cleared between iterations to control memory growth. Uses C++23 features (`std::views::zip`). Supports both tanh and ReLU activations.
- **Rust** — not yet implemented.

## Building & running

### Python notebook

```bash
jupyter notebook micrograd.ipynb
```

Requires `graphviz` for computation graph visualization.

### C++ engine

```bash
cd cpp-engine
mkdir -p build && cd build
cmake ..
make
./micrograd              # defaults: 20 epochs, 200 iterations, lr=0.05
./micrograd 10 200 0.25  # explicit: [epochs] [iterations] [learning_rate]
```

Requires CMake 3.15+ and a C++23-compatible compiler.

## Status

- **Python notebook** — complete
- **C++ engine** — complete, in final review
- **Rust engine** — not started

## Credit

Based on [micrograd](https://github.com/karpathy/micrograd) by Andrej Karpathy.

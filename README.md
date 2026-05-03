# micrograd ports

Multi-language ports of [Andrej Karpathy's micrograd](https://github.com/karpathy/micrograd): a minimal scalar-valued autograd engine and neural network library. The original Python implementation is included as a Jupyter notebook, with a C++ port that runs **17× faster** through deliberate design decisions described below.

## Performance

Training a 3→4→4→1 MLP on 4 samples with MSE loss for 2,000 iterations:

| Language | Time    | Speedup |
|----------|---------|---------|
| Python   | ~2.6 s  | 1×      |
| C++      | ~0.15 s | ~17×    |
| Rust     | TBD     | TBD     |

The speedup comes from how the C++ engine owns memory, runs backprop, and avoids repeated work, and not necessarily from C++ being fast by default.

## Project structure

```
micrograd_ports/
├── micrograd.ipynb     # Python implementation (Jupyter notebook)
├── cpp-engine/
│   ├── CMakeLists.txt
│   ├── main.cpp        # Training loop & benchmark
│   ├── engine.h/.cpp   # Value node & Manager class
│   └── model/
│       ├── neuron.h/.cpp
│       ├── layer.h/.cpp
│       └── mlp.h/.cpp
└── rust-engine/        # Not yet implemented
```

---

## C++ Design Decisions

The Python implementation is about 100 lines and is a clean, idiomatic port: operator overloading via dunder methods, a per-node `_backward` closure that captures local derivatives at operation time, and a topological sort built fresh inside `backward()` on every call. It works beautifully for an educational engine and is certainly much easier to implement with about 5x less code.

The C++ port keeps the same mathematical semantics: same graph, same chain rule, same full-batch gradient descent, but makes explicit choices about memory ownership, backprop structure, and what work can be done once rather than every iteration. Five decisions account for nearly all of the speedup.

---

### Decision 1: Manager-owned graph (`unique_ptr` arena)

**What the C++ version does:**

A `Manager` class holds `std::vector<std::unique_ptr<Value>>`. Every node in the computation graph is created exclusively through `Manager::create()`, which constructs a `unique_ptr<Value>`, assigns the node its index, and returns a plain reference to the caller:

```cpp
Value& Manager::create(double data, std::string label, OpType op) {
    auto node = std::make_unique<Value>(data, this, label, op);
    node->idx = m_all_nodes.size();
    Value& ref = *node;
    m_all_nodes.push_back(std::move(node));
    return ref;
}
```

No node can outlive the Manager. Callers work with `Value&`: no ownership, no reference counting, no `delete`.

**Why:**

Python Value nodes are heap-allocated objects managed by the garbage collector. C++ requires ownership to be declared explicitly, and the choice of `unique_ptr` inside a vector directly enables the rest of the optimizations.

`vector<unique_ptr<Value>>` stores the `Value` objects on the heap. The `unique_ptr` itself lives inside the vector and holds a pointer to that heap object. When the vector reallocates, the `unique_ptr` objects move to new vector storage, but the `Value` objects they point to stay put. So any `Value*` or `Value&` derived from the graph is stable for the Manager's lifetime. The `m_prev[2]` pointers wiring the computation graph together, and the references returned by `create()`, are all valid as long as the Manager exists.

`shared_ptr` would provide similar lifetime guarantees but adds an atomic reference count increment/decrement on every copy. Every forward-pass operation creates new nodes and copies parent pointers; across 2,000 training iterations that adds up. `unique_ptr` has zero runtime overhead over a raw pointer.

**Tradeoff:**

`reserve(400)` pre-allocates vector capacity at startup, keeping allocation out of the hot loop. For larger models this number would need to be tuned to match the graph size.

---

### Decision 2: Manager-level backprop (not per-node closures)

**What the C++ version does:**

All gradient computation lives in a single `Manager::backward(Value& loss)` method. Each Value node stores an `OpType` enum (`NONE`, `ADD`, `MUL`, `POW`, `TANH`, `RELU`). Backward iterates the topo list in reverse and applies the correct gradient rule with an `if/else` on `OpType`:

```cpp
} else if(node->m_op == OpType::MUL) {
    node->m_prev[0]->m_grad += node->m_prev[1]->m_data * node->m_grad;
    node->m_prev[1]->m_grad += node->m_prev[0]->m_data * node->m_grad;
} else if(node->m_op == OpType::TANH) {
    double t = node->m_data;
    node->m_prev[0]->m_grad += (1.0 - t*t) * node->m_grad;
}
// ... etc
```

**Why:**

In Python, each operation stores its backward logic as a closure on the output node:

```python
def _backward():
    self.grad += other.data * out.grad
    other.grad += self.data * out.grad
out._backward = _backward
```

The closure captures `self`, `other`, and `out` by reference at the time the operation runs. In Python this is natural: dynamic dispatch comes for free and closures are a first-class construct.

Porting this directly to C++ means `std::function<void()>` per node. `std::function` involves a heap allocation for each closure object (to store the captured references), and calling it requires virtual dispatch through a type-erased function pointer. With a graph of ~100 nodes and 2,000 iterations, that is 200,000 closure allocations and 200,000 indirect calls just for backprop. Moving all backward logic into one Manager method eliminates every allocation, inlines the gradient math, and lets the branch predictor warm up on the same `if/else` pattern every iteration.

**Tradeoff:**

Adding a new operation requires editing both the `OpType` enum and `Manager::backward()`. In the Python version you just add a method to the Value class. The centralization is a mild extensibility cost for a clear performance gain, and it keeps all gradient math in one auditable place.

---

### Decision 3: Index-based topo sort (not a pointer list)

**What the C++ version does:**

`m_topo` is `std::vector<size_t>`. `build_topo()` stores `v->idx` (the node's position in `m_all_nodes`) rather than the node's address. Lookup during backward is `m_all_nodes[node_id].get()`:

```cpp
void Manager::build_topo(Value* v, std::unordered_set<Value*>& visited) {
    if(v == nullptr || visited.count(v)) return;
    visited.insert(v);
    for(Value* prev : v->m_prev) {
        if(prev != nullptr) build_topo(prev, visited);
    }
    m_topo.push_back(v->idx);   // index, not pointer
}
```

**Why:**

`m_topo` is cached and reused across training iterations (see Decision 5). That caching is what makes indices the right choice over pointers.

When `clear_ephemeral_nodes()` runs at the end of each iteration, it destroys the `unique_ptr` for every ephemeral node, which frees the underlying heap `Value`. Any `Value*` pointing to a deleted ephemeral node is dangling at that point. On the next forward pass, new ephemeral nodes are created and pushed back into `m_all_nodes`, but at fresh heap addresses. A pointer-based topo cached from iteration 1 would hold addresses from the old ephemeral nodes, which are now freed.

Using indices avoids this. `m_all_nodes[idx]` always returns the current occupant of position `idx` in the vector. Because the graph is deterministic (same architecture, same input structure, same creation order), parameter nodes stay at positions 0 through P-1 and new ephemeral nodes fill positions P onwards in the same order every iteration. The cached index topo is valid every time backward runs.

**Tradeoff:**

Lookup requires `m_all_nodes[node_id].get()` instead of a direct dereference: one array index per node per backward pass. That overhead is negligible. The correctness of the cached topo depends on the graph being deterministic across iterations, which is true for a standard fixed-architecture training loop.

---

### Decision 4: `erase_if` to remove ephemeral nodes

**What the C++ version does:**

Every `Value` carries `bool m_is_parameter = false`. When a Neuron is constructed, it marks its weights and bias as parameters:

```cpp
Value& wi = vm->create(val, "w");
wi.m_is_parameter = true;
```

After each training iteration, `Manager::clear_ephemeral_nodes()` compacts the vector in a single pass:

```cpp
void Manager::clear_ephemeral_nodes(const std::vector<Value*>& parameters) {
    std::erase_if(m_all_nodes, [](const std::unique_ptr<Value>& node) {
        return !node->m_is_parameter;
    });
}
```

Every intermediate node from the forward pass (activations, dot-product accumulators, loss terms) is destroyed. The parameter nodes (weights and biases) remain.

**Why:**

In Python the graph is rebuilt from scratch each iteration because the GC collects old nodes as soon as the forward-pass references go out of scope. The Python version keeps `Neuron.w` and `Neuron.b` as long-lived Python objects — operations on them produce new short-lived graph nodes that GC cleans up automatically, with no explicit cleanup required.

The C++ Manager owns every node indefinitely unless told otherwise. Rebuilding the Manager and model from scratch each iteration would re-randomize the parameters, so instead the engine separates the two kinds of nodes by a boolean flag on the Value itself. A boolean flag is the cheapest way to distinguish them: O(1) to check per node, and `erase_if` compacts the vector in a single pass with a lambda. Weights and biases survive; everything else is freed.

An earlier version of this engine used an `unordered_set<Value*>` inside `clear_ephemeral_nodes()` to identify which nodes to keep, rebuilding that set every iteration. The boolean flag on the node itself eliminates that per-iteration allocation entirely.

**Tradeoff:**

`erase_if` on a `vector<unique_ptr>` is O(n) with destructor calls and move operations to compact the remainder. For the graph sizes in this project this is negligible. For very large models, maintaining a separate compute-node vector alongside the parameter vector would avoid compaction at the cost of a more complex topo sort spanning both.

---

### Decision 5: Cached topo vector (built once per training session)

**What the C++ version does:**

`m_is_topo_built` is a `bool` member of Manager, initially `false`. The first call to `Manager::backward()` runs `build_topo()` and sets the flag:

```cpp
void Manager::backward(Value& loss) {
    if(!m_is_topo_built) {
        std::unordered_set<Value*> visited;
        build_topo(&loss, visited);
        m_is_topo_built = true;
    }
    // zero grads, then iterate m_topo in reverse...
}
```

Every subsequent call skips the traversal. `m_topo` is populated once and reused for the entire training run.

**Why:**

In Python, `backward()` rebuilds `topo` and `visited` from scratch on every call:

```python
def backward(self):
    topo = []
    visited = set()
    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)
    build_topo(self)
    ...
```

This is O(n) list and set allocation plus DFS traversal on every backward pass. Python has no choice: the graph is rebuilt each forward pass, so the nodes at iteration N are entirely different objects from the nodes at iteration N+1. There is nothing stable to cache.

In the C++ version the graph topology is **deterministic**: for a fixed architecture and input structure, the same nodes connect in the same order every iteration. Parameter nodes persist (Decision 4 never erases them), and ephemeral nodes are recreated in the same order each forward pass (same Neuron constructors, same loop order). The topo ordering computed on iteration 1 is valid on iteration 2,000. Caching it means the DFS traversal and visited-set allocation run once for the entire training session instead of once per backward call.

**Tradeoff:**

The cached topo stays correct as long as the graph structure does not change after the first backward call, which is true for any training loop where the architecture is fixed at construction time.

---

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
./micrograd              # defaults: 10 epochs x 200 iters, lr=0.25
./micrograd 10 200 0.25  # explicit: [epochs] [iterations] [learning_rate]
```

Requires CMake 3.15+ and a C++23-compatible compiler (uses `std::erase_if`, `std::views::zip`).

---

## Status

| Component       | Status      |
|-----------------|-------------|
| Python notebook | Complete    |
| C++ engine      | Complete    |
| Rust engine     | Not started |

---

## Credit

Based on [micrograd](https://github.com/karpathy/micrograd) by Andrej Karpathy.

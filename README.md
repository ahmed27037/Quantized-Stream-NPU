# Pipelined INT8 NPU Core

A neural processing unit (NPU) slice that focuses on matrix multiply-accumulate (MAC) throughput. The design targets a 4×4 INT8 outer-product engine, but everything is parameterized so you can scale array size, data width, accumulator guard bits, or swap the activation function.

---

## Architecture Overview

### Top-Level Block Diagram

![NPU Top-Level Architecture](diagrams/npu_top_level.jpg)

**What this shows:** The complete NPU system architecture including:
- **Input Control**: `start`, `in_valid`, and data streaming interface
- **Two-Stage Pipeline**: Operand pipeline stages for matrix A columns and matrix B rows
- **PE Array**: 4×4 array of Processing Elements performing MAC operations
- **Activation Function**: Configurable activation (Identity or ReLU)
- **Output Generation**: Dual output paths - packed matrix output and streaming interface

**Key Features Visible:**
- Dual-stage pipeline for operands (stage0 and stage1)
- Processing Element (PE) array performing outer-product computation
- State machine controlling `active` and `streaming` states
- Streaming interface with `result_valid`, `result_ready`, and `result_index`

### PE Array Detail

![PE Array Architecture](diagrams/npu_pe_array.jpg)

**What this shows:** Detailed view of the Processing Element array and the outer-product algorithm:
- **Matrix Multiplication Concept**: How C = A × B is computed
- **Outer-Product Algorithm**: Breaking matrix multiplication into outer products
- **PE Array Structure**: 4×4 grid of Processing Elements
- **Individual PE Logic**: Each PE performs multiply-accumulate operations

**Algorithm:**
- For each column k of matrix A and row k of matrix B
- Each PE computes: `PE[i][j] += A[i][k] × B[k][j]`
- Accumulation happens over multiple cycles as k varies
- Final result is the complete matrix C

### Pipeline and Timing

![NPU Pipeline](diagrams/npu_pipeline_timing.jpg)

**What this shows:** The two-stage operand pipeline operation:
- **Stage 0**: First pipeline stage receives input operands
- **Stage 1**: Second pipeline stage feeds the PE array
- **Pipeline Flow**: How operands move through stages and into PEs
- **Valid Signal Propagation**: Pipeline valid signals tracking data flow

**Pipeline Benefits:**
- Keeps PE array busy once primed
- Reduces latency by overlapping operand loading and computation
- Enables continuous data streaming

---

## Repository Structure

```
Quantized-Stream-NPU/
├── rtl/                    # RTL (Register Transfer Level) design
│   ├── npu_core.sv        # Top-level NPU module
│   └── pe.sv              # Processing Element (PE) – core MAC unit
├── tb/                    # Testbenches for simulation
│   └── npu_core_tb.sv     # Testbench for npu_core
├── sw/                    # Software reference implementation
│   └── host_demo.cpp      # C++ reference model and demo
└── build/                 # Build artifacts
    └── npu_tb             # Compiled testbench executable
```

---

## Highlights

- **Two-stage operand pipeline** broadcasts one column of `A` and one row of `B` each cycle and keeps all MACs busy once primed
- **Automatic accumulator sizing** derives the minimum number of bits from `ARRAY_SIZE` and `DATA_WIDTH`. An extra guard parameter lets you add safety margin; the RTL will `$error` if you under-configure it
- **Optional activation block** applies ReLU (or leaves the data untouched) before results are observed
- **Dual output paths**: a packed matrix (`c_out_flat`) and a ready/valid streaming interface (`result_valid/result_ready`) that emits one element per cycle in row-major order
- **Self-checking SystemVerilog testbench** drives three representative matrix pairs, validates packed outputs, validates the streaming path, and cross-checks a second NPU instance configured with ReLU
- **Host-side C++ reference tool** (`sw/host_demo.cpp`) that mirrors the hardware math, prints matrices, reports accumulator widths, and shows the streaming order you should observe in waveform viewers

---

## File Map

- `rtl/pe.sv` - Multiply-accumulate processing element with synchronous clear and sign-extended products
- `rtl/npu_core.sv` - Top-level pipelined array with activation and streaming logic
- `tb/npu_core_tb.sv` - Simulation driver that covers math correctness, ReLU behaviour, and stream ordering
- `sw/host_demo.cpp` - Standalone C++ helper for generating host reference data

---

## Quick Start

### Prerequisites

**Icarus Verilog:**
- Windows: `choco install icarus-verilog`
- macOS: `brew install icarus-verilog`
- Linux: `sudo apt install iverilog`

**GTKwave** (optional, for waveform viewing):
- Windows: Download from http://gtkwave.sourceforge.net/
- macOS: `brew install gtkwave`
- Linux: `sudo apt install gtkwave`

**C++ Compiler** (optional, for host reference):
- GCC/Clang with C++17 support

Verify installation:
```powershell
iverilog -V
vvp -V
gtkwave --version
```

---

## Simulation

**IMPORTANT:** Run all commands from the `Quantized-Stream-NPU/` directory.

### Basic Simulation

```powershell
cd Quantized-Stream-NPU
iverilog -g2012 -o build/npu_tb rtl/pe.sv rtl/npu_core.sv tb/npu_core_tb.sv
vvp build/npu_tb
```

**Expected output:**
```
[TB] identity_passthrough passed
[TB] random_mix passed
[TB] zero_case passed
[TB] All testcases passed
```

### Simulation with GTKwave Waveforms

The testbench needs modification to generate VCD files. Add these lines to `tb/npu_core_tb.sv` in the `initial` block:

```systemverilog
initial begin
    $dumpfile("npu_core_tb.vcd");
    $dumpvars(0, npu_core_tb);
    // ... rest of testbench code
end
```

Then compile and run:
```powershell
cd Quantized-Stream-NPU
iverilog -g2012 -o build/npu_tb rtl/pe.sv rtl/npu_core.sv tb/npu_core_tb.sv
vvp build/npu_tb
gtkwave npu_core_tb.vcd
```

### Viewing Waveforms

Launch GTKwave:
```powershell
cd Quantized-Stream-NPU
gtkwave npu_core_tb.vcd
```

**Key Signals to Inspect:**

**Input Control:**
- `start` - Pulse high to begin computation
- `in_valid` - Valid signal for input operands
- `a_stream` - Matrix A column data (ARRAY_SIZE elements)
- `b_stream` - Matrix B row data (ARRAY_SIZE elements)

**Status Signals:**
- `busy` - High while core is active (loading data and streaming results)
- `done` - Asserted when computation completes
- `c_valid` - Valid signal for packed output

**State Machine:**
- `active` - State machine state indicating computation in progress
- `streaming` - State machine state indicating result streaming

**Pipeline Stages:**
- `a_stage0[*]`, `a_stage1[*]` - Pipeline stages for matrix A columns
- `b_stage0[*]`, `b_stage1[*]` - Pipeline stages for matrix B rows
- `valid_stage0`, `valid_stage1` - Valid signals for each pipeline stage

**Counters:**
- `feed_count` - Tracks how many operand pairs have been fed
- `processed_count` - Tracks processed elements

**Outputs:**
- `result_valid` - Valid signal for streaming output
- `result_ready` - Ready signal from consumer (can be throttled)
- `result_data` - One element of result matrix per cycle
- `result_index` - Index of current output element (0 to ARRAY_SIZE²-1)
- `c_out_flat` - Packed output (all 16 elements in row-major order)

**What to Look For:**
1. **Pipeline filling**: Watch operands move through stage0 → stage1 → PE array
2. **MAC accumulation**: See PE outputs accumulate over multiple cycles
3. **Streaming output**: Observe `result_index` incrementing 0→15 with corresponding `result_data`
4. **Back-pressure**: Set `result_ready=0` to see how streaming stalls
5. **ReLU activation**: Compare results with and without ReLU (negative values clamped to 0)

---

## Interface Protocol

The testbench (and RTL) accept the following protocol:

1. **Start**: Pulse `start` high for one cycle
2. **Feed Operands**: For `ARRAY_SIZE` consecutive cycles:
   - Assert `in_valid` high
   - Present column `k` of matrix A in `a_stream`
   - Present row `k` of matrix B in `b_stream`
   - Wait for next cycle
3. **Wait for Completion**: Wait for `done` or `c_valid` to go high
4. **Read Results**: Either:
   - Read packed output `c_out_flat` (all elements at once)
   - Read streaming output using `result_valid`/`result_ready` handshaking

---

## Streaming Interface

The streaming interface mirrors common AXI-Stream handshake semantics:

- **`result_valid`**: Asserted when a new element is available
- **`result_ready`**: Set high by consumer to indicate readiness to accept data
- **`result_index`**: Increments from 0 to `ARRAY_SIZE*ARRAY_SIZE-1`, matching row-major order
- **`busy`**: Stays high while the core is loading data **and** while data is draining through the stream

**Streaming Order (4×4 array):**
```
Index:  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
Matrix: C[0,0] C[0,1] C[0,2] C[0,3] C[1,0] C[1,1] ... C[3,3]
```

---

## Activation Functions

Configure the `ACT_FUNC` parameter when instantiating `npu_core`:

- **`0` - Identity**: Raw matrix product (no modification)
- **`1` - ReLU**: Rectified Linear Unit (negative values clamped to zero)

To add new activations, extend the `gen_relu/gen_identity` generate block inside `rtl/npu_core.sv`.

### ReLU Operation

```systemverilog
result = (input < 0) ? 0 : input;
```

---

## Accumulator Sizing

For signed INT8 (`DATA_WIDTH=8`), the largest product is 127×127 = 16,129. The core computes the minimum accumulator width as:

```
MIN_ACC_WIDTH = (2 × DATA_WIDTH) + ceil(log2(ARRAY_SIZE))
```

For 4×4 INT8:
```
MIN_ACC_WIDTH = (2 × 8) + ceil(log2(4))
              = 16 + 2
              = 18 bits
```

If you leave `ACC_WIDTH` at its default, the module adds `EXTRA_ACC_BITS` on top of the minimum. Supplying a smaller width triggers a run-time `$error` so you can catch configuration issues in simulation.

### Accumulator Cheat Sheet

| Array Size | Data Width | Min Acc Width | Typical Acc Width |
|------------|------------|---------------|-------------------|
| 4×4        | 8 bits     | 18 bits       | 20-24 bits        |
| 8×8        | 8 bits     | 19 bits       | 21-24 bits        |
| 16×16      | 8 bits     | 20 bits       | 22-24 bits        |

---

## Host Reference Program

The helper in `sw/host_demo.cpp` prints the exact numbers the hardware should produce and lets you experiment with the accumulator guard bits:

**Compile:**
```powershell
cd Quantized-Stream-NPU
g++ -std=c++17 -O2 -o build/host_demo sw/host_demo.cpp
```

**Run:**
```powershell
./build/host_demo                    # Deterministic demo matrices
./build/host_demo --random           # Generate random INT8-friendly matrices
./build/host_demo --random --seed=123 --extra-bits=6
```

**Sample output** includes:
- Input matrices A and B
- Raw matrix product C = A × B
- ReLU-activated version
- Streaming order (element indices)
- Accumulator width calculations

Use this to cross-check hardware simulation results.

---

## Debug Tips

1. **Pipeline Inspection**: Watch `valid_stage0` and `valid_stage1` to see operand flow
2. **Back-Pressure Testing**: Set `result_ready=0` to watch stream stall propagation
3. **Packed vs Stream**: Use `c_out_flat` for quick scoreboarding; use stream for integration testing
4. **Accumulator Overflow**: Modify `EXTRA_ACC_BITS` to stress accumulator sizing
5. **ReLU Verification**: Compare results from Identity vs ReLU configurations
6. **Index Tracking**: Watch `result_index` to verify streaming order matches row-major format

---

## Extending the Design

- **Scale Array Size**: Change `ARRAY_SIZE` parameter (ensure accumulator width is sufficient)
- **Change Data Width**: Modify `DATA_WIDTH` parameter (update accumulator calculations)
- **Add Activations**: Extend activation function block with new functions (Sigmoid, Tanh, etc.)
- **Multiple Arrays**: Instantiate multiple NPU cores for larger matrix operations
- **AXI-Stream Interface**: Wrap streaming interface in AXI-Stream protocol for SoC integration

---

## Troubleshooting

**"Unknown module type" errors:**
- Ensure all source files are included: `rtl/pe.sv rtl/npu_core.sv tb/npu_core_tb.sv`

**VCD file not generated:**
- Add `$dumpfile()` and `$dumpvars()` to testbench `initial` block

**Simulation hangs:**
- Check that `result_ready` is asserted to allow streaming output
- Verify `start` is pulsed correctly

**Wrong results:**
- Compare against `host_demo.cpp` reference implementation
- Check accumulator width is sufficient for array size
- Verify operand feed order matches protocol (column k of A, row k of B)

**GTKwave not found:**
- Install GTKwave and ensure it's in PATH
- Or manually open VCD file: `gtkwave npu_core_tb.vcd`

---

## License

[Specify your license here]

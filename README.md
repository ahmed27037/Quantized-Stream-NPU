## Pipelined INT8 NPU Core

This repository contains a easily debuggable neural processing unit (NPU) slice that focuses on matrix multiply-accumulate (MAC) throughput. The design targets a 4x4 INT8 outer-product engine, but everything is parameterised so you can scale array size, data width, accumulator guard bits, or swap the activation.

### Highlights
- **Two-stage operand pipeline** broadcasts one column of `A` and one row of `B` each cycle and keeps all MACs busy once primed.
- **Automatic accumulator sizing** derives the minimum number of bits from `ARRAY_SIZE` and `DATA_WIDTH`. An extra guard parameter lets you add safety margin; the RTL will `\$error` if you under-configure it.
- **Optional activation block** applies ReLU (or leaves the data untouched) before results are observed.
- **Dual output paths**: a packed matrix (`c_out_flat`) and a ready/valid streaming interface (`result_valid/result_ready`) that emits one element per cycle in row-major order.
- **Self-checking SystemVerilog testbench** drives three representative matrix pairs, validates packed outputs, validates the streaming path, and cross-checks a second NPU instance configured with ReLU.
- **Host-side C++ reference tool** (`sw/host_demo.cpp`) that mirrors the hardware math, prints matrices, reports accumulator widths, and shows the streaming order you should observe in waveform viewers.

### File Map
- `rtl/pe.sv` - multiply-accumulate processing element with synchronous clear and sign-extended products.
- `rtl/npu_core.sv` - top-level pipelined array with activation and streaming logic.
- `tb/npu_core_tb.sv` - simulation driver that covers math correctness, ReLU behaviour, and stream ordering.
- `sw/host_demo.cpp` - standalone C++ helper for generating host reference data.

### Quick Start (Simulation)
```bash
iverilog -g2012 -o build/npu_tb rtl/pe.sv rtl/npu_core.sv tb/npu_core_tb.sv
vvp build/npu_tb
```

Expected output ends with:
```
[TB] identity_passthrough passed
[TB] random_mix passed
[TB] zero_case passed
[TB] All testcases passed
```

The testbench accepts the same start/stream protocol the RTL expects:
1. Pulse `start` for one cycle.
2. For `ARRAY_SIZE` consecutive cycles assert `in_valid` and present column `k` of `A` and row `k` of `B`.
3. Wait for `c_valid`/`done`, then either grab the packed matrix or read the stream (`result_valid`/`result_ready`).

### Streaming Interface
The streaming interface mirrors common AXI-Stream handshake semantics:
- `result_valid` is asserted when a new element is available.
- Set `result_ready` high to consume elements; throttling is supported.
- `result_index` increments from 0 to `ARRAY_SIZE*ARRAY_SIZE-1`, matching row-major order.
- `busy` stays high while the core is loading data **and** while data is draining through the stream.

### Activation Options
Configure the `ACT_FUNC` parameter when instantiating `npu_core`:
- `0` - Identity (raw matrix product).
- `1` - ReLU (negative values clamped to zero).

To add new activations, extend the `gen_relu/gen_identity` generate block inside `rtl/npu_core.sv`.

### Accumulator Sizing Cheat Sheet
For signed INT8 (`DATA_WIDTH=8`) the largest product is 127*127. The core computes the minimum accumulator width as:

```
MIN_ACC_WIDTH = (2 * DATA_WIDTH) + ceil(log2(ARRAY_SIZE))
```

If you leave `ACC_WIDTH` at its default, the module adds `EXTRA_ACC_BITS` on top of the minimum. Supplying a smaller width triggers a run-time `\$error` so you can catch configuration issues in simulation.

### Host Reference Program
The helper in `sw/host_demo.cpp` prints the exact numbers the hardware should produce and lets you experiment with the accumulator guard bits:

```bash
g++ -std=c++17 -O2 -o build/host_demo sw/host_demo.cpp
./build/host_demo               # deterministic demo matrices
./build/host_demo --random      # generate random INT8-friendly matrices
./build/host_demo --random --seed=123 --extra-bits=6
```

Sample output includes the raw GEMM, the ReLU-activated version, and the streaming order so you can correlate against waveforms or log files.

### Debug Tips
- Inspect the stream with `result_ready=0` to watch back-pressure stall propagation.
- Use the packed output (`c_out_flat`) for quick scoreboarding; switch to the stream when integrating with DMA/FIFO logic.
- Modify `EXTRA_ACC_BITS` in the testbench to stress accumulator overflow or to emulate lower-precision accumulators.

### Next Ideas
- Swap ReLU for a programmable activation/LUT.
- Replace the simple streaming logic with an AXI-Stream shell and add skid buffers.
- Hook the C++ host into a cocotb/Verilator harness to co-simulate software and RTL.

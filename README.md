# Pipelined INT8 NPU Core

A 4×4 INT8 neural processing unit focused on matrix multiply-accumulate throughput. The design uses a two-stage outer-product pipeline with configurable parameters for array size, data width, and activation functions.

---

## Architecture

### Top-Level Block Diagram

```mermaid
flowchart TB
    CLK[clk]
    RST[rst]
    START[start]
    INVALID[in_valid]
    ASTREAM["a_stream[31:0]"]
    BSTREAM["b_stream[31:0]"]
    RREADY[result_ready]
    
    subgraph NPU ["npu_core Module"]
        subgraph PIPE0 ["Stage 0 Pipeline"]
            AS0["a_stage0[3:0]<br/>8-bit each"]
            BS0["b_stage0[3:0]<br/>8-bit each"]
            VS0[valid_stage0]
        end
        
        subgraph PIPE1 ["Stage 1 Pipeline"]
            AS1["a_stage1[3:0]<br/>8-bit each"]
            BS1["b_stage1[3:0]<br/>8-bit each"]
            VS1["valid_stage1<br/>enable signal"]
        end
        
        subgraph ARRAY ["4x4 PE Array"]
            PE00["PE[0][0]"]
            PE01["PE[0][1]"]
            PE10["PE[1][0]"]
            PE11["PE[1][1]"]
            PE20["PE[2][0]"]
            PE21["PE[2][1]"]
            PE30["PE[3][0]"]
            PE31["PE[3][1]"]
        end
        
        CLEAR[clear_acc<br/>start signal]
        ACTFUNC["Activation<br/>ReLU / Identity"]
        STREAMER["Stream FSM<br/>Result Output"]
    end
    
    BUSY[busy]
    DONE[done]
    CVALID[c_valid]
    COUT["c_out_flat[319:0]"]
    RVALID[result_valid]
    RDATA["result_data[19:0]"]
    RIDX["result_index[3:0]"]
    
    CLK --> NPU
    RST --> NPU
    START --> CLEAR
    INVALID --> VS0
    ASTREAM --> AS0
    BSTREAM --> BS0
    
    AS0 --> AS1
    BS0 --> BS1
    VS0 --> VS1
    
    AS1 -->|broadcast rows| PE00
    AS1 -->|broadcast rows| PE10
    BS1 -->|broadcast cols| PE00
    BS1 -->|broadcast cols| PE01
    
    CLEAR --> PE00
    CLEAR --> PE11
    VS1 --> PE00
    VS1 --> PE11
    
    PE00 --> ACTFUNC
    PE01 --> ACTFUNC
    PE10 --> ACTFUNC
    PE11 --> ACTFUNC
    PE20 --> ACTFUNC
    PE21 --> ACTFUNC
    PE30 --> ACTFUNC
    PE31 --> ACTFUNC
    
    ACTFUNC --> COUT
    ACTFUNC --> STREAMER
    RREADY --> STREAMER
    
    NPU --> BUSY
    NPU --> DONE
    NPU --> CVALID
    STREAMER --> RVALID
    STREAMER --> RDATA
    STREAMER --> RIDX
```

The system includes a dual-stage operand pipeline that broadcasts matrix A columns and matrix B rows to a 4×4 processing element array. Each PE performs multiply-accumulate operations. Results can be read through either a packed output bus or a streaming interface with ready/valid handshaking.

### PE Array Detail

```mermaid
flowchart TD
    subgraph INPUT ["INPUT: Operand Broadcast"]
        direction LR
        A_COL["A Column Broadcast<br/>a_stage1 4x8-bit INT8"]
        B_ROW["B Row Broadcast<br/>b_stage1 4x8-bit INT8"]
    end
    
    subgraph CTRL ["CONTROL: Control Signals"]
        direction LR
        ENABLE["valid_stage1<br/>PE enable"]
        CLEAR["clear_acc<br/>reset"]
    end
    
    subgraph PE_DETAIL ["PE STRUCTURE: Each PE Component"]
        direction LR
        MULT["Multiplier<br/>a_value x b_value"]
        EXTEND["Sign Extend<br/>16-bit to 20-bit"]
        ACCUM["Accumulator<br/>acc += product"]
        
        MULT --> EXTEND
        EXTEND --> ACCUM
    end
    
    subgraph PE_ARRAY ["PE ARRAY: 4x4 Processing Elements"]
        subgraph ROW0 ["Row 0"]
            direction LR
            PE00["PE 0,0"]
            PE01["PE 0,1"]
            PE02["PE 0,2"]
            PE03["PE 0,3"]
            
            PE00 --> PE01
            PE01 --> PE02
            PE02 --> PE03
        end
        
        subgraph ROW1 ["Row 1"]
            direction LR
            PE10["PE 1,0"]
            PE11["PE 1,1"]
            PE12["PE 1,2"]
            PE13["PE 1,3"]
            
            PE10 --> PE11
            PE11 --> PE12
            PE12 --> PE13
        end
        
        subgraph ROW2 ["Row 2"]
            direction LR
            PE20["PE 2,0"]
            PE21["PE 2,1"]
            PE22["PE 2,2"]
            PE23["PE 2,3"]
            
            PE20 --> PE21
            PE21 --> PE22
            PE22 --> PE23
        end
        
        subgraph ROW3 ["Row 3"]
            direction LR
            PE30["PE 3,0"]
            PE31["PE 3,1"]
            PE32["PE 3,2"]
            PE33["PE 3,3"]
            
            PE30 --> PE31
            PE31 --> PE32
            PE32 --> PE33
        end
        
        ROW0 --> ROW1
        ROW1 --> ROW2
        ROW2 --> ROW3
    end
    
    subgraph ACT ["ACTIVATION: Output Processing"]
        direction LR
        ACTFN["ReLU or Identity"]
    end
    
    subgraph OUT ["OUTPUT: Result Matrix C"]
        direction LR
        COUT0["C Row 0<br/>C00 C01 C02 C03<br/>20-bit each"]
        COUT1["C Row 1<br/>C10 C11 C12 C13<br/>20-bit each"]
        COUT2["C Row 2<br/>C20 C21 C22 C23<br/>20-bit each"]
        COUT3["C Row 3<br/>C30 C31 C32 C33<br/>20-bit each"]
        
        COUT0 --> COUT1
        COUT1 --> COUT2
        COUT2 --> COUT3
    end
    
    INPUT --> PE_ARRAY
    CTRL --> PE_ARRAY
    PE_ARRAY --> ACT
    ACT --> OUT
```

The outer-product algorithm computes C = A × B by streaming columns of A and rows of B. For each column k, the PE at position [i][j] computes `PE[i][j] += A[i][k] × B[k][j]`. Accumulation happens over multiple cycles as k increments from 0 to ARRAY_SIZE-1.

### Pipeline

```mermaid
flowchart LR
    subgraph S0 ["STAGE 0: Operand Loading"]
        S0K0["Cycle 0<br/><b>Load k=0</b><br/>a_stream → a_stage0<br/>b_stream → b_stage0"]
        S0K1["Cycle 1<br/><b>Load k=1</b><br/>a_stream → a_stage0<br/>b_stream → b_stage0"]
        S0K2["Cycle 2<br/><b>Load k=2</b><br/>a_stream → a_stage0<br/>b_stream → b_stage0"]
        S0K3["Cycle 3<br/><b>Load k=3</b><br/>a_stream → a_stage0<br/>b_stream → b_stage0"]
        
        S0K0 --> S0K1
        S0K1 --> S0K2
        S0K2 --> S0K3
    end
    
    subgraph S1 ["STAGE 1: Broadcast to PE Array"]
        S1K0["Cycle 1<br/><b>Broadcast k=0</b><br/>a_stage1 → PE rows<br/>b_stage1 → PE cols"]
        S1K1["Cycle 2<br/><b>Broadcast k=1</b><br/>a_stage1 → PE rows<br/>b_stage1 → PE cols"]
        S1K2["Cycle 3<br/><b>Broadcast k=2</b><br/>a_stage1 → PE rows<br/>b_stage1 → PE cols"]
        S1K3["Cycle 4<br/><b>Broadcast k=3</b><br/>a_stage1 → PE rows<br/>b_stage1 → PE cols"]
        
        S1K0 --> S1K1
        S1K1 --> S1K2
        S1K2 --> S1K3
    end
    
    subgraph PE ["PE ARRAY: MAC Operations"]
        PEK0["Cycle 1<br/><b>MAC k=0</b><br/>All PEs compute<br/>acc += A × B"]
        PEK1["Cycle 2<br/><b>MAC k=1</b><br/>All PEs compute<br/>acc += A × B"]
        PEK2["Cycle 3<br/><b>MAC k=2</b><br/>All PEs compute<br/>acc += A × B"]
        PEK3["Cycle 4<br/><b>MAC k=3</b><br/>All PEs compute<br/>acc += A × B"]
        
        PEK0 --> PEK1
        PEK1 --> PEK2
        PEK2 --> PEK3
    end
    
    subgraph OUT ["RESULT OUTPUT: Streaming"]
        STREAM["Cycles 5-20<br/><b>Stream 16 Results</b><br/>c_out_flat[319:0]<br/>→ result_data[19:0]<br/>→ result_index[3:0]"]
    end
    
    S0K0 -.Pipeline delay.-> S1K0
    S0K1 -.Pipeline delay.-> S1K1
    S0K2 -.Pipeline delay.-> S1K2
    S0K3 -.Pipeline delay.-> S1K3
    
    S1K0 -.Broadcast.-> PEK0
    S1K1 -.Broadcast.-> PEK1
    S1K2 -.Broadcast.-> PEK2
    S1K3 -.Broadcast.-> PEK3
    
    PEK3 --> STREAM
```

The two-stage pipeline keeps the PE array busy by overlapping operand loading and computation. Stage 0 receives inputs while stage 1 feeds the array, reducing latency and enabling continuous streaming.

---

## Quantization

The NPU processes INT8 data, but most neural networks train in floating-point. Quantization happens in software before feeding data to the hardware.

The implementation uses symmetric quantization with zero-point = 0:
- Scale factor: `scale = max(|min|, |max|) / 127`
- Quantize: `q = clamp(round(x / scale), -128, 127)`
- Dequantize: `x ≈ q × scale`

The flow is: FP32 data → quantize in software → send INT8 to hardware → hardware computes → optionally dequantize output.

---

## Repository Structure

```
Quantized-Stream-NPU/
├── rtl/
│   ├── npu_core.sv           # Top-level NPU module
│   └── pe.sv                 # Processing element (MAC unit)
├── tb/
│   └── npu_integrated_tb.sv  # Testbench
├── sw/
│   ├── gen_test_vectors.cpp  # Generates quantized test vectors
│   └── host_demo.cpp         # Reference model (optional)
└── build/                    # Build artifacts
```

---

## Quick Start

### Prerequisites

- Icarus Verilog: `choco install icarus-verilog` (Windows)
- C++ compiler with C++17 support

Verify:
```powershell
iverilog -V
g++ --version
```

### Running the Test

This runs the full integration: quantize FP32 matrices, feed INT8 to RTL, verify output.

```powershell
cd "Quantized-Stream-NPU"; g++ -std=c++17 -O2 -o build/gen_test_vectors.exe sw/gen_test_vectors.cpp; .\build\gen_test_vectors.exe; iverilog -g2012 -o build/npu_integrated_tb rtl/pe.sv rtl/npu_core.sv tb/npu_integrated_tb.sv; vvp build/npu_integrated_tb
```

Expected output:
```
Quantization Parameters:
  Matrix A: scale=0.0354331, zero_point=0
  Matrix B: scale=0.0433071, zero_point=0

[...matrices displayed...]

[TB] Loaded test vectors from build/test_vectors.hex
[TB] Streaming quantized INT8 data to RTL...
[PASS] All 16 outputs match golden reference

Integration Test PASSED
```

Use `--random` for random test data:
```powershell
cd "Quantized-Stream-NPU"; g++ -std=c++17 -O2 -o build/gen_test_vectors.exe sw/gen_test_vectors.cpp; .\build\gen_test_vectors.exe --random; iverilog -g2012 -o build/npu_integrated_tb rtl/pe.sv rtl/npu_core.sv tb/npu_integrated_tb.sv; vvp build/npu_integrated_tb
```

---

## Interface

### Input Side

```systemverilog
input                               start,        // Pulse to begin computation
input                               in_valid,     // Valid signal for operands
input  [ARRAY_SIZE*DATA_WIDTH-1:0] a_stream,    // Column of matrix A
input  [ARRAY_SIZE*DATA_WIDTH-1:0] b_stream,    // Row of matrix B
```

Feed operands column-by-column. For column k, present A[:,k] on `a_stream` and B[k,:] on `b_stream` with `in_valid` high. Repeat for ARRAY_SIZE cycles.

### Output Side

Two output modes:

**Packed output:**
```systemverilog
output                              done,         // Computation complete
output                              c_valid,      // Output valid
output [OUTPUT_COUNT*ACC_WIDTH-1:0] c_out_flat, // All results at once
```

**Streaming output:**
```systemverilog
output                              result_valid, // Data available
output [ACC_WIDTH-1:0]              result_data,  // One element per cycle
output [INDEX_WIDTH-1:0]            result_index, // Current index (0 to 15)
input                               result_ready  // Consumer ready
```

Streaming follows row-major order: index 0 = C[0,0], index 1 = C[0,1], ..., index 15 = C[3,3].

---

## Parameters

Configure at instantiation:

```systemverilog
npu_core #(
    .ARRAY_SIZE     (4),    // Matrix dimension (4×4 default)
    .DATA_WIDTH     (8),    // Bits per operand (INT8)
    .EXTRA_ACC_BITS (2),    // Guard bits for accumulator
    .ACT_FUNC       (0)     // 0=identity, 1=ReLU
) u_npu (...);
```

The accumulator width is computed automatically: `ACC_WIDTH = 2*DATA_WIDTH + log2(ARRAY_SIZE) + EXTRA_ACC_BITS`. For 4×4 INT8, minimum is 18 bits. The RTL will error if you configure it too small.

---

## Waveform Viewing

To generate VCD files, add these lines to the testbench's `initial` block:

```systemverilog
initial begin
    $dumpfile("npu.vcd");
    $dumpvars(0, npu_integrated_tb);
    // ... rest of test
end
```

Then view with GTKwave:
```powershell
gtkwave npu.vcd
```

Key signals to inspect:
- Pipeline stages: `a_stage0`, `a_stage1`, `valid_stage0`, `valid_stage1`
- State machine: `active`, `streaming`, `busy`, `done`
- Accumulators: `acc_matrix[row][col]` inside PE instances
- Stream control: `result_valid`, `result_ready`, `result_index`

---

## Accumulator Sizing

For signed INT8, the worst-case product is 127 × 127 = 16,129. Accumulating ARRAY_SIZE of these requires:

```
MIN_ACC_WIDTH = 2*DATA_WIDTH + ceil(log2(ARRAY_SIZE))
```

For 4×4: `MIN = 16 + 2 = 18 bits`

The `EXTRA_ACC_BITS` parameter adds safety margin. If your configuration is insufficient, simulation will error during elaboration.

---

## Extending the Design

**Scale the array:** Change `ARRAY_SIZE` parameter. Ensure accumulator width is sufficient.

**Different data widths:** Modify `DATA_WIDTH` and update accumulator calculation.

**Add activation functions:** Extend the generate block in `npu_core.sv` that creates `act_matrix`. Current options are identity (0) and ReLU (1).

**Multiple arrays:** Instantiate several NPU cores for larger matrix operations.

**SoC integration:** The streaming interface is similar to AXI-Stream but simplified. Wrap it with full AXI-Stream signals (TLAST, TKEEP, etc.) for integration with interconnects.

---

## Troubleshooting

**"Unknown module type":** Include all files: `rtl/pe.sv rtl/npu_core.sv tb/npu_integrated_tb.sv`

**"Could not open test_vectors.hex":** Run `gen_test_vectors.exe` first to create the file.

**Simulation hangs:** Check that `result_ready` is asserted to allow streaming output to drain.

**Wrong results:** Compare against the golden output printed by `gen_test_vectors.exe`.

---

## Files

**RTL:**
- `rtl/pe.sv` - Processing element with multiply-accumulate and synchronous clear
- `rtl/npu_core.sv` - Top-level with pipeline, PE array, activation, and streaming logic

**Test:**
- `tb/npu_integrated_tb.sv` - Reads quantized vectors, runs simulation, verifies output
- `sw/gen_test_vectors.cpp` - Quantizes FP32 matrices to INT8, computes golden reference

**Reference:**
- `sw/host_demo.cpp` - Optional reference model for cross-checking

`timescale 1ns/1ps

// Integrated testbench: Tests quantized data flow through RTL
module npu_integrated_tb;

    localparam integer ARRAY_SIZE      = 4;
    localparam integer DATA_WIDTH      = 8;
    localparam integer EXTRA_ACC_BITS  = 4;
    localparam integer ACC_WIDTH       = (2*DATA_WIDTH) + $clog2(ARRAY_SIZE) + EXTRA_ACC_BITS;
    localparam integer OUTPUT_COUNT    = ARRAY_SIZE * ARRAY_SIZE;
    localparam integer INDEX_WIDTH     = (OUTPUT_COUNT > 1) ? $clog2(OUTPUT_COUNT) : 1;

    reg clk;
    reg rst;
    reg start;
    reg in_valid;
    reg [ARRAY_SIZE*DATA_WIDTH-1:0] a_stream;
    reg [ARRAY_SIZE*DATA_WIDTH-1:0] b_stream;
    reg result_ready;

    wire busy;
    wire done;
    wire c_valid;
    wire [OUTPUT_COUNT*ACC_WIDTH-1:0] c_out_flat;
    wire result_valid;
    wire [ACC_WIDTH-1:0] result_data;
    wire [INDEX_WIDTH-1:0] result_index;

    reg signed [DATA_WIDTH-1:0] matrix_a [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1];
    reg signed [DATA_WIDTH-1:0] matrix_b [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1];
    reg signed [ACC_WIDTH-1:0]  golden [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1];

    npu_core #(
        .ARRAY_SIZE     (ARRAY_SIZE),
        .DATA_WIDTH     (DATA_WIDTH),
        .EXTRA_ACC_BITS (EXTRA_ACC_BITS),
        .ACT_FUNC       (0)
    ) dut (
        .clk          (clk),
        .rst          (rst),
        .start        (start),
        .in_valid     (in_valid),
        .a_stream     (a_stream),
        .b_stream     (b_stream),
        .busy         (busy),
        .done         (done),
        .c_valid      (c_valid),
        .c_out_flat   (c_out_flat),
        .result_valid (result_valid),
        .result_data  (result_data),
        .result_index (result_index),
        .result_ready (result_ready)
    );

    // 100 MHz clock
    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    task apply_reset;
        begin
            rst         <= 1'b1;
            start       <= 1'b0;
            in_valid    <= 1'b0;
            a_stream    <= '0;
            b_stream    <= '0;
            result_ready<= 1'b0;
            repeat (4) @(negedge clk);
            rst         <= 1'b0;
            @(negedge clk);
        end
    endtask

    task load_test_vectors;
        integer i, r, c, idx;
        reg [31:0] vec_data;
        integer fd;
        integer scan_result;
        begin
            fd = $fopen("build/test_vectors.hex", "r");
            if (fd == 0) begin
                $display("[ERROR] Could not open build/test_vectors.hex");
                $display("[INFO] Run gen_test_vectors first:");
                $display("  cd Quantized-Stream-NPU");
                $display("  g++ -std=c++17 -O2 -o build/gen_test_vectors.exe sw/gen_test_vectors.cpp");
                $display("  .\\build\\gen_test_vectors.exe");
                $fatal(1, "Missing test vector file");
            end

            // Read matrix A
            for (r = 0; r < ARRAY_SIZE; r = r + 1) begin
                for (c = 0; c < ARRAY_SIZE; c = c + 1) begin
                    scan_result = $fscanf(fd, "%h\n", vec_data);
                    matrix_a[r][c] = vec_data[7:0];
                end
            end

            // Read matrix B
            for (r = 0; r < ARRAY_SIZE; r = r + 1) begin
                for (c = 0; c < ARRAY_SIZE; c = c + 1) begin
                    scan_result = $fscanf(fd, "%h\n", vec_data);
                    matrix_b[r][c] = vec_data[7:0];
                end
            end

            // Read golden results
            for (r = 0; r < ARRAY_SIZE; r = r + 1) begin
                for (c = 0; c < ARRAY_SIZE; c = c + 1) begin
                    scan_result = $fscanf(fd, "%h\n", vec_data);
                    golden[r][c] = vec_data;
                end
            end

            $fclose(fd);
            $display("[TB] Loaded test vectors from build/test_vectors.hex");
        end
    endtask

    task stream_operands;
        integer k, row;
        begin
            @(negedge clk);
            start    <= 1'b1;
            in_valid <= 1'b0;
            a_stream <= '0;
            b_stream <= '0;

            @(negedge clk);
            start <= 1'b0;
            for (k = 0; k < ARRAY_SIZE; k += 1) begin
                in_valid <= 1'b1;
                for (row = 0; row < ARRAY_SIZE; row += 1) begin
                    a_stream[(row*DATA_WIDTH) +: DATA_WIDTH] <= matrix_a[row][k];
                    b_stream[(row*DATA_WIDTH) +: DATA_WIDTH] <= matrix_b[k][row];
                end
                @(negedge clk);
            end
            in_valid <= 1'b0;
            a_stream <= '0;
            b_stream <= '0;
        end
    endtask

    task wait_for_done;
        integer cycles_waited;
        integer max_cycles;
        begin
            max_cycles = (ARRAY_SIZE * 3) + OUTPUT_COUNT + 20;
            cycles_waited = 0;
            while (!c_valid) begin
                @(posedge clk);
                cycles_waited += 1;
                if (cycles_waited > max_cycles) begin
                    $fatal(1, "[TB] Timeout waiting for c_valid");
                end
            end
            @(posedge clk);
        end
    endtask

    task check_results;
        integer row;
        integer col;
        integer errors;
        reg signed [ACC_WIDTH-1:0] observed;
        begin
            errors = 0;
            for (row = 0; row < ARRAY_SIZE; row = row + 1) begin
                for (col = 0; col < ARRAY_SIZE; col = col + 1) begin
                    observed = c_out_flat[((row*ARRAY_SIZE)+col)*ACC_WIDTH +: ACC_WIDTH];
                    if (observed !== golden[row][col]) begin
                        $display("[FAIL] Mismatch at C[%0d][%0d]: RTL=%0d  Golden=%0d",
                                 row, col, observed, golden[row][col]);
                        errors = errors + 1;
                    end
                end
            end

            if (errors == 0) begin
                $display("[PASS] ✓ All %0d outputs match golden reference", OUTPUT_COUNT);
            end else begin
                $fatal(1, "[FAIL] %0d mismatches found", errors);
            end
        end
    endtask

    initial begin
        $display("========================================");
        $display("  Integrated Quantization → RTL Test");
        $display("========================================\n");

        load_test_vectors();
        apply_reset();
        
        $display("[TB] Streaming quantized INT8 data to RTL...");
        stream_operands();
        
        $display("[TB] Waiting for computation to complete...");
        wait_for_done();
        
        $display("[TB] Verifying RTL output against golden reference...");
        check_results();

        $display("\n========================================");
        $display("  Integration Test PASSED");
        $display("========================================");
        $finish;
    end

endmodule


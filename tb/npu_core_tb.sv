`timescale 1ns/1ps

module npu_core_tb;

    localparam integer ARRAY_SIZE      = 4;
    localparam integer DATA_WIDTH      = 8;
    localparam integer EXTRA_ACC_BITS  = 4;
    localparam integer ACC_WIDTH       = (2*DATA_WIDTH) + $clog2(ARRAY_SIZE) + EXTRA_ACC_BITS;
    localparam integer OUTPUT_COUNT    = ARRAY_SIZE * ARRAY_SIZE;
    localparam integer INDEX_WIDTH     = (OUTPUT_COUNT > 1) ? $clog2(OUTPUT_COUNT) : 1;
    localparam integer TOTAL_LATENCY   = (ARRAY_SIZE * 3) - 2;

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

    wire relu_busy;
    wire relu_done;
    wire relu_c_valid;
    wire [OUTPUT_COUNT*ACC_WIDTH-1:0] relu_c_out_flat;
    wire relu_result_valid;
    wire [ACC_WIDTH-1:0] relu_result_data;
    wire [INDEX_WIDTH-1:0] relu_result_index;
    wire relu_result_ready = 1'b1;

    reg signed [DATA_WIDTH-1:0] matrix_a [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1];
    reg signed [DATA_WIDTH-1:0] matrix_b [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1];
    reg signed [ACC_WIDTH-1:0]  golden_raw [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1];
    reg signed [ACC_WIDTH-1:0]  golden_relu [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1];
    reg signed [ACC_WIDTH-1:0]  stream_matrix [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1];

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

    npu_core #(
        .ARRAY_SIZE     (ARRAY_SIZE),
        .DATA_WIDTH     (DATA_WIDTH),
        .EXTRA_ACC_BITS (EXTRA_ACC_BITS),
        .ACT_FUNC       (1)
    ) dut_relu (
        .clk          (clk),
        .rst          (rst),
        .start        (start),
        .in_valid     (in_valid),
        .a_stream     (a_stream),
        .b_stream     (b_stream),
        .busy         (relu_busy),
        .done         (relu_done),
        .c_valid      (relu_c_valid),
        .c_out_flat   (relu_c_out_flat),
        .result_valid (relu_result_valid),
        .result_data  (relu_result_data),
        .result_index (relu_result_index),
        .result_ready (relu_result_ready)
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

    task compute_golden;
        integer i, j, k;
        integer signed sum;
        begin
            for (i = 0; i < ARRAY_SIZE; i += 1) begin
                for (j = 0; j < ARRAY_SIZE; j += 1) begin
                    sum = 0;
                    for (k = 0; k < ARRAY_SIZE; k += 1) begin
                        sum += matrix_a[i][k] * matrix_b[k][j];
                    end
                    golden_raw[i][j]  = sum;
                    golden_relu[i][j] = (sum < 0) ? '0 : sum;
                end
            end
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
        begin
            cycles_waited = 0;
            while (!c_valid) begin
                @(posedge clk);
                cycles_waited += 1;
                if (cycles_waited > (TOTAL_LATENCY + OUTPUT_COUNT + 10)) begin
                    $fatal(1, "[TB] Timeout waiting for c_valid");
                end
            end
            @(posedge clk);
        end
    endtask

    task collect_stream;
        integer captured;
        integer row;
        integer col;
        begin
            for (row = 0; row < ARRAY_SIZE; row += 1) begin
                for (col = 0; col < ARRAY_SIZE; col += 1) begin
                    stream_matrix[row][col] = '0;
                end
            end

            captured = 0;
            result_ready <= 1'b1;
            while (captured < OUTPUT_COUNT) begin
                @(posedge clk);
                if (result_valid && result_ready) begin
                    if (result_index !== captured[INDEX_WIDTH-1:0]) begin
                        $fatal(1, "[TB] stream index mismatch: observed=%0d expected=%0d",
                               result_index, captured);
                    end
                    row = captured / ARRAY_SIZE;
                    col = captured % ARRAY_SIZE;
                    stream_matrix[row][col] = result_data;
                    captured += 1;
                end
            end
            result_ready <= 1'b0;

            wait (!busy);
            @(posedge clk);
        end
    endtask

    task check_results(input [8*32-1:0] label);
        integer row;
        integer col;
        reg signed [ACC_WIDTH-1:0] observed;
        reg signed [ACC_WIDTH-1:0] relu_observed;
        begin
            compute_golden();
            stream_operands();
            wait_for_done();
            collect_stream();

            for (row = 0; row < ARRAY_SIZE; row += 1) begin
                for (col = 0; col < ARRAY_SIZE; col += 1) begin
                    observed = c_out_flat[((row*ARRAY_SIZE)+col)*ACC_WIDTH +: ACC_WIDTH];
                    if (observed !== golden_raw[row][col]) begin
                        $fatal(1,
                               "[TB] %s mismatch (flat) at C[%0d][%0d]: observed=%0d expected=%0d",
                               label, row, col, observed, golden_raw[row][col]);
                    end

                    if (stream_matrix[row][col] !== golden_raw[row][col]) begin
                        $fatal(1,
                               "[TB] %s mismatch (stream) at C[%0d][%0d]: observed=%0d expected=%0d",
                               label, row, col, stream_matrix[row][col], golden_raw[row][col]);
                    end

                    relu_observed = relu_c_out_flat[((row*ARRAY_SIZE)+col)*ACC_WIDTH +: ACC_WIDTH];
                    if (relu_observed !== golden_relu[row][col]) begin
                        $fatal(1,
                               "[TB] %s mismatch (ReLU) at C[%0d][%0d]: observed=%0d expected=%0d",
                               label, row, col, relu_observed, golden_relu[row][col]);
                    end
                end
            end

            $display("[TB] %s passed", label);
        end
    endtask

    initial begin
        apply_reset();

        // Test 1: Identity * Random (checks raw math + ReLU no-op on positives)
        matrix_a[0][0] = 8'sd1; matrix_a[0][1] = 8'sd0; matrix_a[0][2] = 8'sd0; matrix_a[0][3] = 8'sd0;
        matrix_a[1][0] = 8'sd0; matrix_a[1][1] = 8'sd1; matrix_a[1][2] = 8'sd0; matrix_a[1][3] = 8'sd0;
        matrix_a[2][0] = 8'sd0; matrix_a[2][1] = 8'sd0; matrix_a[2][2] = 8'sd1; matrix_a[2][3] = 8'sd0;
        matrix_a[3][0] = 8'sd0; matrix_a[3][1] = 8'sd0; matrix_a[3][2] = 8'sd0; matrix_a[3][3] = 8'sd1;

        matrix_b[0][0] = 8'sd4;  matrix_b[0][1] = -8'sd3; matrix_b[0][2] = 8'sd2;  matrix_b[0][3] = 8'sd1;
        matrix_b[1][0] = 8'sd0;  matrix_b[1][1] = 8'sd5;  matrix_b[1][2] = -8'sd1; matrix_b[1][3] = 8'sd7;
        matrix_b[2][0] = 8'sd1;  matrix_b[2][1] = 8'sd2;  matrix_b[2][2] = 8'sd3;  matrix_b[2][3] = 8'sd4;
        matrix_b[3][0] = -8'sd2; matrix_b[3][1] = 8'sd0;  matrix_b[3][2] = 8'sd1;  matrix_b[3][3] = 8'sd2;
        check_results("identity_passthrough");

        // Test 2: Random small values including negatives to exercise ReLU
        matrix_a[0][0] = 8'sd2;  matrix_a[0][1] = 8'sd1;  matrix_a[0][2] = -8'sd3; matrix_a[0][3] = 8'sd4;
        matrix_a[1][0] = 8'sd0;  matrix_a[1][1] = -8'sd1; matrix_a[1][2] = 8'sd2;  matrix_a[1][3] = 8'sd3;
        matrix_a[2][0] = 8'sd1;  matrix_a[2][1] = 8'sd0;  matrix_a[2][2] = 8'sd1;  matrix_a[2][3] = 8'sd0;
        matrix_a[3][0] = -8'sd1; matrix_a[3][1] = 8'sd2;  matrix_a[3][2] = -8'sd2; matrix_a[3][3] = 8'sd1;

        matrix_b[0][0] = 8'sd1;  matrix_b[0][1] = 8'sd2;  matrix_b[0][2] = 8'sd3;  matrix_b[0][3] = 8'sd4;
        matrix_b[1][0] = 8'sd0;  matrix_b[1][1] = -8'sd1; matrix_b[1][2] = 8'sd2;  matrix_b[1][3] = 8'sd0;
        matrix_b[2][0] = -8'sd2; matrix_b[2][1] = 8'sd1;  matrix_b[2][2] = -8'sd3; matrix_b[2][3] = 8'sd2;
        matrix_b[3][0] = 8'sd3;  matrix_b[3][1] = 8'sd2;  matrix_b[3][2] = 8'sd1;  matrix_b[3][3] = -8'sd1;
        check_results("random_mix");

        // Test 3: All-zero matrices ensure accumulators/reset work
        matrix_a[0][0] = 8'sd0; matrix_a[0][1] = 8'sd0; matrix_a[0][2] = 8'sd0; matrix_a[0][3] = 8'sd0;
        matrix_a[1][0] = 8'sd0; matrix_a[1][1] = 8'sd0; matrix_a[1][2] = 8'sd0; matrix_a[1][3] = 8'sd0;
        matrix_a[2][0] = 8'sd0; matrix_a[2][1] = 8'sd0; matrix_a[2][2] = 8'sd0; matrix_a[2][3] = 8'sd0;
        matrix_a[3][0] = 8'sd0; matrix_a[3][1] = 8'sd0; matrix_a[3][2] = 8'sd0; matrix_a[3][3] = 8'sd0;

        matrix_b[0][0] = 8'sd0; matrix_b[0][1] = 8'sd0; matrix_b[0][2] = 8'sd0; matrix_b[0][3] = 8'sd0;
        matrix_b[1][0] = 8'sd0; matrix_b[1][1] = 8'sd0; matrix_b[1][2] = 8'sd0; matrix_b[1][3] = 8'sd0;
        matrix_b[2][0] = 8'sd0; matrix_b[2][1] = 8'sd0; matrix_b[2][2] = 8'sd0; matrix_b[2][3] = 8'sd0;
        matrix_b[3][0] = 8'sd0; matrix_b[3][1] = 8'sd0; matrix_b[3][2] = 8'sd0; matrix_b[3][3] = 8'sd0;
        check_results("zero_case");

        $display("[TB] All testcases passed");
        $finish;
    end

endmodule

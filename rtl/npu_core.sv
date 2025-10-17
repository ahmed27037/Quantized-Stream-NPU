// Pipelined outer-product NPU core with configurable activation and streaming output
module npu_core #(
    parameter integer ARRAY_SIZE      = 4,
    parameter integer DATA_WIDTH      = 8,
    parameter integer EXTRA_ACC_BITS  = 2,
    parameter integer ACC_WIDTH       = (2*DATA_WIDTH) + $clog2(ARRAY_SIZE) + EXTRA_ACC_BITS,
    parameter integer ACT_FUNC        = 0, // 0 = identity, 1 = ReLU
    parameter integer OUTPUT_COUNT    = ARRAY_SIZE * ARRAY_SIZE,
    parameter integer INDEX_WIDTH     = (OUTPUT_COUNT > 1) ? $clog2(OUTPUT_COUNT) : 1
) (
    input                               clk,
    input                               rst,
    input                               start,
    input                               in_valid,
    input      [ARRAY_SIZE*DATA_WIDTH-1:0] a_stream,
    input      [ARRAY_SIZE*DATA_WIDTH-1:0] b_stream,
    output                              busy,
    output reg                          done,
    output reg                          c_valid,
    output     [OUTPUT_COUNT*ACC_WIDTH-1:0] c_out_flat,
    output reg                          result_valid,
    output reg  [ACC_WIDTH-1:0]         result_data,
    output reg  [INDEX_WIDTH-1:0]       result_index,
    input                               result_ready
);

    localparam integer COUNT_WIDTH   = (ARRAY_SIZE > 1) ? $clog2(ARRAY_SIZE + 1) : 1;
    localparam integer ROWCOL_WIDTH  = (ARRAY_SIZE > 1) ? $clog2(ARRAY_SIZE)     : 1;
    localparam integer MIN_ACC_WIDTH = (2*DATA_WIDTH) + $clog2(ARRAY_SIZE);

    initial begin
        if (ACC_WIDTH < MIN_ACC_WIDTH) begin
            $error("ACC_WIDTH (%0d) is insufficient. Minimum required is %0d for ARRAY_SIZE=%0d DATA_WIDTH=%0d",
                   ACC_WIDTH, MIN_ACC_WIDTH, ARRAY_SIZE, DATA_WIDTH);
        end
    end

    reg active;
    reg streaming;
    reg [COUNT_WIDTH-1:0] feed_count;
    reg [COUNT_WIDTH-1:0] processed_count;

    reg signed [DATA_WIDTH-1:0] a_stage0 [0:ARRAY_SIZE-1];
    reg signed [DATA_WIDTH-1:0] b_stage0 [0:ARRAY_SIZE-1];
    reg signed [DATA_WIDTH-1:0] a_stage1 [0:ARRAY_SIZE-1];
    reg signed [DATA_WIDTH-1:0] b_stage1 [0:ARRAY_SIZE-1];

    reg valid_stage0;
    reg valid_stage1;

    wire clear_acc = start;

    wire signed [ACC_WIDTH-1:0] acc_matrix [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1];
    wire signed [ACC_WIDTH-1:0] act_matrix [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1];

    assign busy = active | streaming;

    genvar row;
    genvar col;
    generate
        for (row = 0; row < ARRAY_SIZE; row = row + 1) begin : gen_rows
            for (col = 0; col < ARRAY_SIZE; col = col + 1) begin : gen_cols
                pe #(
                    .DATA_WIDTH(DATA_WIDTH),
                    .ACC_WIDTH (ACC_WIDTH)
                ) u_pe (
                    .clk     (clk),
                    .rst     (rst),
                    .clear   (clear_acc),
                    .enable  (valid_stage1),
                    .a_value (a_stage1[row]),
                    .b_value (b_stage1[col]),
                    .acc_out (acc_matrix[row][col])
                );

                if (ACT_FUNC == 1) begin : gen_relu
                    assign act_matrix[row][col] = acc_matrix[row][col][ACC_WIDTH-1] ? {ACC_WIDTH{1'b0}} : acc_matrix[row][col];
                end else begin : gen_identity
                    assign act_matrix[row][col] = acc_matrix[row][col];
                end
            end
        end
    endgenerate

    integer i_row;

    // Pipeline for operand broadcast and accumulation
    always @(posedge clk) begin
        if (rst) begin
            active          <= 1'b0;
            feed_count      <= {COUNT_WIDTH{1'b0}};
            processed_count <= {COUNT_WIDTH{1'b0}};
            valid_stage0    <= 1'b0;
            valid_stage1    <= 1'b0;
            done            <= 1'b0;
            c_valid         <= 1'b0;
            for (i_row = 0; i_row < ARRAY_SIZE; i_row = i_row + 1) begin
                a_stage0[i_row] <= {DATA_WIDTH{1'b0}};
                b_stage0[i_row] <= {DATA_WIDTH{1'b0}};
                a_stage1[i_row] <= {DATA_WIDTH{1'b0}};
                b_stage1[i_row] <= {DATA_WIDTH{1'b0}};
            end
        end else begin
            done    <= 1'b0;
            c_valid <= 1'b0;

            if (start && !active && !streaming) begin
                active          <= 1'b1;
                feed_count      <= {COUNT_WIDTH{1'b0}};
                processed_count <= {COUNT_WIDTH{1'b0}};
                valid_stage0    <= 1'b0;
                valid_stage1    <= 1'b0;
            end

            if (active) begin
                if (in_valid && (feed_count < ARRAY_SIZE)) begin
                    feed_count <= feed_count + {{(COUNT_WIDTH-1){1'b0}}, 1'b1};
                    for (i_row = 0; i_row < ARRAY_SIZE; i_row = i_row + 1) begin
                        a_stage0[i_row] <= $signed(a_stream[(i_row*DATA_WIDTH) +: DATA_WIDTH]);
                        b_stage0[i_row] <= $signed(b_stream[(i_row*DATA_WIDTH) +: DATA_WIDTH]);
                    end
                    valid_stage0 <= 1'b1;
                end else begin
                    valid_stage0 <= 1'b0;
                end

                for (i_row = 0; i_row < ARRAY_SIZE; i_row = i_row + 1) begin
                    a_stage1[i_row] <= a_stage0[i_row];
                    b_stage1[i_row] <= b_stage0[i_row];
                end
                valid_stage1 <= valid_stage0;

                if (valid_stage1) begin
                    processed_count <= processed_count + {{(COUNT_WIDTH-1){1'b0}}, 1'b1};
                    if (processed_count == ARRAY_SIZE-1) begin
                        done        <= 1'b1;
                        c_valid     <= 1'b1;
                        active      <= 1'b0;
                    end
                end
            end else begin
                valid_stage0 <= 1'b0;
                valid_stage1 <= 1'b0;
            end
        end
    end

    // Streaming state
    reg [ROWCOL_WIDTH-1:0] stream_row;
    reg [ROWCOL_WIDTH-1:0] stream_col;
    reg [INDEX_WIDTH-1:0]  stream_index;
    reg                    final_word_pending;

    // Launch streaming whenever computation finishes
    always @(posedge clk) begin
        if (rst) begin
            streaming          <= 1'b0;
            result_valid       <= 1'b0;
            result_data        <= {ACC_WIDTH{1'b0}};
            result_index       <= {INDEX_WIDTH{1'b0}};
            stream_row         <= {ROWCOL_WIDTH{1'b0}};
            stream_col         <= {ROWCOL_WIDTH{1'b0}};
            stream_index       <= {INDEX_WIDTH{1'b0}};
            final_word_pending <= 1'b0;
        end else begin
            if (done) begin
                streaming          <= 1'b1;
                stream_row         <= {ROWCOL_WIDTH{1'b0}};
                stream_col         <= {ROWCOL_WIDTH{1'b0}};
                stream_index       <= {INDEX_WIDTH{1'b0}};
                final_word_pending <= 1'b0;
                result_valid       <= 1'b0;
            end

            if (streaming) begin
                if (!result_valid || (result_valid && result_ready)) begin
                    result_valid <= 1'b1;
                    result_data  <= act_matrix[stream_row][stream_col];
                    result_index <= stream_index;

                    if (stream_index == OUTPUT_COUNT-1) begin
                        final_word_pending <= 1'b1;
                    end else begin
                        final_word_pending <= 1'b0;
                        stream_index <= stream_index + {{(INDEX_WIDTH-1){1'b0}}, 1'b1};
                        if (stream_col == ARRAY_SIZE-1) begin
                            stream_col <= {ROWCOL_WIDTH{1'b0}};
                            stream_row <= stream_row + {{(ROWCOL_WIDTH-1){1'b0}}, 1'b1};
                        end else begin
                            stream_col <= stream_col + {{(ROWCOL_WIDTH-1){1'b0}}, 1'b1};
                        end
                    end
                end

                if (final_word_pending && result_valid && result_ready) begin
                    streaming          <= 1'b0;
                    result_valid       <= 1'b0;
                    final_word_pending <= 1'b0;
                end
            end else begin
                if (result_valid && result_ready) begin
                    result_valid <= 1'b0;
                end
                final_word_pending <= 1'b0;
            end
        end
    end

    // Flatten activated matrix into packed bus
    generate
        for (row = 0; row < ARRAY_SIZE; row = row + 1) begin : gen_flatten_rows
            for (col = 0; col < ARRAY_SIZE; col = col + 1) begin : gen_flatten_cols
                localparam integer idx = (row*ARRAY_SIZE) + col;
                assign c_out_flat[(idx+1)*ACC_WIDTH-1 : idx*ACC_WIDTH] = act_matrix[row][col];
            end
        end
    endgenerate

endmodule

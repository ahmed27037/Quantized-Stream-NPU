// Processing element for the pipelined NPU outer-product stage
// Performs an enable-gated multiply-accumulate on broadcast operands
module pe #(
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH  = 24
) (
    input                       clk,
    input                       rst,
    input                       clear,
    input                       enable,
    input      signed [DATA_WIDTH-1:0] a_value,
    input      signed [DATA_WIDTH-1:0] b_value,
    output reg signed [ACC_WIDTH-1:0]  acc_out
);

    wire signed [(2*DATA_WIDTH)-1:0] product;
    wire signed [ACC_WIDTH-1:0]      product_ext;

    assign product     = a_value * b_value;
    assign product_ext = {{(ACC_WIDTH-(2*DATA_WIDTH)){product[(2*DATA_WIDTH)-1]}}, product};

    always @(posedge clk) begin
        if (rst) begin
            acc_out <= 0;
        end else if (clear) begin
            acc_out <= 0;
        end else if (enable) begin
            acc_out <= acc_out + product_ext;
        end
    end

endmodule

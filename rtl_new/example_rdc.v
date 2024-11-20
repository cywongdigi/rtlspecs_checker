module example_rdc(
    input wire clk_a,
    input wire rst_a_n,
    input wire clk_b,
    input wire rst_b_n,
    input wire data_in,
    output reg data_out
);

    reg data_sync_a;
    reg data_sync_b;

    // First clock domain: clk_a and rst_a_n
    always @(posedge clk_a or negedge rst_a_n) begin
        if (!rst_a_n)
            data_sync_a <= 0;
        else
            data_sync_a <= data_in;
    end

    // Clock domain crossing from clk_a to clk_b
    reg data_cross;
    always @(posedge clk_b or negedge rst_b_n) begin
        if (!rst_b_n)
            data_cross <= 0;
        else
            data_cross <= data_sync_a;
    end

    // Second clock domain: clk_b and rst_b_n
    always @(posedge clk_b or negedge rst_b_n) begin
        if (!rst_b_n)
            data_sync_b <= 0;
        else
            data_sync_b <= data_cross;
    end

    // Output data in clk_b domain
    always @(posedge clk_b or negedge rst_b_n) begin
        if (!rst_b_n)
            data_out <= 0;
        else
            data_out <= data_sync_b;
    end

endmodule

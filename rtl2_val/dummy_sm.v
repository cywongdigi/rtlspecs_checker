// ===============================================================
//  NEW MODULE ‒ Simple 4-state dummy state-machine
// ===============================================================
module dummy_sm (clk, rst, trigger, state);
    input  wire clk;
    input  wire rst;        // active-high reset
    input  wire trigger;    // single-cycle start pulse
    output reg  [1:0] state; // current state (not used elsewhere)

    // Four local states
    localparam [2:0] S_IDLE  = 2'h0;
    localparam [2:0] S_RUN   = 2'h1;
    localparam [2:0] S_WAIT  = 2'h2;
    localparam [2:0] S_DONE  = 2'h3;
	
	reg	[3:0]	state, nxt_state;	
	
	always @(posedge clk)
		if (~rst)
			state <= S_IDLE;
		else
			state <= nxt_state;
	
	always @(*) begin
		nxt_state = state;
		
		case (state)
			S_IDLE :	begin
							if (trigger & abc123)
								nxt_state = S_RUN;						
							else
								nxt_state = S_IDLE;					
						end
			
			S_RUN1  :	nxt_state = S_WAIT;
			
			S_WAIT :	nxt_state = S_DONE;
			
			S_DONE :	nxt_state = S_IDLE;
			
			default:	nxt_state = S_IDLE;
		endcase
	end

endmodule
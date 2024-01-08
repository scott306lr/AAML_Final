/*
func_7: 
  0: Execute TPU (input0: (K, M, N), input1: input_offset)
  1: Write A (input0: data, input1: index)
  2: Write B (input0: data, input1: index)
  3: Read C (input0: which_value_in_a_row, input1: index)
  4: Stall until TPU is idle (input0: 0, input1: 0)
*/

`define BUFFER_A_DEPTH  10 
`define BUFFER_B_DEPTH  10
`define BUFFER_C_DEPTH  10

module Cfu (
    input               cmd_valid,
    output              cmd_ready,
    input      [9:0]    cmd_payload_function_id,
    input      [31:0]   cmd_payload_inputs_0,
    input      [31:0]   cmd_payload_inputs_1,
    output              rsp_valid,
    input               rsp_ready,
    output     [31:0]   rsp_payload_outputs_0,
    input               reset,
    input               clk
  );
  
  // Parameter Definitions
  parameter IDLE = 0, CAL_PREPARE = 1, CAL = 2, DONE = 3, SET_Z = 4, ADD = 5, SET_OS = 6;
  reg [2:0] state, next_state, simd_state;

  // Control signals
  wire rst_n = ~reset;
  wire [6:0]  funct_7 = cmd_payload_function_id[9:3];
  wire [2:0]  funct_3 = cmd_payload_function_id[2:0];
  
  // TPU & Global buffer signals
  wire TPU_busy;
  wire TPU_in_valid;
  wire TPU_A_wr_en;
  wire TPU_B_wr_en;
  wire [31:0] TPU_A_data_in;
  wire [31:0] TPU_B_data_in;
  wire [127:0] TPU_C_data_out;
  
  wire [15:0]  A_index;
  wire [31:0]  A_data_in;
  wire [31:0]  A_data_out;
  wire [15:0]  TPU_A_index;
  wire [31:0]  TPU_A_data_out;
  
  wire [15:0]  B_index;
  wire [31:0]  B_data_in;
  wire [31:0]  B_data_out;
  wire [15:0]  TPU_B_index;
  wire [31:0]  TPU_B_data_out;
  
  wire TPU_C_wr_en;
  wire [15:0]  C_index;
  wire [127:0] C_data_in;
  wire [127:0] C_data_out;
  wire [15:0]  TPU_C_index;
  wire [127:0] TPU_C_data_in;
    
  //used on CALC_PREPARE
  wire [7:0] K, M, N;
  wire [31:0] input_offset;

  // SIMD
  reg signed [8:0] simd_input_offset;
  reg [31:0]   rsp_payload_outputs_simd;
  // reg [6:0] reg_func7;

  TPU My_TPU(
        .clk            (clk),
        .rst_n          (rst_n),
        .in_valid       (TPU_in_valid),
        .K              (K),
        .M              (M),
        .N              (N),
        .busy           (TPU_busy),
        .A_index        (TPU_A_index),
        .A_data_out     (TPU_A_data_out),
        .B_index        (TPU_B_index),
        .B_data_out     (TPU_B_data_out),
        .C_wr_en        (TPU_C_wr_en),
        .C_index        (TPU_C_index),
        .C_data_in      (TPU_C_data_in),
        .input_offset   (input_offset),
        .A_wr_en        (TPU_A_wr_en),
        .B_wr_en        (TPU_B_wr_en),
        .A_data_in      (TPU_A_data_in),
        .B_data_in      (TPU_B_data_in),
        .C_data_out     (TPU_C_data_out)
      );

  global_buffer #(
                  .ADDR_BITS(`BUFFER_A_DEPTH),
                  .DATA_BITS(32)
                )
                gbuff_A(
                  .clk(clk),
                  .rst_n(rst_n),
                  .wr_en(A_wr_en),
                  .index(A_index),
                  .data_in(A_data_in),
                  .data_out(A_data_out)
                );

  global_buffer #(
                  .ADDR_BITS(`BUFFER_B_DEPTH),
                  .DATA_BITS(32)
                ) gbuff_B(
                  .clk(clk),
                  .rst_n(rst_n),
                  .wr_en(B_wr_en),
                  .index(B_index),
                  .data_in(B_data_in),
                  .data_out(B_data_out)
                );

  global_buffer #(
                  .ADDR_BITS(`BUFFER_C_DEPTH),
                  .DATA_BITS(128)
                ) gbuff_C(
                  .clk(clk),
                  .rst_n(rst_n),
                  .wr_en(C_wr_en),
                  .index(C_index),
                  .data_in(C_data_in),
                  .data_out(C_data_out)
                );

  // FSM
  always @(posedge clk, negedge rst_n)
  begin
    if (!rst_n) begin
      state <= IDLE;
    end
    else begin
      state <= next_state;
    end
  end

  always @(*)
  begin
    case (state)
      IDLE: begin
        next_state = (!cmd_valid || (cmd_valid && (funct_7 == 5 || funct_7 == 6 || funct_7 == 7))) ? IDLE : 
                     (funct_7 == 0) ? CAL_PREPARE : DONE;
                    //  (funct_7 == 5) ? SET_Z : 
                    //  (funct_7 == 6) ? ADD : 
                    //  (funct_7 == 7) ? SET_OS : DONE;
        // simd_state = (!cmd_valid) ? IDLE :                      
        //              (funct_7 == 5) ? SET_Z : 
        //              (funct_7 == 6) ? ADD : 
        //              (funct_7 == 7) ? SET_OS : IDLE;
      end
      CAL_PREPARE :
        next_state = CAL;
      CAL :
        next_state = (TPU_busy) ? CAL : DONE;
      // SET_Z :
      //   next_state = DONE;
      // ADD :
      //   next_state = DONE;
      // SET_OS : 
      //   next_state = DONE;
      DONE :
        next_state = IDLE;
    endcase
  end


  // store input signal
  reg  [31:0] cmd_payload_inputs_0_ff;
  reg  [31:0] cmd_payload_inputs_1_ff;
  wire [31:0] cmd_payload_inputs_0_comb;
  wire [31:0] cmd_payload_inputs_1_comb;
  
  wire [31:0] rsp_payload_outputs_simd_comb;

  always @(posedge clk, negedge rst_n)
  begin
    if (!rst_n)
    begin
      cmd_payload_inputs_0_ff <= 0;
      cmd_payload_inputs_1_ff <= 0;
      // simd_input_offset <= 9'd128;
      rsp_payload_outputs_simd <= 32'b0;
    end
    else
    begin
      cmd_payload_inputs_0_ff <= cmd_payload_inputs_0_comb;
      cmd_payload_inputs_1_ff <= cmd_payload_inputs_1_comb;

      if (cmd_valid && funct_7 == 7) begin
        simd_input_offset <= cmd_payload_inputs_0[8:0];
      end
      if (cmd_valid && funct_7 == 5 || funct_7 == 6) begin
        rsp_payload_outputs_simd <= rsp_payload_outputs_simd_comb;
      end
      // else if (state == SET_Z) begin
      //   rsp_payload_outputs_simd <= 32'b0;
      // end 
      // else if (state == ADD) begin
      //   rsp_payload_outputs_simd <= rsp_payload_outputs_simd + sum_prods;
      // end 
    end
  end
  assign cmd_payload_inputs_0_comb = (cmd_valid) ? cmd_payload_inputs_0: cmd_payload_inputs_0_ff;
  assign cmd_payload_inputs_1_comb = (cmd_valid) ? cmd_payload_inputs_1: cmd_payload_inputs_1_ff;

  assign rsp_payload_outputs_simd_comb = (cmd_valid && funct_7 == 6) ? rsp_payload_outputs_simd + sum_prods : 
                                         (cmd_valid && funct_7 == 5) ? 32'b0 : rsp_payload_outputs_simd;
  


  // TPU signals
  assign TPU_A_data_out = A_data_out;
  assign TPU_B_data_out = B_data_out;

  // Control signals
  assign A_wr_en = (cmd_valid && funct_7 == 1) ? 1: 0;//(state == WA);
  assign B_wr_en = (cmd_valid && funct_7 == 2) ? 1: 0;//(state == WB);
  assign C_wr_en = TPU_C_wr_en;

  assign A_index = (state == CAL) ? TPU_A_index : cmd_payload_inputs_1_comb[15:0];
  assign B_index = (state == CAL) ? TPU_B_index : cmd_payload_inputs_1_comb[15:0];
  assign C_index = (state == CAL) ? TPU_C_index : cmd_payload_inputs_1_comb[15:0];
  
  assign A_data_in = cmd_payload_inputs_0_ff;
  assign B_data_in = cmd_payload_inputs_0_ff;
  assign C_data_in = TPU_C_data_in;

  //used on CALC_PREPARE
  assign K = cmd_payload_inputs_0_ff[23:16];
  assign M = cmd_payload_inputs_0_ff[15:8];
  assign N = cmd_payload_inputs_0_ff[7:0];
  assign input_offset = cmd_payload_inputs_1_ff[31:0];

  // used on READ
  wire [1:0]  which_value_in_a_row;
  assign which_value_in_a_row = cmd_payload_inputs_0_comb[1:0];

  // Trivial handshaking for a combinational CFU
  assign rsp_valid = (state == DONE || state == CAL_PREPARE || (state==IDLE && cmd_valid==1 && (funct_7 == 5 || funct_7 == 6 || funct_7 == 7))) ? 1: 0;
  assign cmd_ready = rsp_valid;
  assign TPU_in_valid = (state == CAL_PREPARE) ? 1: 0;

  // SIMD multiply step:
  wire signed [15:0] prod_0, prod_1, prod_2, prod_3;
  assign prod_0 =  ($signed(cmd_payload_inputs_0[7 : 0]) + simd_input_offset)
         * $signed(cmd_payload_inputs_1[7 : 0]);
  assign prod_1 =  ($signed(cmd_payload_inputs_0[15: 8]) + simd_input_offset)
         * $signed(cmd_payload_inputs_1[15: 8]);
  assign prod_2 =  ($signed(cmd_payload_inputs_0[23:16]) + simd_input_offset)
         * $signed(cmd_payload_inputs_1[23:16]);
  assign prod_3 =  ($signed(cmd_payload_inputs_0[31:24]) + simd_input_offset)
         * $signed(cmd_payload_inputs_1[31:24]);

  wire signed [31:0] sum_prods;
  assign sum_prods = prod_0 + prod_1 + prod_2 + prod_3;
  
  // assign rsp_payload_outputs_0 with 32 bits of C_data_out according to which_value_in_a_row
  assign rsp_payload_outputs_0 = (cmd_valid && funct_7 == 5) ? 32'b0 :
                                 (cmd_valid && funct_7 == 6) ? rsp_payload_outputs_simd_comb : //rsp_payload_outputs_simd_comb
                                 (which_value_in_a_row == 0) ? C_data_out[127:96] :
                                 (which_value_in_a_row == 1) ? C_data_out[95:64]  :
                                 (which_value_in_a_row == 2) ? C_data_out[63:32]  : C_data_out[31:0];

endmodule

module TPU(
    input clk,
    input rst_n,
    input            in_valid,
    input [7:0]      K,
    input [7:0]      M,
    input [7:0]      N,
    output     busy,

    output           A_wr_en,
    output [15:0]    A_index,
    output [31:0]    A_data_in,
    input  [31:0]    A_data_out,

    output           B_wr_en,
    output [15:0]    B_index,
    output [31:0]    B_data_in,
    input  [31:0]    B_data_out,

    output           C_wr_en,
    output [15:0]    C_index,
    output [127:0]   C_data_in,
    input  [127:0]   C_data_out,
    input  [7:0]     input_offset
  );
  // FSM
  parameter IDLE = 0, PASS_DATA = 1, WAIT_6 = 2, WAIT_FOR_WRITE = 3;
  reg [1:0] state, next_state;

  // Counters
  reg [7:0] n;
  reg [7:0] m;
  reg [8:0] counter;
  wire [7:0] N_tiles = reg_N[7:2] + (reg_N[1:0] != 0);
  wire [7:0] M_tiles = reg_M[7:2] + (reg_M[1:0] != 0);

  // Communicate logic
  assign busy = (state != IDLE) || in_valid;


  // Controller
  wire [15:0] write_start_index;
  assign A_index = A_index_reg;
  assign B_index = B_index_reg;
  assign write_start_index = C_index_reg;

  // wire on_count_k = (counter >= reg_K-1);
  // wire on_count_5 = (counter == 5); // 1 cycle before next state
  // wire on_count_6 = (counter == 6); // counter == 3-1
  wire all_done = (n == N_tiles);
  wire SA_done = (state == WAIT_6) && (counter == 6); //(next_state != WAIT_6)
  wire prev_next_tile = (state != PASS_DATA) && (counter == 5);
  wire next_tile = (state != PASS_DATA) && (counter == 6);
  wire m_reset = (m == M_tiles-1);

  // Save K, M, N to registers
  reg [7:0] reg_K, reg_M, reg_N;
  reg [31:0] reg_input_offset;
  always @(posedge clk)
  begin
    if  (!rst_n)
    begin
      reg_K <= 8'd0;
      reg_M <= 8'd0;
      reg_N <= 8'd0;
      reg_input_offset <= 32'd0;
    end
    else if (in_valid)
    begin
      reg_K <= K;
      reg_M <= M;
      reg_N <= N;
      reg_input_offset <= input_offset;
    end
  end

  reg [15:0] A_index_reg, B_index_reg, C_index_reg;
  wire [15:0] A_index_comb, B_index_comb, C_index_comb;
  wire [7:0] n_comb, m_comb;
  // wire calc_valid;
  always @(posedge clk)
  begin
    if (!rst_n)
    begin
      n <= 8'd0;
      m <= 8'd0;
      A_index_reg <= 0;
      B_index_reg <= 0;
      C_index_reg <= 0;
    end
    else
    begin
      n = n_comb;
      m = m_comb;
      A_index_reg <= A_index_comb;
      B_index_reg <= B_index_comb;
      C_index_reg <= C_index_comb;
    end
  end
  assign n_comb = (prev_next_tile && m_reset) ? n + 8'd1 : (state == IDLE) ? 8'd0 : n;
  assign m_comb = (prev_next_tile) ? ((m_reset) ? 8'd0 : m + 8'd1) : (state == IDLE) ? 8'd0 : m;
  assign A_index_comb = (next_tile) ? m*reg_K : (state == PASS_DATA && !(counter >= reg_K-1)) ? A_index_reg + 16'd1 : (state == IDLE) ? 16'd0 : A_index_reg;
  assign B_index_comb = (next_tile) ? n*reg_K : (state == PASS_DATA && !(counter >= reg_K-1)) ? B_index_reg + 16'd1 : (state == IDLE) ? 16'd0 : B_index_reg;
  assign C_index_comb = (next_tile) ? n*reg_M + m*16'd4 : (state == IDLE) ? 16'd0 : C_index_reg;

  always @(posedge clk)
  begin
    if (!rst_n)
      counter <= 9'd0;
    else
      counter <= (state == IDLE || state != next_state) ? 9'd0 : counter + 9'd1;
  end

  //systolic array
  wire [127:0] PE_data[3:0];
  wire send_data = (state == PASS_DATA);

  systolic_array SA (
                   .clk(clk),
                   .rst_n(rst_n),
                   .read(send_data),
                   .input_offset(reg_input_offset),
                   .A_data(A_data_out),
                   .B_data(B_data_out),
                   .PE_data0(PE_data[0]),
                   .PE_data1(PE_data[1]),
                   .PE_data2(PE_data[2]),
                   .PE_data3(PE_data[3])
                 );

  //output buffer to C
  wire write_busy;
  wire start_write = (state == WAIT_FOR_WRITE);
  write_buffer buffer (
                 .clk(clk),
                 .rst_n(rst_n),
                 .in_valid(SA_done),
                 .busy(write_busy),
                 .PE_data0(PE_data[0]),
                 .PE_data1(PE_data[1]),
                 .PE_data2(PE_data[2]),
                 .PE_data3(PE_data[3]),
                 .write_start_index(write_start_index),
                 .wr_en(C_wr_en),
                 .write_index(C_index),
                 .write_data(C_data_in)
               );


  // state machine
  always @(posedge clk or negedge rst_n)
  begin
    if (!rst_n)
      state <= IDLE;
    else
      state <= next_state;
  end

  always @(*)
  begin
    case(state)
      IDLE:
        next_state = (in_valid) ? PASS_DATA : IDLE;
      PASS_DATA:
        next_state = ((counter >= reg_K-1)) ? WAIT_6 : PASS_DATA;
      WAIT_6:
        next_state = ((counter == 6)) ? ((all_done) ? WAIT_FOR_WRITE : PASS_DATA) : WAIT_6;
      WAIT_FOR_WRITE:
        next_state = (write_busy) ? WAIT_FOR_WRITE : IDLE;
      default:
        next_state = IDLE;
    endcase
  end

  // SRAM interface, not used in this design, set to 0.
  // Not writing any data to A/B SRAM, only reading
  assign A_wr_en = 1'b0;
  assign A_data_in = 32'd0;
  assign B_wr_en = 1'b0;
  assign B_data_in = 32'd0;
endmodule

module systolic_array(
    input clk,
    input rst_n,
    input          read,
    input [31:0]   input_offset,

    input  [31:0]  A_data,
    input  [31:0]  B_data,

    output [127:0] PE_data0,
    output [127:0] PE_data1,
    output [127:0] PE_data2,
    output [127:0] PE_data3
  );

  // Diagonal buffers
  wire [31:0] A_diag_out;
  wire [31:0] B_diag_out;
  wire [3:0] B_offset_mask_diag_out;

  // PE signals
  wire [7:0] PE_wire_X[4:0][4:0];
  wire [7:0] PE_wire_Y[4:0][4:0];
  wire PE_wire_Y_mask[4:0][4:0];
  wire [127:0] PE_data[3:0];
  reg read_ff, PE_clear;

  assign PE_data0 = PE_data[0];
  assign PE_data1 = PE_data[1];
  assign PE_data2 = PE_data[2];
  assign PE_data3 = PE_data[3];

  // PE_clear set to 1 on the first clock when read is 1, and set to 0 on the next clock
  always @(posedge clk)
  begin
    if (!rst_n)
      read_ff <= 1'b0;
    else
      read_ff <= read;
  end

  always @(posedge clk)
  begin
    if (!rst_n)
      PE_clear <= 1'b0;
    else if (read && !read_ff)
      PE_clear <= 1'b1;
    else if (PE_clear)
      PE_clear <= 1'b0;
  end

  diag_buffer A_buffer (
                .clk(clk),
                .rst_n(rst_n),
                .read(read),
                .in_data(A_data),
                .out_data(A_diag_out)
              );
  diag_buffer B_buffer (
                .clk(clk),
                .rst_n(rst_n),
                .read(read),
                .in_data(B_data),
                .out_data(B_diag_out)
              );
  mask_buffer B_offset_mask_buffer (
                .clk(clk),
                .rst_n(rst_n),
                .read(read),
                .out_data(B_offset_mask_diag_out)
              );

  generate
    genvar i, j;
    for (i = 0; i < 4; i = i + 1)
    begin
      for (j = 0; j < 4; j = j + 1)
      begin
        PE PE_inst(
             .clk(clk),
             .rst_n(rst_n),
             .clear(PE_clear),
             .input_offset(input_offset),
             .offset_mask_from_top(PE_wire_Y_mask[i][j]),
             .offset_mask_to_bottom(PE_wire_Y_mask[i+1][j]),
             .data_from_top(PE_wire_Y[i][j]),
             .data_to_bottom(PE_wire_Y[i+1][j]),
             .data_from_left(PE_wire_X[i][j]),
             .data_to_right(PE_wire_X[i][j+1]),
             .accum_out(PE_data[i][(3-j)*32 +: 32])
           );
      end
      assign PE_wire_X[i][0] = A_diag_out[8*(3-i) +: 8];
      assign PE_wire_Y[0][i] = B_diag_out[8*(3-i) +: 8];
      assign PE_wire_Y_mask[0][i] = B_offset_mask_diag_out[3-i];
    end
  endgenerate


  //for debugging
  wire mask1 = B_offset_mask_diag_out[3];
  wire mask2 = B_offset_mask_diag_out[2];
  wire mask3 = B_offset_mask_diag_out[1];
  wire mask4 = B_offset_mask_diag_out[0];
  wire [7:0] A1 = A_diag_out[31:24];
  wire [7:0] A2 = A_diag_out[23:16];
  wire [7:0] A3 = A_diag_out[15:8];
  wire [7:0] A4 = A_diag_out[7:0];

  wire [7:0] B1 = B_diag_out[31:24];
  wire [7:0] B2 = B_diag_out[23:16];
  wire [7:0] B3 = B_diag_out[15:8];
  wire [7:0] B4 = B_diag_out[7:0];

  wire [31:0] A1B1 = PE_data[0][127:96];
  wire [31:0] A1B2 = PE_data[0][95:64];
  wire [31:0] A1B3 = PE_data[0][63:32];
  wire [31:0] A1B4 = PE_data[0][31:0];
endmodule

module global_buffer #(parameter ADDR_BITS=8, parameter DATA_BITS=8)(clk, rst_n, wr_en, index, data_in, data_out);
  input clk;
  input rst_n;
  input wr_en; // Write enable: 1->write 0->read
  input      [ADDR_BITS-1:0]  index;
  input      [DATA_BITS-1:0]  data_in;
  output reg [DATA_BITS-1:0]  data_out;

  integer i;
  parameter DEPTH = 2**ADDR_BITS;
  (* RAM_STYLE="block"*) reg [DATA_BITS-1:0] gbuff [DEPTH-1:0];
  always @ (negedge clk)
  begin
    if(wr_en)
      gbuff[index] = data_in;
    else
      data_out = gbuff[index];
  end
endmodule

module write_buffer(
    input clk,
    input rst_n,
    input in_valid,
    output busy,

    input [127:0] PE_data0,
    input [127:0] PE_data1,
    input [127:0] PE_data2,
    input [127:0] PE_data3,

    input [15:0] write_start_index,

    output wr_en,
    output reg [15:0] write_index,
    output [127:0] write_data
  );

  // Load data from in_matrix_data, save to reg
  wire [127:0] in_matrix_data[3:0];
  assign in_matrix_data[0] = PE_data0;
  assign in_matrix_data[1] = PE_data1;
  assign in_matrix_data[2] = PE_data2;
  assign in_matrix_data[3] = PE_data3;
  reg [128:0] data_buffer[3:0];

  // FSM
  localparam IDLE = 2'd0,  LOAD_DATA = 2'd1, WRITE = 2'd2;
  reg [1:0] state, next_state;
  reg [2:0] counter;

  // Control signals
  assign busy = (state != IDLE) || in_valid;
  assign wr_en = (state == WRITE);
  assign write_data = data_buffer[counter];

  //load data from in_matrix_data, save to reg
  always @(posedge state)
  begin
    if (!rst_n)
    begin
      data_buffer[0] <= 128'd0;
      data_buffer[1] <= 128'd0;
      data_buffer[2] <= 128'd0;
      data_buffer[3] <= 128'd0;
    end
    else if (state == IDLE)
    begin
      data_buffer[0] <= 128'd0;
      data_buffer[1] <= 128'd0;
      data_buffer[2] <= 128'd0;
      data_buffer[3] <= 128'd0;
    end
    else if (state == LOAD_DATA)
    begin
      data_buffer[0] <= in_matrix_data[0];
      data_buffer[1] <= in_matrix_data[1];
      data_buffer[2] <= in_matrix_data[2];
      data_buffer[3] <= in_matrix_data[3];
    end
  end

  //start counting when state is WRITE
  always @(posedge clk)
  begin
    if (!rst_n)
    begin
      write_index <= 16'd0;
      counter <= 3'd0;
    end
    else
    begin
      write_index = write_index_comb;
      counter = counter_comb;
    end
  end
  wire [15:0] write_index_comb = (in_valid) ? write_start_index : ((counter == 3) || state != WRITE) ? write_index : (write_index + 16'd1);
  wire [2:0] counter_comb = ((counter == 3) || state != WRITE) ? 3'd0 : (counter + 3'd1);

  // state machine
  always @(posedge clk or negedge rst_n)
  begin
    if (!rst_n)
      state <= IDLE;
    else
      state <= next_state;
  end

  always @(*)
  begin
    case(state)
      IDLE:
        next_state <= in_valid ? LOAD_DATA : IDLE;
      LOAD_DATA:
        next_state <= WRITE;
      WRITE:
        next_state <= (counter == 3) ? IDLE : WRITE;
    endcase
  end
endmodule

module diag_buffer(
    input clk,
    input rst_n,
    input read,
    input [31:0] in_data,
    output [31:0] out_data
  );
  reg [7:0] data_buffer0;
  reg [15:0] data_buffer1;
  reg [23:0] data_buffer2;
  reg [31:0] data_buffer3;
  assign out_data = {data_buffer0, data_buffer1[15:8], data_buffer2[23:16], data_buffer3[31:24] };

  always @(posedge clk or negedge rst_n)
  begin
    if (!rst_n)
    begin
      data_buffer0 <= 8'd0;
      data_buffer1 <= 16'd0;
      data_buffer2 <= 24'd0;
      data_buffer3 <= 32'd0;
    end
    else
    begin
      data_buffer0 <= (read ? in_data[31:24] : 8'd0);
      data_buffer1 <= {data_buffer1[7:0], (read ? in_data[23:16] : 8'd0)};
      data_buffer2 <= {data_buffer2[15:0], (read ? in_data[15:8] : 8'd0)};
      data_buffer3 <= {data_buffer3[24:0], (read ? in_data[7:0] : 8'd0)};
    end
  end
endmodule

module mask_buffer(
    input clk,
    input rst_n,
    input read,
    output [3:0] out_data
  );
  reg data_buffer0;
  reg [1:0] data_buffer1;
  reg [2:0] data_buffer2;
  reg [3:0] data_buffer3;
  // assign out_data = {data_buffer3[3], data_buffer2[2], data_buffer1[1], data_buffer0};
  assign out_data = {data_buffer0, data_buffer1[1], data_buffer2[2], data_buffer3[3]};

  always @(posedge clk or negedge rst_n)
  begin
    if (!rst_n)
    begin
      data_buffer0 <= 1'b0;
      data_buffer1 <= 2'b0;
      data_buffer2 <= 3'b0;
      data_buffer3 <= 4'b0;
    end
    else
    begin
      data_buffer0 <= (read ? 1'd1 : 1'd0);
      data_buffer1 <= {data_buffer1[0], (read ? 1'd1 : 1'd0)};
      data_buffer2 <= {data_buffer2[1:0], (read ? 1'd1 : 1'd0)};
      data_buffer3 <= {data_buffer3[2:0], (read ? 1'd1 : 1'd0)};
    end
  end
endmodule

module PE(
    input clk,
    input rst_n,
    input clear,
    input [31:0] input_offset, // only the leftmost PE has non-zero input_offset
    input offset_mask_from_top,
    input [7:0] data_from_top,
    input [7:0] data_from_left,

    output reg offset_mask_to_bottom,
    output reg [7:0] data_to_bottom,
    output reg [7:0] data_to_right,
    output reg signed [31:0] accum_out
  );
  wire signed [31:0] multi;
  wire signed [31:0] accum_out_comb;

  always @(posedge clk)
  begin
    if (!rst_n)
    begin
      accum_out <= 0;
      offset_mask_to_bottom <= 1'b0;
      data_to_right <= 8'd0;
      data_to_bottom <= 8'd0;
    end
    else
    begin
      accum_out <= accum_out_comb;
      offset_mask_to_bottom <= offset_mask_from_top;
      data_to_right <= data_from_left;
      data_to_bottom <= data_from_top;
    end
  end

  assign accum_out_comb = (clear) ? multi : (accum_out + multi);
  assign multi = (offset_mask_from_top) ?
         ($signed(input_offset) + $signed(data_from_top))*$signed(data_from_left)
         : $signed(data_from_top)*$signed(data_from_left);
endmodule

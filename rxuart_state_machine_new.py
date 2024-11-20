import re
from graphviz import Digraph

# Verilog code of the state machine (same as before)
verilog_code = '''
    always @(posedge i_clk) begin
        if (i_reset)
            state <= RXU_RESET_IDLE;
        else
            state <= nxt_state;
    end

    always @(*) begin
        nxt_state = state;

        case (state)
            RXU_RESET_IDLE:     begin
                                    if (line_synch)
                                        nxt_state = RXU_IDLE;                      // Goto idle state from a reset
                                    else
                                        nxt_state = RXU_RESET_IDLE;                // Otherwise, stay in this condition 'til reset
                                end

            RXU_BREAK:          begin
                                    if (o_break)
                                        nxt_state = RXU_BREAK;
                                    else if (ck_uart)
                                        nxt_state = RXU_IDLE;                      // Goto idle state following return ck_uart going high
                                    else
                                        nxt_state = RXU_BREAK;
                                end

            RXU_IDLE:           begin
                                    if (o_break)
                                        nxt_state = RXU_BREAK;
                                    else if (~ck_uart & half_baud_time) begin      // We are in the center of a valid start bit
                                        case (data_bits)
                                            2'b00:      nxt_state = RXU_BIT_ZERO;
                                            2'b01:      nxt_state = RXU_BIT_ONE;
                                            2'b10:      nxt_state = RXU_BIT_TWO;
                                            2'b11:      nxt_state = RXU_BIT_THREE;
                                            default:    nxt_state = RXU_BIT_ZERO;
                                        endcase
                                    end
                                    else
                                        nxt_state = RXU_IDLE;                      // Otherwise, just stay here in idle
                                end

            RXU_BIT_ZERO:       begin
                                    if (o_break)
                                        nxt_state = RXU_BREAK;
                                    else if (zero_baud_counter)
                                        nxt_state = RXU_BIT_ONE;
                                    else
                                        nxt_state = RXU_BIT_ZERO;
                                end

            RXU_BIT_ONE:        begin
                                    if (o_break)
                                        nxt_state = RXU_BREAK;
                                    else if (zero_baud_counter)
                                        nxt_state = RXU_BIT_TWO;
                                    else
                                        nxt_state = RXU_BIT_ONE;
                                end

            RXU_BIT_TWO:        begin
                                    if (o_break)
                                        nxt_state = RXU_BREAK;
                                    else if (zero_baud_counter)
                                        nxt_state = RXU_BIT_THREE;
                                    else
                                        nxt_state = RXU_BIT_TWO;
                                end

            RXU_BIT_THREE:      begin
                                    if (o_break)
                                        nxt_state = RXU_BREAK;
                                    else if (zero_baud_counter)
                                        nxt_state = RXU_BIT_FOUR;
                                    else
                                        nxt_state = RXU_BIT_THREE;
                                end

            RXU_BIT_FOUR:       begin
                                    if (o_break)
                                        nxt_state = RXU_BREAK;
                                    else if (zero_baud_counter)
                                        nxt_state = RXU_BIT_FIVE;
                                    else
                                        nxt_state = RXU_BIT_FOUR;
                                end

            RXU_BIT_FIVE:       begin
                                    if (o_break)
                                        nxt_state = RXU_BREAK;
                                    else if (zero_baud_counter)
                                        nxt_state = RXU_BIT_SIX;
                                    else
                                        nxt_state = RXU_BIT_FIVE;
                                end

            RXU_BIT_SIX:        begin
                                    if (o_break)
                                        nxt_state = RXU_BREAK;
                                    else if (zero_baud_counter)
                                        nxt_state = RXU_BIT_SEVEN;
                                    else
                                        nxt_state = RXU_BIT_SIX;
                                end

            RXU_BIT_SEVEN:      begin
                                    if (o_break)
                                        nxt_state = RXU_BREAK;
                                    else if (zero_baud_counter)
                                        if (use_parity)
                                            nxt_state = RXU_PARITY;
                                        else
                                            nxt_state = RXU_STOP;
                                    else
                                        nxt_state = RXU_BIT_SEVEN;
                                end

            RXU_PARITY:         begin
                                    if (o_break)
                                        nxt_state = RXU_BREAK;
                                    else if (zero_baud_counter)
                                        nxt_state = RXU_STOP;
                                    else
                                        nxt_state = RXU_PARITY;
                                end

            RXU_STOP:           begin
                                    if (o_break)
                                        nxt_state = RXU_BREAK;
                                    else if (zero_baud_counter)
                                        if (~ck_uart)
                                            nxt_state = RXU_RESET_IDLE;
                                        else if (dblstop)
                                            nxt_state = RXU_SECOND_STOP;
                                        else
                                            nxt_state = RXU_IDLE;
                                    else
                                        nxt_state = RXU_STOP;
                                end

            RXU_SECOND_STOP:    begin
                                    if (o_break)
                                        nxt_state = RXU_BREAK;
                                    else if (zero_baud_counter)
                                        if (~ck_uart)
                                            nxt_state = RXU_RESET_IDLE;
                                        else
                                            nxt_state = RXU_IDLE;
                                    else
                                        nxt_state = RXU_SECOND_STOP;
                                end
        endcase

    end
'''

# Parsing and generating the state machine diagram
def parse_verilog_state_machine(verilog_code):
    # Regular expressions to parse the Verilog code
    state_regex = re.compile(r'(\w+):\s*begin\s*(.*?)\s*end', re.DOTALL)
    assign_regex = re.compile(r'nxt_state\s*=\s*(\w+);')

    states = []
    transitions = []

    # Extract state blocks
    for state_match in state_regex.finditer(verilog_code):
        state_name = state_match.group(1)
        state_body = state_match.group(2)
        states.append(state_name)

        # Parse transitions within the state
        transitions.extend(parse_state_transitions(state_name, state_body))

    return states, transitions

def parse_state_transitions(state_name, state_body):
    transitions = []
    # Split the state body into lines for processing
    lines = state_body.strip().split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('if'):
            condition, block, i = extract_condition_block(lines, i)
            condition = condition.replace('(', '').replace(')', '')  # Removing left and right brackets
            next_state = extract_next_state(block)
            if next_state:
                transitions.append((state_name, next_state, condition))
            else:
                # Handle nested conditions
                transitions.extend(parse_state_transitions(state_name, block))
        elif line.startswith('else if'):
            condition, block, i = extract_condition_block(lines, i, else_if=True)
            condition = condition.replace('(', '').replace(')', '')  # Removing left and right brackets
            next_state = extract_next_state(block)
            if next_state:
                transitions.append((state_name, next_state, condition))
            else:
                transitions.extend(parse_state_transitions(state_name, block))
        elif line.startswith('else'):
            block = ''
            i += 1
            while i < len(lines) and not lines[i].strip().startswith(('if', 'else', 'end', 'endmodule', 'case', 'RXU_')):
                block += lines[i] + '\n'
                i += 1
            next_state = extract_next_state(block)
            if next_state:
                transitions.append((state_name, next_state, 'else'))
            else:
                transitions.extend(parse_state_transitions(state_name, block))
        else:
            i += 1
    return transitions

def extract_condition_block(lines, start_index, else_if=False):
    condition_line = lines[start_index].strip()
    if else_if:
        condition = condition_line[len('else if('):-1].strip()
    else:
        condition = condition_line[len('if('):-1].strip()
    block = ''
    i = start_index + 1
    # Check if 'i' is within the bounds of 'lines' before accessing 'lines[i]'
    if i < len(lines) and lines[i].strip().startswith('begin'):
        i += 1
        while i < len(lines) and not lines[i].strip().startswith('end'):
            block += lines[i] + '\n'
            i += 1
        i += 1  # Skip the 'end'
    elif i < len(lines):
        block += lines[i].strip()
        i += 1
    else:
        # Handle the case where there is no block following the condition
        pass
    return condition, block, i

def extract_next_state(block):
    match = re.search(r'nxt_state\s*=\s*(\w+);', block)
    if match:
        return match.group(1)
    else:
        return None

def create_state_machine_diagram(states, transitions, output_file='state_machine'):
    fsm = Digraph('FSM', filename=output_file, format='png')
    fsm.attr(rankdir='LR', splines='true', nodesep='0.5', ranksep='0.7')  # Left to right layout

    # Rearrange the states in the sequence you provided
    ordered_states = [
        'RXU_RESET_IDLE',
        'RXU_BREAK',
        'RXU_IDLE',
        'RXU_BIT_ZERO',
        'RXU_BIT_ONE',
        'RXU_BIT_TWO',
        'RXU_BIT_THREE',
        'RXU_BIT_FOUR',
        'RXU_BIT_FIVE',
        'RXU_BIT_SIX',
        'RXU_BIT_SEVEN',
        'RXU_PARITY',
        'RXU_STOP',
        'RXU_SECOND_STOP'
    ]

    # Add states in the specified order
    for state in ordered_states:
        if state in states:
            if state == 'RXU_RESET_IDLE':  # Mark initial state with bold outline
                fsm.node(state, style='bold')
            else:
                fsm.node(state)

    # Add invisible edges to enforce node order
    for i in range(len(ordered_states) - 1):
        from_state = ordered_states[i]
        to_state = ordered_states[i+1]
        if from_state in states and to_state in states:
            fsm.edge(from_state, to_state, style='invis')

    # Add transitions
    for from_state, to_state, condition in transitions:
        fsm.edge(from_state, to_state, label=condition)

    fsm.render(view=True)

# Main execution
if __name__ == '__main__':
    states, transitions = parse_verilog_state_machine(verilog_code)
    create_state_machine_diagram(states, transitions, output_file='rxuart_fsm')

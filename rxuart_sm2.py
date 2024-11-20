import re
from graphviz import Digraph

# Verilog code of the state machine
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

# Simplify condition labels for readability
def simplify_condition(condition):
    # Replace logical operators with symbols
    condition = condition.replace('&&', '∧').replace('||', '∨')
    condition = condition.replace('and', '∧').replace('or', '∨')
    condition = condition.replace('~', '¬')
    # Remove redundant parentheses
    condition = re.sub(r'\(([^()]+)\)', r'\1', condition)
    # Truncate long conditions
    if len(condition) > 30:
        condition = condition[:27] + '...'
    return condition

# Parsing and generating the state machine diagram
def parse_verilog_state_machine(verilog_code):
    # Regular expressions to parse the Verilog code
    state_regex = re.compile(r'(\w+):\s*begin\s*(.*?)\s*end', re.DOTALL)

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

def parse_state_transitions(state_name, state_body, parent_condition=''):
    transitions = []
    # Remove comments
    state_body = re.sub(r'//.*', '', state_body)
    # Split the state body into lines for processing
    lines = state_body.strip().split('\n')
    lines = [line.strip() for line in lines if line.strip()]
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith('if'):
            condition, block, i = extract_condition_block(lines, i)
            condition = condition.strip('() ')
            full_condition = f"{parent_condition} and {condition}" if parent_condition else condition
            nested_transitions = parse_state_transitions(state_name, block, full_condition)
            transitions.extend(nested_transitions)
        elif line.startswith('else if'):
            condition, block, i = extract_condition_block(lines, i, else_if=True)
            condition = condition.strip('() ')
            full_condition = f"{parent_condition} and {condition}" if parent_condition else condition
            nested_transitions = parse_state_transitions(state_name, block, full_condition)
            transitions.extend(nested_transitions)
        elif line.startswith('else'):
            block, i = extract_else_block(lines, i)
            full_condition = parent_condition  # 'else' retains parent condition
            nested_transitions = parse_state_transitions(state_name, block, full_condition)
            transitions.extend(nested_transitions)
        elif line.startswith('case'):
            case_expr = line[len('case'):].strip(' ()')
            case_block, i = extract_case_block(lines, i)
            nested_transitions = parse_case_block(state_name, case_expr, case_block, parent_condition)
            transitions.extend(nested_transitions)
        else:
            # Check for assignment to nxt_state
            next_state = extract_next_state(line)
            if next_state and next_state != state_name:
                condition = parent_condition.strip('() ') if parent_condition else ''
                transitions.append((state_name, next_state, condition))
            i += 1
    return transitions

def extract_condition_block(lines, start_index, else_if=False):
    condition_line = lines[start_index]
    if else_if:
        condition = condition_line[len('else if'):].strip()
    else:
        condition = condition_line[len('if'):].strip()
    block = ''
    i = start_index + 1
    if i < len(lines) and lines[i].startswith('begin'):
        i += 1  # Skip 'begin'
        brace_count = 1
        while i < len(lines) and brace_count > 0:
            line = lines[i]
            if line.startswith('begin'):
                brace_count += 1
            elif line.startswith('end'):
                brace_count -= 1
            if brace_count > 0:
                block += line + '\n'
            i += 1
    else:
        # Single-line statement
        if i < len(lines):
            block = lines[i]
            i += 1
    return condition, block, i

def extract_else_block(lines, start_index):
    block = ''
    i = start_index + 1
    if i < len(lines) and lines[i].startswith('begin'):
        i += 1  # Skip 'begin'
        brace_count = 1
        while i < len(lines) and brace_count > 0:
            line = lines[i]
            if line.startswith('begin'):
                brace_count += 1
            elif line.startswith('end'):
                brace_count -= 1
            if brace_count > 0:
                block += line + '\n'
            i += 1
    else:
        # Single-line statement
        if i < len(lines):
            block = lines[i]
            i += 1
    return block, i

def extract_case_block(lines, start_index):
    case_block = ''
    i = start_index + 1  # Skip 'case' line
    brace_count = 1
    while i < len(lines) and brace_count > 0:
        line = lines[i]
        if line.startswith('case'):
            brace_count += 1
        elif line.startswith('endcase'):
            brace_count -= 1
        if brace_count > 0:
            case_block += line + '\n'
        i += 1
    return case_block, i

def parse_case_block(state_name, case_expr, case_block, parent_condition):
    transitions = []
    lines = case_block.strip().split('\n')
    for line in lines:
        line = line.strip()
        if ':' in line:
            condition_part, statement_part = line.split(':', 1)
            condition = condition_part.strip()
            statement = statement_part.strip()
            next_state = extract_next_state(statement)
            if next_state and next_state != state_name:
                case_condition = f"{case_expr} == {condition}"
                full_condition = f"{parent_condition} and {case_condition}" if parent_condition else case_condition
                transitions.append((state_name, next_state, full_condition))
    return transitions

def extract_next_state(line):
    match = re.search(r'nxt_state\s*=\s*(\w+);', line)
    if match:
        return match.group(1)
    else:
        return None

def create_state_machine_diagram(states, transitions, output_file='state_machine'):
    fsm = Digraph('FSM', filename=output_file, format='png', engine='dot')
    fsm.attr(rankdir='LR', nodesep='1.0', ranksep='1.2', overlap='false', splines='spline')

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
                fsm.node(state, shape='circle', style='bold')
            else:
                fsm.node(state, shape='circle')

    # Remove duplicates from transitions
    transitions = list(set(transitions))

    # Simplify and format conditions
    transitions = [(from_state, to_state, simplify_condition(condition)) for from_state, to_state, condition in transitions]

    # Add transitions
    for from_state, to_state, condition in transitions:
        if condition:
            fsm.edge(from_state, to_state, label=condition)
        else:
            fsm.edge(from_state, to_state)

    fsm.render(view=True)

# Main execution
if __name__ == '__main__':
    states, transitions = parse_verilog_state_machine(verilog_code)
    create_state_machine_diagram(states, transitions, output_file='rxuart_fsm')

import os
from pyverilog.vparser.parser import parse
from pyverilog.vparser.ast import Node, Parameter, Localparam, Reg, Assign, Always, IfStatement, IntConst, Identifier, Decl, Input, Output, Inout, InstanceList, Instance


def get_file_list(directory, extension=".v"):
    """
    Returns a list of Verilog files in the specified directory.

    :param directory: Directory to search for Verilog files.
    :param extension: File extension to filter by (default is ".v").
    :return: List of Verilog file paths.
    """
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(extension)]


def generate_asts(file_list):
    """
    Parses Verilog files and returns their ASTs.

    :param file_list: List of Verilog file paths.
    :return: List of ASTs corresponding to the Verilog files.
    """
    asts = []
    for filename in file_list:
        try:
            ast, _ = parse([filename])
            asts.append(ast)
        except Exception as e:
            print(f"Error parsing {filename}: {e}")
    return asts


def print_ast_contents(asts):
    """
    Prints the AST contents for each module.

    :param asts: List of ASTs corresponding to the Verilog files.
    """
    for ast in asts:
        for definition in ast.description.definitions:
            if hasattr(definition, 'name'):
                module_name = definition.name
                print(f"\nAST for module: {module_name}")
                print_ast_node(definition, "")


def print_ast_node(node, indent):
    """
    Recursively prints the AST node and its children with enhanced readability.

    :param node: The AST node to print.
    :param indent: The indentation level for pretty printing.
    """
    if isinstance(node, Node):
        node_info = f"{indent}{node.__class__.__name__}"

        # Add more detailed information for specific types of nodes
        if isinstance(node, (Parameter, Localparam)):
            node_info += f" (name={node.name}, value={get_rvalue(node.value)})"
        elif isinstance(node, Reg):
            node_info += f" (name={node.name}, width={get_rvalue(node.width)})"
        elif isinstance(node, Assign):
            node_info += f" (LHS={get_rvalue(node.left)}, RHS={get_rvalue(node.right)})"
        elif isinstance(node, Always):
            node_info += f" (sensitivity={get_rvalue(node.sens_list)})"
        elif isinstance(node, IfStatement):
            node_info += f" (condition={get_rvalue(node.cond)})"
        else:
            if hasattr(node, 'name'):
                node_info += f" (name={node.name})"
            if hasattr(node, 'value'):
                node_info += f" (value={node.value})"

        print(node_info)

        # Recursively print each child node with more indentation
        for child in node.children():
            print_ast_node(child, indent + "    ")
    else:
        # Print non-Node types directly (e.g., strings, integers)
        print(f"{indent}{node}")


def get_rvalue(node):
    """
    Recursively get the value of an AST node, converting it to a human-readable format.

    :param node: The AST node to extract the value from.
    :return: A string representation of the value.
    """
    if isinstance(node, IntConst):
        return node.value
    elif isinstance(node, Identifier):
        return node.name
    elif isinstance(node, Parameter):
        return get_rvalue(node.value)
    elif node.__class__.__name__ == 'Rvalue':
        return get_rvalue(node.var)
    elif node.__class__.__name__ == 'Minus':
        left = get_rvalue(node.left)
        right = get_rvalue(node.right)
        return f"{left}-{right}"
    elif hasattr(node, 'children'):
        # Handle complex nodes by getting the values of their children
        return f"{node.__class__.__name__}({', '.join([get_rvalue(child) for child in node.children()])})"
    else:
        return str(node)


def extract_parameters_from_asts(asts):
    parameters_dict = {}
    for ast in asts:
        for definition in ast.description.definitions:
            if hasattr(definition, 'name'):
                module_name = definition.name
                parameters_dict[module_name] = {}
                for item in definition.items:
                    if isinstance(item, Decl):
                        for decl in item.list:
                            if isinstance(decl, Parameter) and not isinstance(decl, Localparam):
                                parameters_dict[module_name][decl.name] = get_rvalue(decl.value)
                    elif isinstance(item, Parameter) and not isinstance(item, Localparam):
                        parameters_dict[module_name][item.name] = get_rvalue(item.value)
    return parameters_dict


def print_parameters(parameters_dict):
    """
    Prints the parameters for each module.

    :param parameters_dict: Dictionary with module names as keys and their parameters as values.
    """
    for module_name, params in parameters_dict.items():
        print(f"\nModule: {module_name}")
        if params:
            for param_name, param_value in params.items():
                print(f"  Parameter: {param_name} = {param_value}")
        else:
            print("  No parameters found.")


def extract_interfaces_from_asts(asts):
    interfaces_dict = {}
    for ast in asts:
        for definition in ast.description.definitions:
            if hasattr(definition, 'name'):
                module_name = definition.name
                interfaces_dict[module_name] = {'input': [], 'output': [], 'inout': []}
                for item in definition.items:
                    if isinstance(item, Decl):
                        for decl in item.list:
                            if isinstance(decl, (Identifier, Input, Output, Inout)):
                                port_type = decl.__class__.__name__.lower()
                                port_name = decl.name
                                port_width = extract_port_width(decl)
                                interfaces_dict[module_name][port_type].append((port_name, port_width))
    return interfaces_dict


def extract_port_width(decl):
    if hasattr(decl, 'width') and decl.width is not None:
        msb = get_rvalue(decl.width.msb)
        lsb = get_rvalue(decl.width.lsb)
        if msb == '0' and lsb == '0':
            return ""  # No need to print width for 1-bit signals
        return f"[{msb}:{lsb}]"
    else:
        return ""  # Default case for 1-bit signals


def print_interfaces(interfaces_dict):
    """
    Prints the interface ports (input, output, inout) for each module with their widths.

    :param interfaces_dict: Dictionary with module names as keys and their interface ports and widths as values.
    """
    for module_name, ports in interfaces_dict.items():
        print(f"\nModule: {module_name}")
        for port_type, port_list in ports.items():
            if port_list:
                print(f"  {port_type.capitalize()} ports:")
                for port_name, port_width in port_list:
                    print(f"    {port_name} {port_width}")
            else:
                print(f"  No {port_type} ports found.")


def extract_module_hierarchy_with_usage(asts):
    hierarchy_dict = {}
    used_modules = set()
    all_modules = set()

    for ast in asts:
        for definition in ast.description.definitions:
            if hasattr(definition, 'name'):
                module_name = definition.name
                all_modules.add(module_name)
                hierarchy_dict[module_name] = []
                for item in definition.items:
                    if isinstance(item, InstanceList):
                        for instance in item.instances:
                            if isinstance(instance, Instance):
                                instantiated_module = instance.module
                                instance_name = instance.name
                                hierarchy_dict[module_name].append((instantiated_module, instance_name))
                                used_modules.add(instantiated_module)

    # Determine unused modules and top-level modules
    unused_modules = all_modules - used_modules
    top_level_modules = set(module for module in hierarchy_dict.keys() if module not in used_modules)

    return hierarchy_dict, unused_modules, top_level_modules
0

def print_module_hierarchy_with_usage(hierarchy_dict, unused_modules, top_level_modules):
    """
    Prints the module hierarchy with used and unused modules in the specified format.

    :param hierarchy_dict: Dictionary where keys are module names and values are lists of instantiated modules and instance names.
    :param unused_modules: Set of modules that are not instantiated in any other module.
    :param top_level_modules: Set of modules that are top-level (i.e., not instantiated by any other module).
    """
    # Print unused modules
    print("Unused Module")
    for module in sorted(unused_modules):
        print(f"- {module}")

    print("\nUsed Module")

    def print_tree(module, hierarchy_dict, indent="L1"):
        if module in hierarchy_dict:
            for child_module, instance_name in hierarchy_dict[module]:
                print(f"{indent} - {child_module}  {instance_name}")
                print_tree(child_module, hierarchy_dict, indent + " ")

    # Print the hierarchy starting from top-level modules
    for top_module in sorted(top_level_modules):
        print(f"Top - {top_module}")
        print_tree(top_module, hierarchy_dict)


if __name__ == "__main__":
    # Get the list of Verilog files
    file_list = get_file_list("rtl")

    # Generate ASTs for the files
    asts = generate_asts(file_list)

    # print_ast_contents(asts)
    #
    # # Extract parameters from each AST
    # parameters_dict = extract_parameters_from_asts(asts)
    #
    # # Print the extracted parameters
    # print_parameters(parameters_dict)

    # # Extract interfaces from each AST
    # interfaces_dict = extract_interfaces_from_asts(asts)
    #
    # # Print the extracted interfaces
    # print_interfaces(interfaces_dict)

    # Extract module hierarchy and usage information
    hierarchy_dict, unused_modules, top_level_modules = extract_module_hierarchy_with_usage(asts)

    # Print the extracted module hierarchy with usage information
    print_module_hierarchy_with_usage(hierarchy_dict, unused_modules, top_level_modules)

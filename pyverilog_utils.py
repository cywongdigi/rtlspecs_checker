import os
from pyverilog.vparser.parser import parse
import io
import sys

def get_file_list():
    """
    Returns a list of Verilog files to process.
    """
    return [
        "test.v",
        # "rtl/txuart.v",
        # "rtl/txuartlite.v",
        # "rtl/ufifo.v",
        # "rtl/wbuart.v",
        # "rtl/rxuart.v",
        # "rtl/rxuartlite.v",
    ]

def save_vparser_output(file_list, output_dir):
    """
    Parses a list of Verilog files and saves their vparser output to files in the specified output directory.

    :param file_list: List of Verilog file paths.
    :param output_dir: Directory where the vparser output files will be saved.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    for filename in file_list:
        # Parse the Verilog file
        ast, _ = parse([filename])

        # Capture the output of ast.show() in a string
        old_stdout = sys.stdout
        new_stdout = io.StringIO()
        sys.stdout = new_stdout
        ast.show(buf=new_stdout)
        ast_str = new_stdout.getvalue()
        sys.stdout = old_stdout

        # Define the output file path
        vparser_output_filename = os.path.join(output_dir, os.path.basename(filename) + "_vparser.txt")

        # Write the captured string to the output file
        with open(vparser_output_filename, 'w') as f:
            f.write(ast_str)

        print(f"vparser output for {filename} saved to {vparser_output_filename}")

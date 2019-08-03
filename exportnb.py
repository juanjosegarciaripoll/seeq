#
# exportnb.py
#
#  Library for exporting multiple Jupyter notebooks into a series of
#  Python files. Main interface provided by function `export_notebooks`
#  below.
#
#  Author: Juan José García Ripoll
#  License: See http://opensource.org/licenses/MIT
#  Version: 1.0 (15/07/2018)
#
import sys, json, re, pathlib
def file_cell(lines):
    #
    # Determine whether a cell is to be saved as code. This
    # is done by inspecting the lines of the cell and looking
    # for a line with a comment of the form # file: xxxxx
    # If so, it eliminates this line and collects the remaining
    # text as code.
    #
    if len(lines):
        ok = re.search('^#[ ]+file:[ ]+("[^\\"]*"|[^ \n]*)[ \n]*$', lines[0])
        if ok:
            return ok.group(1), lines[1:]
    return False, lines

def register_cell(dictionary, cell_lines, add_newline=True):
    #
    # Input:
    #  - dictionary: a map from file names to lists of lines
    #    of code that will be written to the file
    #  - cell_lines: lines of a cell in a Jupyter notebook
    #  - add_newline: add empty line after each cell
    #
    # Output:
    #  - updated dictionary
    #
    file, lines = file_cell(cell_lines)
    if file:
        if file in dictionary:
            lines = dictionary[file] + lines
        if add_newline:
            lines += ['\n']
        dictionary[file] = lines
    return dictionary

def read_notebook(dictionary, notebook, add_newline=True):    
    with open(notebook, 'r', encoding='utf8') as f:
        j = json.load(f)
        if j["nbformat"] >=4:
            for i,cell in enumerate(j["cells"]):
                dictionary = register_cell(dictionary, cell["source"], add_newline)
        else:
            for i,cell in enumerate(j["worksheets"][0]["cells"]):
                dictionary = register_cell(dictionary, cell["input"], add_newline)

def write_notebooks(dictionary, root='', mkdirs=True):
    #
    # Input:
    #  - dictionary: a map from file names to list of lines of codes
    #    to be written
    #  - root: prefix to be added to all file names
    #  - mkdirs: create parent directories if they do not exist
    #
    for file in dictionary.keys():
        path = pathlib.Path(file)
        if mkdirs:
            path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding='utf8', newline='\n') as f:
            for line in dictionary[file]:
                f.write(line)

def export_notebooks(notebooks, root='', add_newline=True, mkdirs=True):
    #
    # Input:
    #  - notebooks: list of notebooks as file names
    #  - root: prefix for exporting all notebooks
    #  - add_linewline: add empty lines between cells
    #
    dictionary = {}
    for nb in notebooks:
        read_notebook(dictionary, nb, add_newline=add_newline)
    write_notebooks(dictionary, root=root, mkdirs=mkdirs)

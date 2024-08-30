from ROOT import RDataFrame
import numpy as np

from . import logger

#TODO: add logger
_logger = logger.Logger(__name__, 'info').get_logger()

def get_tree_branch_as_np(tree_name: str, 
                          root_file: str, 
                          branch_names:str, 
                          filter_string: str = None) -> dict[str, np.ndarray]:
    '''
    Loads a TTree from a .root File, applies a filter and returns defined 
    branches (columns) as a dictionary.
    The column names are the keys of the dictionary. 
    If the column contains only integers, a 1D Numpy array
    is the value. If the column contains C++ Vectors, it is cast to a python 
    list. The value is a 1D
    Numpy array of objects (python lists).
    It is recommended to filter the data needed, since loading to memory it 
    quite slow. A "filter_string" can
    be provided. The content must be C++ code. The name of the variables must 
    be the name of the branches.
    eg: filter_string = 'NSignals >= 2 && FrameIndex == 8'
    Args:
        tree_name: name of a TTree
        root_file: path of the .root file
        branch_names: list of strings containing names of branches to load
        filter_string: string for a filter 
    Returns:
        Dictionary of numpy arrays
    '''
    #TODO: add better exception handling
    try:
        df = RDataFrame(tree_name,root_file)
        print(f'Tree loaded succesfully: \n {df.Describe()}')
    except:
        print('Tree could not be loaded.')
        return 0
    #get a dictionary of types
    column_names = list(df.GetColumnNames())
    column_type = {col: df.GetColumnType(col) for col in column_names}
    
    #Apply a filter string. It is recommended to use one to avoid loading unneccesary data.
    #The content must be C++ code. The name of the variables must be the name of the branches.
    if filter_string is not None:
        try:
            df = df.Filter(filter_string)
        except:
            print('Filter {filter_string} could not be applied')
            return 0
    
    #now get the branches in a numy array and transform C++ Objects to Python Lists
    output = df.AsNumpy(columns=branch_names)
    for key in output.keys():
        branch_entry_type = column_type[key]
        if 'RVec' in branch_entry_type:
            for index, vec in enumerate(output[key]):
                content = list(vec)
                output[key][index] = content
    return output

def get_count_map_from_pixel_index(inp_dict: 
                                   dict[str, np.ndarray]) -> np.ndarray:
    '''
    Takes a dictionary (the output of get_tree_branch_as_np()) with 'Column' 
    and 'Row' keys and returns a 64 by 64 map with the counts.
    Args:
        inp_dict: dictionary with a list of lists for 'Column' and 'Row'
    Returns:
        64 by 64 numpy array
    '''
    #TODO: add better exception handling
    if type(inp_dict) != dict:
        print('Not a dictionary!')
        return
    output = np.zeros((64,64))
    keys = inp_dict.keys()
    if 'Column' not in keys or 'Row' not in keys :
        print(f'No Pixel indizes in keys of dictionary {inp_dict.keys()}')
        return
    cols = inp_dict['Column']
    rows = inp_dict['Row']
    if len(cols) != len(rows):
        print('Something is wrong with the event tree.')
        return
    else:
        #loop through events
        for event_no in range(len(cols)):
            #get the indizes for that event
            cols_event = cols[event_no]
            rows_event = rows[event_no]
            if len(cols_event) != len(rows_event):
                print('Something is wrong with the event tree.')
                return
            else:
                #loop throug indices for that event and fill map
                for i in range(len(cols_event)):
                    col = cols_event[i]
                    row = rows_event[i]
                    output[col,row] += 1
    return output
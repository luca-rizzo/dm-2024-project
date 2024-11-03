# We use 'Int64' to correctly manage also NaN values
import numpy as np

def cast_column_to_Int64(races, column_name):
    prev_column_type = races[column_name].dtype
    try:
        res = np.all(races[column_name] == races[column_name].astype('Int64'))
        if res:
            races[column_name] = races[column_name].astype('Int64')
            print(f'Column casted from {prev_column_type} to {races[column_name].dtype}')

    except TypeError:
        print("Not castable!")
        
        
def cast_column_to_string(races, column_name):
    prev_column_type = races[column_name].dtype
    races[column_name] = races[column_name].astype('string')
    print(f'Column casted from {prev_column_type} to {races[column_name].dtype}')

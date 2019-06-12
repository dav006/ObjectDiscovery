import os
import h5py

def loadh5(dump_file_full_name):
    ''' Loads a h5 file as dictionary '''

    with h5py.File(dump_file_full_name, 'r') as h5file:
        dict_from_file = readh5(h5file)

    return dict_from_file


def readh5(h5node):
    ''' Recursive function to read h5 nodes as dictionary '''

    dict_from_file = {}
    for _key in h5node.keys():
        if isinstance(h5node[_key], h5py._hl.group.Group):
            dict_from_file[_key] = readh5(h5node[_key])
        else:
            dict_from_file[_key] = h5node[_key].value

    return dict_from_file

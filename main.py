

from database_handling.database_extract import get_all_subject_data_from_internal_task_id
from collections import defaultdict

import jax.numpy as jnp


def _swaplist(_list):
        """ Put the various factors / modalities as the leading dimension for a 2D list of lists."""
        if _list is None :
            return None
        
        # Ensure we have more than one iterable dimension :
        for el in _list :
            if (type(el) != list) and (type(el) != tuple) and (type(el) != dict):
                # There is a single factor here ! 
                print(el)
                return _list

        if type(_list[0]==tuple):
            print("yes !")
            # return tuple([[_swaplist(_el) for _el in _list])
        
        
        
        
        
        if type(_list[0]==list):
            # Assume 1 level of recursive depth :
            return [[_el[factor] for _el in _list] for factor in _list[0]]                    
            
        if type(_list[0]==dict):
            _swapped_list = {}
            for key in _list[0].keys():
                _swapped_list[key] = [_el[key] for _el in _list]
            return _swapped_list

        raise NotImplementedError("Unrecognized data type")


# def transpose_list_of_objects():
    

if __name__=="__main__":   
            
    swap_this = [(10,{"angle":jnp.array([1,0,0]),"position":jnp.array([1,0,0])}),
                 (9,{"angle":jnp.array([0,1,0]),"position":jnp.array([1,0,0])}),
                (8,{"angle":jnp.array([0,0,1]),"position":jnp.array([1,0,0])})]

    
    # print(_swaplist(swap_this))

    transposed_once = list(zip(*swap_this))
    for k,el in enumerate(transposed_once):
        if type(el[0])==dict:
            transposed_dict = {}
            for key in el[0].keys():
                transposed_dict[key] = [subel[key] for subel in el]
            transposed_once[k] = transposed_dict
    print(transposed_once)
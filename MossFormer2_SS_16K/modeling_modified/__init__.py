# clearvoice/__init__.py
__version__ = "0.1.0"
__all__ = ['ClearVoice']

from .network_wrapper import network_wrapper
import os
import numpy as np
import torch
import warnings
import argparse
warnings.filterwarnings("ignore")

class ClearVoice:
    """ The main class inferface to the end users for performing speech processing
        this class provides the desired model to perform the given task
    """
    def __init__(self, task, model_names, model_path="./"):
        """ Load the desired models for the specified task. Perform all the given models and return all results.
   
        Parameters:
        ----------
        task: str
            the task matching any of the provided tasks: 
            'speech_enhancement'
            'speech_separation'
            'target_speaker_extraction'
        model_names: str or list of str
            the model names matching any of the provided models: 
            'FRCRN_SE_16K'
            'MossFormer2_SE_48K'
            'MossFormerGAN_SE_16K'
            'MossFormer2_SS_16K'
            'AV_MossFormer2_TSE_16K'

        Returns:
        --------
        A ModelsList object, that can be run to get the desired results
        """        
        self.network_wrapper = network_wrapper(model_path=model_path)
        self.models = []
        for model_name in model_names:
            model = self.network_wrapper(task, model_name)
            self.models += [model]  
            
    def __call__(self, input_path, online_write=False, output_path=None):
        if isinstance(input_path, str):
            return self.call_io_mode(input_path=input_path, online_write=online_write, output_path=output_path)
        elif isinstance(input_path, np.ndarray) or isinstance(input_path, torch.Tensor):
            return self.call_t2t_mode(input_data=input_path)
        else:
            print("Unknown type:", type(input_path))
        
    def call_t2t_mode(self, input_data):
        if len(self.models) > 1:
    	    print('This tensor-to-tensor mode supports only one model!')
    	    return
        else:
            return self.models[0].decode_data(input_data)
                
    def call_io_mode(self, input_path, online_write=False, output_path=None):
        results = {}
        for model in self.models:
            result = model.process(input_path, online_write, output_path)
            if not online_write:
                results[model.name] = result

        if not online_write:
            if len(results) == 1:
                return results[model.name]
            else:
                return results
        else:
       	    return
               
    def write(self, results, output_path):
        add_subdir = False
        use_key = False
        if len(self.models) > 1: add_subdir = True #multi_model is True        
        for model in self.models:
            if isinstance(results, dict):
                if model.name in results: 
                   if len(results[model.name]) > 1: use_key = True
                       
                else:
                   if len(results) > 1: use_key = True #multi_input is True
            break

        for model in self.models:
            model.write(output_path, add_subdir, use_key)

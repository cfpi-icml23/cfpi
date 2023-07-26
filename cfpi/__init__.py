# type: ignore[attr-defined]
"""Offline Reinforcement Learning with Closed-Form Policy Improvement Operators"""

from importlib import metadata as importlib_metadata
import pickle
class RenamingUnpickler(pickle.Unpickler):
    def find_class(self, module, name):

        if module == 'rlkit.core.pythonplusplus':
            module = 'eztils.torch'
        
        if module == 'rlkit.envs.wrappers':
            module = 'cfpi.envs'
            
        if 'rlkit.torch' in module:
            module = module.replace('rlkit.torch', 'cfpi.pytorch')
                    
        if 'rlkit' in module:
            module = module.replace('rlkit', 'cfpi')
        
            
        return super().find_class(module, name)

pickle.Unpickler = RenamingUnpickler

def get_version() -> str:
    try:
        return importlib_metadata.version(__name__)
    except importlib_metadata.PackageNotFoundError:  # pragma: no cover
        return "unknown"


version: str = get_version()
__version__ = version

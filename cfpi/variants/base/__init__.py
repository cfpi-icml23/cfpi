from pydantic import BaseModel as PydanticBaseModel


class BaseModel(PydanticBaseModel):
    class Config:
        arbitrary_types_allowed = True
        allow_mutation = False
        validate_all = True


class FuncWrapper:  # Wrap a function or class definition as an object since we can't directly include functions
    def __init__(self, f) -> None:
        self.f = f


w = FuncWrapper  # alias


#! VAE Kwargs
class VaeKwargs(BaseModel):
    hidden_dim = 750

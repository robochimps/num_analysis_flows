from  enum import Enum

class nnType(Enum):
    """
    Enumeration class representing different types of neural networks.
    """

    resnet = "Residual NN"
    recnet = "Recurrent NN"
    ode = "ODE"
    rnvp = "RNVP"

class activationType(Enum):
    """
    Enum class representing different activation types.

    Attributes:
        sigmoid (str): Sigmoid activation type.
        relu (str): Relu activation type.
        lipswish (str): Lipswish activation type.
    """
    sigmoid = "Sigmoid"
    relu = "Relu"
    lipswish = "Lipswish"
    sinh = "Sinh"
    cosh = "Cosh"
    erf = "Erf"
    scaled_erf = "Scaled_Erf"
    
class evaluationMode(Enum):
    """
    Enum class representing different activation types. modes of evaluation for
    an invertible model.

    Attributes:
        direct (str): Represents the direct mode of evaluation.
        inverse (str): Represents the inverse mode of evaluation.
    """
    direct = "Direct"
    inverse = "Inverse"

class svdType(Enum):
    fourier = "Fourier"
    direct  = "Direct"
    direct_indiv = "Direct_indiv"
    flax = "Flax"  ## To be added.
    svd_multiple = (
        "Allow for different SVD values to be equal to the Lipschitz constant"
    )
    
class fixPoint(Enum):
    """Choice of different computations of the fixed-point method for the inversion of the NN"""
    REGULAR = "Use the iteration x_{k+1} = T x_{k}"
    HALPERN = "Use the iteration x_{k+1} = x_0/(k+2)  + (1 - 1/(k+2)) * T x_{k}"
    PICARD = "Use the iteration x_{k+1} = x_k/(k+2)  + (1 - 1/(k+2)) * T x_{k}"

from .resnet_3d import ControlPredictor as ControlPredictorResNet3D
from .cnn_3d import ControlPredictor as ControlPredictorCNN3D
from .st_transformer import STInverseDynamics

def get_model_cls(model_id):
    match model_id:
        case 'resnet3d':
            return ControlPredictorResNet3D
        case 'cnn3d':
            return ControlPredictorCNN3D
        case 'st_transformer':
            return STInverseDynamics
        case _:
            raise NotImplementedError
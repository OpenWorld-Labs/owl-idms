from .resnet_3d import ControlPredictor as ControlPredictorResNet3D

def get_model_cls(model_id):
    match model_id:
        case 'resnet3d':
            return ControlPredictorResNet3D
        case _:
            raise NotImplementedError
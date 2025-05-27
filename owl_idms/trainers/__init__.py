from typing import Literal

from .idm_trainer import IDMTrainer


def get_trainer_cls(trainer_id: Literal):
    match trainer_id:
        case "idm_trainer":
            return IDMTrainer
        case _:
            raise NotImplementedError
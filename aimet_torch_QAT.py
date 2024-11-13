# https://quic.github.io/aimet-pages/releases/latest/Examples/torch/quantization/qat.html

import os
import torch
from Examples.common import image_net_config
from Examples.torch.utils.image_net_evaluator import ImageNetEvaluator
from Examples.torch.utils.image_net_trainer import ImageNetTrainer
from Examples.torch.utils.image_net_data_loader import ImageNetDataLoader

from torchvision.models import resnet18
from torchvision import models
from aimet_torch.model_preparer import prepare_model

import onnxruntime as ort



DATASET_DIR = '/home/yanwh/workspace/QNN_Tools/mammals_data'   # Please replace this with a real directory

class ImageNetDataPipeline:
    
    @staticmethod
    def get_val_dataloader() -> torch.utils.data.DataLoader:
        """
        Instantiates a validation dataloader for ImageNet dataset and returns it
        """
        data_loader = ImageNetDataLoader(DATASET_DIR,
                                         image_size=image_net_config.dataset['image_size'],
                                         batch_size=image_net_config.evaluation['batch_size'],
                                         is_training=False,
                                         num_workers=image_net_config.evaluation['num_workers']).data_loader
        return data_loader

    @staticmethod
    def evaluate(model: torch.nn.Module, use_cuda: bool) -> float:
        """
        Given a torch model, evaluates its Top-1 accuracy on the dataset
        :param model: the model to evaluate
        :param use_cuda: whether or not the GPU should be used.
        """
        evaluator = ImageNetEvaluator(DATASET_DIR, image_size=image_net_config.dataset['image_size'],
                                      batch_size=image_net_config.evaluation['batch_size'],
                                      num_workers=image_net_config.evaluation['num_workers'])

        return evaluator.evaluate(model, iterations=None, use_cuda=use_cuda)

    @staticmethod
    def finetune(model: torch.nn.Module, epochs, learning_rate, learning_rate_schedule, use_cuda):
        """
        Given a torch model, finetunes the model to improve its accuracy
        :param model: the model to finetune
        :param epochs: The number of epochs used during the finetuning step.
        :param learning_rate: The learning rate used during the finetuning step.
        :param learning_rate_schedule: The learning rate schedule used during the finetuning step.
        :param use_cuda: whether or not the GPU should be used.
        """
        trainer = ImageNetTrainer(DATASET_DIR, image_size=image_net_config.dataset['image_size'],
                                  batch_size=image_net_config.train['batch_size'],
                                  num_workers=image_net_config.train['num_workers'])

        trainer.train(model, max_epochs=epochs, learning_rate=learning_rate,
                      learning_rate_schedule=learning_rate_schedule, use_cuda=use_cuda)
        
    @staticmethod
    def onnx_evaluate(ort_session, use_cuda: bool) -> float:
        """
        Given an onnx model, evaluates its Top-1 accuracy on the dataset
        :param ort_session: the onnx model to evaluate
        :param use_cuda: whether or not the GPU should be used.
        """
        evaluator = ImageNetEvaluator(DATASET_DIR, image_size=image_net_config.dataset['image_size'],
                                      batch_size=image_net_config.evaluation['batch_size'],
                                      num_workers=image_net_config.evaluation['num_workers'])

        return evaluator.onnx_evaluate(ort_session, iterations=None, use_cuda=use_cuda)
    


                

# Load the model
model = resnet18(num_classes=45)

weight_path = "resnet18.pth"
assert os.path.exists(weight_path), "weight {} does not exist.".format(weight_path)
model.load_state_dict(torch.load(weight_path, map_location=torch.device('cuda')))

model = prepare_model(model)

use_cuda = False
if torch.cuda.is_available():
    use_cuda = True
    model.to(torch.device('cuda'))
    
accuracy = ImageNetDataPipeline.evaluate(model, use_cuda)
# print(accuracy)

model_path = "resnet18.onnx"
ort_session = ort.InferenceSession(model_path)

print(ort.get_device())

onnx_accuracy = ImageNetDataPipeline.onnx_evaluate(ort_session, use_cuda)
    
# https://quic.github.io/aimet-pages/releases/latest/Examples/torch/quantization/qat.html

import os
import pdb
import torch
import onnx
from Examples.common import image_net_config
from Examples.torch.utils.image_net_evaluator import ImageNetEvaluator
from Examples.torch.utils.image_net_trainer import ImageNetTrainer
from Examples.torch.utils.image_net_data_loader import ImageNetDataLoader

from torchvision.models import resnet18
from torchvision import models
from aimet_onnx.batch_norm_fold import fold_all_batch_norms_to_weight
from aimet_common.defs import QuantScheme
from aimet_onnx.quantsim import QuantizationSimModel
from aimet_torch.model_preparer import prepare_model

import onnxruntime as ort
from onnxsim import simplify



DATASET_DIR = 'mammals_data'   # Please replace this with a real directory

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
                                      num_workers=image_net_config.evaluation['num_workers'],
                                      num_val_samples_per_class=16)

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
                                      num_workers=image_net_config.evaluation['num_workers'],
                                      num_val_samples_per_class=16)

        return evaluator.onnx_evaluate(ort_session, iterations=None, use_cuda=use_cuda)
    


# 指定设备，device是str类型（字符串）
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using {} device".format(device)) #将设备信息输出


# Load the model
model = resnet18(num_classes=45)

weight_path = "resnet18.pth"
assert os.path.exists(weight_path), "weight {} does not exist.".format(weight_path)
model.load_state_dict(torch.load(weight_path, map_location=torch.device(device)))

# model = prepare_model(model)

use_cuda = False
if torch.cuda.is_available():
    use_cuda = True
    model.to(torch.device('cuda'))
    
print("++++++++++++++++++++++++++++++++++++++++++++ Torch Model Accuracy ++++++++++++++++++++++++++++++++++++++++++")
accuracy = ImageNetDataPipeline.evaluate(model, use_cuda)

input_shape = (1, 3, 224, 224)    # Shape for each ImageNet sample is (3 channels) x (224 height) x (224 width)

train_data = ImageNetDataPipeline.get_val_dataloader()

dummy_input = next(iter(train_data))[0][0]

dummy_input = dummy_input.to(device)
dummy_input = dummy_input.reshape(input_shape)
filename = "./resnet18.onnx"

# Export the torch model to onnx
torch.onnx.export(model.eval(),
                dummy_input,
                filename,
                training = torch.onnx.TrainingMode.PRESERVE,
                export_params = True,
                do_constant_folding = False,
                input_names = ['input'],
                output_names = ['output'],
                dynamic_axes = {
                    'input' : {0 : 'batch_size'},
                    'output' : {0 : 'batch_size'},
                }
                )   

# 加载并检查 ONNX 模型
onnx_model = onnx.load(filename)
onnx.checker.check_model(onnx_model)
print("ONNX Check Success！")

print("finished tarining")

model_path = "resnet18.onnx"
ort_session = ort.InferenceSession(model_path)

print("++++++++++++++++++++++++++++++++++++++++++++ ONNX Model Accuracy +++++++++++++++++++++++++++++++++++++++++++")
onnx_accuracy = ImageNetDataPipeline.onnx_evaluate(ort_session, use_cuda)

model = onnx.load_model(model_path)
filename4sim = "./resnet18_sim.onnx"
model, check = simplify(model, skip_fuse_bn=True, perform_optimization=False)
onnx.save(model, filename4sim)
sim_ort_session = ort.InferenceSession(filename4sim)

print("++++++++++++++++++++++++++++++++++++++++++++ ONNX Model Accuracy after Simplification +++++++++++++++++++++++++++++++++++++++++++")
onnx_accuracy = ImageNetDataPipeline.onnx_evaluate(sim_ort_session, use_cuda)

print("+++++++++++++++++++++++++++++++++++++++++ Start Quant +++++++++++++++++++++++++++++++++++++++++")

model = onnx.load_model(filename4sim)

_ = fold_all_batch_norms_to_weight(model)

sim = QuantizationSimModel(model=model,
                           quant_scheme=QuantScheme.post_training_tf_enhanced,
                           default_activation_bw=8,
                           default_param_bw=8,
                           use_cuda=use_cuda)


def pass_calibration_data(session, samples):
    data_loader = ImageNetDataPipeline.get_val_dataloader()
    batch_size = data_loader.batch_size
    input_name = session.get_inputs()[0].name

    batch_cntr = 0
    for input_data, target_data in data_loader:

        inputs_batch = input_data.numpy()
        session.run(None, {input_name : inputs_batch})

        batch_cntr += 1
        if (batch_cntr * batch_size) > samples:
            break


sim.compute_encodings(forward_pass_callback=pass_calibration_data,
                      forward_pass_callback_args=1000)

print("++++++++++++++++++++++++++ Quan Model Accuracy ++++++++++++++++++++++++++++++++++")
accuracy = ImageNetDataPipeline.onnx_evaluate(sim.session, use_cuda)

# Export the model which saves pytorch model without any simulation nodes and saves encodings file for both
# activations and parameters in JSON format
sim.export(path='./', filename_prefix='quantized_resnet18')
    
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023, Qualcomm Innovation Center, Inc. All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
#  3. Neither the name of the copyright holder nor the names of its contributors
#     may be used to endorse or promote products derived from this software
#     without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
#
#  SPDX-License-Identifier: BSD-3-Clause
#
#  @@-COPYRIGHT-END-@@
# =============================================================================


"""
Creates Evaluator for Image-Net dataset
"""
import logging

import progressbar
import torch
import onnxruntime as ort

from Examples.common.utils import accuracy
from Examples.torch.utils.image_net_data_loader import ImageNetDataLoader

logger = logging.getLogger('Eval')


class ImageNetEvaluator:
    """
    For validation of a trained model using the ImageNet dataset.
    """

    def __init__(self, images_dir: str, image_size: int, batch_size: int = 128,
                 num_workers: int = 32, num_val_samples_per_class: int = None):
        """
        :param images_dir: The path to the data directory
        :param image_size: The length of the image
        :param batch_size: The batch size to use for training and validation
        :param num_workers: Indiicates to the data loader how many sub-processes to use for data loading.
        :param num_train_samples_per_class: Number of samples to use per class.
        """
        self._val_data_loader = ImageNetDataLoader(images_dir,
                                                   image_size=image_size,
                                                   batch_size=batch_size,
                                                   is_training=False,
                                                   num_workers=num_workers,
                                                   num_samples_per_class=num_val_samples_per_class).data_loader

    def evaluate(self, sess: ort.InferenceSession, iterations: int = None) -> float:
        """
        Evaluate the specified model using the specified number of samples batches from the
        validation set.
        :param sess: The model to be evaluated.
        :param iterations: The number of batches to use from the validation set.
        :return: The accuracy for the sample with the maximum accuracy.
        """

        if iterations is None:
            logger.info('No value of iteration is provided, running evaluation on complete dataset.')
            iterations = len(self._val_data_loader)
        if iterations <= 0:
            logger.error('Cannot evaluate on %d iterations', iterations)
        input_name = sess.get_inputs()[0].name
        acc_top1 = 0
        acc_top5 = 0

        logger.info("Evaluating nn.Module for %d iterations with batch_size %d",
                    iterations, self._val_data_loader.batch_size)

        batch_cntr = 1
        with progressbar.ProgressBar(max_value=iterations) as progress_bar:
            for input_data, target_data in self._val_data_loader:

                inputs_batch = input_data.numpy()

                predicted_batch = sess.run(None, {input_name : inputs_batch})[0]

                batch_avg_top_1_5 = accuracy(output=torch.from_numpy(predicted_batch), target=target_data,
                                             topk=(1, 5))

                acc_top1 += batch_avg_top_1_5[0].item()
                acc_top5 += batch_avg_top_1_5[1].item()

                progress_bar.update(batch_cntr)

                batch_cntr += 1
                if batch_cntr > iterations:
                    break

        acc_top1 /= iterations
        acc_top5 /= iterations

        logger.info('Avg accuracy Top 1: %f Avg accuracy Top 5: %f on validation Dataset',
                    acc_top1, acc_top5)

        return acc_top1

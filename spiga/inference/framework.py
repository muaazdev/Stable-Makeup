import os
import pkg_resources
import copy
import torch
import numpy as np

# Paths
weights_path_dft = pkg_resources.resource_filename('spiga', 'models/weights')

import spiga.inference.pretreatment as pretreat
from spiga.models.spiga import SPIGA
from spiga.inference.config import ModelConfig


class SPIGAFramework:

    def __init__(self, model_cfg: ModelConfig(), gpus=[0], load3DM=True):
        # Parameters
        self.model_cfg = model_cfg
        self.gpus = gpus
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Pretreatment initialization
        self.transforms = pretreat.get_transformers(self.model_cfg)

        # SPIGA model
        self.model_inputs = ['image', "model3d", "cam_matrix"]
        self.model = SPIGA(num_landmarks=model_cfg.dataset.num_landmarks,
                           num_edges=model_cfg.dataset.num_edges)

        # Load weights and set model
        weights_path = self.model_cfg.model_weights_path
        if weights_path is None:
            weights_path = weights_path_dft

        if self.model_cfg.load_model_url:
            model_state_dict = torch.hub.load_state_dict_from_url(self.model_cfg.model_weights_url,
                                                                  model_dir=weights_path,
                                                                  file_name=self.model_cfg.model_weights)
        else:
            weights_file = os.path.join(weights_path, self.model_cfg.model_weights)
            model_state_dict = torch.load(weights_file, map_location=self.device)

        self.model.load_state_dict(model_state_dict)
        self.model = self.model.to(self.device)
        self.model.eval()
        print('SPIGA model loaded!')

        # Load 3D model and camera intrinsic matrix
        if load3DM:
            loader_3DM = pretreat.AddModel3D(model_cfg.dataset.ldm_ids,
                                             ftmap_size=model_cfg.ftmap_size,
                                             focal_ratio=model_cfg.focal_ratio,
                                             totensor=True)
            params_3DM = self._data2device(loader_3DM())
            self.model3d = params_3DM['model3d']
            self.cam_matrix = params_3DM['cam_matrix']

    def inference(self, image, bboxes):
        batch_crops, crop_bboxes = self.pretreat(image, bboxes)
        outputs = self.net_forward(batch_crops)
        features = self.postreatment(outputs, crop_bboxes, bboxes)
        return features

    def pretreat(self, image, bboxes):
        crop_bboxes = []
        crop_images = []
        for bbox in bboxes:
            sample = {'image': copy.deepcopy(image),
                      'bbox': copy.deepcopy(bbox)}
            sample_crop = self.transforms(sample)
            crop_bboxes.append(sample_crop['bbox'])
            crop_images.append(sample_crop['image'])

        batch_images = torch.tensor(np.array(crop_images), dtype=torch.float)
        batch_images = self._data2device(batch_images)

        batch_model3D = self.model3d.unsqueeze(0).repeat(len(bboxes), 1, 1)
        batch_cam_matrix = self.cam_matrix.unsqueeze(0).repeat(len(bboxes), 1, 1)

        model_inputs = [batch_images, batch_model3D, batch_cam_matrix]
        return model_inputs, crop_bboxes

    def net_forward(self, inputs):
        outputs = self.model(inputs)
        return outputs

    def postreatment(self, output, crop_bboxes, bboxes):
        features = {}
        crop_bboxes = np.array(crop_bboxes)
        bboxes = np.array(bboxes)

        if 'Landmarks' in output.keys():
            landmarks = output['Landmarks'][-1].cpu().detach().numpy()
            landmarks = landmarks.transpose((1, 0, 2))
            landmarks = landmarks * self.model_cfg.image_size
            landmarks_norm = (landmarks - crop_bboxes[:, 0:2]) / crop_bboxes[:, 2:4]
            landmarks_out = (landmarks_norm * bboxes[:, 2:4]) + bboxes[:, 0:2]
            landmarks_out = landmarks_out.transpose((1, 0, 2))
            features['landmarks'] = landmarks_out.tolist()

        if 'Pose' in output.keys():
            pose = output['Pose'].cpu().detach().numpy()
            features['headpose'] = pose.tolist()

        return features

    def select_inputs(self, batch):
        inputs = []
        for ft_name in self.model_inputs:
            data = batch[ft_name]
            inputs.append(self._data2device(data.type(torch.float)))
        return inputs

    def _data2device(self, data):
        if isinstance(data, list):
            return [self._data2device(v) for v in data]
        if isinstance(data, dict):
            return {k: self._data2device(v) for k, v in data.items()}
        with torch.no_grad():
            return data.to(self.device, non_blocking=torch.cuda.is_available())

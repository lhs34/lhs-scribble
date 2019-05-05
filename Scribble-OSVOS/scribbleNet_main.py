import os
import copy
import timeit

import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import scipy.misc as sm
import numpy as np

from davisinteractive import utils as interactive_utils

from dataloaders import davis_2017 as db
from dataloaders import davis_custom as db_scribblenet

from mypath import Path
from dataloaders import custom_transforms as tr
from layers.osvos_layers import class_balanced_cross_entropy_loss

from networks.scribble_net import ScribbleNet


class ScribbleNetMain(object):
    def __init__(self, parent_model, save_model_dir, gpu_id, time_budget, save_result_dir=None):
        self.save_model_dir = save_model_dir
        self.parent_model = parent_model
        self.save_res_dir = save_result_dir
        self.net = ScribbleNet()
        if gpu_id >= 0:
            torch.cuda.set_device(device=gpu_id)
            self.net.cuda()
        self.gpu_id = gpu_id
        self.time_budget = time_budget
        self.meanval = (104.00699, 116.66877, 122.67892)
        self.train_batch = 1
        self.test_batch = 1
        self.prev_models = {}
        # self.parent_model_state = torch.load(os.path.join(Path.models_dir(), self.parent_model),
        #                                      map_location=lambda storage, loc: storage)

    def train(self, first_frame, n_interaction, obj_id, scribbles_data, scribble_iter, subset, use_previous_mask=False):
        nAveGrad = 1
        num_workers = 4
        train_batch = min(n_interaction, self.train_batch)

        frames_list = interactive_utils.scribbles.annotated_frames_object(scribbles_data, obj_id)
        scribbles_list = scribbles_data['scribbles']
        seq_name = scribbles_data['sequence']

        if obj_id == 1 and n_interaction == 1:
            self.prev_models = {}

        # # Network definition
        # if n_interaction == 1:
        #     print('Loading weights from: {}'.format(self.parent_model))
        #     self.net.load_state_dict(self.parent_model_state)
        #     self.prev_models[obj_id] = None
        # else:
        #     print('Loading weights from previous network: objId-{}_interaction-{}_scribble-{}.pth'
        #           .format(obj_id, n_interaction-1, scribble_iter))
        #     self.net.load_state_dict(self.prev_models[obj_id])

        lr = 1e-5
        wd = 0.0002
        # optimizer = optim.SGD([
        #     {'params': [pr[1] for pr in self.net.stages.named_parameters() if 'weight' in pr[0]], 'weight_decay': wd},
        #     {'params': [pr[1] for pr in self.net.stages.named_parameters() if 'bias' in pr[0]], 'lr': lr * 2},
        #     {'params': [pr[1] for pr in self.net.side_prep.named_parameters() if 'weight' in pr[0]], 'weight_decay': wd},
        #     {'params': [pr[1] for pr in self.net.side_prep.named_parameters() if 'bias' in pr[0]], 'lr': lr * 2},
        #     {'params': [pr[1] for pr in self.net.upscale.named_parameters() if 'weight' in pr[0]], 'lr': 0},
        #     {'params': [pr[1] for pr in self.net.upscale_.named_parameters() if 'weight' in pr[0]], 'lr': 0},
        #     {'params': self.net.fuse.weight, 'lr': lr / 100, 'weight_decay': wd},
        #     {'params': self.net.fuse.bias, 'lr': 2 * lr / 100},
        # ], lr=lr, momentum=0.9)

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.net.parameters()), lr=lr, momentum=0.9)

        prev_mask_path = os.path.join(self.save_res_dir, 'interaction-{}'.format(n_interaction-1),
                                      'scribble-{}'.format(scribble_iter))
        composed_transforms_tr = transforms.Compose([tr.CenterCrop((480,832)),tr.SubtractMeanImage(self.meanval),
                                                     tr.CustomScribbleInteractive(scribbles_list, first_frame,
                                                                                  use_previous_mask=use_previous_mask,
                                                                                  previous_mask_path=prev_mask_path),
                                                     tr.RandomHorizontalFlip(),
                                                     tr.ScaleNRotate(rots=(-30, 30), scales=(.75, 1.25)),
                                                     tr.ToTensor()])
        # Training dataset and its iterator
        # db_train = db.DAVIS2017(split=subset, transform=composed_transforms_tr,
        #                         custom_frames=frames_list, seq_name=seq_name,
        #                         obj_id=obj_id, no_gt=True, retname=True)
        db_train = db_scribblenet.DAVIS2017(split=subset, transform=composed_transforms_tr,
                                custom_frames=frames_list, seq_name=seq_name,
                                obj_id=obj_id, no_gt=True, retname=True)       
        trainloader = DataLoader(db_train, batch_size=train_batch, shuffle=True, num_workers=num_workers)
        num_img_tr = len(trainloader)
        loss_tr = []
        aveGrad = 0

        # List of all previous masks and aggregated features
        prev_masks = []
        prev_aggs = []

        start_time = timeit.default_timer()
        # Main Training and Testing Loop
        epoch = 0
        while 1:
            # One training epoch
            running_loss_tr = 0
            for ii, sample_batched in enumerate(trainloader):
                optimizer.zero_grad()

                # Parse from dataset loader
                inputs = sample_batched['images'].cuda()
                gts = sample_batched['scribble_gt'].cuda()
                scribbles = sample_batched['scribble_raw'].cuda()
                scribbles_idx = sample_batched['scribble_idx'].cuda()

                # Forward-Backward of the mini-batch
                # prev_masks = torch.tensor(prev_masks).unsqueeze(0)
                # prev_aggs = torch.tensor(prev_aggs).unsqueeze(0)
                masks, agg = self.net.forward(inputs, scribbles, scribble_idx, prev_masks, prev_agg)

                # Compute the fuse loss
                loss = class_balanced_cross_entropy_loss(masks, gts, scribble_idx)
                running_loss_tr += loss.item()

                # Print stuff
                if epoch % 10 == 0:
                    running_loss_tr /= num_img_tr
                    loss_tr.append(running_loss_tr)

                    print('[Epoch: %d, numImages: %5d]' % (epoch + 1, ii + 1))
                    print('Loss: %f' % running_loss_tr)

                # Backward the averaged gradient
                loss.backward()
                optimizer.step()

                # Update the current round data
                prev_masks.append(masks.detach())
                prev_aggs.append(agg.detach())

            epoch += train_batch
            stop_time = timeit.default_timer()
            if stop_time - start_time > self.time_budget:
                break

        # Save the model into dictionary
        self.prev_models[obj_id] = copy.deepcopy(self.net.state_dict())

    def test(self,  sequence, n_interaction, obj_id, subset, scribble_iter=0):
        if self.save_res_dir:
            save_dir_res = os.path.join(self.save_res_dir, 'interaction-{}'.format(n_interaction),
                                        'scribble-{}'.format(scribble_iter),
                                        sequence, str(obj_id))
            if not os.path.exists(save_dir_res):
                os.makedirs(save_dir_res)

        composed_transforms_ts = transforms.Compose([tr.CenterCrop((480,832)),
                                                     tr.SubtractMeanImage(self.meanval),
                                                     tr.ToTensor()])

        # Testing dataset and its iterator
        # db_test = db.DAVIS2017(split=subset, transform=composed_transforms_ts, seq_name=sequence, no_gt=True, retname=True)
        db_test = db_scribblenet.DAVIS2017(split=subset, transform=composed_transforms_ts, seq_name=sequence, no_gt=True, retname=True)
        testloader = DataLoader(db_test, batch_size=self.test_batch, shuffle=False, num_workers=2)

        print('Testing Network for obj_id={}'.format(obj_id))
        print('Loading weights from objId-{}_interaction-{}_scribble-{}.pth'
              .format(obj_id, n_interaction, scribble_iter))

        # Main Testing Loop
        masks = []
        for ii, sample_batched in enumerate(testloader):

            img, gt, meta = sample_batched['image'], sample_batched['gt'], sample_batched['meta']

            # Forward of the mini-batch
            inputs, gts = Variable(img, volatile=True), Variable(gt, volatile=True)
            if self.gpu_id >= 0:
                inputs, gts = inputs.cuda(), gts.cuda()

            outputs = self.net.forward(inputs)[-1].cpu().data.numpy()

            for jj in range(int(inputs.size()[0])):
                pred = np.transpose(outputs[jj, :, :, :], (1, 2, 0))
                pred = 1 / (1 + np.exp(-pred))
                pred = np.squeeze(pred)

                if self.save_res_dir:
                    # Save the result, attention to the index jj
                    sm.imsave(os.path.join(save_dir_res, os.path.basename(meta['frame_id'][jj]) + '.png'), pred)
                masks.append(pred)

        return masks

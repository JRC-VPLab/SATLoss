import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.cubical_complex import CubicalComplex
from utils.PDMatching import SpatialAware_WassersteinDistance

class PDMatchingLoss(nn.Module):
    def __init__(self, opt, p=2):
        super().__init__()
        # Cubical complex constructor for persistent homology computation
        self.getPersistentInfo = CubicalComplex(dim=2)

        # distance between persistent diagrams
        self.criterion = SpatialAware_WassersteinDistance(p=p)

        # For precomputed ground truth persistent diagram.
        self.precal_PD = opt.precal_PD
        self.PD_target = {}
        self.pad_dims = (1, 1, 1, 1)

    def _pad_to_square(self, x1, x2, H, W):
        margin = abs(H - W)
        pad1, pad2 = margin // 2, margin - margin // 2

        if H > W:
            paddings = (pad1, pad2, 0, 0)
        else:
            paddings = (0, 0, pad1, pad2)

        if x1 is not None:
            x1 = F.pad(x1, paddings, "constant", 0.0)
        if x2 is not None:
            x2 = F.pad(x2, paddings, "constant", 0.0)

        return x1, x2

    def _pre_compute_PD(self, target, img_names):
        """
            Pre-compute and save the
        """
        # pad the boundary of the images by 1
        padded_target = F.pad(target, self.pad_dims, mode='constant', value=1)

        N, _, H, W = target.size()

        # pad the image to square
        if H != W:
            _, padded_target = self._pad_to_square(None, padded_target, H, W)

        padded_target = torch.clamp(padded_target, min=0.0, max=1.0)
        padded_target = 1.0 - padded_target

        for i in range(N):
            img = padded_target[i,0,:,:].unsqueeze(0).unsqueeze(0)
            self.PD_target[img_names[i]] = self.getPersistentInfo(img)

    def forward(self, input, target, img_names=None):
        N, C, H, W = input.size()
        assert input.size() == target.size()
        assert input.device == target.device
        assert C == 1

        self.device = input.device

        input = input.to(torch.float32)
        target = target.to(torch.float32)

        # pad the boundary of the images by 1 (see Hu et al. NIPS 19' for reasons)
        padded_input = F.pad(input, self.pad_dims, mode='constant', value=1)
        padded_target = F.pad(target, self.pad_dims, mode='constant', value=1)

        N, C, H, W = input.size()

        # pad the image to square
        if H != W:
            input, target = self._pad_to_square(padded_input, padded_target, H, W)

        input = torch.clamp(input, min=0.0, max=1.0)
        target = torch.clamp(target, min=0.0, max=1.0)

        loss = torch.tensor(0, dtype=torch.float32, device=self.device)

        # invert the image color to fit the computation in CubicalComplex (super-level filtration)
        input = 1.0 - input
        target = 1.0 - target

        pi_x = self.getPersistentInfo(input)
        if self.precal_PD:  # read ground truth persistent diagram from pre-computed
            pi_y = [self.PD_target[img_names[0]][0]]
            for idx in range(1, N):
                pi_y.append(self.PD_target[img_names[idx]][0])
        else:
            pi_y = self.getPersistentInfo(target)

        for i in range(N):
            # 0-th persistent diagram (connected components)
            pd_x_0 = pi_x[i][0][0]
            pd_y_0 = pi_y[i][0][0]

            # 1-st persistent diagram (loops)
            pd_x_1 = pi_x[i][0][1]
            pd_y_1 = pi_y[i][0][1]

            wd_0 = self.criterion(pd_x_0, pd_y_0, H, W)
            wd_1 = self.criterion(pd_x_1, pd_y_1, H, W)

            loss += (wd_0 + wd_1)

        loss /= N

        return loss
import ot
import torch
import torch.nn.functional as F

from torch_topological.utils import wrap_if_not_iterable

class SpatialAware_WassersteinDistance(torch.nn.Module):
    """
        Given the persistent diagram of the prediction and ground truth,
        compute the spatially-weighted Wasserstein distance between the PDs
    """

    def __init__(self, p=torch.inf, q=1):
        """
            p : order of the norm.
            q: order of the Wasserstein distance.
        """
        super().__init__()
        self.p = p
        self.q = q

    def _project_to_diagonal(self, diagram):
        x = diagram[:, 0]
        y = diagram[:, 1]

        return 0.5 * torch.stack(((x + y), (x + y)), 1)

    def _distance_to_diagonal(self, diagram):
        return torch.linalg.vector_norm(
            diagram - self._project_to_diagonal(diagram),
            self.p,
            dim=1
        )

    def _make_distance_matrix(self, D1, D2, C1, C2):
        # Smallest distance from (all) persistent features to the diagonal of the persistent diagram (birth=death)
        dist_D11 = self._distance_to_diagonal(D1)
        dist_D22 = self._distance_to_diagonal(D2)

        # distance matrix with global topological info only (creation and destruction time)
        PD_dist = torch.cdist(D1, D2, p=self.p)

        # spatial distance matrix
        Spatial_dist = torch.cdist(C1, C2, p=self.p)
        Spatial_dist = torch.clamp(Spatial_dist, 0.05, 1)

        # weight the distance matrix
        Weighted_dist = Spatial_dist * PD_dist

        # Extend the matrix to include also matching cost to diagonal
        upper_blocks = torch.hstack((Weighted_dist, dist_D11[:, None]))
        lower_blocks = torch.cat(
            (dist_D22, torch.tensor(0, device=dist_D22.device).unsqueeze(0))
        )
        M = torch.vstack((upper_blocks, lower_blocks))

        M = M.pow(self.q)

        return M

    def forward(self, X, Y, H, W):
        """Calculate Spatially-weighted Wasserstein metric based on input tensors.
            X, Y : Persistent diagram of class:`PersistenceInformation`
            H, W : Height and weight of the image to normalize the spatial distance
        """
        total_cost = 0.0

        X = wrap_if_not_iterable(X)
        Y = wrap_if_not_iterable(Y)

        for pers_info in zip(X, Y):
            # Persistent diagram
            D1 = pers_info[0].diagram
            D2 = pers_info[1].diagram

            # Creator pixel coordinate for each homology class
            # We don't use the destroyer since infinite persistent might occur
            C1 = pers_info[0].pairing[:,:2].float()
            C2 = pers_info[1].pairing[:,:2].float()

            # Normalize pixel coordinate (0->H, 0->W) to (0->1, 0->1)
            C1[:, 0] /= H
            C2[:, 0] /= H
            C1[:, 1] /= W
            C2[:, 1] /= W

            n = len(D1)
            m = len(D2)

            # Spatially-weighted cost matrix for the Wasserstein matching
            dist = self._make_distance_matrix(D1, D2, C1, C2)

            # weight vectors for Wasserstein matching
            a = torch.ones(n + 1, device=dist.device)
            b = torch.ones(m + 1, device=dist.device)

            a[-1] = m
            b[-1] = n

            # Wasserstein (Earth Moving Distance) computation.
            total_cost += ot.emd2(a, b, dist)

        return total_cost.pow(1.0 / self.q)
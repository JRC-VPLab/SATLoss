"""Cubical complex computation module, adapted from the torch_topological library"""
import numpy as np
import torch
from torch import nn

from torch_topological.nn import PersistenceInformation
import gudhi

class CubicalComplex(nn.Module):
    def __init__(self, superlevel=False, dim=2):
        super().__init__()
        self.superlevel = superlevel
        self.dim = dim

    def forward(self, x):
        # Dimension was provided; this makes calculating the *effective*
        # dimension of the tensor much easier: take everything but the
        # last `self.dim` dimensions.
        if self.dim is not None:
            shape = x.shape[:-self.dim]
            dims = len(shape)

        # No dimension was provided; just use the shape provided by the
        # client.
        else:
            dims = len(x.shape) - 2

        # No additional dimensions present: a single image
        if dims == 0:
            return self._forward(x)

        # Handle image with channels, such as a tensor of the form `(C, H, W)`
        elif dims == 1:
            return [
                self._forward(x_) for x_ in x
            ]

        # Handle image with channels and batch index, such as a tensor of
        # the form `(B, C, H, W)`.
        elif dims == 2:
            return [
                    [self._forward(x__) for x__ in x_] for x_ in x
            ]

    def _forward(self, x):
        if self.superlevel:
            x = -x

        cubical_complex = gudhi.CubicalComplex(
            dimensions=x.shape,
            top_dimensional_cells=x.flatten()
        )

        # We need the persistence pairs first, even though we are *not*
        # using them directly here.
        cubical_complex.persistence()
        cofaces = cubical_complex.cofaces_of_persistence_pairs()

        max_dim = len(x.shape)

        persistence_information = [
            self._extract_generators_and_diagrams(
                x,
                cofaces,
                dim
            ) for dim in range(0, max_dim)
        ]

        return persistence_information

    def _extract_generators_and_diagrams(self, x, cofaces, dim):
        pairs = torch.empty((0, 2), dtype=torch.long)

        try:
            regular_pairs = torch.as_tensor(
                cofaces[0][dim], dtype=torch.long
            )
            pairs = torch.cat(
                (pairs, regular_pairs)
            )
        except IndexError:
            pass

        try:
            infinite_pairs = torch.as_tensor(
                cofaces[1][dim], dtype=torch.long
            )
        except IndexError:
            infinite_pairs = None

        if infinite_pairs is not None:
            # 'Pair off' all the indices
            max_index = torch.argmax(x)
            fake_destroyers = torch.empty_like(infinite_pairs).fill_(max_index)

            infinite_pairs = torch.stack(
                (infinite_pairs, fake_destroyers), 1
            )

            pairs = torch.cat(
                (pairs, infinite_pairs)
            )

        return self._create_tensors_from_pairs(x, pairs, dim)

    # Internal utility function to handle the 'heavy lifting:'
    # creates tensors from sets of persistence pairs.
    def _create_tensors_from_pairs(self, x, pairs, dim):

        xs = x.shape

        # Notice that `creators` and `destroyers` refer to pixel
        # coordinates in the image.
        creators = torch.as_tensor(
                np.column_stack(
                    np.unravel_index(pairs[:, 0], xs)
                ),
                dtype=torch.long
        )
        destroyers = torch.as_tensor(
                np.column_stack(
                    np.unravel_index(pairs[:, 1], xs)
                ),
                dtype=torch.long
        )
        gens = torch.as_tensor(torch.hstack((creators, destroyers)))

        persistence_diagram = torch.stack((
            x.ravel()[pairs[:, 0]],
            x.ravel()[pairs[:, 1]]
        ), 1)

        return PersistenceInformation(
                pairing=gens,
                diagram=persistence_diagram,
                dimension=dim
        )

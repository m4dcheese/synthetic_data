"""File containing the SCM dataset logic."""
from data.scm import SCM


class SCMDataset:
    """Dataset class loading batched data from one random SCM.

    Why not torch Dataset Subclass? -> We can sample infinitely many data points that
    are already in batches. A dataloader would try to load them one by one, which is
    inefficient as we sample from a batch-capable MLP.
    """
    def __init__(self, num_x: int, num_y: int, batch_size: int):
        """Initializes a new SCM."""
        self.scm = SCM(num_x, num_y)
        self.batch_size = batch_size

    def __iter__(self):
        while True:
            yield self.scm.generate_data(size=self.batch_size)

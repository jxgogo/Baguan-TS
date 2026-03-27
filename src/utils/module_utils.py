import math
from collections.abc import Generator
import torch.nn.functional as F

def get_openai_lr(transformer_model):
    num_params = sum(p.numel() for p in transformer_model.parameters())
    return 0.003239 - 0.0001395 * math.log(num_params)

def get_cosine_schedule_with_warmup(num_training_steps, num_warmup_steps=100, num_cycles=0.5):
    """ Create a schedule with a learning rate that decreases following the
    values of the cosine function between 0 and `pi * cycles` after a warmup
    period during which it increases linearly between 0 and 1.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps)) # (current_step - 100) / (num_training_steps - 100)  # 归一化到 [0, 1]
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))) # 0.5 * (1 + cos(pi * progress)) num_warmup_steps -> num_training_steps 1 -> 0 余弦下降

    return lr_lambda

import torch
from contextlib import contextmanager


@contextmanager
def isolate_torch_rng(seed: int, device: torch.device) -> Generator[None, None, None]:
    torch_rng_state = torch.get_rng_state()
    if torch.cuda.is_available():
        torch_cuda_rng_state = torch.cuda.get_rng_state(device=device)
    torch.cuda.manual_seed(seed)
    try:
        yield
    finally:
        torch.set_rng_state(torch_rng_state)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(torch_cuda_rng_state, device=device)



def halfnormal_with_p_weight_before(
    range_max: float,
    p: float = 0.5,
) -> torch.distributions.HalfNormal:
    s = range_max / torch.distributions.HalfNormal(torch.tensor(1.0)).icdf(
        torch.tensor(p),
    )
    return torch.distributions.HalfNormal(s)


def _map_to_bucket_ix(y: torch.Tensor, borders: torch.Tensor) -> torch.Tensor:
    ix = torch.searchsorted(sorted_sequence=borders, input=y) - 1
    ix[y == borders[0]] = 0
    ix[y == borders[-1]] = len(borders) - 2
    return ix


# TODO (eddiebergman): Can probably put this back to the Bar distribution.
# However we don't really need the full BarDistribution class and this was
# put here to make that a bit more obvious in terms of what was going on.
def _cdf(logits: torch.Tensor, borders: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
    ys = ys.repeat(logits.shape[:-1] + (1,))
    n_bars = len(borders) - 1
    y_buckets = _map_to_bucket_ix(ys, borders).clamp(0, n_bars - 1).to(logits.device)

    probs = torch.softmax(logits, dim=-1)
    prob_so_far = torch.cumsum(probs, dim=-1) - probs
    prob_left_of_bucket = prob_so_far.gather(index=y_buckets, dim=-1)

    bucket_widths = borders[1:] - borders[:-1]
    share_of_bucket_left = ys - borders[y_buckets] / bucket_widths[y_buckets]
    share_of_bucket_left = share_of_bucket_left.clamp(0.0, 1.0)

    prob_in_bucket = probs.gather(index=y_buckets, dim=-1) * share_of_bucket_left
    prob_left_of_ys = prob_left_of_bucket + prob_in_bucket

    prob_left_of_ys[ys <= borders[0]] = 0.0
    prob_left_of_ys[ys >= borders[-1]] = 1.0
    return prob_left_of_ys.clip(0.0, 1.0)


def translate_probs_across_borders(
    logits: torch.Tensor,
    *,
    frm: torch.Tensor,
    to: torch.Tensor,
) -> torch.Tensor:
    """Translate the probabilities across the borders.

    Args:
        logits: The logits defining the distribution to translate.
        frm: The borders to translate from.
        to: The borders to translate to.

    Returns:
        The translated probabilities.
    """
    prob_left = _cdf(logits, borders=frm, ys=to)
    prob_left[..., 0] = 0.0
    prob_left[..., -1] = 1.0

    return (prob_left[..., 1:] - prob_left[..., :-1]).clamp_min(0.0)


class FullSupportBarDistribution(torch.nn.Module):
    def __init__(
        self,
        borders: torch.Tensor,
        ignore_nan_targets: bool = True,
    ):
        # here borders should start with min and end with max, where all values
        # lie in (min,max) and are sorted
        """:param borders:"""
        super().__init__()
        self.ignore_nan_targets = ignore_nan_targets
        self.borders = borders
        self.num_bars = len(borders) - 1
        self.bucket_widths = self.borders[1:] - self.borders[:-1]
        self.to(borders.device)
        
    def map_to_bucket_idx(self, y: torch.Tensor) -> torch.Tensor:
        # assert the borders are actually sorted
        assert (self.borders[1:] - self.borders[:-1] >= 0.0).all()
        # print(self.borders.device)
        # print(y.device)
        target_sample = torch.searchsorted(self.borders, y) - 1
        target_sample[y == self.borders[0]] = 0
        target_sample[y == self.borders[-1]] = self.num_bars - 1
        return target_sample
    
    def compute_scaled_log_probs(self, logits: torch.Tensor) -> torch.Tensor:
        # this is equivalent to log(p(y)) of the density p
        bucket_log_probs = torch.log_softmax(logits, -1)
        return bucket_log_probs - torch.log(self.bucket_widths)
    
    @staticmethod
    def halfnormal_with_p_weight_before(
        range_max: float,
        p: float = 0.5,
    ) -> torch.distributions.HalfNormal:
        s = range_max / torch.distributions.HalfNormal(torch.tensor(1.0)).icdf(
            torch.tensor(p),
        )
        return torch.distributions.HalfNormal(s)

    def forward(
        self,
        logits: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Returns the negative log density (the _loss_).

        y: T x B, logits: T x B x self.num_bars.

        :param logits: Tensor of shape T x B x self.num_bars
        :param y: Tensor of shape T x B
        :param mean_prediction_logits:
        :return:
        """
        self.borders = self.borders.to(logits.device)
        self.bucket_widths = self.bucket_widths.to(logits.device)

        bucket_means = self.borders[:-1] + self.bucket_widths / 2 # shape: num_bars
        bucket_means = bucket_means.unsqueeze(0).unsqueeze(0).expand_as(logits)  # shape: T x B x num_bars
        bucket_means = bucket_means.to(logits.device)
        

        assert self.num_bars > 1
        y = y.clone().view(*logits.shape[:-1])  # no trailing one dimension
        target_sample = self.map_to_bucket_idx(y)  # shape: T x B (same as y)
        target_sample.clamp_(0, self.num_bars - 1)

        assert (
            logits.shape[-1] == self.num_bars
        ), f"{logits.shape[-1]} vs {self.num_bars}"
        assert (target_sample >= 0).all()
        assert (
            target_sample < self.num_bars
        ).all(), f"y {y} not in support set for borders (min_y, max_y) {self.borders}"
        last_dim = logits.shape[-1]
        assert last_dim == self.num_bars, f"{last_dim} vs {self.num_bars}"
        # ignore all position with nan values

        scaled_bucket_log_probs = self.compute_scaled_log_probs(logits) # shape: T x B x num_bars

        assert len(scaled_bucket_log_probs) == len(target_sample), (
            len(scaled_bucket_log_probs),
            len(target_sample),
        )
        log_probs = scaled_bucket_log_probs.gather(
            -1,
            target_sample.unsqueeze(-1),
        ).squeeze(-1)

        target_bucket_mean = bucket_means.gather(
            -1,
            target_sample.unsqueeze(-1),
        ) # shape: T x B x 1
        bucket_distances = bucket_means - target_bucket_mean # shape: T x B x num_bars
        bucket_distances = bucket_distances ** 2 # shape: T x B x num_bars
        std = torch.mean(bucket_distances * torch.exp(scaled_bucket_log_probs), dim=-1) # shape: T x B

        side_normals = (
            self.halfnormal_with_p_weight_before(self.bucket_widths[0]),
            self.halfnormal_with_p_weight_before(self.bucket_widths[-1]),
        )

        log_probs[target_sample == 0] += side_normals[0].log_prob(
            (self.borders[1] - y[target_sample == 0]).clamp(min=0.00000001),
        ) + torch.log(self.bucket_widths[0])
        log_probs[target_sample == self.num_bars - 1] += side_normals[1].log_prob(
            (y[target_sample == self.num_bars - 1] - self.borders[-2]).clamp(
                min=0.00000001,
            ),
        ) + torch.log(self.bucket_widths[-1])

        nll_loss = -log_probs



        return nll_loss #+ std


class EntropyLoss(torch.nn.Module):
    def __init__(self, borders: torch.Tensor = None):
        super().__init__()
        self.borders = borders if borders is not None else torch.linspace(-10, 10, 5001)
        self.num_bars = len(borders) - 1
        self.bucket_widths = self.borders[1:] - self.borders[:-1]

    def map_to_bucket_idx(self, y: torch.Tensor) -> torch.Tensor:
        assert (self.borders[1:] - self.borders[:-1] >= 0.0).all()
        target_sample = torch.searchsorted(self.borders, y) - 1
        target_sample[y == self.borders[0]] = 0
        target_sample[y == self.borders[-1]] = self.num_bars - 1
        return target_sample


    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self.borders = self.borders.to(logits.device)
        assert self.num_bars > 1
        y = y.clone().view(*logits.shape[:-1])  # no trailing one dimension
        target_sample = self.map_to_bucket_idx(y)  # shape: T x B (same as y)
        target_sample.clamp_(0, self.num_bars - 1)

        return torch.nn.functional.cross_entropy(logits.permute(0, 4, 1, 2, 3), target_sample, reduction='mean')


class CRLoss(torch.nn.Module):
    def __init__(self, borders: torch.Tensor = None):
        super().__init__()
        self.borders = borders if borders is not None else torch.linspace(-10, 10, 5001)
        self.num_bars = len(self.borders) - 1
        self.bin_centers = (self.borders[:-1] + self.borders[1:]) / 2  # shape: [num_bars]
        self.centers = self.bin_centers.unsqueeze(0)  # [1, K]
        centers_expanded = self.bin_centers.unsqueeze(1)  # [K, 1]
        self.abs_diff_centers = (self.centers - centers_expanded).abs()  # [K, K]

    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        device = logits.device
        self.abs_diff_centers = self.abs_diff_centers.to(device)
        self.centers = self.centers.to(device)
        prob = F.softmax(logits, dim=-1)  # shape: [..., num_bars]
        prob_flat = prob.view(-1, self.num_bars)  # [N, K]
        y_flat = y.view(-1)                        # [N]
        diff_to_y = (self.centers - y_flat.unsqueeze(1)).abs()  # [N, K]
        term1 = (prob_flat * diff_to_y).sum(dim=1)  # [N]
        Dp = torch.matmul(prob_flat, self.abs_diff_centers)
        term2 = (prob_flat * Dp).sum(dim=1)
        crps = term1 - 0.5 * term2  # [N]
        return crps.mean()


class z_score_normalizer:
    def __init__(self):
        self.mean = None
        self.var = None
    def fit(self, X, dim=0):
        self.mean = torch.nanmean(X, dim=dim, keepdim=True)
        mean2 = torch.nanmean(X**2, dim=dim, keepdim=True)
        var = (mean2 - self.mean**2).clamp(min=0)
        self.std = torch.sqrt(var)
        self.std = torch.where(self.std<1e-8,torch.ones_like(self.std),self.std)
    def transform(self, X):
        return (X-self.mean)/(self.std)
    def transform_inverse(self, X):
        return X * (self.std) + self.mean
    def fit_transform(self, X, dim=0):
        self.fit(X,dim=dim)
        return self.transform(X)
    def fit_transform_inverse(self, X, dim=0):
        return self.transform_inverse(X)

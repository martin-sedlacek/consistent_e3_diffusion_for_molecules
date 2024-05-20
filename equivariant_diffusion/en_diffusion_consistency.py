from equivariant_diffusion import utils
import numpy as np
import math
import torch
from egnn import models
from torch.nn import functional as F
from equivariant_diffusion import utils as diffusion_utils
from equivariant_diffusion.en_diffusion import (PredefinedNoiseSchedule, GammaNetwork, sum_except_batch, expm1,
                                                softplus, gaussian_KL, gaussian_KL_for_dimension,
                                                cdf_standard_gaussian, EnVariationalDiffusion)


class ConsistentEnVariationalDiffusion(EnVariationalDiffusion):
    def __init__(
            self,
            dynamics: models.EGNN_dynamics_QM9,
            in_node_nf: int,
            n_dims: int,
            timesteps: int = 1000,
            parametrization='eps',
            noise_schedule='learned',
            noise_precision=1e-4,
            loss_type='vlb',
            norm_values=(1., 1., 1.),
            norm_biases=(None, 0., 0.),
            include_charges=True
    ):
        super().__init__(dynamics, in_node_nf, n_dims, timesteps, parametrization, noise_schedule, noise_precision,
                         loss_type, norm_values, norm_biases, include_charges)

    def compute_loss(self, x, h, node_mask, edge_mask, context, t0_always, generative_model_ema=None, N=None, boundaries=None):
        """Computes an estimator for the variational lower bound, or the simple loss (MSE)."""

        # This part is about whether to include loss term 0 always.
        if t0_always:
            # loss_term_0 will be computed separately.
            # estimator = loss_0 + loss_t,  where t ~ U({1, ..., T})
            lowest_t = 1
        else:
            # estimator = loss_t,           where t ~ U({0, ..., T})
            lowest_t = 0

        # Sample a timestep t.
        #t_int = torch.randint(
        #    lowest_t, self.T + 1, size=(x.size(0), 1), device=x.device).float()
        #s_int = t_int - 1
        #t_is_zero = (t_int == 0).float()  # Important to compute log p(x | z0).

        bs, n_nodes, n_dims = x.size()
        rand_int_t = torch.randint(0, N - 1, (bs, 1), device=x.device)
        s_int = boundaries[rand_int_t]
        t_int = boundaries[rand_int_t + 1]

        t_is_zero = (rand_int_t == 0).float()  # Important to compute log p(x | z0).

        # Normalize t to [0, 1]. Note that the negative
        # step of s will never be used, since then p(x | z0) is computed.
        s = s_int / self.T
        t = t_int / self.T

        # Compute gamma_s and gamma_t via the network. (NOISE)
        gamma_s = self.inflate_batch_array(self.gamma(s), x)
        gamma_t = self.inflate_batch_array(self.gamma(t), x)

        # Compute alpha_t and sigma_t from gamma.
        alpha_t = self.alpha(gamma_t, x)
        sigma_t = self.sigma(gamma_t, x)

        # Compute alpha_s and sigma_s from gamma.
        alpha_s = self.alpha(gamma_s, x)
        sigma_s = self.sigma(gamma_s, x)

        # Sample zt ~ Normal(alpha_t x, sigma_t)
        eps = self.sample_combined_position_feature_noise(n_samples=x.size(0), n_nodes=x.size(1), node_mask=node_mask)

        # Concatenate x, h[integer] and h[categorical].
        xh = torch.cat([x, h['categorical'], h['integer']], dim=2)

        # Sample z_t given x, h for timestep t, from q(z_t | x, h)
        z_t = alpha_t * xh + sigma_t * eps
        z_s = alpha_s * xh + sigma_s * eps

        diffusion_utils.assert_mean_zero_with_mask(z_t[:, :, :self.n_dims], node_mask)
        diffusion_utils.assert_mean_zero_with_mask(z_s[:, :, :self.n_dims], node_mask)

        net_out = self.make_pred(z_t, t, t_int, node_mask, edge_mask, context)
        with torch.no_grad():
            if self.training:
                pred_ema = generative_model_ema.make_pred(z_s, s, s_int, node_mask, edge_mask, context)
            else:
                pred_ema = xh

        # Compute the error.
        error = self.compute_error(net_out, gamma_t, pred_ema)

        if self.training and self.loss_type == 'l2':
            SNR_weight = torch.ones_like(error)
        else:
            # Compute weighting with SNR: (SNR(s-t) - 1) for epsilon parametrization.
            SNR_weight = (self.SNR(gamma_s - gamma_t) - 1).squeeze(1).squeeze(1)
        assert error.size() == SNR_weight.size()
        loss_t_larger_than_zero = 0.5 * SNR_weight * error

        # The _constants_ depending on sigma_0 from the
        # cross entropy term E_q(z0 | x) [log p(x | z0)].
        neg_log_constants = -self.log_constants_p_x_given_z0(x, node_mask)

        # Reset constants during training with l2 loss.
        if self.training and self.loss_type == 'l2':
            neg_log_constants = torch.zeros_like(neg_log_constants)

        # The KL between q(z1 | x) and p(z1) = Normal(0, 1). Should be close to zero.
        kl_prior = self.kl_prior(xh, node_mask)

        # TODO: check this for ocnsistency
        # Combining the terms
        if t0_always:
            loss_t = loss_t_larger_than_zero
            num_terms = self.T  # Since t=0 is not included here.
            estimator_loss_terms = num_terms * loss_t

            # Compute noise values for t = 0.
            t_zeros = torch.zeros_like(s)
            gamma_0 = self.inflate_batch_array(self.gamma(t_zeros), x)
            alpha_0 = self.alpha(gamma_0, x)
            sigma_0 = self.sigma(gamma_0, x)

            # Sample z_0 given x, h for timestep t, from q(z_t | x, h)
            eps_0 = self.sample_combined_position_feature_noise(
                n_samples=x.size(0), n_nodes=x.size(1), node_mask=node_mask)
            z_0 = alpha_0 * xh + sigma_0 * eps_0

            net_out = self.phi(z_0, t_zeros, node_mask, edge_mask, context)

            loss_term_0 = -self.log_pxh_given_z0_without_constants(
                x, h, z_0, gamma_0, eps_0, net_out, node_mask)

            assert kl_prior.size() == estimator_loss_terms.size()
            assert kl_prior.size() == neg_log_constants.size()
            assert kl_prior.size() == loss_term_0.size()

            loss = kl_prior + estimator_loss_terms + neg_log_constants + loss_term_0

        else:
            # Computes the L_0 term (even if gamma_t is not actually gamma_0)
            # and this will later be selected via masking.
            loss_term_0 = -self.log_pxh_given_z0_without_constants(x, h, z_t, gamma_t, eps, net_out, node_mask)

            t_is_not_zero = 1 - t_is_zero

            loss_t = loss_term_0 * t_is_zero.squeeze() + t_is_not_zero.squeeze() * loss_t_larger_than_zero

            # Only upweigh estimator if using the vlb objective.
            if self.training and self.loss_type == 'l2':
                estimator_loss_terms = loss_t
            else:
                num_terms = self.T + 1  # Includes t = 0.
                estimator_loss_terms = num_terms * loss_t

            assert kl_prior.size() == estimator_loss_terms.size()
            assert kl_prior.size() == neg_log_constants.size()

            loss = kl_prior + estimator_loss_terms + neg_log_constants

        assert len(loss.shape) == 1, f'{loss.shape} has more than only batch dim.'

        return loss, {'t': t_int.squeeze(), 'loss_t': loss.squeeze(),
                      'error': error.squeeze()}

    def make_pred(self, z_t, t, t_int, node_mask, edge_mask, context):
        diffusion_utils.assert_mean_zero_with_mask(z_t[:, :, :self.n_dims], node_mask)

        net_out = self.phi(z_t, t, node_mask, edge_mask, context)

        orig_eps = 0.002
        t = t_int - orig_eps  # should we use t_int or something else?
        c_skip_t = 0.25 / (t.pow(2) + 0.25)
        c_out_t = 0.25 * t / ((t + orig_eps).pow(2) + 0.25).pow(0.5)
        net_out = c_skip_t[:, :, None] * z_t + c_out_t[:, :, None] * net_out
        return net_out

    def forward(self, x, h, node_mask=None, edge_mask=None, context=None, generative_model_ema=None, N=None, boundaries=None):
        """
        Computes the loss (type l2 or NLL) if training. And if eval then always computes NLL.
        """
        # Normalize data, take into account volume change in x.
        x, h, delta_log_px = self.normalize(x, h, node_mask)

        # Reset delta_log_px if not vlb objective.
        if self.training and self.loss_type == 'l2':
            delta_log_px = torch.zeros_like(delta_log_px)

        if self.training:
            # Only 1 forward pass when t0_always is False.
            loss, loss_dict = self.compute_loss(x, h, node_mask, edge_mask, context, t0_always=False,
                                                generative_model_ema=generative_model_ema, N=N, boundaries=boundaries)
        else:
            # Less variance in the estimator, costs two forward passes.
            loss, loss_dict = self.compute_loss(x, h, node_mask, edge_mask, context, t0_always=True,
                                                generative_model_ema=generative_model_ema, N=N, boundaries=boundaries)

        neg_log_pxh = loss

        # Correct for normalization on x.
        assert neg_log_pxh.size() == delta_log_px.size()
        neg_log_pxh = neg_log_pxh - delta_log_px

        return neg_log_pxh

    # TODO: this needs to work correctly for consistency models
    @torch.no_grad()
    def sample(self, n_samples, n_nodes, node_mask, edge_mask, context, fix_noise=False):
        """
        Draw samples from the generative model.
        """
        if fix_noise:
            # Noise is broadcasted over the batch axis, useful for visualizations.
            z = self.sample_combined_position_feature_noise(1, n_nodes, node_mask)
        else:
            z = self.sample_combined_position_feature_noise(n_samples, n_nodes, node_mask)

        diffusion_utils.assert_mean_zero_with_mask(z[:, :, :self.n_dims], node_mask)

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        #for s in reversed(range(0, self.T)):
        s = 0  # TODO: this might be bullshit, idk how to sample properly here
        s_array = torch.full((n_samples, 1), fill_value=s, device=z.device)
        t_array = s_array + 1
        s_array = s_array / self.T
        t_array = t_array / self.T

        z = self.sample_p_zs_given_zt(s_array, t_array, z, node_mask, edge_mask, context, fix_noise=fix_noise)

        # Finally sample p(x, h | z_0).
        x, h = self.sample_p_xh_given_z0(z, node_mask, edge_mask, context, fix_noise=fix_noise)

        diffusion_utils.assert_mean_zero_with_mask(x, node_mask)

        max_cog = torch.sum(x, dim=1, keepdim=True).abs().max().item()
        if max_cog > 5e-2:
            print(f'Warning cog drift with error {max_cog:.3f}. Projecting '
                  f'the positions down.')
            x = diffusion_utils.remove_mean_with_mask(x, node_mask)

        return x, h

    # TODO: add sampling for consistency models
    @torch.no_grad()
    def sample_chain(self, n_samples, n_nodes, node_mask, edge_mask, context, keep_frames=None):
        """
        Draw samples from the generative model, keep the intermediate states for visualization purposes.
        """
        z = self.sample_combined_position_feature_noise(n_samples, n_nodes, node_mask)

        diffusion_utils.assert_mean_zero_with_mask(z[:, :, :self.n_dims], node_mask)

        if keep_frames is None:
            keep_frames = self.T
        else:
            assert keep_frames <= self.T
        chain = torch.zeros((keep_frames,) + z.size(), device=z.device)

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        #for s in reversed(range(0, self.T)):
        s = 0 # TODO: again is this BS or can I do it like this in consistency?
        s_array = torch.full((n_samples, 1), fill_value=s, device=z.device)
        t_array = s_array + 1
        s_array = s_array / self.T
        t_array = t_array / self.T

        z = self.sample_p_zs_given_zt(s_array, t_array, z, node_mask, edge_mask, context)

        diffusion_utils.assert_mean_zero_with_mask(z[:, :, :self.n_dims], node_mask)

        # Write to chain tensor.
        write_index = (s * keep_frames) // self.T
        chain[write_index] = self.unnormalize_z(z, node_mask)

        # Finally sample p(x, h | z_0).
        x, h = self.sample_p_xh_given_z0(z, node_mask, edge_mask, context)

        diffusion_utils.assert_mean_zero_with_mask(x[:, :, :self.n_dims], node_mask)

        xh = torch.cat([x, h['categorical'], h['integer']], dim=2)
        chain[0] = xh  # Overwrite last frame with the resulting x and h.

        chain_flat = chain.view(n_samples * keep_frames, *z.size()[1:])

        return chain_flat


class DummyTestingEMA(torch.nn.Module):
    """
    TBA
    """

    def __init__(
            self,
            n_dims: int,
            loss_type='l2',
            norm_values=(1., 1., 1.),
            norm_biases=(None, 0., 0.),
            include_charges=True
    ):
        super().__init__()
        self.loss_type = loss_type
        self.training = False
        self.norm_values = norm_values
        self.norm_biases = norm_biases
        self.n_dims = n_dims
        self.include_charges = include_charges

    def subspace_dimensionality(self, node_mask):
        """Compute the dimensionality on translation-invariant linear subspace where distributions on x are defined."""
        number_of_nodes = torch.sum(node_mask.squeeze(2), dim=1)
        return (number_of_nodes - 1) * self.n_dims

    def normalize(self, x, h, node_mask):
        x = x / self.norm_values[0]
        delta_log_px = -self.subspace_dimensionality(node_mask) * np.log(self.norm_values[0])

        # Casting to float in case h still has long or int type.
        h_cat = (h['categorical'].float() - self.norm_biases[1]) / self.norm_values[1] * node_mask
        h_int = (h['integer'].float() - self.norm_biases[2]) / self.norm_values[2]

        if self.include_charges:
            h_int = h_int * node_mask

        # Create new h dictionary.
        h = {'categorical': h_cat, 'integer': h_int}

        return x, h, delta_log_px

    def make_pred(self, z_t, t, t_int, node_mask, edge_mask, context):
        return z_t


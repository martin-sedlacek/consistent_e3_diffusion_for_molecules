import torch


def sum_except_batch(x):
    return x.view(x.size(0), -1).sum(dim=-1)


def assert_correctly_masked(variable, node_mask):
    assert (variable * (1 - node_mask)).abs().sum().item() < 1e-8


def compute_loss_and_nll(args, generative_model, nodes_dist, x, h, node_mask, edge_mask, context):
    bs, n_nodes, n_dims = x.size()

    if args.probabilistic_model == 'diffusion':
        edge_mask = edge_mask.view(bs, n_nodes * n_nodes)

        assert_correctly_masked(x, node_mask)

        # Here x is a position tensor, and h is a dictionary with keys
        # 'categorical' and 'integer'.
        nll = generative_model(x, h, node_mask, edge_mask, context)

        N = node_mask.squeeze(2).sum(1).long()

        log_pN = nodes_dist.log_prob(N)

        assert nll.size() == log_pN.size()
        nll = nll - log_pN

        # Average over batch.
        nll = nll.mean(0)

        reg_term = torch.tensor([0.]).to(nll.device)
        mean_abs_z = 0.
    else:
        raise ValueError(args.probabilistic_model)

    return nll, reg_term, mean_abs_z


def compute_loss_and_nll_consistency(args, generative_model, generative_model_ema, nodes_dist, x, h, node_mask,
                                     edge_mask, context, boundaries, N):
    bs, n_nodes, n_dims = x.size()

    if args.probabilistic_model == 'diffusion':
        edge_mask = edge_mask.view(bs, n_nodes * n_nodes)

        # Normalize data, take into account volume change in x.
        x, h, delta_log_px = generative_model_ema.normalize(x, h, node_mask)

        assert_correctly_masked(x, node_mask)

        # TODO: 1 or 0? idk what to set this to
        #lowest_t = 0
        #t_int = torch.randint(lowest_t, generative_model.T, size=(x.size(0), 1), device=x.device).float()

        t = torch.randint(0, N - 1, (bs, 1), device=x.device)
        t_0 = boundaries[t]
        t_1 = boundaries[t + 1]

        pred = generative_model.make_pred(x, h, t_1, node_mask, edge_mask, context)
        with torch.no_grad():
            pred_ema = generative_model_ema.make_pred(x, h, t_0, node_mask, edge_mask, context)

        mse_loss = torch.nn.functional.mse_loss(pred_ema, pred)

#        return loss
#    else:
#        raise ValueError(args.probabilistic_model)

        # Reset delta_log_px if not vlb objective.
        if generative_model_ema.training and generative_model_ema.loss_type == 'l2':
            delta_log_px = torch.zeros_like(delta_log_px)

        with torch.no_grad():
            loss, loss_dict = generative_model.compute_loss(x, h, node_mask, edge_mask, context, t0_always=False)

        assert loss.size() == delta_log_px.size()
        nll = loss - delta_log_px

        N = node_mask.squeeze(2).sum(1).long()

        log_pN = nodes_dist.log_prob(N)

        assert nll.size() == log_pN.size()
        nll = nll - log_pN

        # Average over batch.
        nll = nll.mean(0)

        reg_term = torch.tensor([0.]).to(nll.device)
        mean_abs_z = 0.
    else:
        raise ValueError(args.probabilistic_model)

    return nll, reg_term, mean_abs_z, mse_loss
'''
'''
import torch

def calculate_knn_org_sim(target, windows):
    def z_normalize(x, dim=-1):
        mean = x.mean(dim=dim, keepdim=True)
        std = x.std(dim=dim, unbiased=False, keepdim=True) + 1e-8
        return (x - mean) / std

    test_z = z_normalize(target)
    cand_z = z_normalize(windows)

    similarities = torch.einsum('vmd,md->vm', cand_z, test_z).mean(dim=-1)
    # scale to merge
    return similarities / max((similarities.max() - similarities.min()), 1e-8)

def calculate_knn_cos_sim(target, windows):
    """:
    target: [1, F] or [M, F] 
    windows: [V, 1, F] or [V, M, F] 
    similarity: [V] 
    """
    # 2D and 3D
    assert target.shape[-1] == windows.shape[-1]
    # target [1, F] and windows [V, 1, F]
    if target.shape[0] == 1 and windows.shape[1] == 1: 
        target = target.squeeze(0)  # [F]
        windows = windows.squeeze(1)  # [V, F]
        cos_similarity = torch.cosine_similarity(windows, target.unsqueeze(0), dim=-1)  # [V]
    # target [M, F] and windows [V, M, F]
    else:
        cos_similarity = torch.cosine_similarity(windows, target.unsqueeze(0), dim=-1).mean(dim=-1)  # [V]
    
    return cos_similarity

def calculate_knn_l2_sim(target, windows):
    """
    target: [1, F] or [M, F] 
    windows: [V, 1, F] or [V, M, F] 
    similarity: [V] l-2 similarity (-1, 1)
    """
    # 2D and 3D
    assert target.shape[-1] == windows.shape[-1]
    if target.shape[0] == 1 and windows.shape[1] == 1:
        # target [1, F] and windows [V, 1, F]
        target = target.squeeze(0)  # [F]
        windows = windows.squeeze(1)  # [V, F]
        l2_distances = torch.norm(windows - target, p=2, dim=-1)  # [V]
        l2_similarity = -l2_distances  # [V]
    else:
        # target [M, F] and windows [V, M, F]
        l2_distances = torch.norm(windows - target.unsqueeze(0), p=2, dim=-1)  # [V, M]
        avg_l2_distances = l2_distances.mean(dim=-1)  # [V]
        l2_similarity = -avg_l2_distances  # [V]
    # (-inf, 0] -> (-1, 1]
    return torch.tanh(l2_similarity) 

def calculate_knn_similarity(target, windows, rag_type='Yscl', weights=None):
    """
    target [1, F] or [M, F]
    windows [V, 1, F] or [V, M, F] 
    return [V]
    """
    sims = []
    if 's' in rag_type:
        sims.append(calculate_knn_org_sim(target, windows))
    if 'c' in rag_type:
        sims.append(calculate_knn_cos_sim(target, windows))
    if 'l' in rag_type:
        sims.append(calculate_knn_l2_sim(target, windows))
    
    if weights is not None:
        w = torch.tensor(weights).to(target.device).to(target.dtype)
        assert len(sims) == len(weights)
        assert torch.isclose(torch.sum(w), torch.tensor(1.0))
        sims = (torch.stack(sims, dim=0) * w.reshape(-1,1)).sum(dim=0)
    else:
        sims = torch.stack(sims, dim=0).mean(dim=0)
    assert sims.shape == (windows.shape[0],)
    return sims

def get_test_with_knn_contexts(ts, M, P, K, period=1, rag_type='Yscl', rag_window_step=1, rag_w=None):
        """
        Args:
            ts: [N, F] time series tensor
            M: context window length
            P: prediction length
            K: neighbor number
            period: period length
            rag_type: 'Y' target, 'X' feature(covariates), 'B' both, ‘NA’ No RAG | XT_scl, XF_scl, Y_scl, B_scl
        """
        assert ts.dim() == 2, "ts must be [N, F]" 
        N, F = ts.shape
        if 'Y' in rag_type:
            assert P < M and N >= M
            test_sample = ts[-M:-P, -1:].reshape(1, -1)     # [M-P]
            test_sample_full = ts[-M:]       # [M, F]
        elif 'X' in rag_type:
            # select based on X
            assert N >= M, "N must be >= M"
            test_sample = ts[-M:, :-1]  # [M, (F-1)] flatten
            test_sample_full = ts[-M:]             # [M, F]
        elif 'B' in rag_type:
            # select based on X+Y
            assert P < M and N >= M
            test_sample = ts[-M:-P, :]      # [M-P]
            test_sample_full = ts[-M:]       # [M, F]
        elif "NA" in rag_type:
            # ========== No RAG ==========
            test_sample_full = ts[-M:]  # [M, F]
            # create non-overlapping contexts from -M position backwards with length M
            num_contexts = (N - M) // M
            if num_contexts == 0:
                raise ValueError(f"Not enough data to create non-overlapping contexts of length {M}")
            
            non_overlap_contexts = []
            for i in range(num_contexts):
                start_idx = num_contexts * M - (i + 1) * M
                context = ts[start_idx:start_idx + M]
                non_overlap_contexts.append(context)
            
            
            knn_contexts = torch.stack(non_overlap_contexts, dim=0)
            result = torch.cat([knn_contexts, test_sample_full.unsqueeze(0)], dim=0)
            # print(f"No RAG, data shape is {result.shape}")
            return result
        else:
            raise ValueError(f"Invalid rag_type: {rag_type}")
        assert period is not None and period > 0

    
        max_start = N - P - M 
        
        if max_start < 0:
            raise ValueError(f"Not enough data")
        
        
        valid_starts = torch.arange(0, max_start + 1, step=rag_window_step, device=ts.device)
        V = len(valid_starts)
        if V == 0:
            raise ValueError("No valid context windows")


        period_ids = valid_starts // period
        unique_period_ids, inverse_indices = torch.unique(period_ids, return_inverse=True)
        num_periods = len(unique_period_ids)


        if 'Y' in rag_type:
            offsets = torch.arange(M-P, device=ts.device).unsqueeze(0)
            window_indices = valid_starts.unsqueeze(1) + offsets
            windows_data = ts[window_indices, -1].unsqueeze(1)  # [V, M]
        elif 'XF' in rag_type:  
            offsets = torch.arange(M, device=ts.device).unsqueeze(0)
            window_indices = valid_starts.unsqueeze(1) + offsets
            windows_full = ts[window_indices]  # [V, M, F]
            windows_data = windows_full[..., :-1] #.reshape(V, -1)  # [V, M*(F-1)] flatten
        elif 'XT' in rag_type:  
            offsets = torch.arange(M, device=ts.device).unsqueeze(0)
            window_indices = valid_starts.unsqueeze(1) + offsets
            windows_full = ts[window_indices]
            windows_data = windows_full[..., :-1].permute(0, 2, 1)  # [V, F, M]
            test_sample = test_sample.permute(1, 0)
        elif 'BF' in rag_type:
            offsets = torch.arange(M-P, device=ts.device).unsqueeze(0)
            window_indices = valid_starts.unsqueeze(1) + offsets
            windows_data = ts[window_indices]  # [V, M, F]
        elif 'BT' in rag_type:
            offsets = torch.arange(M-P, device=ts.device).unsqueeze(0)
            window_indices = valid_starts.unsqueeze(1) + offsets
            windows_data = ts[window_indices].permute(0, 2, 1)
            test_sample = test_sample.permute(1, 0)



        similarities = calculate_knn_similarity(test_sample, windows_data, rag_type, weights=rag_w)


        dtype = similarities.dtype
        device = ts.device


        max_similarities = torch.full(
            (num_periods,), 
            -torch.inf,
            dtype=dtype,
            device=device
        )
        max_similarities.scatter_reduce_(
            0, 
            inverse_indices, 
            similarities, 
            reduce='amax', 
            include_self=False
        )


        indices = torch.arange(V, dtype=torch.long, device=device)
        mask = (similarities == max_similarities[inverse_indices])
        
        invalid_marker = V
        indices_masked = torch.where(mask, indices, invalid_marker)
        
        best_local_idx = torch.full(
            (num_periods,), 
            invalid_marker, 
            dtype=torch.long, 
            device=device
        )
        best_local_idx.scatter_reduce_(
            0,
            inverse_indices,
            indices_masked,
            reduce='amin',
            include_self=False
        )


        valid_mask = best_local_idx != invalid_marker
        if not torch.all(valid_mask):
            raise RuntimeError("Found invalid indices in period grouping")


        best_global_starts = valid_starts[best_local_idx]
        best_similarities = max_similarities

        K_actual = min(K, num_periods)
        _, topk_period_idx = torch.topk(best_similarities, K_actual)
        topk_starts = best_global_starts[topk_period_idx]

        knn_contexts = torch.stack([ts[i:i + M] for i in topk_starts], dim=0)
        result = torch.cat([knn_contexts, test_sample_full.unsqueeze(0)], dim=0)
        return result
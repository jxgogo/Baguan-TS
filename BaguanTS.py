from src.pipeline.factory import ModelFactory
import torch
from src.utils.rag import get_test_with_knn_contexts


class global_normalizer:
    def __init__(self,):
        self.mean = None
        self.std = None

    def fit_transform(self, X, num_test, predict_len, eps=1e-8):
        X_m = X.clone()
        max_x = torch.amax(X_m.nan_to_num(nan=-float('inf')), dim=(-3, -2), keepdim=True)
        min_x = torch.amin(X_m.nan_to_num(nan=float('inf')), dim=(-3, -2), keepdim=True)
        X_m = (X_m - min_x) / (max_x - min_x + 1e-15) # 
        # target mask
        X_m[..., -num_test:, -predict_len:, :] = float('nan')
        X_m = X_m.masked_fill(torch.isinf(X_m), float('nan'))

        #norm
        mean = torch.nanmean(X_m, dim=(-3, -2), keepdim=True)
        mean = torch.where(torch.isnan(mean), torch.zeros_like(mean), mean)
        # std = sqrt(nanmean((x - mean)^2))
        centered = X_m - mean
        var = torch.nanmean(centered ** 2, dim=(-3, -2), keepdim=True)
        var = torch.clamp(var, min=0.0)
        std = torch.sqrt(var + eps)  
        std = torch.where(torch.isnan(std) | (std < eps), torch.ones_like(std), std)

        self.std = std * (max_x - min_x + 1e-15)
        self.mean = mean * (max_x - min_x + 1e-15) + min_x
        
        assert torch.isnan(self.mean).sum() == 0 and torch.isnan(self.std).sum() == 0 

        return (X - self.mean) / self.std

    def fit(self, X, num_test, predict_len, eps=1e-8):
        X_m = X.clone()
        # target mask
        X_m[..., -num_test:, -predict_len:, :] = float('nan')
        X_m = X_m.masked_fill(torch.isinf(X_m), float('nan'))
        
        # compute mean
        mean = torch.nanmean(X_m, dim=(-3, -2), keepdim=True)
        mean = torch.where(torch.isnan(mean), torch.zeros_like(mean), mean)
        # compute standard deviation: std = sqrt(nanmean((x - mean)^2))
        centered = X_m - mean
        var = torch.nanmean(centered ** 2, dim=(-3, -2), keepdim=True)
        var = torch.clamp(var, min=0.0)
        std = torch.sqrt(var + eps) 
        std = torch.where(torch.isnan(std) | (std < eps), torch.ones_like(std), std)

        self.mean = mean
        self.std = std
        assert torch.isnan(self.mean).sum() == 0 and torch.isnan(self.std).sum() == 0 

    def transform(self, X):
        return (X - self.mean) / self.std

    def inverse_transform(self, X):
        return X * self.std  + self.mean

class BaguanTS:
    def __init__(self, ckpt_path='PATHTOCKPT.ckpt', config_path='PATHTOMODELCFG.yml', 
                        device='cuda:0'):
        self.ckpt_path = ckpt_path
        self.config_path = config_path
        self.device = device

        self.model = ModelFactory.from_config(self.config_path)
        checkpoint = torch.load(self.ckpt_path, weights_only=False)
        new_state_dict = {
            key[4:] if key.startswith('net.') else key: value for key, value in checkpoint['state_dict'].items()
            }
        self.model.load_state_dict(new_state_dict)
        self.model = self.model.to(device)
        self.model.eval()
        self.normalizer = StandardScaler()
        

    def predict_n_forward(self, feature_ts, target_ts, num_test, predict_len, quantiles, n_repeat=8, mask_per_size=8, mask_ratio=0.2):
        device = self.device
        b_c = feature_ts.shape[0]  # original batch size

        feature_ts = feature_ts.to(torch.float32).to(device)
        target_ts = target_ts.to(torch.float32).to(device)

        # ==============================
        # Step 1: Try full-batch anti-symmetric mode
        # ==============================
        try:
            # Full replication
            feat_rep = feature_ts.repeat(n_repeat, 1, 1, 1)
            targ_rep = target_ts.repeat(n_repeat, 1, 1, 1)

            feat_b = torch.cat([feat_rep, feat_rep], dim=0)
            targ_b = torch.cat([targ_rep, -targ_rep], dim=0)

            if feat_b.shape[-1] >= 5:
                f = feat_b.shape[-1]
                perm = torch.randperm(f, device=device)
                feat_b = feat_b[..., perm]

            b_total, n, s, f = targ_b.shape
            mask = torch.rand((b_total, n, s // mask_per_size, f), device=device) <= mask_ratio
            mask = mask.unsqueeze(-2).repeat(1, 1, 1, mask_per_size, 1)
            mask = mask.reshape(b_total, n, -1, f)
            if mask.shape[-2] < s:
                pad_len = s - mask.shape[-2]
                mask = torch.cat([
                    mask,
                    torch.zeros((b_total, n, pad_len, f), device=device, dtype=mask.dtype)
                ], dim=-2)
            targ_b.masked_fill_(mask > 0.5, float('nan'))

            with torch.no_grad():
                pred, pfn_pred = self.model(
                    feat_b, targ_b, num_test, predict_len,
                    quantiles=quantiles
                )

            pred = pred.cpu().squeeze(-1)
            pfn_pred = pfn_pred.cpu().squeeze(-2)

            pred_pos = pred[:b_total//2].view(n_repeat, b_c, *pred.shape[1:])
            pred_neg = pred[b_total//2:].view(n_repeat, b_c, *pred.shape[1:])
            pfn_pred_pos = pfn_pred[:b_total//2].view(n_repeat, b_c, *pfn_pred.shape[1:])
            pfn_pred_neg = pfn_pred[b_total//2:].view(n_repeat, b_c, *pfn_pred.shape[1:])

            pred_diff = (pred_pos - pred_neg) / 2
            pfn_pred_diff = (pfn_pred_pos - pfn_pred_neg.flip(-1)) / 2

            return pred_diff.mean(dim=0), pfn_pred_diff.mean(dim=0)

        except Exception as e:
            # Catch ANY exception (including DSA, cuDNN, custom errors, etc.)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[Warning] Full-batch inference failed with: {e}")
            print(f"Falling back to safe looped anti-symmetric inference (n_repeat={n_repeat}).")

            # ==============================
            # Step 2: Safe fallback — loop over n_repeat, 1 at a time (anti-symmetric preserved)
            # Each iteration: process (B, ...) -> (2*B, ...) but B is original, so total = 2*B
            # ==============================
            all_pred_diff = []
            all_pfn_pred_diff = []

            for i in range(n_repeat):
                try:
                    # One repeat: (B, ...) -> (2*B, ...)
                    feat_i = torch.cat([feature_ts, feature_ts], dim=0)      # (2B, n, s, f)
                    targ_i = torch.cat([target_ts, -target_ts], dim=0)       # (2B, n, s, f)

                    # Random feature perm
                    if feat_i.shape[-1] >= 5:
                        perm = torch.randperm(feat_i.shape[-1], device=device)
                        feat_i = feat_i[..., perm]

                    # Masking
                    b2, n, s, f = targ_i.shape
                    mask = torch.rand((b2, n, s // mask_per_size, f), device=device) <= mask_ratio
                    mask = mask.unsqueeze(-2).repeat(1, 1, 1, mask_per_size, 1)
                    mask = mask.reshape(b2, n, -1, f)
                    if mask.shape[-2] < s:
                        pad_len = s - mask.shape[-2]
                        mask = torch.cat([
                            mask,
                            torch.zeros((b2, n, pad_len, f), device=device, dtype=mask.dtype)
                        ], dim=-2)
                    targ_i.masked_fill_(mask > 0.5, float('nan'))

                    with torch.no_grad():
                        pred_i, pfn_pred_i = self.model(
                            feat_i, targ_i, num_test, predict_len, quantiles=quantiles
                        )

                    pred_i = pred_i.cpu().squeeze(-1)        # (2B, ...)
                    pfn_pred_i = pfn_pred_i.cpu().squeeze(-2)  # (2B, ...)

                    B = b_c
                    pred_pos = pred_i[:B]
                    pred_neg = pred_i[B:]
                    pfn_pred_pos = pfn_pred_i[:B]
                    pfn_pred_neg = pfn_pred_i[B:]

                    pred_diff_i = (pred_pos - pred_neg) / 2
                    pfn_pred_diff_i = (pfn_pred_pos - pfn_pred_neg.flip(-1)) / 2

                    all_pred_diff.append(pred_diff_i)
                    all_pfn_pred_diff.append(pfn_pred_diff_i)

                    # Optional: clear cache
                    del feat_i, targ_i, mask, pred_i, pfn_pred_i
                    torch.cuda.empty_cache()

                except Exception as inner_e:
                    # If even single repeat fails, skip or raise
                    print(f"[Error] Repeat {i} failed: {inner_e}. Skipping.")
                    # You could choose to raise here, but we continue for robustness
                    continue

            if not all_pred_diff:
                raise RuntimeError("All fallback repeats failed.")

            final_pred = torch.stack(all_pred_diff, dim=0).mean(dim=0)
            final_pfn_pred = torch.stack(all_pfn_pred_diff, dim=0).mean(dim=0)

            return final_pred, final_pfn_pred

    
    def predict(self, 
                X_train, 
                y_train, 
                X_test, 
                context_len=10000,
                K=20,
                period=1,
                rag_type = 'Yscl',
                rag_window_step = 1,
                rag_w = None,
                data_type='TS-tabular',
                quantiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                mF = 1,
                ):
        """
        X_train, y_train, X_test under batch size = 1, single task prediction
        """

        if data_type == 'TS-tabular':
            predict_len = X_test.shape[0]
            num_test = 1
            assert context_len > predict_len
            feature_ts = torch.cat((
                torch.from_numpy(X_train),
                torch.from_numpy(X_test),
            ), dim=0)
            target_ts = torch.cat((
                torch.from_numpy(y_train.flatten()).unsqueeze(-1),
                torch.zeros((X_test.shape[0],1)),
            ), dim=0)
            data_org = get_test_with_knn_contexts(
                                    torch.cat((feature_ts,target_ts),dim=-1),
                                    M=context_len,
                                    P=predict_len,
                                    K=K,
                                    period=period,
                                    rag_type=rag_type,
                                    rag_window_step=rag_window_step,
                                    rag_w=rag_w,
                                    ).unsqueeze(0)
            feature_ts = data_org[...,:-1] # b c t f
            target_ts = data_org[...,-1:]


        elif data_type == 'tabular':
            predict_len = 1
            num_test = X_test.shape[0]
            feature_ts = torch.cat((
                torch.from_numpy(X_train),
                torch.from_numpy(X_test),
            ), dim=0) # all_length, feature_num
            target_ts = torch.cat((
                torch.from_numpy(y_train.flatten()).unsqueeze(-1),
                torch.zeros((X_test.shape[0],1)),
            ), dim=0)
            feature_ts = feature_ts.unsqueeze(0).unsqueeze(-2) # 1, all_length, 1, feature_num
            target_ts = target_ts.unsqueeze(0).unsqueeze(-2) # 1, all_length, 1, 1
           

        # Normalization - Global and Y in Slice/Patch
        normalizer = global_normalizer()
        feature_ts = normalizer.fit_transform(feature_ts, num_test, predict_len)
        target_ts = normalizer.fit_transform(target_ts, num_test, predict_len)

        
        self.model.eval()
        with torch.no_grad():
            # add randomness and multiple forwards
            if mF >= 1:
                pred, pfn_pred = self.predict_n_forward(feature_ts, target_ts, num_test, predict_len, quantiles, n_repeat=mF)
            else:
                b = feature_ts.shape[0]
                feature_ts_b = torch.cat((feature_ts,feature_ts)).to(torch.float32).to(self.device)
                target_ts_b = torch.cat((target_ts,-target_ts)).to(torch.float32).to(self.device)
                pred, pfn_pred = self.model(feature_ts_b, target_ts_b, num_test, predict_len, quantiles=quantiles)
                pred, pfn_pred = pred.cpu().squeeze(-1), pfn_pred.cpu().squeeze(-2)
                pred = (pred[:b,...] - pred[b:,...])/2
                pfn_pred = (pfn_pred[:b,...] - pfn_pred[b:,...].flip(-1)) / 2
        pred = normalizer.inverse_transform(pred).squeeze((-3, -2, -1)).numpy()
        pfn_pred = normalizer.inverse_transform(pfn_pred).squeeze(-3, -2).numpy()
        return pred.squeeze(0), pfn_pred.squeeze(0)
        


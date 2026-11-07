# Implementation Plan - Fix for Systematic Bias in Probabilistic Loss Strategy

## Description
When running fits using `--probabilistic_sampling_target`, the target is stochastically sampled from the probabilistic model. Stochastic fluctuations (e.g., Poisson/Bernoulli under-fluctuations) can cause a hit to be missed. In the current implementation of `ProbabilisticLossStrategy.compute`, observed hits are matched to predicted slots using `trigger_nb = compute_occurrence_indices(target_pixel_ids)`, which assigns sequential occurrence indices (0, 1, 2, ...) to the target hits on each pixel.

If a hit is missed in the target realization, the index matching shifts for all subsequent hits on that pixel by $-1$. This mismatch compares observed target hits against incorrect predicted simulation slots (which expect hits at different times/charges), resulting in near-zero likelihoods and introducing a massive systematic penalty. The optimizer is forced to bias charge-related parameters (specifically `kb` and `lifetime`) away from their true values to minimize this mismatch penalty.

To eliminate this systematic bias, we propose replacing the hard occurrence-index matching with a **marginalized sum (logsumexp) over all predicted slots**. This match-free formulation is physically and mathematically correct for a Poisson Point Process, as the total intensity at any tick is simply the sum of intensities of all possible hits on that pixel.

---

## User Review Required
We are modifying the core loss computation in `ProbabilisticLossStrategy.compute` in [strategies.py](file:///sdf/group/neutrino/pgranger/larnd-sim-jax/optimize/strategies.py). 
This change will only affect the likelihood calculation when `apply_deadtime=False` and `first_hit_only=False` (which is the default configuration for standard 1D parameter fits). If either of those options is enabled, the loss will fallback to the original sequential slot matching logic to preserve backward compatibility.

> [!NOTE]
> Per the user request, this work is being carried out on a dedicated branch: `marginalized_loss`.

> [!IMPORTANT]
> This mathematically correct marginalized formulation eliminates the need to match a target hit to a specific sequential slot number. By summing probabilities over all predicted slots, the loss function is robust to stochastic hit omissions.

---

## Proposed Changes

### Calibration Loss Strategies

#### [MODIFY] [strategies.py](file:///sdf/group/neutrino/pgranger/larnd-sim-jax/optimize/strategies.py)

Modify the `compute` method in `ProbabilisticLossStrategy` to use marginalized slot summation when sequential tracking (deadtime/first-hit) is not requested.

```diff
         trigger_nb = compute_occurrence_indices(target_pixel_ids)
         
         # Identify which target hits are valid and relevant for the current mode
         is_relevant_target_hit = (target_pixel_ids >= 0)
         if self.first_hit_only:
             is_relevant_target_hit = is_relevant_target_hit & (trigger_nb == 0)
         
         # Update match validity to only include relevant hits
         pixel_match_valid = pixel_match_valid & is_relevant_target_hit
         
         Nticks = ticks_prob.shape[-1]
         
-        if self.apply_deadtime:
-            prev_target_ticks = jnp.roll(target_ticks, shift=1)
-            tick_indices = jnp.arange(Nticks)
-            
-            # Mask out the impossible region for the second hit onwards
-            is_valid_mask = jnp.where(
-                trigger_nb[:, None] == 0,
-                True,
-                tick_indices[None, :] > (prev_target_ticks[:, None] + self.deadtime_ticks)
-            )
-            
-            # Renormalize probability mass in the valid zone
-            ticks_prob_slice = ticks_prob[pixel_indices_safe, trigger_nb, :]
-            log_Z_all = jax.nn.logsumexp(ticks_prob_slice, axis=-1)
-            masked_ticks_prob = jnp.where(is_valid_mask, ticks_prob_slice, self.min_log_prob)
-            log_Z_valid = jax.nn.logsumexp(masked_ticks_prob, axis=-1)
-            log_renorm_boost = log_Z_all - log_Z_valid
-            
-            # Identify if the target hit itself violates the deadtime rule
-            is_valid_target_hit = jnp.where(
-                trigger_nb == 0,
-                True,
-                target_ticks > (prev_target_ticks + self.deadtime_ticks)
-            )
-        else:
-            log_renorm_boost = 0.0
-            is_valid_target_hit = jnp.ones_like(target_ticks, dtype=bool)
-
-        if self.time_window > 0:
-            # Create a window of ticks around each target hit
-            # shape: (Nhits, 2*W + 1)
-            delta = jnp.arange(-self.time_window, self.time_window + 1)
-            window_ticks = target_ticks[:, None] + delta[None, :]
-            window_ticks = jnp.clip(window_ticks, 0, Nticks - 1)
-            
-            # Extract log-probs for the window
-            # shape: (Nhits, 2*W + 1)
-            window_log_probs = ticks_prob[pixel_indices_safe[:, None], trigger_nb[:, None], window_ticks]
-            
-            # Expected charge at each tick in the window
-            window_expected_charges = adc2charge(adcs_distrib[pixel_indices_safe[:, None], trigger_nb[:, None], window_ticks], params)
-            
-            # Compute Gaussian log-likelihood of charge FOR EACH TICK in the window
-            charge_diffs = target_charge[:, None] - window_expected_charges
-            window_log_charge_intensity = (
-                -0.5 * (charge_diffs / (self.sigma_charge/1000)) ** 2 
-                - 0.5 * jnp.log(2 * jnp.pi * (self.sigma_charge/1000)**2)
-            )
-            
-            # Total joint log probability for each tick in the window
-            # log P(t | pixel) + log K(t_obs - t) + log P(q_obs | q_pred(t))
-            joint_window_log_probs = window_log_probs + self.log_time_weights[None, :] + window_log_charge_intensity
-            
-            # Marginalize over the time window using logsumexp
-            joint_hit_log_probs = jax.nn.logsumexp(joint_window_log_probs, axis=1)
-            
-        else:
-            # Exact point-wise extraction
-            hit_tick_probs = ticks_prob[pixel_indices_safe, trigger_nb, target_ticks]
-            hit_expected_charges = adc2charge(adcs_distrib[pixel_indices_safe, trigger_nb, target_ticks], params)
-            
-            charge_diff = target_charge - hit_expected_charges
-            log_charge_intensity = (
-                -0.5 * (charge_diff / (self.sigma_charge/1000)) ** 2 
-                - 0.5 * jnp.log(2 * jnp.pi * (self.sigma_charge/1000)**2)
-            )
-            joint_hit_log_probs = hit_tick_probs + log_charge_intensity
-        
-        if self.apply_deadtime:
-            # If the target hit is inside the blacklist zone, give it a massive penalty.
-            # Otherwise, give it the standard log-likelihood + the renormalization boost.
-            joint_hit_log_probs = jnp.where(
-                is_valid_target_hit, 
-                joint_hit_log_probs + log_renorm_boost, 
-                self.min_log_prob
-            )
+        if self.apply_deadtime or self.first_hit_only:
+            if self.apply_deadtime:
+                prev_target_ticks = jnp.roll(target_ticks, shift=1)
+                tick_indices = jnp.arange(Nticks)
+                
+                # Mask out the impossible region for the second hit onwards
+                is_valid_mask = jnp.where(
+                    trigger_nb[:, None] == 0,
+                    True,
+                    tick_indices[None, :] > (prev_target_ticks[:, None] + self.deadtime_ticks)
+                )
+                
+                # Renormalize probability mass in the valid zone
+                ticks_prob_slice = ticks_prob[pixel_indices_safe, trigger_nb, :]
+                log_Z_all = jax.nn.logsumexp(ticks_prob_slice, axis=-1)
+                masked_ticks_prob = jnp.where(is_valid_mask, ticks_prob_slice, self.min_log_prob)
+                log_Z_valid = jax.nn.logsumexp(masked_ticks_prob, axis=-1)
+                log_renorm_boost = log_Z_all - log_Z_valid
+                
+                # Identify if the target hit itself violates the deadtime rule
+                is_valid_target_hit = jnp.where(
+                    trigger_nb == 0,
+                    True,
+                    target_ticks > (prev_target_ticks + self.deadtime_ticks)
+                )
+            else:
+                log_renorm_boost = 0.0
+                is_valid_target_hit = jnp.ones_like(target_ticks, dtype=bool)
+
+            if self.time_window > 0:
+                delta = jnp.arange(-self.time_window, self.time_window + 1)
+                window_ticks = target_ticks[:, None] + delta[None, :]
+                window_ticks = jnp.clip(window_ticks, 0, Nticks - 1)
+                
+                window_log_probs = ticks_prob[pixel_indices_safe[:, None], trigger_nb[:, None], window_ticks]
+                window_expected_charges = adc2charge(adcs_distrib[pixel_indices_safe[:, None], trigger_nb[:, None], window_ticks], params)
+                
+                charge_diffs = target_charge[:, None] - window_expected_charges
+                window_log_charge_intensity = (
+                    -0.5 * (charge_diffs / (self.sigma_charge/1000)) ** 2 
+                    - 0.5 * jnp.log(2 * jnp.pi * (self.sigma_charge/1000)**2)
+                )
+                joint_window_log_probs = window_log_probs + self.log_time_weights[None, :] + window_log_charge_intensity
+                joint_hit_log_probs = jax.nn.logsumexp(joint_window_log_probs, axis=1)
+                
+            else:
+                hit_tick_probs = ticks_prob[pixel_indices_safe, trigger_nb, target_ticks]
+                hit_expected_charges = adc2charge(adcs_distrib[pixel_indices_safe, trigger_nb, target_ticks], params)
+                
+                charge_diff = target_charge - hit_expected_charges
+                log_charge_intensity = (
+                    -0.5 * (charge_diff / (self.sigma_charge/1000)) ** 2 
+                    - 0.5 * jnp.log(2 * jnp.pi * (self.sigma_charge/1000)**2)
+                )
+                joint_hit_log_probs = hit_tick_probs + log_charge_intensity
+            
+            if self.apply_deadtime:
+                joint_hit_log_probs = jnp.where(
+                    is_valid_target_hit, 
+                    joint_hit_log_probs + log_renorm_boost, 
+                    self.min_log_prob
+                )
+        else:
+            # Marginalized sum over all slots
+            Nhits_sim = ticks_prob.shape[1]
+            
+            if self.time_window > 0:
+                delta = jnp.arange(-self.time_window, self.time_window + 1)
+                window_ticks = target_ticks[:, None] + delta[None, :]
+                window_ticks = jnp.clip(window_ticks, 0, Nticks - 1)
+                
+                # shape: (Nhits_target, Nhits_sim, 2*W + 1)
+                window_log_probs = ticks_prob[
+                    pixel_indices_safe[:, None, None],
+                    jnp.arange(Nhits_sim)[None, :, None],
+                    window_ticks[:, None, :]
+                ]
+                window_expected_charges = adc2charge(
+                    adcs_distrib[
+                        pixel_indices_safe[:, None, None],
+                        jnp.arange(Nhits_sim)[None, :, None],
+                        window_ticks[:, None, :]
+                    ],
+                    params
+                )
+                
+                charge_diffs = target_charge[:, None, None] - window_expected_charges
+                window_log_charge_intensity = (
+                    -0.5 * (charge_diffs / (self.sigma_charge/1000)) ** 2 
+                    - 0.5 * jnp.log(2 * jnp.pi * (self.sigma_charge/1000)**2)
+                )
+                joint_window_log_probs = (
+                    window_log_probs
+                    + self.log_time_weights[None, None, :]
+                    + window_log_charge_intensity
+                )
+                # Marginalize over slots (axis 1) and time window (axis 2)
+                joint_hit_log_probs = jax.nn.logsumexp(joint_window_log_probs, axis=(1, 2))
+            else:
+                # shape: (Nhits_target, Nhits_sim)
+                hit_tick_probs = ticks_prob[
+                    pixel_indices_safe[:, None],
+                    jnp.arange(Nhits_sim)[None, :],
+                    target_ticks[:, None]
+                ]
+                hit_expected_charges = adc2charge(
+                    adcs_distrib[
+                        pixel_indices_safe[:, None],
+                        jnp.arange(Nhits_sim)[None, :],
+                        target_ticks[:, None]
+                    ],
+                    params
+                )
+                
+                charge_diff = target_charge[:, None] - hit_expected_charges
+                log_charge_intensity = (
+                    -0.5 * (charge_diff / (self.sigma_charge/1000)) ** 2 
+                    - 0.5 * jnp.log(2 * jnp.pi * (self.sigma_charge/1000)**2)
+                )
+                joint_slot_log_probs = hit_tick_probs + log_charge_intensity
+                # Marginalize over slots (axis 1)
+                joint_hit_log_probs = jax.nn.logsumexp(joint_slot_log_probs, axis=1)
```

---

## Verification Plan

### Automated Tests
1. **Unit Test for Loss Strategy**: We will write a test script in the scratch directory `/sdf/home/p/pgranger/.gemini/antigravity-cli/brain/c72f957b-eef3-451c-9987-d4c6713c615e/scratch/test_marginalized_loss.py`. It will:
   - Create mock target outputs and simulation predictions with multi-hit scenarios (including stochastically omitted hits).
   - Compute loss using the original `ProbabilisticLossStrategy` implementation vs the new marginalized implementation.
   - Verify that when a hit is omitted in the target, the marginalized loss does not receive a massive penalty and tracks the true parameter gradient correctly, while the original loss is heavily biased.
   - Verify that when there is no missed hit, the marginalized loss behaves close to the original loss.
   
2. **Execute Fit Test**: Run a quick optimization test (e.g. 50 iterations) using Apptainer on one of the parameters (e.g. `kb` or `lifetime`) to confirm the code runs without JAX tracing or JIT compilation errors.
   - `apptainer exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch /sdf/group/neutrino/pgranger/larnd-sim-jax/larndsim-jax_main.sif python3 -m optimize.example_run --params kb --iterations 50 --probabilistic_sim --probabilistic_sampling_target --loss_fn llhd --data_sz 1 --max_nbatch 1 --max_batch_len 200`

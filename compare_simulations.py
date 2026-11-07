import h5py
import numpy as np
import sys

def load_results(fname):
    with h5py.File(fname, 'r') as f:
        data = {'eventID': [], 'adc_clean': [], 'pixels': [], 'ticks': [], 'Q': []}
        for batch_idx in f.keys():
            group = f[batch_idx]
            # Under each batch, there are event groups: e.g. event_0, event_1...
            for event_key in group.keys():
                event_group = group[event_key]
                data['eventID'].append(event_group['eventID'][:])
                data['pixels'].append(event_group['pixels'][:])
                data['adc_clean'].append(event_group['adc_clean'][:])
                data['ticks'].append(event_group['ticks'][:])
                if 'Q' in event_group:
                    data['Q'].append(event_group['Q'][:])
                else:
                    data['Q'].append(np.zeros_like(event_group['adc_clean'][:]))
                    
    data['eventID'] = np.concatenate(data['eventID'])
    data['pixels'] = np.concatenate(data['pixels'])
    data['adc_clean'] = np.concatenate(data['adc_clean'])
    data['ticks'] = np.concatenate(data['ticks'])
    data['Q'] = np.concatenate(data['Q'])
    return data

def main():
    if len(sys.argv) < 3:
        print("Usage: python compare_simulations.py <target_h5> <guess_h5>")
        sys.exit(1)
        
    tgt_path = sys.argv[1]
    guess_path = sys.argv[2]
    
    print(f"Loading target results from: {tgt_path}")
    tgt = load_results(tgt_path)
    
    print(f"Loading guess results from: {guess_path}")
    guess = load_results(guess_path)
    
    # 1. Overall comparison
    print("\n=================== OVERALL SUMMARY ===================")
    print(f"{'Metric':<30} | {'Target':<15} | {'Guess':<15} | {'Ratio (Guess/Tgt)':<18}")
    print("-" * 85)
    
    tgt_pixels = len(tgt['adc_clean'])
    guess_pixels = len(guess['adc_clean'])
    print(f"{'Total Active Pixels-Ticks':<30} | {tgt_pixels:<15,d} | {guess_pixels:<15,d} | {guess_pixels/tgt_pixels:<18.4f}")
    
    # Unique pixels activated
    tgt_uniq_pixels = len(np.unique(tgt['pixels']))
    guess_uniq_pixels = len(np.unique(guess['pixels']))
    print(f"{'Unique Pixels Activated':<30} | {tgt_uniq_pixels:<15,d} | {guess_uniq_pixels:<15,d} | {guess_uniq_pixels/tgt_uniq_pixels:<18.4f}")
    
    # Total ADC (sum of clean adcs)
    tgt_tot_adc = np.sum(tgt['adc_clean'])
    guess_tot_adc = np.sum(guess['adc_clean'])
    print(f"{'Total Clean ADC sum':<30} | {tgt_tot_adc:<15,.2f} | {guess_tot_adc:<15,.2f} | {guess_tot_adc/tgt_tot_adc:<18.4f}")
    
    # Total Charge Q
    tgt_tot_q = np.sum(tgt['Q'])
    guess_tot_q = np.sum(guess['Q'])
    print(f"{'Total Charge Q (ke-)':<30} | {tgt_tot_q:<15,.2f} | {guess_tot_q:<15,.2f} | {guess_tot_q/tgt_tot_q:<18.4f}")
    
    # Average ADC per active pixel-tick
    tgt_avg_adc = np.mean(tgt['adc_clean'])
    guess_avg_adc = np.mean(guess['adc_clean'])
    print(f"{'Mean ADC per active pixel-tick':<30} | {tgt_avg_adc:<15,.4f} | {guess_avg_adc:<15,.4f} | {guess_avg_adc/tgt_avg_adc:<18.4f}")
    
    # Standard deviation of ADC
    tgt_std_adc = np.std(tgt['adc_clean'])
    guess_std_adc = np.std(guess['adc_clean'])
    print(f"{'Std dev of ADC':<30} | {tgt_std_adc:<15,.4f} | {guess_std_adc:<15,.4f} | {guess_std_adc/tgt_std_adc:<18.4f}")

    # 2. Ticks distribution
    print("\n================= TICKS (TIME) DISTRIBUTION =================")
    print(f"{'Metric':<30} | {'Target':<15} | {'Guess':<15} | {'Difference (Guess-Tgt)':<22}")
    print("-" * 89)
    print(f"{'Min tick':<30} | {np.min(tgt['ticks']):<15.1f} | {np.min(guess['ticks']):<15.1f} | {np.min(guess['ticks'])-np.min(tgt['ticks']):<22.1f}")
    print(f"{'Max tick':<30} | {np.max(tgt['ticks']):<15.1f} | {np.max(guess['ticks']):<15.1f} | {np.max(guess['ticks'])-np.max(tgt['ticks']):<22.1f}")
    print(f"{'Mean tick':<30} | {np.mean(tgt['ticks']):<15.4f} | {np.mean(guess['ticks']):<15.4f} | {np.mean(guess['ticks'])-np.mean(tgt['ticks']):<22.4f}")
    print(f"{'Std dev of ticks':<30} | {np.std(tgt['ticks']):<15.4f} | {np.std(guess['ticks']):<15.4f} | {np.std(guess['ticks'])-np.std(tgt['ticks']):<22.4f}")

    # Check distribution in quantiles
    for q in [10, 25, 50, 75, 90]:
        tgt_q = np.percentile(tgt['ticks'], q)
        guess_q = np.percentile(guess['ticks'], q)
        print(f"{f'{q}th percentile tick':<30} | {tgt_q:<15.2f} | {guess_q:<15.2f} | {guess_q-tgt_q:<22.2f}")

    # Per-event comparison
    print("\n================== PER-EVENT CHARGE RATIOS ==================")
    tgt_events = np.unique(tgt['eventID'])
    guess_events = np.unique(guess['eventID'])
    common_events = sorted(list(set(tgt_events) & set(guess_events)))
    
    print(f"{'EventID':<10} | {'Tgt Active Px':<15} | {'Guess Active Px':<15} | {'Tgt Charge':<15} | {'Guess Charge':<15} | {'Charge Ratio':<12}")
    print("-" * 92)
    for eid in common_events[:10]:
        t_mask = tgt['eventID'] == eid
        g_mask = guess['eventID'] == eid
        
        t_px = np.sum(t_mask)
        g_px = np.sum(g_mask)
        t_ch = np.sum(tgt['Q'][t_mask])
        g_ch = np.sum(guess['Q'][g_mask])
        
        ratio = g_ch / t_ch if t_ch > 0 else 0.0
        print(f"{eid:<10d} | {t_px:<15,d} | {g_px:<15,d} | {t_ch:<15,.2f} | {g_ch:<15,.2f} | {ratio:<12.4f}")

if __name__ == '__main__':
    main()

# Talk animations

Eleven animations for the Keynote talk on calibrating a LArTPC with a differentiable simulation, in talk order.

- `dark/` and `light/`: the same videos in each theme. H.264, 3840x2160, 30 fps, in the MP4 container.
- `posters/`: the end frame of each video as a PNG, and the pause points as JSON.
- Scripts: `../scripts/talk_anim/` (`BRIEF.md` is the shared design brief). Working data, encoder and fonts:
  `/sdf/scratch/neutrino/pgranger/talk_animations/`. Scratch space may be purged, so copy `data/` if you plan to re-render much later.

All "data" here is noisy simulation at known truth. Each video's footer says so and names its source.

## In Keynote
Each video holds on a static end frame. To stop at a pause point below, press K during the show to pause and resume, or
split the video into separate clips and trim them. Not yet tested in Keynote itself.

## The animations

| # | File | Length | Idea |
|---|---|---|---|
| 1 | a01_muon_to_hits | 36.0 s | A real event, from muons to charge, drift, hits and back to 3-D |
| 2 | a02_drift | 37.0 s | One real segment; where the five calibration parameters act |
| 3 | a03_cloud_to_current | 38.0 s | Charge induces current on a pixel and its neighbours, using the real field table |
| 4 | a04_current_to_hit | 42.0 s | A self-triggering pixel: threshold, hold, digitise, reset |
| 5 | a05_random_to_differentiable | 47.3 s | 10,000 throws build the distribution the model computes directly |
| 6 | a06_five_knobs | 48.2 s | Each parameter's effect on the event, with jax.grad tangents |
| 7 | a07_scoring_a_hit | 42.5 s | How one pixel is scored, and how pixels add up to one loss |
| 8 | a08_track_converging | 39.5 s | Calibration, path and dE/dx on one muon, at the real saved fit states |
| 9 | a09_calibration_convergence | 42.5 s | 36 fits from one wrong start, iteration by iteration |
| 10 | a10_parameter_path | 48.0 s | A_b and lifetime share the charge scale; the valley |
| 11 | a11_ensemble_landing | 31.3 s | Where the 36 fits land: mean, spread, biases |

## Pause points (seconds)

- **a01:** 4.5 tracks · 8.5 charge · 12 drift · 22.5 hits · 26 3-D
- **a02:** 3 ionisation · 7 recombination · 12.5 drift speed · 18 attenuation · 23 diffusion · 29.5 all five
- **a03:** 5.5 neighbours · 12 net charge · 16.5 real pixel · 19 pieces add up · 33 waveform
- **a04:** 6.2 integrate · 10.5 threshold · 16.2 digitise · 19.5 reset · 24.6 second crossing · 35.6 a hit is a time and a charge
- **a05:** 6.2 replay · 11.6 speeding up · 22.7 10,000 throws · 25.7 analytic P(t) · 29.3 agreement · 35.3 scale test · 43.7 conclusion
- **a06:** 3.2 A_b · 10.35 lifetime · 17.5 E · 26.35 D_T · 34.35 D_L · 42.65 gradient
- **a07:** 6.5 model and data · 13 scoring · 21 start vs 20000 · 29.5 all pixels · 37.5 one gradient
- **a08:** 5.4 hit seed · 11 fit runs · 19.9 iteration 10000 · 27.4 iteration 14000 · 34.9 iteration 20000
- **a09:** 12.2 iteration 1000 · 21.2 iteration 7500 · 26.4 iteration 10000 · 31.8 iteration 14000 · 39 iteration 20000
- **a10:** 13.6 iteration 1000 · 25.3 iteration 14000 · 33.6 end points · 37 loss surface · 40.5 valley · 44 end points vs valley
- **a11:** 16.4 all landed · 19.3 mean ± SE · 23.3 spread · 27.3 biases

## What the videos rest on

- **Runs:** ensemble = `EXP<s>C75H`, 36 samples with 1 mm endpoint smear. Example event and a08 = sample 8, batch 35, event 2, run
  `KN10N8C75`/`FR`/`H`, with endpoints from simulation.
- **a08 is not continuous:** track shape, dE/dx and pixel charge appear only at real saved states. The hit seed was recomputed with the
  production method and matched the run's own log. Only the right-hand curves are per-iteration.
- **a06:** 5-7 real points per parameter, plus 4 value-only for E. Display curves between them are interpolated through values and
  gradients, as the footer says. The independent test of the gradient is the finite-difference numbers on screen.
- **a09 display:** lines are raw to iteration 1000, then a running mean widening to 100 iterations. Once axes zoom, early lines dim
  and the median is binned. The footer says so.
- **Schematic elements:** a01 anode drawn as floor; a02 dot positions drawn from the analytic cloud; a03 charge height assumes uniform
  drift speed; a07 arrows have equal length, and only their direction and printed numbers are real.

## Checks

- Every on-screen claim was read against the code and paper by an independent fact-checker, whose corrections are included. It reviewed
  sampled frames from the dark videos rather than every beat.
- Ensemble means, SE, SD, ranges, slope and r were recomputed separately from raw traces.
- Every MP4 was measured for single-frame flicker (none) and timestamps.
- Nobody has watched these play in real time or opened them in Keynote.

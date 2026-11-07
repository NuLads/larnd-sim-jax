The Differentiable Local Deflection Chain for LArTPC Track Fitting

A major challenge when calibrating Liquid Argon Time Projection Chambers (LArTPCs) using cosmic ray muons is Multiple Coulomb Scattering (MCS). Because MCS is a stochastic, step-by-step physical process rather than a global continuous curve, simple geometric functions (like parabolas or low-order polynomials) fail to capture the local, random-walk nature of the track.

The Local Deflection Chain solves this by modeling the track as a sequence of connected straight segments where the optimization parameters are the local scattering angles at each step. This formulation is fully differentiable, physically rigorous, and allows for the seamless injection of MCS priors via the Highland formula.

1. Mathematical Formulation

Instead of parameterizing the track using absolute coordinates in the detector frame, we define it autoregressively.

Step 1: Initial Conditions

The track begins at a defined entry point with an initial direction:

Initial Position: $\vec{x}_0 = (x_0, y_0, z_0)^T \in \mathbb{R}^3$

Initial Direction (Unit Vector): $\hat{u}_0 \in S^2$ (typically parameterized by spherical angles $\theta_0, \phi_0$)

Step 2: Local Reference Frames

At any segment $i$, we have a direction vector $\hat{u}_i$. To define a local deflection, we must construct an orthonormal coordinate frame local to that segment: $\{\hat{u}_i, \hat{v}_i, \hat{w}_i\}$.

To do this differentiably and avoid coordinate singularities, we can choose an arbitrary, non-collinear reference vector $\vec{a}$ (e.g., if $\hat{u}_i$ is primarily longitudinal, let $\vec{a} = (0, 1, 0)^T$) and use cross products:

$$\hat{v}_i = \frac{\hat{u}_i \times \vec{a}}{\|\hat{u}_i \times \vec{a}\|}$$

$$\hat{w}_i = \hat{u}_i \times \hat{v}_i$$

Step 3: Local Angular Deflections (The Parameters)

For a chain of $N$ segments of fixed physical length $\Delta s$, the optimization parameters for the track shape are the set of local angular deviations at each joint $i$:

$$\mathbf{\Theta} = \{ (\delta\theta_1, \delta\phi_1), (\delta\theta_2, \delta\phi_2), \dots, (\delta\theta_{N-1}, \delta\phi_{N-1}) \}$$

Where:

$\delta\theta_i \in [0, \pi]$ is the polar scattering angle (the magnitude of the deflection).

$\delta\phi_i \in [0, 2\pi)$ is the azimuthal angle (the direction of the deflection in the transverse plane).

Step 4: Autoregressive Propagation

The direction of the next segment, $\hat{u}_{i+1}$, is calculated by rotating the current direction $\hat{u}_i$ by $\delta\theta_i$ along the direction designated by $\delta\phi_i$ within the local transverse plane spanned by $\{\hat{v}_i, \hat{w}_i\}$:

$$\hat{u}_{i+1} = \cos(\delta\theta_i)\hat{u}_i + \sin(\delta\theta_i)\cos(\delta\phi_i)\hat{v}_i + \sin(\delta\theta_i)\sin(\delta\phi_i)\hat{w}_i$$

The next node position is then:

$$\vec{x}_{i+1} = \vec{x}_i + \Delta s \cdot \hat{u}_i$$

Because all operations (trigonometry, cross products, normalization, and vector addition) have well-defined, continuous analytical derivatives, the position of every point along the track is fully differentiable with respect to the initial state $(\vec{x}_0, \hat{u}_0)$ and all local scattering parameters $(\delta\theta_i, \delta\phi_i)$.

2. Incorporating the Physics Prior (Highland Formula)

Because we have parameterized the track using the physical variables of scattering, we can apply an exact, physically-motivated regularizing prior to our loss function.

According to the Highland formula, a charged particle traveling through a material undergoes multiple small-angle deflections. The width of the angular distribution $\theta_0$ is given by:

$$\theta_0 = \frac{13.6 \text{ MeV}}{\beta p c} z \sqrt{\frac{\Delta s}{X_0}} \left[ 1 + 0.038 \ln\left(\frac{\Delta s}{X_0}\right) \right]$$

For a minimum ionizing muon in Liquid Argon:

Radiation length of LAr: $X_0 \approx 14.0 \text{ cm}$.

If we choose a step length of $\Delta s = 10 \text{ cm}$, the term $\sqrt{\Delta s / X_0} \approx 0.845$.

For a typical cosmic run, you can estimate an average momentum $p$ (e.g., $3 \text{ GeV}/c$).

This gives us an expected standard deviation for our local polar deflection angles, $\sigma_{\text{MCS}}$.

The MCS Regularization Loss

We add a negative log-likelihood penalty to the optimization loss function to act as a soft constraint. Since the transverse scattering coordinates are normally distributed, the polar deflection angle $\delta\theta_i$ follows a 2D Gaussian projection (often modeled as a Rayleigh distribution for the 2D angle, or simply treating the projected orthogonal angles as Gaussian):

$$\mathcal{L}_{\text{MCS}} = \sum_{i=1}^{N-1} \left( \frac{\delta\theta_i^2}{2\sigma_{\text{MCS}}^2} \right)$$

This term strictly penalizes unphysical "kinks" or highly-curved segments, ensuring the optimizer only curves the track as much as physically reasonable for a particle of that energy.

3. Implementation in a Differentiable Framework (PyTorch Concept)

Below is an elegant, vectorized implementation showcasing how to construct the forward kinematics of the local deflection chain.

import torch
import torch.nn as nn

class DifferentiableTrack(nn.Module):
    def __init__(self, num_segments, step_length=10.0, initial_pos=None, initial_dir=None):
        super().__init__()
        self.num_segments = num_segments
        self.step_length = step_length  # in cm
        
        # 1. Optimizeable initial state
        if initial_pos is None:
            initial_pos = torch.tensor([0.0, 0.0, 0.0])
        if initial_dir is None:
            initial_dir = torch.tensor([0.0, 0.0, 1.0]) # along Z-axis
            
        self.x0 = nn.Parameter(initial_pos)
        self.u0 = nn.Parameter(initial_dir) # Will normalize in forward pass
        
        # 2. Optimizeable local scattering angles
        # We initialize them close to zero so the starting guess is a straight line
        self.delta_theta = nn.Parameter(torch.randn(num_segments - 1) * 0.01)
        self.delta_phi = nn.Parameter(torch.rand(num_segments - 1) * 2 * torch.pi)

    def forward(self):
        positions = [self.x0]
        
        # Ensure initial direction is a unit vector
        u_curr = self.u0 / torch.norm(self.u0)
        directions = [u_curr]
        
        # Reference vector to build local coordinate systems (arbitrary but stable)
        ref_vec = torch.tensor([0.0, 1.0, 0.0], device=self.x0.device)
        
        for i in range(self.num_segments - 1):
            # 1. Compute current node position
            x_next = positions[-1] + self.step_length * u_curr
            positions.append(x_next)
            
            # 2. Construct local orthonormal basis {u_curr, v_curr, w_curr}
            # Guard against collinearity
            cross_prod = torch.linalg.cross(u_curr, ref_vec)
            if torch.norm(cross_prod) < 1e-4:
                # Fallback reference vector if track is pointing parallel to Y-axis
                alt_ref = torch.tensor([1.0, 0.0, 0.0], device=self.x0.device)
                cross_prod = torch.linalg.cross(u_curr, alt_ref)
                
            v_curr = cross_prod / torch.norm(cross_prod)
            w_curr = torch.linalg.cross(u_curr, v_curr)
            
            # 3. Retrieve local deflection parameters
            theta = self.delta_theta[i]
            phi = self.delta_phi[i]
            
            # 4. Propagate direction vector via local rotation
            u_curr = (torch.cos(theta) * u_curr + 
                      torch.sin(theta) * torch.cos(phi) * v_curr + 
                      torch.sin(theta) * torch.sin(phi) * w_curr)
            
            # Ensure unit length to prevent numerical drift
            u_curr = u_curr / torch.norm(u_curr)
            directions.append(u_curr)
            
        return torch.stack(positions), torch.stack(directions)

    def get_mcs_loss(self, sigma_mcs):
        """Computes the physics prior regularization loss."""
        # Penalty is proportional to the deviation of theta from 0
        return torch.sum((self.delta_theta / sigma_mcs) ** 2)


4. Preventing Co-Optimization Degeneracy

When performing detector calibration, your physical parameters (e.g., Space Charge Effect E-field maps, electron lifetime $\tau$, wire field response) are optimized jointly with the track trajectories. This presents a critical hazard: the model could "cheat" by bending the track coordinates to fit uncalibrated detector anomalies.

To enforce strict boundary conditions and decouple detector effects from tracking effects:

A. The "Bilateral Constraint" Strategy (Fixed Endpoints)

Muons crossing the detector typically exit through the physical boundaries. If your track is bounded by precise optical flash triggers (PMTs/SiPMs) or external cathode/anode crossings:

Fix the entry point $\vec{x}_0$ and the final exit point $\vec{x}_N$ using hard constraints.

The optimizer is permitted to "wiggle" the internal nodes of the chain, but cannot drift the global trajectory endpoints. This heavily restricts the degeneracy phase space.

B. Spectral & Frequency Separation

Detector Distortions (e.g., Space Charge Effects): These are spatially static, smooth, low-frequency vector fields.

MCS Bending: This is highly localized, high-frequency, and varies entirely from track to track.

Optimization Solution: Optimize your detector parameters using a massive ensemble of tracks simultaneously (e.g., 500 tracks per batch). While a single track can easily bend to accommodate a bad SCE calibration, 500 distinct tracks pointing in different directions cannot all satisfy their individual MCS priors under a faulty global SCE E-field map.

C. The "Freeze-Thaw" Optimization Schedule

Do not optimize everything in a single gradient step. Instead, utilize a multi-phase optimization loop:

  ┌────────────────────────────────────────────────────────┐
  │ 1. INITIALIZE: Assume straight tracks & nominal TPC   │
  └───────────────────────────┬────────────────────────────┘
                              │
                              ▼
  ┌────────────────────────────────────────────────────────┐
  │ 2. FREEZE TPC: Optimize only Track Parameters          │
  │    (Local deflections converge to true physical path)  │
  └───────────────────────────┬────────────────────────────┘
                              │
                              ▼
  ┌────────────────────────────────────────────────────────┐
  │ 3. FREEZE TRACKS: Optimize only TPC Calibration        │
  │    (Adjusts global E-field, lifetime, and response)   │
  └───────────────────────────┬────────────────────────────┘
                              │
             Iterate until    │
             convergence     │
                              └────────────────────────────┘


This ensures that the highly-flexible track parameters are updated only in a context where the global detector terms are stable, preventing runaway gradients.
#!/usr/bin/env python3
"""
================================================================================
5MW WIND TURBINE BLADE DESIGN TOOL
================================================================================
Complete blade design including:
- NACA64-A17 airfoil geometry generation
- BEM (Blade Element Momentum) aerodynamic analysis
- Structural stress analysis
- Iterative optimization balancing efficiency and structural integrity
- Output: PNG plots, CSV data, CAD files (STL/OBJ/IGES)
- Interactive visualization (Matplotlib)

Author: Wind Turbine Design Tool
Turbine: 5MW, 126m rotor diameter, 3 blades
Airfoil: NACA64-A17
Material: Epoxy-GFRP (Premium)
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D
import json
import os

# ============================================================================
# CONFIGURATION - TURBINE SPECIFICATIONS
# ============================================================================
class TurbineConfig:
    """Wind turbine configuration parameters"""
    # Power and geometry
    P_rated = 5e6              # Rated power [W]
    R = 63.0                   # Rotor radius [m] (126m diameter)
    R_hub = 1.5                # Hub radius [m]
    B = 3                      # Number of blades
    
    # Operating conditions
    rho = 1.225                # Air density [kg/m³]
    V_rated = 11.4             # Rated wind speed [m/s]
    V_cutin = 3.0              # Cut-in wind speed [m/s]
    V_cutout = 25.0            # Cut-out wind speed [m/s]
    
    # Design parameters
    lambda_design = 7.0        # Design tip speed ratio
    n_stations = 20            # Number of blade stations
    n_iterations = 80          # Total optimization iterations
    n_rule_based = 15          # Rule-based iterations (phase 1)
    
    # Genetic Algorithm parameters
    ga_population_size = 30    # Population size for GA
    ga_crossover_rate = 0.8    # Crossover probability
    ga_mutation_rate = 0.1     # Mutation probability
    ga_elite_count = 2         # Number of elite individuals to preserve
    
    # Derivative-based optimization parameters
    optimization_method = 'derivative'  # 'GA', 'derivative', or 'hybrid'
    deriv_method = 'SLSQP'     # Scipy optimizer: 'SLSQP', 'L-BFGS-B', 'trust-constr'
    deriv_maxiter = 200        # Maximum iterations for derivative optimizer
    deriv_ftol = 1e-6          # Function tolerance
    deriv_eps = 1e-3           # Step size for finite difference gradients (larger for stability)
    use_smooth_penalty = False # Use smooth penalty instead of hard constraint (for L-BFGS-B)
    penalty_weight = 10.0      # Weight for constraint penalty
    use_variable_scaling = True  # Scale variables to [0, 1] range
    
    # Initial constant values for optimization
    initial_chord = 3.0        # Initial constant chord [m]
    initial_twist = 8.0        # Initial constant twist/pitch [degrees]

# ============================================================================
# CONFIGURATION - MATERIAL PROPERTIES (Epoxy-GFRP Premium)
# ============================================================================
class MaterialConfig:
    """Material properties for Epoxy-GFRP"""
    name = "Epoxy-GFRP (Premium)"
    
    # Strength properties [Pa]
    sigma_tensile = 1100e6     # Tensile strength
    sigma_compress = 600e6     # Compressive strength
    tau_shear = 80e6           # Shear strength
    
    # Elastic properties
    E = 45e9                   # Young's modulus [Pa]
    G = 5e9                    # Shear modulus [Pa]
    
    # Physical properties
    rho = 1900                 # Density [kg/m³]
    
    # Safety factor
    SF = 2.5
    
    @classmethod
    def get_allowable(cls):
        """Return allowable stresses with safety factor"""
        return {
            'tensile': cls.sigma_tensile / cls.SF,
            'compress': cls.sigma_compress / cls.SF,
            'shear': cls.tau_shear / cls.SF
        }

# ============================================================================
# NACA64-A17 AIRFOIL DATA
# ============================================================================
class AirfoilData:
    """NACA64-A17 airfoil aerodynamic data and geometry"""
    
    # Cl, Cd, Cm data from DOWEC (alpha in degrees)
    _raw_data = """
    -180.00,0.000,0.0198,0.0000
    -175.00,0.374,0.0341,0.1880
    -170.00,0.749,0.0955,0.3770
    -160.00,0.659,0.2807,0.2747
    -155.00,0.736,0.3919,0.3130
    -150.00,0.783,0.5086,0.3428
    -145.00,0.803,0.6267,0.3654
    -140.00,0.798,0.7427,0.3820
    -135.00,0.771,0.8537,0.3935
    -130.00,0.724,0.9574,0.4007
    -125.00,0.660,1.0519,0.4042
    -120.00,0.581,1.1355,0.4047
    -115.00,0.491,1.2070,0.4025
    -110.00,0.390,1.2656,0.3981
    -105.00,0.282,1.3104,0.3918
    -100.00,0.169,1.3410,0.3838
    -95.00,0.052,1.3572,0.3743
    -90.00,-0.067,1.3587,0.3636
    -85.00,-0.184,1.3456,0.3517
    -80.00,-0.299,1.3181,0.3388
    -75.00,-0.409,1.2765,0.3248
    -70.00,-0.512,1.2212,0.3099
    -65.00,-0.606,1.1532,0.2940
    -60.00,-0.689,1.0731,0.2772
    -55.00,-0.759,0.9822,0.2595
    -50.00,-0.814,0.8820,0.2409
    -45.00,-0.850,0.7742,0.2212
    -40.00,-0.866,0.6610,0.2006
    -35.00,-0.860,0.5451,0.1789
    -30.00,-0.829,0.4295,0.1563
    -25.00,-0.853,0.3071,0.1156
    -24.00,-0.870,0.2814,0.1040
    -23.00,-0.890,0.2556,0.0916
    -22.00,-0.911,0.2297,0.0785
    -21.00,-0.934,0.2040,0.0649
    -20.00,-0.958,0.1785,0.0508
    -19.00,-0.982,0.1534,0.0364
    -18.00,-1.005,0.1288,0.0218
    -17.00,-1.082,0.1037,0.0129
    -16.00,-1.113,0.0786,-0.0028
    -15.00,-1.105,0.0535,-0.0251
    -14.00,-1.078,0.0283,-0.0419
    -13.50,-1.053,0.0158,-0.0521
    -13.00,-1.015,0.0151,-0.0610
    -12.00,-0.904,0.0134,-0.0707
    -11.00,-0.807,0.0121,-0.0722
    -10.00,-0.711,0.0111,-0.0734
    -9.00,-0.595,0.0099,-0.0772
    -8.00,-0.478,0.0091,-0.0807
    -7.00,-0.375,0.0086,-0.0825
    -6.00,-0.264,0.0082,-0.0832
    -5.00,-0.151,0.0079,-0.0841
    -4.00,-0.017,0.0072,-0.0869
    -3.00,0.088,0.0064,-0.0912
    -2.00,0.213,0.0054,-0.0946
    -1.00,0.328,0.0052,-0.0971
    0.00,0.442,0.0052,-0.1014
    1.00,0.556,0.0052,-0.1076
    2.00,0.670,0.0053,-0.1126
    3.00,0.784,0.0053,-0.1157
    4.00,0.898,0.0054,-0.1199
    5.00,1.011,0.0058,-0.1240
    6.00,1.103,0.0091,-0.1234
    7.00,1.181,0.0113,-0.1184
    8.00,1.257,0.0124,-0.1163
    8.50,1.293,0.0130,-0.1163
    9.00,1.326,0.0136,-0.1160
    9.50,1.356,0.0143,-0.1154
    10.00,1.382,0.0150,-0.1149
    10.50,1.400,0.0267,-0.1145
    11.00,1.415,0.0383,-0.1143
    11.50,1.425,0.0498,-0.1147
    12.00,1.434,0.0613,-0.1158
    12.50,1.443,0.0727,-0.1165
    13.00,1.451,0.0841,-0.1153
    13.50,1.453,0.0954,-0.1131
    14.00,1.448,0.1065,-0.1112
    14.50,1.444,0.1176,-0.1101
    15.00,1.445,0.1287,-0.1103
    15.50,1.447,0.1398,-0.1109
    16.00,1.448,0.1509,-0.1114
    16.50,1.444,0.1619,-0.1111
    17.00,1.438,0.1728,-0.1097
    17.50,1.439,0.1837,-0.1079
    18.00,1.448,0.1947,-0.1080
    18.50,1.452,0.2057,-0.1090
    19.00,1.448,0.2165,-0.1086
    19.50,1.438,0.2272,-0.1077
    20.00,1.428,0.2379,-0.1099
    21.00,1.401,0.2590,-0.1169
    22.00,1.359,0.2799,-0.1190
    23.00,1.300,0.3004,-0.1235
    24.00,1.220,0.3204,-0.1393
    25.00,1.168,0.3377,-0.1440
    26.00,1.116,0.3554,-0.1486
    28.00,1.015,0.3916,-0.1577
    30.00,0.926,0.4294,-0.1668
    32.00,0.855,0.4690,-0.1759
    35.00,0.800,0.5324,-0.1897
    40.00,0.804,0.6452,-0.2126
    45.00,0.793,0.7573,-0.2344
    50.00,0.763,0.8664,-0.2553
    55.00,0.717,0.9708,-0.2751
    60.00,0.656,1.0693,-0.2939
    65.00,0.582,1.1606,-0.3117
    70.00,0.495,1.2438,-0.3285
    75.00,0.398,1.3178,-0.3444
    80.00,0.291,1.3809,-0.3593
    85.00,0.176,1.4304,-0.3731
    90.00,0.053,1.4565,-0.3858
    95.00,-0.074,1.4533,-0.3973
    100.00,-0.199,1.4345,-0.4075
    105.00,-0.321,1.4004,-0.4162
    110.00,-0.436,1.3512,-0.4231
    115.00,-0.543,1.2874,-0.4280
    120.00,-0.640,1.2099,-0.4306
    125.00,-0.723,1.1196,-0.4304
    130.00,-0.790,1.0179,-0.4270
    135.00,-0.840,0.9064,-0.4196
    140.00,-0.868,0.7871,-0.4077
    145.00,-0.872,0.6627,-0.3903
    150.00,-0.850,0.5363,-0.3665
    155.00,-0.798,0.4116,-0.3349
    160.00,-0.714,0.2931,-0.2942
    170.00,-0.749,0.0971,-0.3771
    175.00,-0.374,0.0334,-0.1879
    180.00,0.000,0.0198,0.0000
    """
    
    def __init__(self):
        self._parse_data()
        self._create_interpolators()
    
    def _parse_data(self):
        """Parse raw airfoil data"""
        self.alpha = []
        self.Cl = []
        self.Cd = []
        self.Cm = []
        
        for line in self._raw_data.strip().split('\n'):
            parts = line.strip().split(',')
            if len(parts) == 4:
                self.alpha.append(float(parts[0]))
                self.Cl.append(float(parts[1]))
                self.Cd.append(float(parts[2]))
                self.Cm.append(float(parts[3]))
        
        self.alpha = np.array(self.alpha)
        self.Cl = np.array(self.Cl)
        self.Cd = np.array(self.Cd)
        self.Cm = np.array(self.Cm)
    
    def _create_interpolators(self):
        """Create interpolation functions"""
        self._cl_interp = interp1d(self.alpha, self.Cl, kind='cubic', fill_value='extrapolate')
        self._cd_interp = interp1d(self.alpha, self.Cd, kind='cubic', fill_value='extrapolate')
        self._cm_interp = interp1d(self.alpha, self.Cm, kind='cubic', fill_value='extrapolate')
    
    def get_Cl(self, alpha):
        """Get lift coefficient at angle of attack"""
        return float(self._cl_interp(np.clip(alpha, -180, 180)))
    
    def get_Cd(self, alpha):
        """Get drag coefficient at angle of attack"""
        return float(self._cd_interp(np.clip(alpha, -180, 180)))
    
    def get_Cm(self, alpha):
        """Get moment coefficient at angle of attack"""
        return float(self._cm_interp(np.clip(alpha, -180, 180)))
    
    def get_optimal_alpha(self):
        """Find angle of attack with maximum L/D ratio"""
        valid = (self.alpha > 0) & (self.alpha < 15) & (self.Cd > 0.001)
        ld_ratio = self.Cl[valid] / self.Cd[valid]
        idx = np.argmax(ld_ratio)
        return self.alpha[valid][idx]

# ============================================================================
# AIRFOIL GEOMETRY GENERATOR
# ============================================================================
class AirfoilGeometry:
    """Generate NACA 64-A17 airfoil coordinates"""
    
    def __init__(self, n_points=100):
        self.n_points = n_points
        self.t_ratio = 0.17  # 17% thickness
        self.generate_coordinates()
    
    def generate_coordinates(self):
        """Generate airfoil coordinates using NACA equations"""
        # Cosine spacing for better resolution at LE/TE
        beta = np.linspace(0, np.pi, self.n_points)
        self.x = 0.5 * (1 - np.cos(beta))
        
        # NACA 4-digit thickness distribution (modified for 6-series)
        t = self.t_ratio
        self.yt = t/0.2 * (0.2969*np.sqrt(self.x) - 0.1260*self.x 
                          - 0.3516*self.x**2 + 0.2843*self.x**3 - 0.1015*self.x**4)
        
        # Camber line (low camber for 64-A series)
        m, p = 0.02, 0.4
        self.yc = np.where(self.x <= p,
                          m/p**2 * (2*p*self.x - self.x**2),
                          m/(1-p)**2 * ((1-2*p) + 2*p*self.x - self.x**2))
        
        dyc_dx = np.where(self.x <= p,
                         2*m/p**2 * (p - self.x),
                         2*m/(1-p)**2 * (p - self.x))
        
        theta = np.arctan(dyc_dx)
        
        # Upper and lower surface
        self.xu = self.x - self.yt * np.sin(theta)
        self.yu = self.yc + self.yt * np.cos(theta)
        self.xl = self.x + self.yt * np.sin(theta)
        self.yl = self.yc - self.yt * np.cos(theta)
        
        # Combined coordinates (counterclockwise)
        self.x_coords = np.concatenate([self.xu[::-1], self.xl[1:]])
        self.y_coords = np.concatenate([self.yu[::-1], self.yl[1:]])
    
    def get_scaled_section(self, chord, pitch_deg):
        """Get airfoil section scaled and rotated"""
        theta = np.radians(pitch_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        
        # Scale
        x_scaled = self.x_coords * chord
        y_scaled = self.y_coords * chord
        
        # Rotate
        x_rot = x_scaled * cos_t + y_scaled * sin_t
        y_rot = -x_scaled * sin_t + y_scaled * cos_t
        
        # Offset to pitch axis (25% chord)
        x_final = x_rot - 0.25 * chord * cos_t
        y_final = y_rot + 0.25 * chord * sin_t
        
        return x_final, y_final

# ============================================================================
# SECTION PROPERTIES CALCULATOR
# ============================================================================
class SectionProperties:
    """Calculate structural section properties"""
    
    @staticmethod
    def calculate(chord, t_skin=0.02, t_spar=0.03):
        """
        Calculate section properties for hollow box-beam model
        
        Parameters:
            chord: Chord length [m]
            t_skin: Skin thickness as fraction of chord
            t_spar: Spar cap thickness as fraction of chord
        
        Returns:
            Dictionary with Ixx, Iyy, J, A, y_max, x_max
        """
        t_ratio = 0.17  # Airfoil thickness ratio
        
        h = chord * t_ratio      # Section height
        w = chord * 0.4          # Effective width
        skin = chord * t_skin
        spar = chord * t_spar
        
        # Box dimensions
        h_outer, w_outer = h, w
        h_inner = max(0, h - 2*spar)
        w_inner = max(0, w - 2*skin)
        
        # Area
        A = h_outer * w_outer - h_inner * w_inner
        
        # Second moments of area
        Ixx = (w_outer * h_outer**3 - w_inner * h_inner**3) / 12
        Iyy = (h_outer * w_outer**3 - h_inner * w_inner**3) / 12
        
        # Torsion constant (thin-walled closed section)
        Am = (h_outer - spar) * (w_outer - skin)
        perimeter = 2 * ((h_outer - spar) + (w_outer - skin))
        t_avg = (skin + spar) / 2
        J = 4 * Am**2 * t_avg / (perimeter + 1e-10)
        
        return {
            'Ixx': Ixx, 'Iyy': Iyy, 'J': J, 'A': A,
            'y_max': h_outer / 2, 'x_max': w_outer / 2,
            'h': h, 'w': w
        }

# ============================================================================
# BEM (BLADE ELEMENT MOMENTUM) ANALYSIS
# ============================================================================
class BEMAnalysis:
    """Blade Element Momentum aerodynamic analysis"""
    
    def __init__(self, turbine_config, airfoil_data):
        self.cfg = turbine_config
        self.airfoil = airfoil_data
    
    def analyze(self, r, chord, pitch, omega, V_inf, tip_loss=True):
        """
        Perform BEM analysis
        
        Parameters:
            r: Radial stations [m]
            chord: Chord at each station [m]
            pitch: Pitch angle at each station [deg]
            omega: Rotational speed [rad/s]
            V_inf: Wind speed [m/s]
            tip_loss: Apply Prandtl tip loss correction
        
        Returns:
            Dictionary with aerodynamic results
        """
        n = len(r)
        
        # Initialize outputs
        a = np.zeros(n)           # Axial induction
        a_prime = np.zeros(n)     # Tangential induction
        alpha = np.zeros(n)       # Angle of attack
        Cl = np.zeros(n)
        Cd = np.zeros(n)
        pn = np.zeros(n)          # Normal force [N/m]
        pt = np.zeros(n)          # Tangential force [N/m]
        
        for i, r_i in enumerate(r):
            if r_i < self.cfg.R_hub:
                continue
            
            c = chord[i]
            theta = pitch[i]
            sigma = self.cfg.B * c / (2 * np.pi * r_i)
            
            # Initial guess
            a[i], a_prime[i] = 0.1, 0.01
            
            # Iterative solution
            for _ in range(100):
                V_axial = V_inf * (1 - a[i])
                V_tan = omega * r_i * (1 + a_prime[i])
                W = np.sqrt(V_axial**2 + V_tan**2)
                phi = np.arctan2(V_axial, V_tan)
                
                alpha[i] = np.degrees(phi) - theta
                Cl[i] = self.airfoil.get_Cl(alpha[i])
                Cd[i] = self.airfoil.get_Cd(alpha[i])
                
                Cn = Cl[i] * np.cos(phi) + Cd[i] * np.sin(phi)
                Ct = Cl[i] * np.sin(phi) - Cd[i] * np.cos(phi)
                
                # Tip loss factor
                if tip_loss and r_i < self.cfg.R and np.sin(phi) > 0.01:
                    f = (self.cfg.B/2) * (self.cfg.R - r_i) / (r_i * np.sin(phi))
                    F = (2/np.pi) * np.arccos(np.exp(-min(f, 50))) if f < 50 else 1.0
                else:
                    F = 1.0
                
                # New induction factors
                denom_a = 4 * F * np.sin(phi)**2 / (sigma * Cn + 1e-10) + 1
                a_new = 1 / denom_a if Cn > 0 else 0
                
                denom_ap = 4 * F * np.sin(phi) * np.cos(phi) / (sigma * Ct + 1e-10) - 1
                a_prime_new = 1 / denom_ap if Ct > 0 and denom_ap != 0 else 0
                
                # Glauert correction
                if a_new > 0.4:
                    a_new = 0.5 * (2 + (1 - 2*0.4) - np.sqrt((1-2*0.4)**2 + 4*(1-2*0.4)*0.4))
                
                # Relaxation and limits
                a[i] = 0.5 * a[i] + 0.5 * np.clip(a_new, 0, 0.95)
                a_prime[i] = 0.5 * a_prime[i] + 0.5 * np.clip(a_prime_new, 0, 0.5)
                
                if abs(a_new - a[i]) < 1e-6:
                    break
            
            # Final loads
            V_axial = V_inf * (1 - a[i])
            V_tan = omega * r_i * (1 + a_prime[i])
            W = np.sqrt(V_axial**2 + V_tan**2)
            phi = np.arctan2(V_axial, V_tan)
            
            pn[i] = 0.5 * self.cfg.rho * W**2 * c * (Cl[i] * np.cos(phi) + Cd[i] * np.sin(phi))
            pt[i] = 0.5 * self.cfg.rho * W**2 * c * (Cl[i] * np.sin(phi) - Cd[i] * np.cos(phi))
        
        # Calculate power
        power = self.cfg.B * np.trapz(pt * r, r) * omega
        Cp = power / (0.5 * self.cfg.rho * np.pi * self.cfg.R**2 * V_inf**3)
        
        return {
            'a': a, 'a_prime': a_prime, 'alpha': alpha,
            'Cl': Cl, 'Cd': Cd, 'pn': pn, 'pt': pt,
            'power': power, 'Cp': Cp
        }

# ============================================================================
# STRUCTURAL ANALYSIS
# ============================================================================
class StructuralAnalysis:
    """Structural stress analysis for wind turbine blade"""
    
    def __init__(self, material_config):
        self.mat = material_config
        self.allowable = material_config.get_allowable()
    
    def analyze(self, r, chord, pn, pt, omega):
        """
        Calculate structural loads and stresses
        
        Parameters:
            r: Radial stations [m]
            chord: Chord at each station [m]
            pn: Normal force distribution [N/m]
            pt: Tangential force distribution [N/m]
            omega: Rotational speed [rad/s]
        
        Returns:
            Dictionary with structural results
        """
        n = len(r)
        
        # Section properties
        section_props = [SectionProperties.calculate(c) for c in chord]
        
        # Mass distribution
        m_dist = np.array([sp['A'] * self.mat.rho for sp in section_props])
        
        # Centrifugal force
        Fc = m_dist * omega**2 * r
        
        # Initialize internal loads
        M_flap = np.zeros(n)
        M_edge = np.zeros(n)
        V_flap = np.zeros(n)
        V_edge = np.zeros(n)
        T_axial = np.zeros(n)
        
        # Integrate from tip to root
        for i in range(n-2, -1, -1):
            dr = r[i+1] - r[i]
            V_flap[i] = V_flap[i+1] + pn[i+1] * dr
            V_edge[i] = V_edge[i+1] + pt[i+1] * dr
            M_flap[i] = M_flap[i+1] + V_flap[i+1] * dr + 0.5 * pn[i+1] * dr**2
            M_edge[i] = M_edge[i+1] + V_edge[i+1] * dr + 0.5 * pt[i+1] * dr**2
            T_axial[i] = T_axial[i+1] + Fc[i+1] * dr
        
        # Calculate stresses
        sigma_tensile = np.zeros(n)
        sigma_compress = np.zeros(n)
        tau_shear = np.zeros(n)
        
        for i in range(n):
            sp = section_props[i]
            
            sigma_flap = M_flap[i] * sp['y_max'] / (sp['Ixx'] + 1e-10)
            sigma_edge = M_edge[i] * sp['x_max'] / (sp['Iyy'] + 1e-10)
            sigma_bend = np.sqrt(sigma_flap**2 + sigma_edge**2)
            sigma_axial = T_axial[i] / (sp['A'] + 1e-10)
            
            sigma_tensile[i] = sigma_axial + sigma_bend
            sigma_compress[i] = abs(sigma_axial - sigma_bend)
            
            V_total = np.sqrt(V_flap[i]**2 + V_edge[i]**2)
            tau_shear[i] = V_total * sp['y_max'] / (sp['Ixx'] * 0.1 * chord[i] + 1e-10)
        
        # Stress ratios
        ratio_tensile = sigma_tensile / self.allowable['tensile']
        ratio_compress = sigma_compress / self.allowable['compress']
        ratio_shear = tau_shear / self.allowable['shear']
        
        return {
            'M_flap': M_flap, 'M_edge': M_edge,
            'V_flap': V_flap, 'V_edge': V_edge,
            'T_axial': T_axial,
            'sigma_tensile': sigma_tensile,
            'sigma_compress': sigma_compress,
            'tau_shear': tau_shear,
            'ratio_tensile': ratio_tensile,
            'ratio_compress': ratio_compress,
            'ratio_shear': ratio_shear,
            'section_props': section_props
        }

# ============================================================================
# BLADE OPTIMIZER
# ============================================================================
class BladeOptimizer:
    """Optimize blade design for efficiency and structural integrity"""
    
    def __init__(self, turbine_config, material_config, airfoil_data):
        self.cfg = turbine_config
        self.mat = material_config
        self.airfoil = airfoil_data
        self.bem = BEMAnalysis(turbine_config, airfoil_data)
        self.struct = StructuralAnalysis(material_config)
    
    def create_initial_design(self):
        """Create initial blade design using constant chord and twist"""
        omega = self.cfg.lambda_design * self.cfg.V_rated / self.cfg.R
        r = np.linspace(self.cfg.R_hub + 0.5, self.cfg.R - 0.5, self.cfg.n_stations)
        
        # Use constant chord and twist as initial values
        chord = np.ones_like(r) * self.cfg.initial_chord
        pitch = np.ones_like(r) * self.cfg.initial_twist
        
        return r, chord, pitch, omega
    
    def optimize(self, verbose=True):
        """
        Run optimization based on configuration method:
        - 'GA': Rule-based + Genetic Algorithm
        - 'derivative': Rule-based + Derivative-based (SLSQP/L-BFGS-B)
        - 'hybrid': Rule-based + Derivative-based + GA polish
        
        Returns:
            Tuple of (initial_design, optimized_design, history)
        """
        r, chord, pitch, omega = self.create_initial_design()
        
        # Store initial design
        initial_chord = chord.copy()
        initial_pitch = pitch.copy()
        
        bem_init = self.bem.analyze(r, initial_chord, initial_pitch, omega, self.cfg.V_rated)
        struct_init = self.struct.analyze(r, initial_chord, bem_init['pn'], bem_init['pt'], omega)
        
        initial_design = {
            'r': r, 'chord': initial_chord, 'pitch': initial_pitch, 'omega': omega,
            'bem': bem_init, 'struct': struct_init
        }
        
        # Optimization history
        history = {'Cp': [], 'max_stress_ratio': [], 'iteration': [], 'phase': []}
        
        if verbose:
            print("\n" + "=" * 60)
            print("PHASE 1: Rule-Based Optimization (Iterations 1-{})".format(self.cfg.n_rule_based))
            print("=" * 60)
        
        # ================================================================
        # PHASE 1: Rule-based optimization (iterations 1-15)
        # ================================================================
        for iteration in range(self.cfg.n_rule_based):
            bem = self.bem.analyze(r, chord, pitch, omega, self.cfg.V_rated)
            struct = self.struct.analyze(r, chord, bem['pn'], bem['pt'], omega)
            
            max_ratio = max(
                np.max(struct['ratio_tensile']),
                np.max(struct['ratio_compress']),
                np.max(struct['ratio_shear'])
            )
            
            history['iteration'].append(iteration)
            history['Cp'].append(bem['Cp'])
            history['max_stress_ratio'].append(max_ratio)
            history['phase'].append(1)
            
            if verbose:
                status = "✓" if max_ratio <= 1.0 else "✗"
                print(f"  Iter {iteration+1:2d}: Cp = {bem['Cp']:.4f}, Max σ ratio = {max_ratio:.3f} {status}")
            
            # Adjust chord where stress is high
            for i in range(len(r)):
                local_max = max(struct['ratio_tensile'][i], 
                              struct['ratio_compress'][i],
                              struct['ratio_shear'][i])
                if local_max > 1.0:
                    chord[i] *= 1.0 + 0.1 * (local_max - 1.0)
                elif local_max < 0.7:
                    chord[i] *= 0.98
            
            chord = np.clip(chord, 0.8, 5.0)
            
            # Adjust pitch for optimal AoA
            for i in range(len(r)):
                if bem['alpha'][i] > 10:
                    pitch[i] += 0.5
                elif bem['alpha'][i] < 4:
                    pitch[i] -= 0.5
        
        # ================================================================
        # PHASE 2: Based on optimization_method config
        # ================================================================
        if self.cfg.optimization_method in ['derivative', 'hybrid']:
            chord, pitch, history = self._derivative_optimization(
                r, chord, pitch, omega, history, verbose
            )
        
        if self.cfg.optimization_method == 'GA':
            chord, pitch, history = self._genetic_algorithm_optimization(
                r, chord, pitch, omega, history, verbose
            )
        elif self.cfg.optimization_method == 'hybrid':
            # Additional GA polish after derivative optimization
            if verbose:
                print("\n" + "=" * 60)
                print("PHASE 3: Genetic Algorithm Polish")
                print("=" * 60)
            chord, pitch, history = self._genetic_algorithm_optimization(
                r, chord, pitch, omega, history, verbose, n_generations=20
            )
        
        # Final analysis
        bem_final = self.bem.analyze(r, chord, pitch, omega, self.cfg.V_rated)
        struct_final = self.struct.analyze(r, chord, bem_final['pn'], bem_final['pt'], omega)
        
        optimized_design = {
            'r': r, 'chord': chord, 'pitch': pitch, 'omega': omega,
            'bem': bem_final, 'struct': struct_final
        }
        
        if verbose:
            print("=" * 60)
            print(f"Final: Cp = {bem_final['Cp']:.4f}, Power = {bem_final['power']/1e6:.2f} MW")
        
        return initial_design, optimized_design, history
    
    def _derivative_optimization(self, r, chord, pitch, omega, history, verbose=True):
        """
        Derivative-based optimization using scipy.optimize.minimize
        
        Improvements to reduce oscillation:
        1. Variable scaling to [0,1] range
        2. Smooth penalty functions instead of hard constraints
        3. Larger finite difference step size
        4. Smooth max function for constraint
        
        Supports SLSQP (constrained), L-BFGS-B (box bounds), trust-constr (constrained)
        """
        if verbose:
            print("\n" + "=" * 60)
            print(f"PHASE 2: Derivative-Based Optimization ({self.cfg.deriv_method})")
            print("=" * 60)
            print(f"  Max iterations: {self.cfg.deriv_maxiter}, Tolerance: {self.cfg.deriv_ftol:.1e}")
            print(f"  FD step size: {self.cfg.deriv_eps:.1e}")
            if self.cfg.use_smooth_penalty:
                print(f"  Using smooth penalty (weight={self.cfg.penalty_weight})")
            if self.cfg.use_variable_scaling:
                print("  Using variable scaling [0, 1]")
            print("-" * 60)
        
        n_stations = len(r)
        
        # Bounds for optimization variables
        chord_min, chord_max = 0.8, 5.0
        pitch_min, pitch_max = -5.0, 25.0
        
        # Variable scaling functions
        def scale_vars(x):
            """Scale from physical to [0,1] range"""
            x_scaled = np.zeros_like(x)
            x_scaled[:n_stations] = (x[:n_stations] - chord_min) / (chord_max - chord_min)
            x_scaled[n_stations:] = (x[n_stations:] - pitch_min) / (pitch_max - pitch_min)
            return x_scaled
        
        def unscale_vars(x_scaled):
            """Unscale from [0,1] to physical range"""
            x = np.zeros_like(x_scaled)
            x[:n_stations] = x_scaled[:n_stations] * (chord_max - chord_min) + chord_min
            x[n_stations:] = x_scaled[n_stations:] * (pitch_max - pitch_min) + pitch_min
            return x
        
        def smooth_max(values, sharpness=10.0):
            """Smooth approximation of max function using log-sum-exp"""
            values = np.array(values)
            max_val = np.max(values)
            # Shift for numerical stability
            return max_val + np.log(np.sum(np.exp(sharpness * (values - max_val)))) / sharpness
        
        def smooth_penalty(violation, sharpness=5.0):
            """Smooth penalty function: 0 when feasible, grows smoothly when infeasible"""
            # Softplus-like function: smooth transition around constraint boundary
            return np.log1p(np.exp(sharpness * violation)) / sharpness
        
        # Initial guess from Phase 1
        x0_physical = np.concatenate([chord, pitch])
        
        if self.cfg.use_variable_scaling:
            x0 = scale_vars(x0_physical)
            bounds = [(0.0, 1.0) for _ in range(2 * n_stations)]
        else:
            x0 = x0_physical
            bounds = [(chord_min, chord_max) for _ in range(n_stations)] + \
                     [(pitch_min, pitch_max) for _ in range(n_stations)]
        
        # Iteration counter for callback
        self._deriv_iter = 0
        start_iter = len(history['iteration'])
        self._best_feasible_x = x0.copy()
        self._best_feasible_Cp = -np.inf
        
        def get_physical_vars(x):
            """Get physical variables from scaled/unscaled x"""
            if self.cfg.use_variable_scaling:
                x_phys = unscale_vars(x)
            else:
                x_phys = x
            return x_phys[:n_stations], x_phys[n_stations:]
        
        def compute_stress_ratio(c, p):
            """Compute max stress ratio"""
            bem = self.bem.analyze(r, c, p, omega, self.cfg.V_rated)
            struct = self.struct.analyze(r, c, bem['pn'], bem['pt'], omega)
            
            if self.cfg.use_smooth_penalty:
                # Use smooth max for better gradients
                max_ratio = smooth_max([
                    np.max(struct['ratio_tensile']),
                    np.max(struct['ratio_compress']),
                    np.max(struct['ratio_shear'])
                ])
            else:
                max_ratio = max(
                    np.max(struct['ratio_tensile']),
                    np.max(struct['ratio_compress']),
                    np.max(struct['ratio_shear'])
                )
            return max_ratio, bem
        
        def objective_with_penalty(x):
            """Objective: -Cp + penalty for constraint violation"""
            c, p = get_physical_vars(x)
            bem = self.bem.analyze(r, c, p, omega, self.cfg.V_rated)
            struct = self.struct.analyze(r, c, bem['pn'], bem['pt'], omega)
            
            # Use smooth max for stress ratio
            if self.cfg.use_smooth_penalty:
                max_ratio = smooth_max([
                    np.max(struct['ratio_tensile']),
                    np.max(struct['ratio_compress']),
                    np.max(struct['ratio_shear'])
                ])
                # Smooth penalty: grows from 0 at ratio=1
                penalty = self.cfg.penalty_weight * smooth_penalty(max_ratio - 1.0)
            else:
                max_ratio = max(
                    np.max(struct['ratio_tensile']),
                    np.max(struct['ratio_compress']),
                    np.max(struct['ratio_shear'])
                )
                # Hard penalty
                penalty = self.cfg.penalty_weight * max(0, max_ratio - 1.0) ** 2
            
            return -bem['Cp'] + penalty
        
        def objective(x):
            """Objective function: -Cp (we minimize, so negate Cp to maximize)"""
            c, p = get_physical_vars(x)
            bem = self.bem.analyze(r, c, p, omega, self.cfg.V_rated)
            return -bem['Cp']
        
        def stress_constraint(x):
            """
            Inequality constraint: max_stress_ratio <= 1.0
            Scipy expects g(x) >= 0 for inequality constraints
            So we return: 1.0 - max_stress_ratio
            """
            c, p = get_physical_vars(x)
            max_ratio, _ = compute_stress_ratio(c, p)
            return 1.0 - max_ratio
        
        def callback(x):
            """Callback to record history and print progress"""
            c, p = get_physical_vars(x)
            bem = self.bem.analyze(r, c, p, omega, self.cfg.V_rated)
            struct = self.struct.analyze(r, c, bem['pn'], bem['pt'], omega)
            
            max_ratio = max(
                np.max(struct['ratio_tensile']),
                np.max(struct['ratio_compress']),
                np.max(struct['ratio_shear'])
            )
            
            # Track best feasible solution
            if max_ratio <= 1.0 and bem['Cp'] > self._best_feasible_Cp:
                self._best_feasible_Cp = bem['Cp']
                self._best_feasible_x = x.copy()
            
            iteration = start_iter + self._deriv_iter
            history['iteration'].append(iteration)
            history['Cp'].append(bem['Cp'])
            history['max_stress_ratio'].append(max_ratio)
            history['phase'].append(2)
            
            if verbose:
                status = "✓" if max_ratio <= 1.0 else "✗"
                print(f"  Iter {self._deriv_iter+1:3d}: Cp = {bem['Cp']:.4f}, "
                      f"Max σ ratio = {max_ratio:.3f} {status}")
            
            self._deriv_iter += 1
        
        # Choose optimization approach
        if self.cfg.deriv_method == 'L-BFGS-B' or (self.cfg.use_smooth_penalty and self.cfg.deriv_method != 'trust-constr'):
            # Use penalty method (for L-BFGS-B or when smooth penalty is requested)
            result = minimize(
                objective_with_penalty,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                callback=callback,
                options={
                    'maxiter': self.cfg.deriv_maxiter,
                    'ftol': self.cfg.deriv_ftol,
                    'disp': False,
                    'eps': self.cfg.deriv_eps
                }
            )
        elif self.cfg.deriv_method == 'SLSQP':
            # SLSQP with hard constraint (best for this problem)
            constraints = {'type': 'ineq', 'fun': stress_constraint}
            
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                callback=callback,
                options={
                    'maxiter': self.cfg.deriv_maxiter,
                    'ftol': self.cfg.deriv_ftol,
                    'disp': False,
                    'eps': self.cfg.deriv_eps
                }
            )
        elif self.cfg.deriv_method == 'trust-constr':
            from scipy.optimize import NonlinearConstraint
            
            def stress_ratio_func(x):
                c, p = get_physical_vars(x)
                max_ratio, _ = compute_stress_ratio(c, p)
                return max_ratio
            
            constraint = NonlinearConstraint(stress_ratio_func, -np.inf, 1.0)
            
            result = minimize(
                objective,
                x0,
                method='trust-constr',
                bounds=bounds,
                constraints=constraint,
                callback=callback,
                options={
                    'maxiter': self.cfg.deriv_maxiter,
                    'gtol': self.cfg.deriv_ftol,
                    'disp': False
                }
            )
        else:
            raise ValueError(f"Unknown derivative method: {self.cfg.deriv_method}")
        
        # Extract optimized values (use best feasible if available)
        if self._best_feasible_Cp > -np.inf:
            # Check if result is feasible
            c_res, p_res = get_physical_vars(result.x)
            max_ratio_res, bem_res = compute_stress_ratio(c_res, p_res)
            
            if max_ratio_res > 1.0:
                # Result is infeasible, use best feasible
                if verbose:
                    print(f"  Note: Using best feasible solution (Cp={self._best_feasible_Cp:.4f})")
                final_x = self._best_feasible_x
            else:
                final_x = result.x
        else:
            final_x = result.x
        
        chord_opt, pitch_opt = get_physical_vars(final_x)
        
        if verbose:
            print("-" * 60)
            print(f"Optimization {['Failed', 'Succeeded'][result.success]}: {result.message}")
            print(f"Function evaluations: {result.nfev}")
            if hasattr(result, 'njev') and result.njev is not None:
                print(f"Gradient evaluations: {result.njev}")
            
            # Report final objective
            final_bem = self.bem.analyze(r, chord_opt, pitch_opt, omega, self.cfg.V_rated)
            print(f"Final Cp: {final_bem['Cp']:.4f}")
        
        return chord_opt, pitch_opt, history
    def _genetic_algorithm_optimization(self, r, chord, pitch, omega, history, verbose=True, n_generations=None):
        """
        Genetic Algorithm optimization
        """
        if n_generations is None:
            n_generations = self.cfg.n_iterations - self.cfg.n_rule_based
        
        if verbose:
            if self.cfg.optimization_method == 'GA':
                print("\n" + "=" * 60)
                print("PHASE 2: Genetic Algorithm Optimization")
                print("=" * 60)
            print(f"  Population: {self.cfg.ga_population_size}, "
                  f"Crossover: {self.cfg.ga_crossover_rate:.0%}, "
                  f"Mutation: {self.cfg.ga_mutation_rate:.0%}, "
                  f"Elite: {self.cfg.ga_elite_count}")
            print("-" * 60)
        
        n_stations = len(r)
        n_genes = 2 * n_stations  # chord + pitch
        
        # Bounds
        chord_bounds = (0.8, 5.0)
        pitch_bounds = (-5.0, 25.0)
        
        def encode(chord_arr, pitch_arr):
            """Encode chord and pitch arrays into chromosome"""
            return np.concatenate([chord_arr, pitch_arr])
        
        def decode(chromosome):
            """Decode chromosome into chord and pitch arrays"""
            return chromosome[:n_stations], chromosome[n_stations:]
        
        def fitness(chromosome):
            """
            Fitness function: Maximize Cp with penalty for stress violations
            Returns (fitness_value, Cp, max_stress_ratio)
            """
            c, p = decode(chromosome)
            bem = self.bem.analyze(r, c, p, omega, self.cfg.V_rated)
            struct = self.struct.analyze(r, c, bem['pn'], bem['pt'], omega)
            
            max_ratio = max(
                np.max(struct['ratio_tensile']),
                np.max(struct['ratio_compress']),
                np.max(struct['ratio_shear'])
            )
            
            Cp = bem['Cp']
            
            # Penalty for stress constraint violation
            if max_ratio > 1.0:
                penalty = 0.5 * (max_ratio - 1.0) ** 2
                fitness_val = Cp - penalty
            else:
                fitness_val = Cp
            
            return fitness_val, Cp, max_ratio
        
        def tournament_selection(population, fitness_vals, tournament_size=3):
            """Select individual using tournament selection"""
            indices = np.random.choice(len(population), tournament_size, replace=False)
            best_idx = indices[np.argmax(fitness_vals[indices])]
            return population[best_idx].copy()
        
        def blx_alpha_crossover(parent1, parent2, alpha=0.5):
            """BLX-α crossover for real-valued chromosomes"""
            child1 = np.zeros_like(parent1)
            child2 = np.zeros_like(parent2)
            
            for i in range(len(parent1)):
                min_val = min(parent1[i], parent2[i])
                max_val = max(parent1[i], parent2[i])
                range_val = max_val - min_val
                
                low = min_val - alpha * range_val
                high = max_val + alpha * range_val
                
                child1[i] = np.random.uniform(low, high)
                child2[i] = np.random.uniform(low, high)
            
            return child1, child2
        
        def gaussian_mutation(chromosome, mutation_rate, sigma=0.1):
            """Gaussian mutation for real-valued chromosomes"""
            mutated = chromosome.copy()
            for i in range(len(mutated)):
                if np.random.random() < mutation_rate:
                    # Determine bounds based on gene type
                    if i < n_stations:
                        bounds = chord_bounds
                    else:
                        bounds = pitch_bounds
                    
                    # Apply Gaussian mutation
                    range_val = bounds[1] - bounds[0]
                    mutated[i] += np.random.normal(0, sigma * range_val)
                    mutated[i] = np.clip(mutated[i], bounds[0], bounds[1])
            
            return mutated
        
        def repair_chromosome(chromosome):
            """Ensure chromosome is within bounds"""
            repaired = chromosome.copy()
            repaired[:n_stations] = np.clip(repaired[:n_stations], chord_bounds[0], chord_bounds[1])
            repaired[n_stations:] = np.clip(repaired[n_stations:], pitch_bounds[0], pitch_bounds[1])
            return repaired
        
        # Initialize population around current solution
        population = []
        base_chromosome = encode(chord, pitch)
        
        # Add current result as first individual (elitism)
        population.append(base_chromosome.copy())
        
        # Generate rest of population with variations
        for _ in range(self.cfg.ga_population_size - 1):
            new_chromosome = base_chromosome.copy()
            # Add random variation
            for i in range(n_genes):
                if i < n_stations:
                    new_chromosome[i] += np.random.normal(0, 0.3)
                else:
                    new_chromosome[i] += np.random.normal(0, 2.0)
            new_chromosome = repair_chromosome(new_chromosome)
            population.append(new_chromosome)
        
        population = np.array(population)
        
        # Track best solution
        best_chromosome = base_chromosome.copy()
        best_fitness = -np.inf
        
        start_iter = len(history['iteration'])
        
        # Genetic Algorithm main loop
        for generation in range(n_generations):
            iteration = start_iter + generation
            
            # Evaluate fitness for all individuals
            fitness_vals = np.zeros(len(population))
            Cp_vals = np.zeros(len(population))
            stress_vals = np.zeros(len(population))
            
            for i, chromosome in enumerate(population):
                fitness_vals[i], Cp_vals[i], stress_vals[i] = fitness(chromosome)
            
            # Find best in current generation
            best_idx = np.argmax(fitness_vals)
            gen_best_fitness = fitness_vals[best_idx]
            gen_best_Cp = Cp_vals[best_idx]
            gen_best_stress = stress_vals[best_idx]
            
            # Update global best
            if gen_best_fitness > best_fitness:
                best_fitness = gen_best_fitness
                best_chromosome = population[best_idx].copy()
            
            # Record history (best of generation)
            history['iteration'].append(iteration)
            history['Cp'].append(gen_best_Cp)
            history['max_stress_ratio'].append(gen_best_stress)
            phase = 3 if self.cfg.optimization_method == 'hybrid' else 2
            history['phase'].append(phase)
            
            if verbose:
                status = "✓" if gen_best_stress <= 1.0 else "✗"
                avg_Cp = np.mean(Cp_vals)
                print(f"  Gen {generation+1:2d} (Iter {iteration+1:2d}): "
                      f"Best Cp = {gen_best_Cp:.4f}, Avg Cp = {avg_Cp:.4f}, "
                      f"Max σ = {gen_best_stress:.3f} {status}")
            
            # Create new population
            new_population = []
            
            # Elitism: keep best individuals
            elite_indices = np.argsort(fitness_vals)[-self.cfg.ga_elite_count:]
            for idx in elite_indices:
                new_population.append(population[idx].copy())
            
            # Generate offspring
            while len(new_population) < self.cfg.ga_population_size:
                # Selection
                parent1 = tournament_selection(population, fitness_vals)
                parent2 = tournament_selection(population, fitness_vals)
                
                # Crossover
                if np.random.random() < self.cfg.ga_crossover_rate:
                    child1, child2 = blx_alpha_crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                # Mutation
                child1 = gaussian_mutation(child1, self.cfg.ga_mutation_rate)
                child2 = gaussian_mutation(child2, self.cfg.ga_mutation_rate)
                
                # Repair (ensure within bounds)
                child1 = repair_chromosome(child1)
                child2 = repair_chromosome(child2)
                
                new_population.append(child1)
                if len(new_population) < self.cfg.ga_population_size:
                    new_population.append(child2)
            
            population = np.array(new_population)
        
        # Extract best solution
        chord_opt, pitch_opt = decode(best_chromosome)
        
        if verbose:
            print("-" * 60)
            print(f"GA Complete: Best Fitness = {best_fitness:.4f}")
        
        return chord_opt, pitch_opt, history

# ============================================================================
# CAD FILE GENERATOR
# ============================================================================
class CADGenerator:
    """Generate CAD files (STL, OBJ, IGES) from blade design"""
    
    def __init__(self, airfoil_geometry):
        self.airfoil = airfoil_geometry
    
    def generate_stl(self, r, chord, pitch, filename):
        """Generate STL file"""
        n_span = len(r)
        n_pts = min(30, len(self.airfoil.x_coords))
        
        # Sample airfoil
        indices = np.linspace(0, len(self.airfoil.x_coords)-1, n_pts, dtype=int)
        x_sample = self.airfoil.x_coords[indices]
        y_sample = self.airfoil.y_coords[indices]
        
        # Compute all vertices
        all_verts = []
        for i in range(n_span):
            x_final, y_final = self._transform_section(x_sample, y_sample, chord[i], pitch[i])
            all_verts.append([(x_final[j], y_final[j], r[i]) for j in range(n_pts)])
        
        # Generate STL
        lines = ["solid wind_turbine_blade"]
        
        for i in range(n_span - 1):
            for j in range(n_pts - 1):
                v1, v2 = all_verts[i][j], all_verts[i][j+1]
                v3, v4 = all_verts[i+1][j+1], all_verts[i+1][j]
                
                for tri in [(v1, v2, v3), (v1, v3, v4)]:
                    n = self._calc_normal(*tri)
                    lines.append(f"  facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}")
                    lines.append("    outer loop")
                    for v in tri:
                        lines.append(f"      vertex {v[0]:.6e} {v[1]:.6e} {v[2]:.6e}")
                    lines.append("    endloop")
                    lines.append("  endfacet")
        
        lines.append("endsolid wind_turbine_blade")
        
        with open(filename, 'w') as f:
            f.write('\n'.join(lines))
    
    def generate_obj(self, r, chord, pitch, filename):
        """Generate OBJ file"""
        n_span = len(r)
        n_pts = min(30, len(self.airfoil.x_coords))
        
        indices = np.linspace(0, len(self.airfoil.x_coords)-1, n_pts, dtype=int)
        x_sample = self.airfoil.x_coords[indices]
        y_sample = self.airfoil.y_coords[indices]
        
        lines = ["# Wind Turbine Blade - 5MW NACA64-A17", ""]
        
        # Vertices
        for i in range(n_span):
            x_final, y_final = self._transform_section(x_sample, y_sample, chord[i], pitch[i])
            for j in range(n_pts):
                lines.append(f"v {x_final[j]:.6f} {y_final[j]:.6f} {r[i]:.6f}")
        
        lines.append("")
        
        # Faces
        for i in range(n_span - 1):
            base1 = i * n_pts + 1
            base2 = (i + 1) * n_pts + 1
            for j in range(n_pts - 1):
                lines.append(f"f {base1+j} {base1+j+1} {base2+j+1} {base2+j}")
        
        with open(filename, 'w') as f:
            f.write('\n'.join(lines))
    
    def generate_iges(self, r, chord, pitch, filename):
        """Generate simplified IGES file"""
        lines = []
        lines.append("                                                                        S      1")
        lines.append("1H,,1H;,7Hblade.igs,19Hwind_turbine_blade,32,38,6,99,15,               G      1")
        lines.append("7Hblade.igs,1.,1,4HINCH,32767,0.0001,13H241224.120000,                 G      2")
        lines.append("0.001,1000.,14Hblade_designer,10HAnthropic,11,0,;                      G      3")
        
        # Simplified: store as point data
        d_entries = []
        p_entries = []
        d_line, p_line = 1, 1
        
        n_pts = min(30, len(self.airfoil.x_coords))
        indices = np.linspace(0, len(self.airfoil.x_coords)-1, n_pts, dtype=int)
        x_sample = self.airfoil.x_coords[indices]
        y_sample = self.airfoil.y_coords[indices]
        
        for i in range(len(r)):
            x_final, y_final = self._transform_section(x_sample, y_sample, chord[i], pitch[i])
            
            d_entries.append(f"     106{d_line:8d}       1       0       0       0       0       0        1D{d_line:7d}")
            d_entries.append(f"     106       2       3{n_pts:8d}                                        D{d_line+1:7d}")
            
            p_data = f"106,1,{n_pts},"
            for j in range(n_pts):
                p_data += f"{x_final[j]:.4f},{y_final[j]:.4f},{r[i]:.4f},"
            p_data = p_data[:-1] + ";"
            
            while len(p_data) > 64:
                p_entries.append(f"{p_data[:64]:<72}{p_line}P{d_line:7d}")
                p_data = p_data[64:]
                p_line += 1
            if p_data:
                p_entries.append(f"{p_data:<72}{p_line}P{d_line:7d}")
                p_line += 1
            
            d_line += 2
        
        lines.extend(d_entries)
        lines.extend(p_entries)
        lines.append(f"S{1:7d}G{3:7d}D{len(d_entries):7d}P{len(p_entries):7d}{' '*40}T      1")
        
        with open(filename, 'w') as f:
            f.write('\n'.join(lines))
    
    def _transform_section(self, x, y, chord, pitch_deg):
        """Transform airfoil section by chord and pitch"""
        theta = np.radians(pitch_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        
        x_scaled = x * chord
        y_scaled = y * chord
        
        x_rot = x_scaled * cos_t + y_scaled * sin_t
        y_rot = -x_scaled * sin_t + y_scaled * cos_t
        
        x_final = x_rot - 0.25 * chord * cos_t
        y_final = y_rot + 0.25 * chord * sin_t
        
        return x_final, y_final
    
    def _calc_normal(self, p1, p2, p3):
        """Calculate face normal"""
        u = (p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2])
        v = (p3[0]-p1[0], p3[1]-p1[1], p3[2]-p1[2])
        n = (u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0])
        mag = (n[0]**2 + n[1]**2 + n[2]**2)**0.5
        return (n[0]/mag, n[1]/mag, n[2]/mag) if mag > 0 else (0, 0, 1)

# ============================================================================
# RESULTS OUTPUT
# ============================================================================
class ResultsOutput:
    """Generate output files and plots"""
    
    def __init__(self, output_dir="."):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def save_csv(self, design, filename, header_info=""):
        """Save blade design to CSV"""
        r = design['r']
        lines = [
            f"# 5MW Wind Turbine Blade Design - NACA64-A17",
            f"# {header_info}",
            f"# Power Coefficient: {design['bem']['Cp']:.4f}",
            f"# Power: {design['bem']['power']/1e6:.3f} MW",
            "r(m),chord(m),pitch(deg),alpha(deg),Cl,Cd,pn(N/m),pt(N/m),M_flap(Nm),sigma_t(MPa),sigma_c(MPa),tau(MPa),ratio_t,ratio_c,ratio_s"
        ]
        
        for i in range(len(r)):
            lines.append(
                f"{r[i]:.2f},{design['chord'][i]:.3f},{design['pitch'][i]:.2f},"
                f"{design['bem']['alpha'][i]:.2f},{design['bem']['Cl'][i]:.3f},{design['bem']['Cd'][i]:.4f},"
                f"{design['bem']['pn'][i]:.1f},{design['bem']['pt'][i]:.1f},"
                f"{design['struct']['M_flap'][i]:.1f},"
                f"{design['struct']['sigma_tensile'][i]/1e6:.1f},"
                f"{design['struct']['sigma_compress'][i]/1e6:.1f},"
                f"{design['struct']['tau_shear'][i]/1e6:.1f},"
                f"{design['struct']['ratio_tensile'][i]:.3f},"
                f"{design['struct']['ratio_compress'][i]:.3f},"
                f"{design['struct']['ratio_shear'][i]:.3f}"
            )
        
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            f.write('\n'.join(lines))
        return filepath
    
    def save_comparison_csv(self, initial, optimized, filename):
        """Save comparison summary"""
        data = [
            "Parameter,Initial,Optimized,Change(%)",
            f"Power Coefficient,{initial['bem']['Cp']:.4f},{optimized['bem']['Cp']:.4f},{(optimized['bem']['Cp']-initial['bem']['Cp'])/initial['bem']['Cp']*100:.2f}",
            f"Power (MW),{initial['bem']['power']/1e6:.3f},{optimized['bem']['power']/1e6:.3f},{(optimized['bem']['power']-initial['bem']['power'])/initial['bem']['power']*100:.2f}",
            f"Max Stress Ratio,{max(np.max(initial['struct']['ratio_tensile']),np.max(initial['struct']['ratio_compress']),np.max(initial['struct']['ratio_shear'])):.3f},{max(np.max(optimized['struct']['ratio_tensile']),np.max(optimized['struct']['ratio_compress']),np.max(optimized['struct']['ratio_shear'])):.3f},N/A",
            f"Max Chord (m),{np.max(initial['chord']):.3f},{np.max(optimized['chord']):.3f},{(np.max(optimized['chord'])-np.max(initial['chord']))/np.max(initial['chord'])*100:.2f}",
        ]
        
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            f.write('\n'.join(data))
        return filepath
    
    def save_json(self, initial, optimized, airfoil, filename):
        """Save all data to JSON"""
        data = {
            'turbine': {
                'power_rated_W': TurbineConfig.P_rated,
                'rotor_diameter_m': 2*TurbineConfig.R,
                'num_blades': TurbineConfig.B,
                'V_rated_ms': TurbineConfig.V_rated
            },
            'material': {
                'name': MaterialConfig.name,
                'sigma_tensile_Pa': MaterialConfig.sigma_tensile,
                'sigma_compress_Pa': MaterialConfig.sigma_compress,
                'tau_shear_Pa': MaterialConfig.tau_shear,
                'safety_factor': MaterialConfig.SF
            },
            'initial': {
                'r': initial['r'].tolist(),
                'chord': initial['chord'].tolist(),
                'pitch': initial['pitch'].tolist(),
                'Cp': initial['bem']['Cp'],
                'power': initial['bem']['power']
            },
            'optimized': {
                'r': optimized['r'].tolist(),
                'chord': optimized['chord'].tolist(),
                'pitch': optimized['pitch'].tolist(),
                'alpha': optimized['bem']['alpha'].tolist(),
                'Cl': optimized['bem']['Cl'].tolist(),
                'Cd': optimized['bem']['Cd'].tolist(),
                'pn': optimized['bem']['pn'].tolist(),
                'pt': optimized['bem']['pt'].tolist(),
                'sigma_tensile': optimized['struct']['sigma_tensile'].tolist(),
                'sigma_compress': optimized['struct']['sigma_compress'].tolist(),
                'tau_shear': optimized['struct']['tau_shear'].tolist(),
                'Cp': optimized['bem']['Cp'],
                'power': optimized['bem']['power']
            },
            'airfoil': {
                'alpha': airfoil.alpha.tolist(),
                'Cl': airfoil.Cl.tolist(),
                'Cd': airfoil.Cd.tolist()
            }
        }
        
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        return filepath
    
    def save_optimization_history_csv(self, history, filename):
        """
        Save optimization history to CSV file
        
        Parameters:
            history: dict with 'iteration', 'Cp', 'max_stress_ratio', and 'phase' keys
            filename: output filename for the CSV
        
        Returns:
            filepath: path to the saved CSV file
        """
        lines = [
            "# 5MW Wind Turbine Blade Optimization History",
            "# Initial Chord: {:.2f} m, Initial Twist: {:.2f} deg".format(
                TurbineConfig.initial_chord, TurbineConfig.initial_twist),
            "# Phase 1: Rule-based (iterations 1-{}), Phase 2: Genetic Algorithm (iterations {}-{})".format(
                TurbineConfig.n_rule_based, TurbineConfig.n_rule_based + 1, TurbineConfig.n_iterations),
            "# Total Iterations: {}".format(len(history['iteration'])),
            "iteration,phase,method,Cp,max_stress_ratio,stress_status,Cp_improvement_pct"
        ]
        
        initial_Cp = history['Cp'][0] if history['Cp'] else 0
        phase_names = {1: 'Rule-Based', 2: 'GA'}
        
        for i in range(len(history['iteration'])):
            iteration = history['iteration'][i] + 1  # 1-indexed
            phase = history.get('phase', [1] * len(history['iteration']))[i]
            method = phase_names.get(phase, 'Unknown')
            Cp = history['Cp'][i]
            max_stress = history['max_stress_ratio'][i]
            status = "PASS" if max_stress <= 1.0 else "FAIL"
            improvement = (Cp - initial_Cp) / initial_Cp * 100 if initial_Cp > 0 else 0
            
            lines.append(f"{iteration},{phase},{method},{Cp:.6f},{max_stress:.6f},{status},{improvement:.4f}")
        
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            f.write('\n'.join(lines))
        return filepath

# ============================================================================
# PLOTTING
# ============================================================================
class PlotGenerator:
    """Generate visualization plots"""
    
    def __init__(self, output_dir="."):
        self.output_dir = output_dir
    
    def plot_airfoil_data(self, airfoil, airfoil_geom, filename):
        """Plot airfoil characteristics"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Cl vs alpha
        ax = axes[0, 0]
        ax.plot(airfoil.alpha, airfoil.Cl, 'b-', linewidth=2)
        ax.set_xlabel('Angle of Attack (deg)')
        ax.set_ylabel('Lift Coefficient Cl')
        ax.set_title('NACA64-A17 Lift Curve')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([-20, 25])
        
        # Cd vs alpha
        ax = axes[0, 1]
        ax.plot(airfoil.alpha, airfoil.Cd, 'r-', linewidth=2)
        ax.set_xlabel('Angle of Attack (deg)')
        ax.set_ylabel('Drag Coefficient Cd')
        ax.set_title('NACA64-A17 Drag Curve')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([-20, 25])
        
        # L/D ratio
        ax = axes[1, 0]
        valid = (airfoil.alpha > -10) & (airfoil.alpha < 15) & (airfoil.Cd > 0.001)
        ax.plot(airfoil.alpha[valid], airfoil.Cl[valid]/airfoil.Cd[valid], 'g-', linewidth=2)
        ax.set_xlabel('Angle of Attack (deg)')
        ax.set_ylabel('Lift-to-Drag Ratio Cl/Cd')
        ax.set_title('NACA64-A17 Efficiency')
        ax.grid(True, alpha=0.3)
        
        # Airfoil shape
        ax = axes[1, 1]
        ax.plot(airfoil_geom.x_coords, airfoil_geom.y_coords, 'k-', linewidth=2)
        ax.set_xlabel('x/c')
        ax.set_ylabel('y/c')
        ax.set_title('NACA64-A17 Airfoil Shape')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        return filepath
    
    def plot_comparison(self, initial, optimized, filename):
        """Plot initial vs optimized comparison"""
        r = initial['r']
        
        fig, axes = plt.subplots(3, 3, figsize=(16, 14))
        fig.suptitle('5MW Wind Turbine Blade: Initial vs Optimized Design', fontsize=14, fontweight='bold')
        
        plots = [
            (axes[0,0], 'chord', 'Chord (m)', 'Chord Distribution'),
            (axes[0,1], 'pitch', 'Pitch (deg)', 'Pitch Distribution'),
            (axes[0,2], ('bem', 'alpha'), 'AoA (deg)', 'Angle of Attack'),
            (axes[1,0], ('bem', 'pn'), 'Normal Force (N/m)', 'Flapwise Load', 1000, 'kN/m'),
            (axes[1,1], ('bem', 'pt'), 'Tangential Force (N/m)', 'Driving Force', 1000, 'kN/m'),
            (axes[1,2], ('struct', 'M_flap'), 'Moment (MN·m)', 'Bending Moment', 1e6, 'MN·m'),
            (axes[2,0], ('struct', 'ratio_tensile'), 'Stress Ratio', 'Tensile Stress Ratio'),
            (axes[2,1], ('struct', 'ratio_compress'), 'Stress Ratio', 'Compressive Stress Ratio'),
            (axes[2,2], ('struct', 'ratio_shear'), 'Stress Ratio', 'Shear Stress Ratio'),
        ]
        
        for item in plots:
            ax = item[0]
            key = item[1]
            ylabel = item[2]
            title = item[3]
            scale = item[4] if len(item) > 4 else 1
            unit = item[5] if len(item) > 5 else ylabel
            
            if isinstance(key, tuple):
                y_init = initial[key[0]][key[1]] / scale
                y_opt = optimized[key[0]][key[1]] / scale
            else:
                y_init = initial[key] / scale
                y_opt = optimized[key] / scale
            
            ax.plot(r, y_init, 'b--', linewidth=2, label='Initial', marker='o', markersize=3)
            ax.plot(r, y_opt, 'r-', linewidth=2, label='Optimized', marker='s', markersize=3)
            
            if 'ratio' in str(key):
                ax.axhline(y=1.0, color='k', linestyle='--', linewidth=2, label='Limit')
            
            ax.set_xlabel('Radial Position (m)')
            ax.set_ylabel(unit if scale != 1 else ylabel)
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        return filepath
    
    def plot_3d_blade(self, r, chord, pitch, airfoil_geom, filename):
        """Plot 3D blade geometry"""
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        n_span = len(r)
        n_chord = min(50, len(airfoil_geom.x_coords))
        
        indices = np.linspace(0, len(airfoil_geom.x_coords)-1, n_chord, dtype=int)
        x_af = airfoil_geom.x_coords[indices]
        y_af = airfoil_geom.y_coords[indices]
        
        X = np.zeros((n_span, n_chord))
        Y = np.zeros((n_span, n_chord))
        Z = np.zeros((n_span, n_chord))
        
        for i in range(n_span):
            c = chord[i]
            theta = np.radians(pitch[i])
            
            x_scaled = x_af * c
            y_scaled = y_af * c
            
            x_rot = x_scaled * np.cos(theta) + y_scaled * np.sin(theta)
            y_rot = -x_scaled * np.sin(theta) + y_scaled * np.cos(theta)
            
            X[i, :] = x_rot - 0.25 * c * np.cos(theta)
            Y[i, :] = y_rot
            Z[i, :] = r[i]
        
        ax.plot_surface(X, Y, Z, alpha=0.7, cmap='viridis')
        ax.set_xlabel('X (m) - Chordwise')
        ax.set_ylabel('Y (m) - Thickness')
        ax.set_zlabel('Z (m) - Spanwise')
        ax.set_title('3D Blade Geometry')
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        return filepath
    
    def plot_optimization_progress(self, history, filename):
        """
        Plot optimization progress showing efficiency (Cp) and max stress ratio vs iteration
        Shows Phase 1 (rule-based) and Phase 2 (derivative-based) with different colors
        
        Parameters:
            history: dict with 'iteration', 'Cp', 'max_stress_ratio', and 'phase' keys
            filename: output filename for the plot
        
        Returns:
            filepath: path to the saved plot
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
        fig.suptitle('Two-Phase Optimization Progress', fontsize=14, fontweight='bold')
        
        iterations = np.array(history['iteration']) + 1  # 1-indexed for display
        Cp = np.array(history['Cp'])
        max_stress = np.array(history['max_stress_ratio'])
        phase = np.array(history.get('phase', [1] * len(iterations)))
        
        # Separate phases
        phase1_mask = phase == 1
        phase2_mask = phase == 2
        
        # Plot Power Coefficient (Efficiency)
        if np.any(phase1_mask):
            ax1.plot(iterations[phase1_mask], Cp[phase1_mask], 'b-o', linewidth=2, 
                    markersize=8, label='Phase 1: Rule-Based')
        if np.any(phase2_mask):
            ax1.plot(iterations[phase2_mask], Cp[phase2_mask], 'g-s', linewidth=2, 
                    markersize=8, label='Phase 2: Genetic Algorithm')
        
        # Add vertical line at phase transition
        if np.any(phase1_mask) and np.any(phase2_mask):
            phase1_end = iterations[phase1_mask][-1]
            ax1.axvline(x=phase1_end + 0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
            ax1.text(phase1_end + 0.5, ax1.get_ylim()[1] * 0.95, ' Phase 2 →', 
                    fontsize=9, color='gray', va='top')
        
        ax1.set_ylabel('Power Coefficient (Cp)', fontsize=12)
        ax1.set_title('Aerodynamic Efficiency vs Iteration')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='lower right')
        
        # Add improvement annotations
        if len(Cp) > 1:
            # Phase 1 improvement
            if np.any(phase1_mask):
                phase1_Cp = Cp[phase1_mask]
                improvement1 = (phase1_Cp[-1] - phase1_Cp[0]) / phase1_Cp[0] * 100
                ax1.annotate(f'Phase 1: {improvement1:+.2f}%', 
                            xy=(iterations[phase1_mask][-1], phase1_Cp[-1]), 
                            xytext=(-50, 20), textcoords='offset points',
                            fontsize=9, ha='right', color='blue',
                            arrowprops=dict(arrowstyle='->', color='blue', alpha=0.7))
            
            # Total improvement
            total_improvement = (Cp[-1] - Cp[0]) / Cp[0] * 100
            ax1.annotate(f'Total: {total_improvement:+.2f}%', 
                        xy=(iterations[-1], Cp[-1]), 
                        xytext=(-30, -30), textcoords='offset points',
                        fontsize=10, ha='right', color='darkgreen', fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color='darkgreen'))
        
        # Plot Max Stress Ratio
        if np.any(phase1_mask):
            ax2.plot(iterations[phase1_mask], max_stress[phase1_mask], 'b-o', linewidth=2, 
                    markersize=8, label='Phase 1: Rule-Based')
        if np.any(phase2_mask):
            ax2.plot(iterations[phase2_mask], max_stress[phase2_mask], 'g-s', linewidth=2, 
                    markersize=8, label='Phase 2: Genetic Algorithm')
        
        ax2.axhline(y=1.0, color='r', linestyle='--', linewidth=2, label='Stress Limit (1.0)')
        
        # Add vertical line at phase transition
        if np.any(phase1_mask) and np.any(phase2_mask):
            ax2.axvline(x=phase1_end + 0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
        
        ax2.set_xlabel('Iteration', fontsize=12)
        ax2.set_ylabel('Max Stress Ratio', fontsize=12)
        ax2.set_title('Structural Stress Ratio vs Iteration')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right')
        
        # Color code points above/below limit
        above_limit = max_stress > 1.0
        if np.any(above_limit):
            ax2.scatter(iterations[above_limit], max_stress[above_limit], 
                       color='red', s=120, zorder=5, marker='x', linewidths=3, label='_nolegend_')
        below_limit = max_stress <= 1.0
        if np.any(below_limit):
            ax2.scatter(iterations[below_limit], max_stress[below_limit], 
                       color='green', s=60, zorder=5, marker='o', facecolors='none', 
                       linewidths=2, label='_nolegend_')
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        return filepath

# ============================================================================
# INTERACTIVE VISUALIZATION
# ============================================================================
class InteractiveVisualizer:
    """Interactive matplotlib visualization with sliders"""
    
    def __init__(self, initial_design, optimized_design, airfoil_data, airfoil_geom):
        self.initial = initial_design
        self.optimized = optimized_design
        self.airfoil = airfoil_data
        self.airfoil_geom = airfoil_geom
        self.current_design = 'optimized'
        self.selected_station = 10
        self.allowable = MaterialConfig.get_allowable()
    
    def show(self):
        """Display interactive visualization"""
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle('5MW Wind Turbine Blade Design - Interactive Viewer', fontsize=14, fontweight='bold')
        
        # Create axes
        ax_chord = fig.add_subplot(231)
        ax_pitch = fig.add_subplot(232)
        ax_stress = fig.add_subplot(233)
        ax_airfoil = fig.add_subplot(234)
        ax_loads = fig.add_subplot(235)
        ax_info = fig.add_subplot(236)
        
        plt.subplots_adjust(bottom=0.25, hspace=0.35, wspace=0.3)
        
        # Initial plots
        design = self.optimized
        r = design['r']
        
        line_chord, = ax_chord.plot(r, design['chord'], 'b-', linewidth=2, marker='o', markersize=4)
        ax_chord.set_xlabel('Radial Position (m)')
        ax_chord.set_ylabel('Chord (m)')
        ax_chord.set_title('Chord Distribution')
        ax_chord.grid(True, alpha=0.3)
        
        line_pitch, = ax_pitch.plot(r, design['pitch'], 'r-', linewidth=2, marker='o', markersize=4)
        ax_pitch.set_xlabel('Radial Position (m)')
        ax_pitch.set_ylabel('Pitch (deg)')
        ax_pitch.set_title('Pitch Distribution')
        ax_pitch.grid(True, alpha=0.3)
        
        line_st, = ax_stress.plot(r, design['struct']['ratio_tensile'], 'b-', linewidth=2, label='Tensile')
        line_sc, = ax_stress.plot(r, design['struct']['ratio_compress'], 'r-', linewidth=2, label='Compress')
        line_ss, = ax_stress.plot(r, design['struct']['ratio_shear'], 'g-', linewidth=2, label='Shear')
        ax_stress.axhline(y=1.0, color='k', linestyle='--', linewidth=2)
        ax_stress.set_xlabel('Radial Position (m)')
        ax_stress.set_ylabel('Stress Ratio')
        ax_stress.set_title('Stress Ratios')
        ax_stress.legend()
        ax_stress.grid(True, alpha=0.3)
        ax_stress.set_ylim([0, 1.5])
        
        # Airfoil section
        station = self.selected_station
        c = design['chord'][station]
        p = design['pitch'][station]
        x_af, y_af = self.airfoil_geom.get_scaled_section(c, p)
        line_airfoil, = ax_airfoil.plot(x_af, y_af, 'k-', linewidth=2)
        ax_airfoil.set_aspect('equal')
        ax_airfoil.set_xlabel('X (m)')
        ax_airfoil.set_ylabel('Y (m)')
        ax_airfoil.set_title(f'Section at r = {r[station]:.1f} m')
        ax_airfoil.grid(True, alpha=0.3)
        
        # Loads
        line_pn, = ax_loads.plot(r, design['bem']['pn']/1000, 'b-', linewidth=2, label='Normal')
        line_pt, = ax_loads.plot(r, design['bem']['pt']/1000, 'r-', linewidth=2, label='Tangential')
        ax_loads.set_xlabel('Radial Position (m)')
        ax_loads.set_ylabel('Force (kN/m)')
        ax_loads.set_title('Aerodynamic Loads')
        ax_loads.legend()
        ax_loads.grid(True, alpha=0.3)
        
        # Info panel
        ax_info.axis('off')
        info_text = ax_info.text(0.1, 0.9, '', transform=ax_info.transAxes, 
                                  fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        def update_info():
            design = self.optimized if self.current_design == 'optimized' else self.initial
            i = self.selected_station
            text = f"Design: {self.current_design.upper()}\n"
            text += f"{'='*30}\n"
            text += f"Cp = {design['bem']['Cp']:.4f}\n"
            text += f"Power = {design['bem']['power']/1e6:.2f} MW\n\n"
            text += f"Station: r = {design['r'][i]:.1f} m\n"
            text += f"{'='*30}\n"
            text += f"Chord = {design['chord'][i]:.3f} m\n"
            text += f"Pitch = {design['pitch'][i]:.2f}°\n"
            text += f"AoA = {design['bem']['alpha'][i]:.2f}°\n"
            text += f"Cl = {design['bem']['Cl'][i]:.3f}\n"
            text += f"Cd = {design['bem']['Cd'][i]:.4f}\n"
            text += f"L/D = {design['bem']['Cl'][i]/design['bem']['Cd'][i]:.1f}\n\n"
            text += f"σ_tensile = {design['struct']['sigma_tensile'][i]/1e6:.1f} MPa\n"
            text += f"σ_compress = {design['struct']['sigma_compress'][i]/1e6:.1f} MPa\n"
            text += f"τ_shear = {design['struct']['tau_shear'][i]/1e6:.1f} MPa\n"
            info_text.set_text(text)
        
        update_info()
        
        # Sliders
        ax_station = plt.axes([0.25, 0.15, 0.5, 0.03])
        slider_station = Slider(ax_station, 'Station', 0, len(r)-1, valinit=self.selected_station, valstep=1)
        
        # Radio buttons
        ax_radio = plt.axes([0.025, 0.5, 0.1, 0.15])
        radio = RadioButtons(ax_radio, ('Optimized', 'Initial'))
        
        def update_station(val):
            self.selected_station = int(val)
            design = self.optimized if self.current_design == 'optimized' else self.initial
            c = design['chord'][self.selected_station]
            p = design['pitch'][self.selected_station]
            x_af, y_af = self.airfoil_geom.get_scaled_section(c, p)
            line_airfoil.set_data(x_af, y_af)
            ax_airfoil.relim()
            ax_airfoil.autoscale_view()
            ax_airfoil.set_title(f'Section at r = {design["r"][self.selected_station]:.1f} m')
            update_info()
            fig.canvas.draw_idle()
        
        def update_design(label):
            self.current_design = label.lower()
            design = self.optimized if self.current_design == 'optimized' else self.initial
            
            line_chord.set_ydata(design['chord'])
            line_pitch.set_ydata(design['pitch'])
            line_st.set_ydata(design['struct']['ratio_tensile'])
            line_sc.set_ydata(design['struct']['ratio_compress'])
            line_ss.set_ydata(design['struct']['ratio_shear'])
            line_pn.set_ydata(design['bem']['pn']/1000)
            line_pt.set_ydata(design['bem']['pt']/1000)
            
            for ax in [ax_chord, ax_pitch, ax_loads]:
                ax.relim()
                ax.autoscale_view()
            
            update_station(self.selected_station)
            fig.canvas.draw_idle()
        
        slider_station.on_changed(update_station)
        radio.on_clicked(update_design)
        
        plt.show()

# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main(output_dir="./output", interactive=True):
    """
    Main function to run complete blade design workflow
    
    Parameters:
        output_dir: Directory for output files
        interactive: Show interactive visualization
    """
    print("=" * 70)
    print("5MW WIND TURBINE BLADE DESIGN TOOL")
    print("Airfoil: NACA64-A17 | Material: Epoxy-GFRP | Rotor: 126m")
    print("=" * 70)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize components
    print("\n[1/8] Initializing components...")
    airfoil_data = AirfoilData()
    airfoil_geom = AirfoilGeometry(n_points=100)
    
    # Run optimization
    print("\n[2/8] Running blade optimization...")
    print(f"  Initial chord: {TurbineConfig.initial_chord:.2f} m (constant)")
    print(f"  Initial twist: {TurbineConfig.initial_twist:.2f} deg (constant)")
    print(f"  Optimization method: {TurbineConfig.optimization_method.upper()}")
    if TurbineConfig.optimization_method in ['derivative', 'hybrid']:
        print(f"  Derivative solver: {TurbineConfig.deriv_method}")
    optimizer = BladeOptimizer(TurbineConfig, MaterialConfig, airfoil_data)
    initial, optimized, history = optimizer.optimize(verbose=True)
    
    # Generate plots
    print("\n[3/8] Generating plots...")
    plotter = PlotGenerator(output_dir)
    
    f1 = plotter.plot_airfoil_data(airfoil_data, airfoil_geom, "airfoil_data.png")
    print(f"  ✓ {f1}")
    
    f2 = plotter.plot_comparison(initial, optimized, "comparison_plots.png")
    print(f"  ✓ {f2}")
    
    f3 = plotter.plot_3d_blade(optimized['r'], optimized['chord'], optimized['pitch'], 
                               airfoil_geom, "blade_3d.png")
    print(f"  ✓ {f3}")
    
    # Generate optimization progress plot
    f3b = plotter.plot_optimization_progress(history, "optimization_progress.png")
    print(f"  ✓ {f3b}")
    
    # Save CSV files
    print("\n[4/8] Saving CSV files...")
    output = ResultsOutput(output_dir)
    
    f4 = output.save_csv(initial, "initial_design.csv", "Initial Design (Constant Chord/Twist)")
    print(f"  ✓ {f4}")
    
    f5 = output.save_csv(optimized, "optimized_design.csv", "Optimized Design")
    print(f"  ✓ {f5}")
    
    f6 = output.save_comparison_csv(initial, optimized, "comparison_summary.csv")
    print(f"  ✓ {f6}")
    
    f7 = output.save_json(initial, optimized, airfoil_data, "blade_data.json")
    print(f"  ✓ {f7}")
    
    # Save optimization history CSV
    f7b = output.save_optimization_history_csv(history, "optimization_history.csv")
    print(f"  ✓ {f7b}")
    
    # Generate CAD files
    print("\n[5/8] Generating CAD files...")
    cad = CADGenerator(airfoil_geom)
    
    stl_file = os.path.join(output_dir, "blade_geometry.stl")
    cad.generate_stl(optimized['r'], optimized['chord'], optimized['pitch'], stl_file)
    print(f"  ✓ {stl_file}")
    
    obj_file = os.path.join(output_dir, "blade_geometry.obj")
    cad.generate_obj(optimized['r'], optimized['chord'], optimized['pitch'], obj_file)
    print(f"  ✓ {obj_file}")
    
    iges_file = os.path.join(output_dir, "blade_geometry.igs")
    cad.generate_iges(optimized['r'], optimized['chord'], optimized['pitch'], iges_file)
    print(f"  ✓ {iges_file}")
    
    # Save airfoil coordinates
    print("\n[6/8] Saving airfoil coordinates...")
    airfoil_csv = os.path.join(output_dir, "naca64a17_coords.csv")
    with open(airfoil_csv, 'w') as f:
        f.write("# NACA64-A17 Airfoil Coordinates\nx/c,y/c\n")
        for i in range(len(airfoil_geom.x_coords)):
            f.write(f"{airfoil_geom.x_coords[i]:.6f},{airfoil_geom.y_coords[i]:.6f}\n")
    print(f"  ✓ {airfoil_csv}")
    
    # Print optimization history summary
    print("\n[7/8] Optimization History Summary:")
    print("-" * 70)
    print(f"{'Iter':<6} {'Phase':<22} {'Cp':<12} {'Max Stress':<14} {'Status'}")
    print("-" * 70)
    phase_names = {1: 'Rule-Based', 2: 'Derivative/GA', 3: 'Hybrid GA Polish'}
    for i in range(len(history['iteration'])):
        phase = history.get('phase', [1] * len(history['iteration']))[i]
        phase_name = phase_names.get(phase, 'Unknown')
        status = "✓ PASS" if history['max_stress_ratio'][i] <= 1.0 else "✗ FAIL"
        print(f"{history['iteration'][i]+1:<6} {phase_name:<22} {history['Cp'][i]:<12.4f} {history['max_stress_ratio'][i]:<14.3f} {status}")
    print("-" * 70)
    
    # Print summary
    print("\n[8/8] Design Summary:")
    print("=" * 70)
    print(f"{'Parameter':<30} {'Initial':<20} {'Optimized':<20}")
    print("-" * 70)
    print(f"{'Power Coefficient (Cp)':<30} {initial['bem']['Cp']:<20.4f} {optimized['bem']['Cp']:<20.4f}")
    print(f"{'Power (MW)':<30} {initial['bem']['power']/1e6:<20.3f} {optimized['bem']['power']/1e6:<20.3f}")
    
    max_init = max(np.max(initial['struct']['ratio_tensile']),
                   np.max(initial['struct']['ratio_compress']),
                   np.max(initial['struct']['ratio_shear']))
    max_opt = max(np.max(optimized['struct']['ratio_tensile']),
                  np.max(optimized['struct']['ratio_compress']),
                  np.max(optimized['struct']['ratio_shear']))
    
    status_init = "✗ FAIL" if max_init > 1 else "✓ OK"
    status_opt = "✗ FAIL" if max_opt > 1 else "✓ OK"
    
    print(f"{'Max Stress Ratio':<30} {max_init:<15.3f} {status_init:<5} {max_opt:<15.3f} {status_opt}")
    print(f"{'Max Chord (m)':<30} {np.max(initial['chord']):<20.3f} {np.max(optimized['chord']):<20.3f}")
    print("=" * 70)
    
    print(f"\nAll output files saved to: {os.path.abspath(output_dir)}")
    
    # Interactive visualization
    if interactive:
        print("\n[Interactive] Launching visualization...")
        viz = InteractiveVisualizer(initial, optimized, airfoil_data, airfoil_geom)
        viz.show()
    
    return initial, optimized, history

# ============================================================================
# RUN
# ============================================================================
if __name__ == "__main__":
    # Run with output to current directory, interactive mode off for batch processing
    initial, optimized, history = main(output_dir="./", interactive=False)

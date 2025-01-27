from emtworkflow import *

swgaf = swg_adiabatic_filter(
    #For detailed variable descriptions, see emtworkflow.py
    width_wg1=0.44e-6,              
    width_wg1_taper_tip=0.06e-6,    
    width_wg2=0.245e-6,            
    width_wg2_tip=0.1e-6,          
    
    
    width_fishbone=0.0e-6,         
    width_wg_port=0.5e-6,          
    
    swg_pitch=0.25e-6,             
    swg_dc=0.5,                     
    
    length_swg_taper=20e-6,
    length_coupling=80e-6,
    length_bend=150e-6,
    length_bend_output=50e-6,

    gap_start=0.2e-6,
    gap_end=1.3e-6,
    output_pitch=3e-6,
    
    thickness_superstrate=3e-6,
    thickness_device=0.22e-6,
    thickness_substrate=2e-6,

    mat_superstrate="SiO2 (Glass) - Palik",
    mat_device="Si (Silicon) - Palik",
    mat_substrate="SiO2 (Glass) - Palik",
)

#swgaf.FDE_stripwg(folder=True,simfile=True)                          # 1a - FDE Strip Waveguide Simulation    
#swgaf.EMT_swg(simulate_lumerical=True,folder=True,simfile=True)      # 1b - FDE EMT Material Wavgeuide Simulation
#swgaf.FDTD_swg(folder=True,simfile=True)                             # 1c - Optional: FDTD SWG Waveguide Simulation

#swgaf.supermodes(folder=True,simfile=True)                           # 2: SUPERMODE ANALYSIS
#swgaf.EMT_bandstructure(folder=True,simfile=True)                    # 3: FDTD EMT Bandstructure Simulation 
#swgaf.optimize_device(folder=True,simfile=True)                      # 4: EME Optimize coupling lengths
swgaf.generate_spectrum(folder=True,simfile=True)                    # 4a - Optional: EME Spectrum generation to get a rough gauge device is functioning as a filter, NOT VALID for analyzing crossing wavelength.

#swgaf.FDTD_fullsim(folder=True,simfile=True)                         # 5 Final Verification
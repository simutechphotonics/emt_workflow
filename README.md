# Effective Medium Theory Workflow
A Lumerical workflow for designing Sub-Wavelength Grating (SWG) Filters leveraging interoperability between tools to reduce the design time. 
For details on the workflow methodology and device details, SimuTech Group has a whitepaper available [here](https://simutechgroup.com/photonic-sub-wavelength-grating-based-devices-whitepaper/).

## Requirements
- [opticalmaterialspy*](https://github.com/jtambasco/opticalmaterialspy)
- scipy
- numpy
- matplotlib
- six
- Ansys Lumerical Suite (FDE, FDTD)

*The newest version of this library is not working correctly with pulling the index data from online sources. A local library is included in this repo that fixes syntax errors for the web-retrieval of optical material data.

## How to Use
Import `emtworkflow.py` and create a `swgaf` object with various design parameters. Afterwards, call the various subfunctions (listed below) for the `swgaf` object to perform simulations.

Users can use `example_design.py` as a template for their own design.

## Functions
The arguments for each function are avaiable as comments in `emtworkflow.py`.

### Wg_analysis()
Simulates the strip and subwavelength grating waveguides. The SWG can be simulated in FDE (using EMT) or FDTD. This functions calls `FDE_stripwg()` for the strip waveguide and `EMT_swg()` or `FDTD_swg()` for the subwavelength grating.

### FDE_stripwg()
Performs a waveguide simulation using FDE. Outputs the effective and group index of the waveguide for a given wavelength range.

### EMT_swg()
Performs a waveguide simulation for the subwavelength grating waveguide using FDE. a bulk EMT material is utilized to represent the equivalent effective index of the SWG. 
This method is faster but does not capture dispersion of the SWG.
Calculates the effective and group index for the EMT material and simulates EMT-waveguide for a given wavelength range.

### FDTD_swg()
*Performs a FDTD bandstructure simulation for the subwavelength grating using periodic boundaries. Outputs
Outputs the band diagram for the SWG waveguide, the propagation constant vs. angular frequency, the effective index/group index of the SWG waveguide, and the wavevector vs. wavelength plot.

### supermodes()
Performs a supermode analysis for the strip and SWG waveguides using FDE. material data.
Outputs the effective index vs. wavelength for the odd and even supermode. Also outputs the group index vs. wavelength for the odd and even supermode.

### EMT_bandstructure()
*Performs a bandstructure analysis in FDTD using the bulk EMT material to represent the SWG. This function is to compare the differences between the EMT and FDTD bandstructure results.
Outputs the effective and group indices vs. wavelength for the EMT and FDTD simulations.

### optimize_device()
Performs an EME simulation to find the optimal length for the filter coupling. This method uses EMT to represent the SWG.
Outputs the transmitted power at each output port for a range of coupling lengths. Two results are performed, one for each input waveguide.

### generate_spectrum()
Performs a wavelength sweep using EME to obtain the full spectrum of the filter device. This method uses EMT to estimate the device performance. If plots are shown, the plots will need to be manually closed for this function to complete operation.

#### FDTD_fullsim()
Creates a simulation file for the full device in FDTD. The simulation is very large and this function does not automatically run the simulation. Users can use this as a final validation of the design.




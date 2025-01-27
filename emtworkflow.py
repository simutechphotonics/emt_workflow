# %% -*- coding: utf-8 -*-
"""
SWG Adiabatic filter design recipe.

Created on Mon May 29 11:56:32 2023
@author: Stephen Lin (SimuTech Group) and Mustafa Hammood

"""
import sys, os                                                          #pathing for imports
sys.path.append("C:\\Program Files\\Lumerical\\v242\\api\\python\\")    #default path for current release  
sys.path.append(os.path.dirname(__file__))                              #Current directory
import lumapi
#import opticalmaterialspy as mat                                       #for pulling material data from web database
import LIB_opticalmaterialspy as mat                                    #included package as the public version is bugged
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib
from scipy.interpolate import interp1d

class swg_adiabatic_filter:
    """Adaibatic SWG filter design object."""

    def __init__(
        self,
        width_wg1,
        width_wg1_taper_tip,
        width_wg2_tip,
        width_wg2,
        width_fishbone,
        width_wg_port,
        swg_pitch,
        swg_dc,
        length_swg_taper,
        length_coupling,
        length_bend,
        length_bend_output,
        gap_start,
        gap_end,
        output_pitch,
        thickness_superstrate,
        thickness_device,
        thickness_substrate,
        mat_superstrate,
        mat_device,
        mat_substrate,
    ):
        """
        initialization method. All units are SI.

        Parameters
        ----------
        width_wg1 : float
            Waveguide 1 (SWG) width.
        width_wg2_tip : float
            Waveguide 2 (strip) tip, starting, width.
        width_wg2 : float
            Waveguide 2 (strip) width.
        width_fishbone : float
            SWG fishbone waveguide width.
        swg_pitch : float
            Sub wavelength period.
        swg_dc : float
            Sub wavelength duty cylce. value from 0 to 1.
        length_swg_taper : float
            Strip-subwavelength taper length.
        length_coupling : float
            Initial SWG-strip coupling region length.
        length_bend : float
            Waveguide 2 (strip) tapering away bend section length.
        length_bend_output : float
            Waveguide 2 (strip) bend section length to output pitch.
        gap_start : float
            Initial WG1 and WG2 gap.
        gap_end : float
            Final WG1 and WG2 gap.
        output_pitch : float
            Output waveguides pitch.
        thickness_superstrate : float
            Superstrate layer thickness.
        thickness_device : float
            Device layer thickness.
        thickness_substrate : float
            Substrate layer thickness.
        mat_superstrate : string
            Superstrate layer material name.
        mat_device : string
            Device layer material name.
        mat_substrate : string
            Substrate layer material name.

        Returns
        -------
        None.

        """
        self.width_wg1 = width_wg1
        self.width_wg1_taper_tip = width_wg1_taper_tip
        self.width_wg2_tip = width_wg2_tip
        self.width_wg2 = width_wg2
        self.width_fishbone = width_fishbone
        self.width_wg_port = width_wg_port
        self.swg_pitch = swg_pitch
        self.swg_dc = swg_dc
        self.length_swg_taper = length_swg_taper
        self.length_coupling = length_coupling
        self.length_bend = length_bend
        self.length_bend_output = length_bend_output
        self.gap_start = gap_start
        self.gap_end = gap_end
        self.output_pitch = output_pitch
        self.thickness_superstrate = thickness_superstrate
        self.thickness_device = thickness_device
        self.thickness_substrate = thickness_substrate
        self.mat_superstrate = mat_superstrate
        self.mat_device = mat_device
        self.mat_substrate = mat_substrate

    @property
    def device_id(self):
        return f"swg_af_WGA{int(self.width_wg1*1e9)}_WGB{int(self.width_wg2*1e9)}_Gstart{int(self.gap_start*1e9)}_Gend{int(self.gap_end*1e9)}"

    def FDTD_swg(
        self,
        mesh=2,
        finemesh_y=20e-9,
        finemesh_z=20e-9,
        sim_time=501e-15,
        f1=150e12,
        f2=250e12,
        kx_start=0.2,
        kx_stop=0.38,
        kx_pts=5,
        apod_center=0.5,
        apod_width=0.125,
        visual=True,
        folder = False,
        simfile=False,
    ):
        """
        Simulate the SWG waveguide using FDTD bloch-boundary bandstructure method.

        Parameters
        ----------
        mesh : int, optional
            FDTD overall solver mesh accuracy (1-8). The default is 2.
        finemesh_y : float, optional
            Fine mesh region (y) around the waveguide step size. The default is 20e-9.
        finemesh_z : TYPE, optional
            Fine mesh region (z) around the waveguide the waveguide step size. The default is 20e-9.
        sim_time : float, optional
            FDTD simulation time. The default is 501e-15.
        f1 : float, optional
            Frequency start point. The default is 150e12.
        f2 : float, optional
            Frequency stop point. The default is 250e12.
        kx_start : float, optional
            Normalized wavevector start. The default is 0.2.
        kx_stop : float, optional
            Normalized wavevector stop. The default is .38.
        kx_pts : int, optional
            Number of wavevector points in the bandstructure. The default is 5.
        apod_center : float, optional
            Inverse fourier transform's normalized apodization center. The default is 0.5.
        apod_width : float, optional
            Inverse fourier transform's normalized apodization width. The default is 0.125.
        visual : Boolean, optional
            Visualization flag. The default is True.
        folder : Boolean, optional
            If True, a sub folder is created in the current working directory and the generated files are placed there. The default is False.
        simfile : Boolean, optinal
            If True, a sim file will be saved.
        Returns
        -------
        None.

        """
        print("Running: FDTD for SWG.")
        # Create Folder?
        if folder:
            folder_path="1c_FDTD_swg/"        
            # Create Folder if doesn't exist
            isExist = os.path.exists(folder_path)
            if not isExist:
                os.makedirs(folder_path)
        else:
            folder_path=""
            
        sim_buffer_y = 0.1e-6  # buffer distance between material and sim region
        x_span = self.swg_pitch * 4 + 2e-6
        y_span = self.width_wg1 + 3e-6
        y_span_fdtd = y_span - 2 * sim_buffer_y
        
        with lumapi.FDTD(hide=False) as fdtd:
            # Insert Custom Material for simulation (Palik Silicon, etc.)
            self.init_custom_material(fdtd)
            
            # Add Superstrate
            fdtd.addrect()
            fdtd.set("name", "superstrate")
            fdtd.set("material", self.mat_superstrate)
            fdtd.set("x", 0)
            fdtd.set("x span", x_span)
            fdtd.set("z min", 0)
            fdtd.set("z max", self.thickness_substrate)
            fdtd.set("y", 0)
            fdtd.set("y span", y_span)
            fdtd.set("override mesh order from material database", 1)
            fdtd.set("mesh order", 3)
            fdtd.set("alpha", 0.2)

            # Add Substrate
            fdtd.addrect()
            fdtd.set("name", "substrate")
            fdtd.set("material", self.mat_substrate)
            fdtd.set("x", 0)
            fdtd.set("x span", x_span)
            fdtd.set("z min", -self.thickness_substrate)
            fdtd.set("z max", 0)
            fdtd.set("y", 0)
            fdtd.set("y span", y_span)
            fdtd.set("override mesh order from material database", 1)
            fdtd.set("mesh order", 3)
            fdtd.set("alpha", 0.2)

            # Add Fishbone taper section
            fdtd.addrect()
            fdtd.set("name", "fishbone")
            fdtd.set("material", self.mat_device)
            fdtd.set("x", 0)
            fdtd.set("x span", x_span)
            fdtd.set("z min", 0)
            fdtd.set("z max", self.thickness_device)
            fdtd.set("y", 0)
            fdtd.set("y span", self.width_fishbone)

            # Add SWG teeth
            x0 = -2 * self.swg_pitch
            n_swg = 4
            for i in range(n_swg):
                fdtd.addrect()
                fdtd.set("name", f"swg{i}")
                fdtd.set("material", self.mat_device)
                fdtd.set("x min", x0 + i * self.swg_pitch)
                fdtd.set(
                    "x max", x0 + i * self.swg_pitch + self.swg_pitch * self.swg_dc
                )
                fdtd.set("z min", 0)
                fdtd.set("z max", self.thickness_device)
                fdtd.set("y", 0)
                fdtd.set("y span", self.width_wg1)

            # Add FDTD Object
            fdtd.addfdtd()
            fdtd.set("mesh accuracy", mesh)             # 2 is usually okay for >1.4µm
            fdtd.set("simulation time", sim_time)       # 500 fs seems converging
            fdtd.set("x min", -self.swg_pitch)
            fdtd.set("x max", 0)
            fdtd.set("y", 0)
            fdtd.set("y span", y_span_fdtd)
            fdtd.set("z min", -5 * self.thickness_device)
            fdtd.set("z max", 6 * self.thickness_device)
            fdtd.set("x min bc", "Bloch")
            if self.mat_substrate == self.mat_superstrate:
                fdtd.set("y min bc", "Anti-Symmetric")
            else:
                fdtd.set("y min bc", "PML")
            fdtd.set("y max bc", "PML")
            fdtd.set("z min bc", "PML")
            fdtd.set("z max bc", "PML")
            fdtd.set("set based on source angle", 0)

            # Add fine mesh
            fdtd.addmesh()
            fdtd.set("x min", -self.swg_pitch)
            fdtd.set("x max", 0)
            fdtd.set("y", 0)
            fdtd.set("y span", 1.5 * self.width_wg1)
            fdtd.set("z", self.thickness_device / 2)
            fdtd.set("z span", 1.5 * self.thickness_device)
            fdtd.set("override x mesh", False)
            fdtd.set("dy", finemesh_y)
            fdtd.set("dz", finemesh_z)

            # Add source
            fdtd.addplane()
            fdtd.set("injection axis", "x-axis")
            fdtd.set("x", -self.swg_pitch / 2)
            fdtd.set("y", 0)
            fdtd.set("y span", y_span_fdtd / 2) #NOTE: v2023 R2, The value is double when passed. Seems to be a bug.
            fdtd.set("z min", -5 * self.thickness_device)
            fdtd.set("z max", 6 * self.thickness_device)
            fdtd.set("frequency start", f1)
            fdtd.set("frequency stop", f2)

            # Add bandstructure object
            fdtd.addobject("bandstructure")
            fdtd.set("x", -self.swg_pitch / 2)
            fdtd.set("x span", self.swg_pitch)
            if self.mat_substrate == self.mat_superstrate:
                fdtd.set("y", self.width_wg1 / 4)
                fdtd.set("y span", self.width_wg1 / 2)
            else:
                fdtd.set("y", 0)
                fdtd.set("y span", 1.5 * self.width_wg1)
            fdtd.set("z", self.thickness_device / 2)
            fdtd.set("z span", 2 * self.thickness_device)
            
            # NOTE: Using modified bandstructure object analysis script because
            # 2020 R1 changed some details and no longer worked.
            
            # Create analysis group property variables
            fdtd.addanalysisprop("apod_width", 0, apod_width)
            fdtd.addanalysisprop("apod_center", 0, apod_center)
            # Load analysis script into the bandstructure object
            analysis_script = open('bandstructure_analysis_script.lsf', 'r').read()
            fdtd.set("analysis script", analysis_script)

            # Setup model object
            fdtd.eval("select('');")  # selects top level "model"
            fdtd.eval(f"adduserprop('pitch', 2, {self.swg_pitch});")
            fdtd.eval(f"adduserprop('apod_center', 0, {apod_center});")
            fdtd.eval(f"adduserprop('apod_width', 0, {apod_width});")
            fdtd.eval(f"adduserprop('f1', 4, {f1});")
            fdtd.eval(f"adduserprop('f2', 4, {f2});")
            fdtd.eval("adduserprop('kx', 0, 0.5);")
            
            #NOTE: Using adduserprop directly through LumAPI will proc a warning
            # Lumerical is aware of this and updating the API.
            # Lumerical internal team has suggested to use the function rather than .eval()
            # as it will supress useful debug information if and error occurs.
            # fdtd.adduserprop("pitch", 2, self.swg_pitch)
            # fdtd.adduserprop("apod_center", 0, apod_center)
            # fdtd.adduserprop("apod_width", 0, apod_width)
            # fdtd.adduserprop("f1", 4, f1)
            # fdtd.adduserprop("f2", 4, f2)
            # fdtd.adduserprop("kx", 0, 0.5)
            
            setup_script = open('setup_script.lsf', 'r').read()
            fdtd.set("setup script", setup_script)
            
            # Add sweep object
            # https://optics.ansys.com/hc/en-us/articles/360034930413-addsweep
            fdtd.addsweep()
            kx = dict()
            kx["Name"] = "kx"
            kx["Parameter"] = "::model::kx"
            kx["Type"] = "Number"
            kx["Start"] = kx_start
            kx["Stop"] = kx_stop
            
            # Add Sweep parameter
            # https://optics.ansys.com/hc/en-us/articles/360034930493-addsweepparameter
            fdtd.addsweepparameter("sweep", kx)

            # Add a sweep result to report
            # https://optics.ansys.com/hc/en-us/articles/360034410034-addsweepresult
            spectrum = dict()
            spectrum["Name"] = "spectrum"
            spectrum["Result"] = "::model::bandstructure::spectrum"
            fdtd.addsweepresult("sweep", spectrum)
            
            # Set sweep range
            # https://optics.ansys.com/hc/en-us/articles/360034930473-setsweep
            fdtd.setsweep("sweep", "number of points", kx_pts)

            # Save and Run sweep
            swg_wg_id = (
                f"swg_width{int(1e9*self.width_wg1)}nm_pitch{int(1e9*self.swg_pitch)}nm"
            )
            if simfile:
                fdtd.save(folder_path+f"{swg_wg_id}")
            fdtd.runsweep("sweep")

            # Analyze Results
            spectrum = fdtd.getsweepresult("sweep", "spectrum")
            kx = spectrum["kx"][0]
            f = spectrum["f"]
            fs = np.log10(spectrum["fs"])
            beta = kx * 2 * np.pi / self.swg_pitch
            c = 299792458  # m/s

            fband = []
            for i in range(len(kx)):
                peak_pos = np.argmax(np.abs(np.transpose(fs)[i]))
                fband.append(f[peak_pos][0])

            # Fit and resample on higher resolution
            N = 3  # fit order
            npts = 100  # resampling points
            w = 2 * np.pi * np.array(fband)
            p = np.flip(np.polyfit(beta, w, N))
            beta_resample = np.linspace(min(beta), max(beta), npts)
            w_resample = p[0]

            for i in range(1, len(p)):
                w_resample = w_resample + p[i] * beta_resample ** [i]

            lightline_w = np.linspace(
                0.95 * np.min(w_resample), 1.05 * np.max(w_resample), npts
            )
            n_substrate = fdtd.getfdtdindex(
                self.mat_substrate, lightline_w / (2 * np.pi), np.min(f), np.max(f)
            )

            v_phase = w_resample / beta_resample  # phase velocity
            dw_dbeta = p[1] # sample dw/dbeta to get the group velocity
            for i in range(2, len(p)):
                dw_dbeta = dw_dbeta + p[i] * i * beta_resample ** [i - 1]
            v_group = dw_dbeta

            # Calculate neff and ng
            self.swg_neff = c / v_phase
            self.swg_ng = c / v_group

            f_resample = w_resample / (2 * np.pi)
            self.swg_wavl = c / f_resample
            if visual:
                font = {"family": "DejaVu Sans", "size": 18}
                matplotlib.rc("font", **font)
                
                # Plot raw bandstructure
                # Normalize 'fs' to [0, 1] before displaying it.
                fs_normalized = (fs - fs.min()) / (fs.max() - fs.min())

                plt.figure(figsize=(10, 6))
                plt.imshow(
                    fs_normalized,
                    origin="lower",
                    aspect="auto",
                    cmap="hot",
                    extent=[kx.min(), kx.max(), 1e6 * c / f.min(), 1e6 * c / f.max()],
                )
                plt.colorbar(label="Intensity")
                plt.title("Photonic Band Diagram")
                plt.xlabel("Normalized wavevector [2*pi/pitch]")
                plt.ylabel("Wavelength [µm]")
                plt.title("Photonic Band Diagram Analysis For SWG WG")
                plt.savefig(folder_path+f"banstructure_{swg_wg_id}")
                #plt.show() #Enabling will pause script

                plt.figure(figsize=(10, 6))
                plt.scatter(kx, 1e6 * c / np.array(fband))
                plt.plot(kx, 1e6 * c / np.array(fband), "--", color="blue")
                plt.xlabel("Normalized wavevector [2*pi/pitch]")
                plt.ylabel("Wavelength [µm]")
                plt.title("Photonic Band Diagram Analysis For SWG WG")
                plt.savefig(folder_path+f"wavevector_{swg_wg_id}")
                #plt.show() #Enabling will pause script

                lightline_beta = lightline_w * np.real(n_substrate) / c
                lightline_beta = lightline_beta[0]
                plt.figure(figsize=(10, 6))
                plt.scatter(beta, w, color="red", label="Bandstructure data")
                plt.plot(
                    beta_resample, w_resample, "--", color="red", label="Polyfit data"
                )
                plt.plot(
                    lightline_beta, lightline_w, color="blue", label="Lightline data"
                )
                plt.xlabel("Propagation constant, β [1/m]")
                plt.ylabel("Angular frequency, ω [rad/s]")
                plt.title("Photonic Band Diagram Analysis For SWG WG")
                plt.legend()
                plt.savefig(folder_path+f"lightline_{swg_wg_id}")
                #plt.show() #Enabling will pause script

                fig, ax1 = plt.subplots(figsize=(10, 6))
                ax1.plot(self.swg_wavl * 1e6, self.swg_neff, color="r")
                ax1.set_ylabel("Effective index", color="r")
                ax1.set_xlabel("Wavelength [µm]")
                ax1.set_title("SWG Waveguide Model Using FDTD Bandstructure")
                for tl in ax1.get_yticklabels():
                    tl.set_color("r")
                ax2 = ax1.twinx()
                ax2.plot(self.swg_wavl * 1e6, self.swg_ng, color="b")
                ax2.set_ylabel("Group index", color="b")
                for tl in ax2.get_yticklabels():
                    tl.set_color("b")
                fig.savefig(folder_path+f"neff_ng_{swg_wg_id}")
                #fig.show() #Enabling will pause script
        print("Complete: FDTD for SWG.")

    def EMT_swg(
        self,
        wavl_start=1.3e-6,
        wavl_stop=1.8e-6,
        npts=500,
        visual=True,
        simulate_lumerical=True,
        mesh_dx=50e-9,
        mesh_dy=50e-9,
        mode_npts=5,
        folder=False,
        simfile=False,
    ):
        """
        Model an SWG waveguide using the effective medium theory method.

        Parameters
        ----------
        mesh_dx : float, optional
            X Mesh step in the eigenment solver. The default is 25e-9.
        mesh_dy : float, optional
            Y Mesh step in the eigenment solver. The default is 25e-9.
        wavl_start : float, optional
            Wavelength sweep start point. The default is 1.4e-6.
        wavl_stop : float, optional
            Wavelength sweep stop point. The default is 1.7e-6.
        npts : int, optional
            EMT model wavelength points. The default is 500.
        mode_npts : TYPE, optional
            Wavelength sweep points. The default is 5.
        visual : Boolean, optional
            Visualization flag. The default is True.
        folder : Boolean, optional
            If True, a sub folder is created in the current working directory and the generated files are placed there. The default is False.
        simfile : Boolean, optinal
            If True, a sim file will be saved.
            
        Returns
        -------
        None.

        """
        print("Running: FDE for EMT-SWG.")
        # Create Folder?
        if folder:
            folder_path="1b_EMT_swg/"        
            # Create Folder if doesn't exist
            isExist = os.path.exists(folder_path)
            if not isExist:
                os.makedirs(folder_path)
        else:
            folder_path=""
        
        # Pull refractive index information from webdatabase
        # Si Li model 1980 (1.2 µm - 14 µm) - 293 K ) 
        # TODO: Couldn't find Palik data.
        Si = mat.RefractiveIndexWeb(
            "https://refractiveindex.info/?shelf=main&book=Si&page=Li-293K"
        )
        SiO2 = mat.SiO2()
        
        # Volume of material
        vol_dev = self.swg_dc # SWG Duty cycle
        vol_clad = 1 - self.swg_dc # Cladding

        self.wavl_emt = np.linspace(wavl_start, wavl_stop, npts)
        self.swg_n_emt = []
        for i in self.wavl_emt:
            self.swg_n_emt.append(
                Si.n(i) * vol_dev + SiO2.n(i) * vol_clad
            )

        # Calculate group index (irrelevant for this example)
        self.swg_ng_emt = []
        for i in self.wavl_emt:
            self.swg_ng_emt.append(
                Si.ng(i) * vol_dev + SiO2.ng(i) * vol_clad
            )

        # Simulate in Lumerical
        swg_emt_id = f"swg_emt_width{int(1e9*self.width_wg1)}nm"
        if simulate_lumerical:
            # Create an equivalent material model in lumerical
            # Create an array with zeros for the third column (imaginary)
            zeros = np.zeros_like(self.wavl_emt)
            # Combine the variables into a single array
            data = np.column_stack((self.wavl_emt * 1e9, self.swg_n_emt, zeros))
            # Define the filename for the text file
            filename = folder_path+f"{swg_emt_id}_material.txt"
            # Save the data to a text file
            np.savetxt(filename, data, delimiter="\t", fmt="%.6f")
            
            # Simulation setup parameters
            sim_buffer_y = 0.1e-6                   # buffer between material and sim region
            x_span = self.width_wg1 * 8 + 2e-6
            x_span_fde = x_span - 2 * sim_buffer_y
            y_span = self.thickness_device + 8e-6
            y_span_fde = y_span - 2 * sim_buffer_y
            z_span = 1e-6

            with lumapi.MODE(hide=False) as mode:
                # Add material
                mode.eval("swg_material = addmaterial('Sampled 3D data');")
                mode.eval(f"setmaterial(swg_material, 'name', '{swg_emt_id}');")
                # Format the data for Lumerical's format
                c = 299792458  # m/s
                wavl_arr = np.array2string(c / self.wavl_emt, separator=";").replace(
                    "\n", ""
                )
                n_arr = np.array2string(
                    np.array(self.swg_n_emt) ** 2, separator=";"
                ).replace("\n", "")
                mode.eval(
                    f"setmaterial('{swg_emt_id}', 'sampled 3d data', [{wavl_arr},{n_arr}]);"
                )
                
                # Add structures
                mode.addrect()
                mode.set("name", "substrate")
                mode.set("material", self.mat_substrate)
                mode.set("x", 0)
                mode.set("x span", x_span)
                mode.set("y max", 0)
                mode.set("y min", -self.thickness_substrate)
                mode.set("z", 0)
                mode.set("z span", z_span)
                mode.set("override mesh order from material database", 1)
                mode.set("mesh order", 3)
                mode.set("alpha", 0.2)

                mode.addrect()
                mode.set("name", "superstrate")
                mode.set("material", self.mat_superstrate)
                mode.set("x", 0)
                mode.set("x span", x_span)
                mode.set("y min", 0)
                mode.set("y max", self.thickness_superstrate)
                mode.set("z", 0)
                mode.set("z span", z_span)
                mode.set("override mesh order from material database", 1)
                mode.set("mesh order", 3)
                mode.set("alpha", 0.2)

                mode.addrect()
                mode.set("name", "waveguide")
                mode.set("material", swg_emt_id)
                mode.set("x", 0)
                mode.set("x span", self.width_wg1)
                mode.set("y min", 0)
                mode.set("y max", self.thickness_device)
                mode.set("z", 0)
                mode.set("z span", z_span)

                mode.addfde()
                mode.set("x", 0)
                mode.set("x span", x_span_fde)
                mode.set("y", self.thickness_device / 2)
                mode.set("y span", y_span_fde)
                mode.set("z", 0)
                mode.set("define x mesh by", "maximum mesh step")
                mode.set("define y mesh by", "maximum mesh step")
                mode.set("dx", mesh_dx)
                mode.set("dy", mesh_dy)
                if self.mat_substrate == self.mat_superstrate:
                    mode.set("y min bc", "symmetric")
                else:
                    mode.set("y min bc", "PML")
                mode.set("x min bc", "PML")
                mode.set("x max bc", "PML")

                #Run Simulation
                mode.run()
                
                #Run Analysis
                mode.setanalysis("wavelength", wavl_start)
                mode.setanalysis("stop wavelength", wavl_stop)
                mode.setanalysis("number of points", mode_npts)
                mode.setanalysis("number of trial modes", 1)
                
                mode.findmodes()
                mode.selectmode(1)
                mode.setanalysis("track selected mode", 1)
                mode.setanalysis("detailed dispersion calculation", 1)
                mode.frequencysweep()
                
                swg_wg_id = (
                f"swg_width{int(1e9*self.width_wg1)}nm_pitch{int(1e9*self.swg_pitch)}nm"
                )
                if simfile:
                    mode.save(folder_path+"EMT_"+f"{swg_wg_id}")
                    
                f = mode.getdata("frequencysweep", "f_D")
                f = [i[0] for i in f]
                wavl = c / np.array(f)

                vg = np.abs(mode.getdata("frequencysweep", "vg"))
                vg = [i[0] for i in vg]

                ng = c / np.array(vg)
                neff = np.abs(mode.getdata("frequencysweep", "neff"))
                neff = [i[0] for i in neff]

                # Fit and resample on higher resolution
                N = 3  # fit order
                pts = npts  # resampling points

                wavl_poly_coeffs = np.polyfit(np.arange(len(wavl)), wavl, deg=N)
                self.swg_emt_wavl = np.polyval(
                    wavl_poly_coeffs, np.linspace(0, len(wavl) - 1, pts)
                )

                ng_poly_coeffs = np.polyfit(np.arange(len(ng)), ng, deg=N)
                self.swg_emt_ng = np.polyval(
                    ng_poly_coeffs, np.linspace(0, len(ng) - 1, pts)
                )

                neff_poly_coeffs = np.polyfit(np.arange(len(neff)), neff, deg=N)
                self.swg_emt_neff = np.polyval(
                    neff_poly_coeffs, np.linspace(0, len(neff) - 1, pts)
                )

                if visual:
                    font = {"family": "normal", "size": 18}

                    fig, ax1 = plt.subplots(figsize=(10, 6))
                    ax1.plot(self.swg_emt_wavl * 1e6, self.swg_emt_neff, color="r")
                    ax1.set_ylabel("Effective Index", color="r")
                    ax1.set_xlabel("Wavelength [µm]")
                    ax1.set_title(f"EMT Material Waveguide Model for width="+str(int(1e9*self.width_wg1))+"nm")
                    for tl in ax1.get_yticklabels():
                        tl.set_color("r")
                    ax2 = ax1.twinx()
                    ax2.plot(self.swg_emt_wavl * 1e6, self.swg_emt_ng, color="b")
                    ax2.set_ylabel("Group index", color="b")
                    for tl in ax2.get_yticklabels():
                        tl.set_color("b")
                    fig.savefig(folder_path+f"waveguide_model_{swg_emt_id}")
                    fig.show()

        if visual:
            font = {"family": "normal", "size": 18}

            fig, ax1 = plt.subplots(figsize=(10, 6))
            ax1.plot(self.wavl_emt * 1e6, self.swg_n_emt, color="r")
            ax1.set_ylabel("Effective Index", color="r")
            ax1.set_xlabel("Wavelength [µm]")
            ax1.set_title("EMT Material Index")
            for tl in ax1.get_yticklabels():
                tl.set_color("r")
            ax2 = ax1.twinx()
            ax2.plot(self.wavl_emt * 1e6, self.swg_ng_emt, color="b")
            ax2.set_ylabel("Group Index", color="b")
            for tl in ax2.get_yticklabels():
                tl.set_color("b")
            fig.savefig(folder_path+f"material_index_{swg_emt_id}")
            fig.show()
        print("Complete: FDE for EMT-SWG.")
        
        
    def FDE_stripwg(
        self,
        mesh_dx=25e-9,
        mesh_dy=25e-9,
        wavl_start=1.4e-6,
        wavl_stop=1.7e-6,
        npts=5,
        visual=True,
        folder=False,
        simfile=False,
        
    ):
        """
        Simulating the strip waveguide (Waveguide 2) using eigenmode solver.

        Parameters
        ----------
        mesh_dx : float, optional
            X Mesh step in the eigenment solver. The default is 25e-9.
        mesh_dy : float, optional
            Y Mesh step in the eigenment solver. The default is 25e-9.
        wavl_start : float, optional
            Wavelength sweep start point. The default is 1.4e-6.
        wavl_stop : float, optional
            Wavelength sweep stop point. The default is 1.7e-6.
        npts : int, optional
            Wavelength sweep points. The default is 5.
        visual : Boolean, optional
            Visualization flag. The default is True.
        folder : Boolean, optional
            If True, a sub folder is created in the current working directory and the generated files are placed there. The default is False.
        simfile : Boolean, optinal
            If True, a sim file will be saved.

        Returns
        -------
        None.

        """
        
        print("Running: FDE for Strip wg.")
        # Create Folder?
        if folder:
            folder_path="1a_FDE_strip_wg/"        
            # Create Folder if doesn't exist
            isExist = os.path.exists(folder_path)
            if not isExist:
                os.makedirs(folder_path)
        else:
            folder_path=""
            
        sim_buffer_y = 0.1e-6  # buffer between material and sim region
        x_span = self.width_wg2 * 4 + 2e-6
        x_span_fde = x_span - 2 * sim_buffer_y
        y_span = self.thickness_device + 4e-6
        y_span_fde = y_span - 2 * sim_buffer_y
        z_span = 1e-6  # irrelevent
        with lumapi.MODE(hide=False) as mode:
            # add materials
            # self.init_custom_material(mode)
            # add geometry
            mode.addrect()
            mode.set("name", "substrate")
            mode.set("material", self.mat_substrate)
            mode.set("x", 0)
            mode.set("x span", x_span)
            mode.set("y max", 0)
            mode.set("y min", -self.thickness_substrate)
            mode.set("z", 0)
            mode.set("z span", z_span)
            mode.set("override mesh order from material database", 1)
            mode.set("mesh order", 3)
            mode.set("alpha", 0.2)

            mode.addrect()
            mode.set("name", "superstrate")
            mode.set("material", self.mat_superstrate)
            mode.set("x", 0)
            mode.set("x span", x_span)
            mode.set("y min", 0)
            mode.set("y max", self.thickness_superstrate)
            mode.set("z", 0)
            mode.set("z span", z_span)
            mode.set("override mesh order from material database", 1)
            mode.set("mesh order", 3)
            mode.set("alpha", 0.2)

            mode.addrect()
            mode.set("name", "waveguide")
            mode.set("material", self.mat_device)
            mode.set("x", 0)
            mode.set("x span", self.width_wg2)
            mode.set("y min", 0)
            mode.set("y max", self.thickness_device)
            mode.set("z", 0)
            mode.set("z span", z_span)

            mode.addfde()
            mode.set("x", 0)
            mode.set("x span", x_span_fde)
            mode.set("y", self.thickness_device / 2)
            mode.set("y span", y_span_fde)
            mode.set("z", 0)
            mode.set("define x mesh by", "maximum mesh step")
            mode.set("define y mesh by", "maximum mesh step")
            mode.set("dx", mesh_dx)
            mode.set("dy", mesh_dy)
            if self.mat_substrate == self.mat_superstrate:
                mode.set("y min bc", "symmetric")
            else:
                mode.set("y min bc", "PML")
            mode.set("x min bc", "PML")
            mode.set("x max bc", "PML")

            mode.run()
            mode.setanalysis("wavelength", wavl_start)
            mode.setanalysis("stop wavelength", wavl_stop)
            mode.setanalysis("number of points", npts)
            mode.setanalysis("number of trial modes", 1)

            mode.findmodes()
            mode.selectmode(1)
            mode.setanalysis("track selected mode", 1)
            mode.setanalysis("detailed dispersion calculation", 1)
            mode.frequencysweep()
            if simfile:
                mode.save(folder_path+"stripwg")

            c = 299792458  # m/s
            f = mode.getdata("frequencysweep", "f_D")
            f = [i[0] for i in f]
            wavl = c / np.array(f)

            vg = np.abs(mode.getdata("frequencysweep", "vg"))
            vg = [i[0] for i in vg]

            ng = c / np.array(vg)
            neff = np.abs(mode.getdata("frequencysweep", "neff"))
            neff = [i[0] for i in neff]

            # fit and resample on higher resolution
            N = 3  # fit order
            pts = 100  # resampling points

            wavl_poly_coeffs = np.polyfit(np.arange(len(wavl)), wavl, deg=N)
            self.strip_wavl = np.polyval(
                wavl_poly_coeffs, np.linspace(0, len(wavl) - 1, pts)
            )

            ng_poly_coeffs = np.polyfit(np.arange(len(ng)), ng, deg=N)
            self.strip_ng = np.polyval(ng_poly_coeffs, np.linspace(0, len(ng) - 1, pts))

            neff_poly_coeffs = np.polyfit(np.arange(len(neff)), neff, deg=N)
            self.strip_neff = np.polyval(
                neff_poly_coeffs, np.linspace(0, len(neff) - 1, pts)
            )

            strip_wg_id = f"strip_width{int(1e9*self.width_wg2)}nm"

            if visual:

                font = {"family": "normal", "size": 18}

                fig, ax1 = plt.subplots(figsize=(10, 6))
                ax1.scatter(wavl * 1e6, neff, color="r")
                ax1.plot(self.strip_wavl * 1e6, self.strip_neff, color="r")
                ax1.set_ylabel("Effective index", color="r")
                ax1.set_xlabel("Wavelength [µm]")
                ax1.set_title("Strip Waveguide Model")
                for tl in ax1.get_yticklabels():
                    tl.set_color("r")
                ax2 = ax1.twinx()
                ax2.scatter(wavl * 1e6, ng, color="b")
                ax2.plot(self.strip_wavl * 1e6, self.strip_ng, color="b")
                ax2.set_ylabel("Group index", color="b")
                for tl in ax2.get_yticklabels():
                    tl.set_color("b")
                fig.savefig(folder_path+f"neff_ng_{strip_wg_id}")
                fig.show()

    def EMT_bandstructure(self,folder=False,simfile=False):
        """
        Method to visualize the difference between EMT and FDTD methods for
        simulating a sub-wavelength waveguide.

        Parameters
        -------
        folder : Boolean, optional
            If True, a sub folder is created in the current working directory and the generated files are placed there. The default is False.
        simfile : Boolean, optinal
            If True, a sim file will be saved.
            
        Returns
        -------
        None.

        """

        print("Running: FDTD EMT Bandstructure.")
        # Create Folder?
        if folder:
            folder_path="3_FDTD_EMT_bandstructure/"        
            # Create Folder if doesn't exist
            isExist = os.path.exists(folder_path)
            if not isExist:
                os.makedirs(folder_path)
        else:
            folder_path=""
            
        font = {"family": "normal", "size": 18}

        self.EMT_swg(folder=folder,simfile=simfile)
        self.FDTD_swg(folder=folder,simfile=simfile)

        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(
            self.swg_emt_wavl * 1e6, self.swg_emt_neff, color="r", label="EMT neff"
        )
        ax1.plot(self.swg_wavl * 1e6, self.swg_neff, "--", color="r", label="FDTD neff")
        ax1.set_ylabel("Effective index", color="r")
        ax1.set_xlabel("Wavelength [µm]")
        ax1.set_title(f"Comparision of EMT vs Bandstructure method for SWG WG")
        ax1.legend(loc=1)
        for tl in ax1.get_yticklabels():
            tl.set_color("r")
        ax2 = ax1.twinx()
        ax2.plot(self.swg_emt_wavl * 1e6, self.swg_emt_ng, color="b", label="EMT ng")
        ax2.plot(self.swg_wavl * 1e6, self.swg_ng, "--", color="b", label="FDTD ng")
        ax2.set_ylabel("Group index", color="b")
        for tl in ax2.get_yticklabels():
            tl.set_color("b")
        ax2.legend(loc=2)
        fig.show()
        fig.savefig(folder_path+"BandstructureComparison")
        
        print("Complete: FDTD EMT Bandstructure.")

    def Wg_analysis(self, swg_model="EMT",folder=False,simfile=False):
        """
        Perform individual waveguide modal analysis (not supermodal analysis).

        Parameters
        ----------
        swg_model : float, optional
            Method used to model SWG waveguides. The default is 'EMT'.
            Options include ['EMT', 'FDTD']
        folder : Boolean, optional
            If True, a sub folder is created in the current working directory and the generated files are placed there. The default is False.
        simfile : Boolean, optinal
            If True, a sim file will be saved.

        Returns
        -------
        None.

        """
        print("Running: Waveguide Analysis.")
        # Create Folder?
        if folder:
            folder_path="1_waveguide_analysis/"        
            # Create Folder if doesn't exist
            isExist = os.path.exists(folder_path)
            if not isExist:
                os.makedirs(folder_path)
        else:
            folder_path=""
            
        font = {"family": "normal", "size": 18}
        swg_emt_id = f"swg_emt_width{int(1e9*self.width_wg1)}nm"

        self.FDE_stripwg(folder=folder,simfile=simfile)
        if swg_model == "EMT":
            self.EMT_swg(folder=folder,simfile=simfile)

            fig, ax1 = plt.subplots(figsize=(10, 6))
            ax1.plot(
                self.swg_emt_wavl * 1e6,
                self.swg_emt_neff,
                color="r",
                label="SWG Waveguide (EMT)",
            )
            ax1.plot(
                self.strip_wavl * 1e6,
                self.strip_neff,
                "--",
                color="b",
                label="Strip Waveguide",
            )
            ax1.set_ylabel("Effective index")
            ax1.set_xlabel("Wavelength [µm]")
            ax1.set_title(f"SWG Adiabatic Filter Modal Analysis (Effective index)")
            ax1.legend(loc=1)
            fig.savefig(folder_path+f"EMT_modal_analysis_neff_{swg_emt_id}")

            fig, ax1 = plt.subplots(figsize=(10, 6))
            ax1.plot(
                self.swg_emt_wavl * 1e6,
                self.swg_emt_ng,
                color="r",
                label="SWG Waveguide (EMT)",
            )
            ax1.plot(
                self.strip_wavl * 1e6,
                self.strip_ng,
                "--",
                color="b",
                label="Strip Waveguide",
            )
            ax1.set_ylabel("Group index")
            ax1.set_xlabel("Wavelength [µm]")
            ax1.set_title("SWG Adiabatic Filter Modal Analysis (Group index)")
            ax1.legend(loc=1)
            fig.savefig(folder_path+f"EMT_modal_analysis_ng_{swg_emt_id}")
            
        elif swg_model == "FDTD":
            self.FDTD_swg(folder=folder,simfile=simfile)

            fig, ax1 = plt.subplots(figsize=(10, 6))
            ax1.plot(
                self.swg_wavl * 1e6,
                self.swg_neff,
                color="r",
                label="SWG Waveguide (FDTD)",
            )
            ax1.plot(
                self.strip_wavl * 1e6,
                self.strip_neff,
                "--",
                color="b",
                label="Strip Waveguide",
            )
            ax1.set_ylabel("Effective index")
            ax1.set_xlabel("Wavelength [µm]")
            ax1.set_title("SWG Adiabatic Filter Modal Analysis (Effective index)")
            ax1.legend(loc=1)
            fig.savefig(folder_path+f"FDTD_modal_analysis_neff_{swg_emt_id}")

            fig, ax1 = plt.subplots(figsize=(10, 6))
            ax1.plot(
                self.swg_wavl * 1e6,
                self.swg_ng,
                color="r",
                label="SWG Waveguide (FDTD)",
            )
            ax1.plot(
                self.strip_wavl * 1e6,
                self.strip_ng,
                "--",
                color="b",
                label="Strip Waveguide",
            )
            ax1.set_ylabel("Group index")
            ax1.set_xlabel("Wavelength [µm]")
            ax1.set_title(f"SWG Adiabatic Filter Modal Analysis (Group index)")
            ax1.legend(loc=1)
            fig.savefig(folder_path+f"FDTD_modal_analysis_ng_{swg_emt_id}")
            
        print("Complete: Waveguide Analysis.")

    def supermodes(
        self,
        mesh_dx=25e-9,
        mesh_dy=25e-9,
        wavl_start=1.4e-6,
        wavl_stop=1.7e-6,
        npts=10,
        visual=True,
        folder=False,
        simfile=False,
    ):
        """
        Perform supermodes analysis of the two waveguide system.

        Note: This method can only use EMT for now since I can't figure out
        how to make an equivalent material model using from FDTD data.

        Parameter2222s
        ----------
        mesh_dx : float, optional
            X Mesh step in the eigenment solver. The default is 25e-9.
        mesh_dy : float, optional
            Y Mesh step in the eigenment solver. The default is 25e-9.
        wavl_start : float, optional
            Wavelength sweep start point. The default is 1.4e-6.
        wavl_stop : float, optional
            Wavelength sweep stop point. The default is 1.7e-6.
        npts : int, optional
            Wavelength sweep points. The default is 5.
        visual : Boolean, optional
            Visualization flag. The default is True.
        folder : Boolean, optional
            If True, a sub folder is created in the current working directory and the generated files are placed there. The default is False.
        simfile : Boolean, optinal
            If True, a sim file will be saved.

        Returns
        -------
        None.

        """
        
        print("Running: FDE Supermode Analysis.")
        # Create Folder?
        if folder:
            folder_path="2_supermode_analysis/"        
            # Create Folder if doesn't exist
            isExist = os.path.exists(folder_path)
            if not isExist:
                os.makedirs(folder_path)
        else:
            folder_path=""
            
        sim_buffer = 0.2e-6  # buffer between material and sim region
        x_span = self.width_wg1 * 6 + self.width_wg2 * 6 + self.gap_end
        x_span_fde = x_span - 2 * sim_buffer
        y_span = self.thickness_device + 4e-6
        y_span_fde = y_span - 2 * sim_buffer
        z_span = 1e-6  # irrelevant

        with lumapi.MODE(hide=False) as mode:
            # Step 0: Perform EMT and make equivalent material model
            self.EMT_swg(
                wavl_start=wavl_start,
                wavl_stop=wavl_stop,
                visual=visual,
                simulate_lumerical=False,
                folder=folder,
                simfile=simfile
            )
            swg_emt_id = f"swg_emt_width{int(1e9*self.width_wg1)}nm"
            mode.eval("swg_material = addmaterial('Sampled 3D data');")
            mode.eval(f"setmaterial(swg_material, 'name', '{swg_emt_id}');")
            # format the data for lumerical's format
            c = 299792458  # m/s
            wavl_arr = np.array2string(c / self.wavl_emt, separator=";").replace(
                "\n", ""
            )
            n_arr = np.array2string(
                np.array(self.swg_n_emt) ** 2, separator=";"
            ).replace("\n", "")
            mode.eval(
                f"setmaterial('{swg_emt_id}', 'sampled 3d data', [{wavl_arr},{n_arr}]);"
            )

            # Step 1: build geometry
            mode.addrect()
            mode.set("name", "substrate")
            mode.set("material", self.mat_substrate)
            mode.set("x", 0)
            mode.set("x span", x_span)
            mode.set("y max", 0)
            mode.set("y min", -self.thickness_substrate)
            mode.set("z", 0)
            mode.set("z span", z_span)
            mode.set("override mesh order from material database", 1)
            mode.set("mesh order", 3)
            mode.set("alpha", 0.2)

            mode.addrect()
            mode.set("name", "superstrate")
            mode.set("material", self.mat_superstrate)
            mode.set("x", 0)
            mode.set("x span", x_span)
            mode.set("y min", 0)
            mode.set("y max", self.thickness_superstrate)
            mode.set("z", 0)
            mode.set("z span", z_span)
            mode.set("override mesh order from material database", 1)
            mode.set("mesh order", 3)
            mode.set("alpha", 0.2)

            mode.addrect()
            mode.set("name", "WG2_strip")
            mode.set("material", self.mat_device)
            mode.set("x", -self.gap_end / 2 - self.width_wg2 / 2)
            mode.set("x span", self.width_wg2)
            mode.set("y min", 0)
            mode.set("y max", self.thickness_device)
            mode.set("z", 0)
            mode.set("z span", z_span)

            mode.addrect()
            mode.set("name", "WG1_SWG")
            mode.set("material", swg_emt_id)
            mode.set("x", self.gap_end / 2 + self.width_wg1 / 2)
            mode.set("x span", self.width_wg1)
            mode.set("y min", 0)
            mode.set("y max", self.thickness_device)
            mode.set("z", 0)
            mode.set("z span", z_span)

            # Step 2: build solver
            mode.addfde()
            mode.set("x", 0)
            mode.set("x span", x_span_fde)
            mode.set("y", self.thickness_device / 2)
            mode.set("y span", y_span_fde)
            mode.set("z", 0)
            mode.set("define x mesh by", "maximum mesh step")
            mode.set("define y mesh by", "maximum mesh step")
            mode.set("dx", mesh_dx)
            mode.set("dy", mesh_dy)
            if self.mat_substrate == self.mat_superstrate:
                mode.set("y min bc", "symmetric")
            else:
                mode.set("y min bc", "PML")
            mode.set("x min bc", "PML")
            mode.set("x max bc", "PML")

            # Step 3: run solver and frequency sweeps
            # even supermode
            mode.run()
            mode.setanalysis("wavelength", wavl_start)
            mode.setanalysis("stop wavelength", wavl_stop)
            mode.setanalysis("number of points", npts)
            mode.setanalysis("number of trial modes", 1)

            mode.findmodes()
            mode.selectmode(1)
            mode.setanalysis("track selected mode", 1)
            mode.setanalysis("detailed dispersion calculation", 1)
            mode.frequencysweep()

            f = mode.getdata("frequencysweep", "f_D")
            f = [i[0] for i in f]
            wavl = c / np.array(f)

            vg_even = np.abs(mode.getdata("frequencysweep", "vg"))
            vg_even = [i[0] for i in vg_even]
            ng_even = c / np.array(vg_even)
            neff_even = np.abs(mode.getdata("frequencysweep", "neff"))
            neff_even = [i[0] for i in neff_even]

            mode.switchtolayout()
            # odd supermode
            mode.run()
            mode.setanalysis("wavelength", wavl_start)
            mode.setanalysis("stop wavelength", wavl_stop)
            mode.setanalysis("number of points", npts)
            mode.setanalysis("number of trial modes", 2)

            mode.findmodes()
            mode.selectmode(2)
            mode.setanalysis("track selected mode", 1)
            mode.setanalysis("detailed dispersion calculation", 1)
            mode.frequencysweep()

            vg_odd = np.abs(mode.getdata("frequencysweep", "vg"))
            vg_odd = [i[0] for i in vg_odd]
            ng_odd = c / np.array(vg_odd)
            neff_odd = np.abs(mode.getdata("frequencysweep", "neff"))
            neff_odd = [i[0] for i in neff_odd]

            # Step 4: analyze
            # fit and resample on higher resolution
            N = 5  # fit order
            pts = 100  # resampling points

            wavl_poly_coeffs = np.polyfit(np.arange(len(wavl)), wavl, deg=N)
            self.supermode_wavl = np.polyval(
                wavl_poly_coeffs, np.linspace(0, len(wavl) - 1, pts)
            )

            ng_poly_coeffs = np.polyfit(np.arange(len(ng_even)), ng_even, deg=N)
            self.supermode_even_ng = np.polyval(
                ng_poly_coeffs, np.linspace(0, len(ng_even) - 1, pts)
            )

            ng_poly_coeffs = np.polyfit(np.arange(len(ng_odd)), ng_odd, deg=N)
            self.supermode_odd_ng = np.polyval(
                ng_poly_coeffs, np.linspace(0, len(ng_odd) - 1, pts)
            )

            neff_poly_coeffs = np.polyfit(np.arange(len(neff_even)), neff_even, deg=N)
            self.supermode_even_neff = np.polyval(
                neff_poly_coeffs, np.linspace(0, len(neff_even) - 1, pts)
            )

            neff_poly_coeffs = np.polyfit(np.arange(len(neff_odd)), neff_odd, deg=N)
            self.supermode_odd_neff = np.polyval(
                neff_poly_coeffs, np.linspace(0, len(neff_odd) - 1, pts)
            )

            # Step 5: visualize
            if visual:

                font = {"family": "normal", "size": 18}

                # effective index plot
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(wavl * 1e6, neff_even, color="r")
                ax.scatter(wavl * 1e6, neff_odd, color="b")
                ax.plot(
                    self.supermode_wavl * 1e6,
                    self.supermode_even_neff,
                    color="r",
                    label="Even supermode",
                )
                ax.plot(
                    self.supermode_wavl * 1e6,
                    self.supermode_odd_neff,
                    color="b",
                    label="Odd supermode",
                )
                ax.set_ylabel("Effective index")
                ax.set_xlabel("Wavelength [µm]")
                ax.set_title("Supermodes effective indices analysis")
                ax.legend()
                fig.show()
                fig.savefig(folder_path+"supermode_effective_indices")

                # effective index plot
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(wavl * 1e6, ng_even, color="r")
                ax.scatter(wavl * 1e6, ng_odd, color="b")
                ax.plot(
                    self.supermode_wavl * 1e6,
                    self.supermode_even_ng,
                    color="r",
                    label="Even supermode",
                )
                ax.plot(
                    self.supermode_wavl * 1e6,
                    self.supermode_odd_ng,
                    color="b",
                    label="Odd supermode",
                )
                ax.set_ylabel("Group index")
                ax.set_xlabel("Wavelength [µm]")
                ax.set_title("Supermodes group indices analysis")
                ax.legend()
                fig.show()
                fig.savefig(folder_path+"supermode_group_indices")

    def optimize_device(
        self,
        mesh_dy=55e-9,
        mesh_dz=55e-9,
        wavl_start=1.4e-6,
        wavl_stop=1.7e-6,
        num_modes=15,
        cells=[2, 5, 15, 15, 2],
        length_sweeps=[False, [1e-6, 100e-6], [10e-6, 250e-6], [10e-6, 200e-6], False],
        length_pts=[False, 50, 50, 50, False],
        visual=True,
        folder=False,
        simfile=False
    ):
        print("Running: FDE EME Device Optimization.")
        # Create Folder?
        if folder:
            folder_path="4_EME_optimize/"        
            # Create Folder if doesn't exist
            isExist = os.path.exists(folder_path)
            if not isExist:
                os.makedirs(folder_path)
        else:
            folder_path=""
            
        sim_buffer = 0.2e-6  # buffer between material and sim region
        pml_buffer = 4e-6  # distance buffer between waveguide and sim edge

        z_span = self.thickness_device + 4e-6
        z_span_eme = z_span - 2 * sim_buffer
        y_span_eme = self.width_wg1 + self.width_wg2 + self.gap_end + 2 * pml_buffer
        y_span = y_span_eme + 2 * sim_buffer
        y = 0  # y center point of the simulation, substate, and superstate
        y_dev = y + 1.5e-6

        x0 = 0
        x1 = x0 + self.length_swg_taper
        x2 = x1 + self.length_coupling
        x3 = x2 + self.length_bend
        x4 = x3 + self.length_bend_output
        x5 = x4 + self.length_swg_taper
        x_end = x5

        with lumapi.MODE(hide=False) as mode:
            # Perform EMT and make equivalent material model
            self.EMT_swg(
                wavl_start=wavl_start,
                wavl_stop=wavl_stop,
                visual=visual,
                simulate_lumerical=False,
                folder=folder,
                simfile=simfile
            )
            swg_emt_id = f"swg_emt_width{int(1e9*self.width_wg1)}nm"
            mode.eval("swg_material = addmaterial('Sampled 3D data');")
            mode.eval(f"setmaterial(swg_material, 'name', '{swg_emt_id}');")
            
            # Format the data for lumerical's format
            c = 299792458  # m/s
            wavl_arr = np.array2string(c / self.wavl_emt, separator=";").replace(
                "\n", ""
            )
            n_arr = np.array2string(
                np.array(self.swg_n_emt) ** 2, separator=";"
            ).replace("\n", "")
            mode.eval(
                f"setmaterial('{swg_emt_id}', 'sampled 3d data', [{wavl_arr},{n_arr}]);"
            )

            # Step 1: build geometry
            mode.addrect()
            mode.set("name", "substrate")
            mode.set("material", self.mat_substrate)
            mode.set("x min", x0 - sim_buffer)
            mode.set("x max", x_end + sim_buffer)
            mode.set("y", y)
            mode.set("y span", y_span)
            mode.set("z max", 0)
            mode.set("z min", -self.thickness_substrate)
            mode.set("override mesh order from material database", 1)
            mode.set("mesh order", 3)
            mode.set("alpha", 0.2)

            mode.addrect()
            mode.set("name", "superstrate")
            mode.set("material", self.mat_superstrate)
            mode.set("x min", x0 - sim_buffer)
            mode.set("x max", x_end + sim_buffer)
            mode.set("y", y)
            mode.set("y span", y_span)
            mode.set("z min", 0)
            mode.set("z max", self.thickness_superstrate)
            mode.set("override mesh order from material database", 1)
            mode.set("mesh order", 3)
            mode.set("alpha", 0.2)

            # wg1: SWG waveguide
            wg1_y_center = y_dev + self.gap_start / 2 + self.width_wg1 / 2
            mode.addrect()
            mode.set("name", "wg1")
            mode.set("material", swg_emt_id)
            mode.set("x min", x0 - sim_buffer)
            mode.set("x max", x_end + sim_buffer)
            mode.set("y", wg1_y_center)
            mode.set("y span", self.width_wg1)
            mode.set("z min", 0)
            mode.set("z max", self.thickness_device)

            # wg2: strip waveguide end taper section
            wg2_end_y_center = (
                wg1_y_center - self.width_wg1 / 2 - self.gap_end - self.width_wg2 / 2
            )

            # wg2: strip waveguide output pitch end section
            wg2_out_y_center = wg1_y_center - self.output_pitch

            mode.addpoly()
            mode.set("name", "wg2_output_taper")
            mode.set("x", x0)
            mode.set("y", wg2_out_y_center)
            mode.set("z min", 0)
            mode.set("z max", self.thickness_device)

            mode.eval("M=matrix(6,2);")
            mode.eval(f"M(1,1:2)=[{x4}, {self.width_wg2/2}];")
            mode.eval(f"M(2,1:2)=[{x5}, {self.width_wg_port/2}];")
            mode.eval(f"M(3,1:2)=[{x5+sim_buffer}, {self.width_wg_port/2}];")
            mode.eval(f"M(4,1:2)=[{x5+sim_buffer}, {-self.width_wg_port/2}];")
            mode.eval(f"M(5,1:2)=[{x5}, {-self.width_wg_port/2}];")
            mode.eval(f"M(6,1:2)=[{x4}, {-self.width_wg2/2}];")
            mode.eval("set('vertices',M);")
            mode.set("material", self.mat_device)

            # wg2: strip waveguide beginning taper section
            wg2_start_y_center = wg1_y_center - self.width_wg1 / 2
            mode.addpoly()
            mode.set("name", "wg2_start")
            mode.set("x", x0)
            mode.set("y", wg2_start_y_center)
            mode.set("z min", 0)
            mode.set("z max", self.thickness_device)

            mode.eval("M=matrix(4,2);")
            mode.eval(f"M(1,1:2)=[{x1}, {-self.gap_start}];")
            mode.eval(f"M(2,1:2)=[{x2}, {-self.gap_start}];")
            mode.eval(f"M(3,1:2)=[{x2}, {-self.gap_start-self.width_wg2}];")
            mode.eval(f"M(4,1:2)=[{x1}, {-self.gap_start-self.width_wg2_tip}];")
            mode.eval("set('vertices',M);")
            mode.set("material", self.mat_device)

            # wg2: strip waveguide s-bend section (gap_start to gap_end)
            y1 = wg1_y_center - self.width_wg1 / 2 - self.gap_start - self.width_wg2 / 2
            y2 = wg2_end_y_center
            sbend_height = -np.abs(y1 - y2)
            mode.addobject("s_bend_wg")
            mode.set("x", x2)
            mode.set("y", y1)
            mode.set("z", self.thickness_device / 2)
            mode.set("base width", self.width_wg2)
            mode.set("base height", self.thickness_device)
            mode.set("x span", self.length_bend)
            mode.set("y span", sbend_height)
            mode.set("material", self.mat_device)

            # wg2: strip waveguide s-bend section (gap_end to output_pitch)
            y3 = wg2_out_y_center
            y2 = wg2_end_y_center
            sbend_height = -np.abs(y3 - y2)
            mode.addobject("s_bend_wg")
            mode.set("x", x3)
            mode.set("y", y2)
            mode.set("z", self.thickness_device / 2)
            mode.set("base width", self.width_wg2)
            mode.set("base height", self.thickness_device)
            mode.set("x span", self.length_bend_output)
            mode.set("y span", sbend_height)
            mode.set("material", self.mat_device)

            # wg2: strip waveguide bend taper in
            radius = 2e-6
            mode.addobject("90_bend_wg")
            mode.set("start angle", 90)
            mode.set("radius", radius)
            mode.set("x", x1)
            mode.set(
                "y",
                wg1_y_center
                - self.width_wg1 / 2
                - self.gap_start
                - self.width_wg2_tip / 2
                - radius,
            )
            mode.set("z", self.thickness_device / 2)
            mode.set("base width", self.width_wg2_tip)
            mode.set("base height", self.thickness_device)
            mode.set("material", self.mat_device)

            # Step 2: build simulation
            mode.addeme()
            mode.set("x min", x0)
            mode.set("y", y)
            mode.set("y span", y_span_eme)
            mode.set("z", self.thickness_device / 2)
            mode.set("z span", z_span_eme)

            # setup EME cells
            cell_groups = len(cells)
            mode.set("number of cell groups", cell_groups)
            mode.set("number of modes for all cell groups", num_modes)
            # TODO: group spans doesn't scale with added sections in the future
            group_spans = [
                self.length_swg_taper,
                self.length_coupling,
                self.length_bend,
                self.length_bend_output,
                self.length_swg_taper,
            ]
            mode.eval(f"set('group spans',[{';'.join(map(str, group_spans))}]);")
            mode.eval(f"set('cells',[{';'.join(map(str, cells))}]);")
            # TODO: length of the ones array doesn't auto scale with number of cells
            mode.eval("set('subcell method',[1;1;1;1;1]);")  # CVCS for all

            # set simulation mesh
            mode.set("define y mesh by", "maximum mesh step")
            mode.set("define z mesh by", "maximum mesh step")
            mode.set("dy", mesh_dy)
            mode.set("dz", mesh_dz)

            # define symmetry if symmetric stack
            # TODO: handle rib at some point, maybe a waste of time...
            if self.mat_substrate == self.mat_superstrate:
                mode.set("z min bc", "Symmetric")

            # setup ports
            mode.select("EME::Ports::port_1")
            mode.set("use full simulation span", 1)
            mode.set("mode selection", "fundamental TE mode")

            # set up ports: port 2 SWG WG (WG1) port
            mode.select("EME::Ports::port_2")
            mode.set("use full simulation span", 0)
            mode.set("mode selection", "fundamental TE mode")
            mode.set("z span", z_span_eme)
            mode.set("y min", y)
            mode.set("y max", y + y_span_eme / 2)

            # setup ports: port 3 Strip WG  (WG2) port
            mode.addemeport()
            mode.select("EME::Ports::port_3")
            mode.set("use full simulation span", 0)
            mode.set("mode selection", "fundamental TE mode")
            mode.set("port location", "right")
            mode.set("z span", z_span_eme)
            mode.set("y max", y)
            mode.set("y min", -(y + y_span_eme / 2))

            mode.addemeprofile()
            mode.set("x min", x0)
            mode.set("x max", x_end)
            mode.set("z", self.thickness_device / 2)
            mode.set("y", y)
            mode.set("y span", y_span_eme)

            # Step 3: run simulation and sweeps
            # Perform group length sweeps at each wavelength (short and long)
            if simfile:
                mode.save(folder_path+self.device_id+"_shortwl")
            mode.select("EME")
            mode.set("wavelength", wavl_start)
            mode.run()

            # perform length sweeps
            mode.setemeanalysis("propagation sweep", 1)
            for idx, sweep in enumerate(length_sweeps):
                if sweep != False:
                    mode.setemeanalysis("parameter", f"group span {idx+1}")
                    mode.setemeanalysis("start", sweep[0])
                    mode.setemeanalysis("stop", sweep[1])
                    mode.setemeanalysis("number of points", length_pts[idx])
                    mode.emesweep()

                    # get propagation sweep result
                    S = mode.getemesweep("S")
                    s_21 = [S["s21"]]
                    s_31 = [S["s31"]]
                    group_span = S[f"group_span_{idx+1}"]
                    if visual:

                        # plot S21 vs group span
                        fig, ax1 = plt.subplots()
                        ax1.plot(
                            group_span * 1e6,
                            10 * np.log10(abs(s_21[0]) ** 2),
                            color="b",
                            label="S21",
                        )

                        ax1.set_title(
                            f"Length sweep (Group {idx+1}, Wavelength={wavl_start*1e6} µm)"
                        )
                        ax1.set_xlabel("Length [µm]", fontsize=14)
                        ax1.set_ylabel("Transmission S21 [dB]", color="b")
                        ax1.tick_params(axis="y", colors="b")

                        ax2 = ax1.twinx()
                        ax2.plot(
                            group_span * 1e6,
                            10 * np.log10(abs(s_31[0]) ** 2),
                            color="r",
                            label="S31",
                        )
                        ax2.set_ylabel("Transmission S31 [dB]", color="r")
                        ax2.tick_params(axis="y", colors="r")
                        fig.show()
                        fig.savefig(folder_path+"Length_sweep_shortwl")

            mode.switchtolayout()
            if simfile:
                mode.save(self.device_id+"_longwl") #save as to perform same simulation for long wavelength
                #the mode API call sets the working directory to the current file
                #thus, folder_path is not needed.
            mode.select("EME")
            mode.set("wavelength", wavl_stop)
            mode.run()

            # perform length sweeps
            mode.setemeanalysis("propagation sweep", 1)
            for idx, sweep in enumerate(length_sweeps):
                if sweep != False:
                    mode.setemeanalysis("parameter", f"group span {idx+1}")
                    mode.setemeanalysis("start", sweep[0])
                    mode.setemeanalysis("stop", sweep[1])
                    mode.setemeanalysis("number of points", length_pts[idx])
                    mode.emesweep()

                    # get propagation sweep result
                    S = mode.getemesweep("S")
                    s_21 = [S["s21"]]
                    s_31 = [S["s31"]]
                    group_span = S[f"group_span_{idx+1}"]
                    if visual:

                        # plot S21 and S31 vs group span
                        # S21, straight through SWG WG - passes long wavelengths (red)
                        # S31, cross strip WG - passes short wavelengths (blue)
                        fig, ax1 = plt.subplots()
                        ax1.plot(
                            group_span * 1e6,
                            10 * np.log10(abs(s_21[0]) ** 2),
                            color="r",
                            label="S21",
                        )

                        ax1.set_title(
                            f"Length sweep (Group {idx+1}, Wavelength={wavl_stop*1e6} µm)"
                        )
                        ax1.set_xlabel("Length [µm]", fontsize=14)
                        ax1.set_ylabel("Transmission S21 [dB]", color="r")
                        ax1.tick_params(axis="y", colors="r")

                        ax2 = ax1.twinx()
                        ax2.plot(
                            group_span * 1e6,
                            10 * np.log10(abs(s_31[0]) ** 2),
                            color="b",
                            label="S31",
                        )
                        ax2.set_ylabel("Transmission S31 [dB]", color="b")
                        ax2.tick_params(axis="y", colors="b")
                        #fig.show()
                        fig.savefig(folder_path+"Length_sweep_longwl")

    def generate_spectrum(
        self,
        mesh_dy=55e-9,
        mesh_dz=55e-9,
        wavl_start=1.4e-6,
        wavl_stop=1.7e-6,
        wavl_pts=15,
        num_modes=2, #15
        cells=[2, 5, 100, 100, 2], #15 15
        visual=True,
        folder=False,
        simfile=False
    ):
        
        sim_buffer = 0.2e-6  # buffer between material and sim region
        pml_buffer = 4e-6  # distance buffer between waveguide and sim edge

        z_span = self.thickness_device + 4e-6
        z_span_eme = z_span - 2 * sim_buffer
        y_span_eme = self.width_wg1 + self.width_wg2 + self.gap_end + 2 * pml_buffer
        y_span = y_span_eme + 2 * sim_buffer
        y = 0  # y center point of the simulation, substate, and superstate
        y_dev = y + 1.5e-6

        x0 = 0
        x1 = x0 + self.length_swg_taper
        x2 = x1 + self.length_coupling
        x3 = x2 + self.length_bend
        x4 = x3 + self.length_bend_output
        x5 = x4 + self.length_swg_taper
        x_end = x5

        # Create Folder?
        if folder:
            folder_path="4a_spectrum/"        
            # Create Folder if doesn't exist
            isExist = os.path.exists(folder_path)
            if not isExist:
                os.makedirs(folder_path)
        else:
            folder_path=""

        with lumapi.MODE(hide=False) as mode:
            # Step 0: Perform EMT and make equivalent material model
            self.EMT_swg(
                wavl_start=wavl_start,
                wavl_stop=wavl_stop,
                visual=visual,
                simulate_lumerical=False,
                folder=folder,
                simfile=simfile
            )
            swg_emt_id = f"swg_emt_width{int(1e9*self.width_wg1)}nm"
            mode.eval("swg_material = addmaterial('Sampled 3D data');")
            mode.eval(f"setmaterial(swg_material, 'name', '{swg_emt_id}');")
            # format the data for lumerical's silly format
            c = 299792458  # m/s
            wavl_arr = np.array2string(c / self.wavl_emt, separator=";").replace(
                "\n", ""
            )
            n_arr = np.array2string(
                np.array(self.swg_n_emt) ** 2, separator=";"
            ).replace("\n", "")
            mode.eval(
                f"setmaterial('{swg_emt_id}', 'sampled 3d data', [{wavl_arr},{n_arr}]);"
            )

            # Step 1: build geometry
            mode.addrect()
            mode.set("name", "substrate")
            mode.set("material", self.mat_substrate)
            mode.set("x min", x0 - sim_buffer)
            mode.set("x max", x_end + sim_buffer)
            mode.set("y", y)
            mode.set("y span", y_span)
            mode.set("z max", 0)
            mode.set("z min", -self.thickness_substrate)
            mode.set("override mesh order from material database", 1)
            mode.set("mesh order", 3)
            mode.set("alpha", 0.2)

            mode.addrect()
            mode.set("name", "superstrate")
            mode.set("material", self.mat_superstrate)
            mode.set("x min", x0 - sim_buffer)
            mode.set("x max", x_end + sim_buffer)
            mode.set("y", y)
            mode.set("y span", y_span)
            mode.set("z min", 0)
            mode.set("z max", self.thickness_superstrate)
            mode.set("override mesh order from material database", 1)
            mode.set("mesh order", 3)
            mode.set("alpha", 0.2)

            # wg1: SWG waveguide
            wg1_y_center = y_dev + self.gap_start / 2 + self.width_wg1 / 2
            mode.addrect()
            mode.set("name", "wg1")
            mode.set("material", swg_emt_id)
            mode.set("x min", x0 - sim_buffer)
            mode.set("x max", x_end + sim_buffer)
            mode.set("y", wg1_y_center)
            mode.set("y span", self.width_wg1)
            mode.set("z min", 0)
            mode.set("z max", self.thickness_device)

            # wg2: strip waveguide end taper section
            wg2_end_y_center = (
                wg1_y_center - self.width_wg1 / 2 - self.gap_end - self.width_wg2 / 2
            )

            # wg2: strip waveguide output pitch end section
            wg2_out_y_center = wg1_y_center - self.output_pitch
            mode.addpoly()
            mode.set("name", "wg2_output_taper")
            mode.set("x", x0)
            mode.set("y", wg2_out_y_center)
            mode.set("z min", 0)
            mode.set("z max", self.thickness_device)

            mode.eval("M=matrix(6,2);")
            mode.eval(f"M(1,1:2)=[{x4}, {self.width_wg2/2}];")
            mode.eval(f"M(2,1:2)=[{x5}, {self.width_wg_port/2}];")
            mode.eval(f"M(3,1:2)=[{x5+sim_buffer}, {self.width_wg_port/2}];")
            mode.eval(f"M(4,1:2)=[{x5+sim_buffer}, {-self.width_wg_port/2}];")
            mode.eval(f"M(5,1:2)=[{x5}, {-self.width_wg_port/2}];")
            mode.eval(f"M(6,1:2)=[{x4}, {-self.width_wg2/2}];")
            mode.eval("set('vertices',M);")
            mode.set("material", self.mat_device)

            # wg2: strip waveguide beginning taper section
            wg2_start_y_center = wg1_y_center - self.width_wg1 / 2
            mode.addpoly()
            mode.set("name", "wg2_start")
            mode.set("x", x0)
            mode.set("y", wg2_start_y_center)
            mode.set("z min", 0)
            mode.set("z max", self.thickness_device)

            mode.eval("M=matrix(4,2);")
            mode.eval(f"M(1,1:2)=[{x1}, {-self.gap_start}];")
            mode.eval(f"M(2,1:2)=[{x2}, {-self.gap_start}];")
            mode.eval(f"M(3,1:2)=[{x2}, {-self.gap_start-self.width_wg2}];")
            mode.eval(f"M(4,1:2)=[{x1}, {-self.gap_start-self.width_wg2_tip}];")
            mode.eval("set('vertices',M);")
            mode.set("material", self.mat_device)

            # wg2: strip waveguide s-bend section (gap_start to gap_end)
            y1 = wg1_y_center - self.width_wg1 / 2 - self.gap_start - self.width_wg2 / 2
            y2 = wg2_end_y_center
            sbend_height = -np.abs(y1 - y2)
            mode.addobject("s_bend_wg")
            mode.set("x", x2)
            mode.set("y", y1)
            mode.set("z", self.thickness_device / 2)
            mode.set("base width", self.width_wg2)
            mode.set("base height", self.thickness_device)
            mode.set("x span", self.length_bend)
            mode.set("y span", sbend_height)
            mode.set("material", self.mat_device)

            # wg2: strip waveguide s-bend section (gap_end to output_pitch)
            y3 = wg2_out_y_center
            y2 = wg2_end_y_center
            sbend_height = -np.abs(y3 - y2)
            mode.addobject("s_bend_wg")
            mode.set("x", x3)
            mode.set("y", y2)
            mode.set("z", self.thickness_device / 2)
            mode.set("base width", self.width_wg2)
            mode.set("base height", self.thickness_device)
            mode.set("x span", self.length_bend_output)
            mode.set("y span", sbend_height)
            mode.set("material", self.mat_device)

            # wg2: strip waveguide bend taper in
            radius = 2e-6
            mode.addobject("90_bend_wg")
            mode.set("start angle", 90)
            mode.set("radius", radius)
            mode.set("x", x1)
            mode.set(
                "y",
                wg1_y_center
                - self.width_wg1 / 2
                - self.gap_start
                - self.width_wg2_tip / 2
                - radius,
            )
            mode.set("z", self.thickness_device / 2)
            mode.set("base width", self.width_wg2_tip)
            mode.set("base height", self.thickness_device)
            mode.set("material", self.mat_device)

            # Step 2: build simulation
            mode.addeme()
            mode.set("x min", x0)
            mode.set("y", y)
            mode.set("y span", y_span_eme)
            mode.set("z", self.thickness_device / 2)
            mode.set("z span", z_span_eme)

            # setup EME cells
            cell_groups = len(cells)
            mode.set("number of cell groups", cell_groups)
            mode.set("number of modes for all cell groups", num_modes)
            # TODO: group spans doesn't scale with added sections in the future
            group_spans = [
                self.length_swg_taper,
                self.length_coupling,
                self.length_bend,
                self.length_bend_output,
                self.length_swg_taper,
            ]
            mode.eval(f"set('group spans',[{';'.join(map(str, group_spans))}]);")
            mode.eval(f"set('cells',[{';'.join(map(str, cells))}]);")
            # TODO: length of the ones array doesn't auto scale with number of cells
            # find a way to auto populate this array based on number of cell groups
            mode.eval("set('subcell method',[1;1;1;1;1]);")  # CVCS for all

            # set simulation mesh
            mode.set("define y mesh by", "maximum mesh step")
            mode.set("define z mesh by", "maximum mesh step")
            mode.set("dy", mesh_dy)
            mode.set("dz", mesh_dz)

            # define symmetry if symmetric stack
            # TODO: handle rib at some point, maybe a waste of time...
            if self.mat_substrate == self.mat_superstrate:
                mode.set("z min bc", "Symmetric")

            # setup ports
            mode.select("EME::Ports::port_1")
            mode.set("use full simulation span", 1)
            mode.set("mode selection", "fundamental TE mode")

            # set up ports: port 2 SWG WG (WG1) port
            mode.select("EME::Ports::port_2")
            mode.set("use full simulation span", 0)
            mode.set("mode selection", "fundamental TE mode")
            mode.set("z span", z_span_eme)
            mode.set("y min", y)
            mode.set("y max", y + y_span_eme / 2)

            # setup ports: port 3 Strip WG  (WG2) port
            mode.addemeport()
            mode.select("EME::Ports::port_3")
            mode.set("use full simulation span", 0)
            mode.set("mode selection", "fundamental TE mode")
            mode.set("port location", "right")
            mode.set("z span", z_span_eme)
            mode.set("y max", y)
            mode.set("y min", -(y + y_span_eme / 2))

            mode.addemeprofile()
            mode.set("x min", x0)
            mode.set("x max", x_end)
            mode.set("z", self.thickness_device / 2)
            mode.set("y", y)
            mode.set("y span", y_span_eme)

            # Step 3: run simulation and sweeps
            # Perform group length sweeps at each wavelength (short and long)
            mode.save(folder_path+self.device_id)

            wavelength = np.linspace(wavl_start, wavl_stop, wavl_pts)
            s_21 = []
            s_31 = []
            E_field = []
            for wavl in wavelength:
                mode.select("EME")
                mode.set("wavelength", wavl)
                mode.run()
                # get propagation sweep result
                mode.setemeanalysis("start wavelength", wavl - 1e-12)
                mode.setemeanalysis("stop wavelength", wavl + 1e-12)
                mode.setemeanalysis("number of wavelength points", 2)
                mode.setemeanalysis("wavelength sweep", 1)
                mode.emesweep("wavelength sweep")

                S = mode.getemesweep("S_wavelength_sweep")
                s_21.append(
                    S["s21"][0]
                )  # 2 points sweep seperated by 2 pm so just pick 1
                s_31.append(
                    S["s31"][0]
                )  # future me: i spent 3 hours catching a bug here :)

                # run 2 field profile sims one at short and one at long wavls
                if wavl == wavelength[0] or wavl == wavelength[-1]:
                    mode.emepropagate()
                    field_prof = mode.getresult("monitor", "field profile")
                    x_domain = field_prof["x"]
                    # x_domain = [i[0] for i in x_domain]
                    y_domain = field_prof["y"]
                    # y_domain = [i[0] for i in y_domain]
                    E_field.append(field_prof["E"])

                mode.switchtolayout()

            self.result_eme_s21 = s_21
            self.result_eme_s31 = s_31
            self.result_eme_wavl = wavelength
            if visual:

                # plot S21 and S31 vs wavelength
                # S21, straight through SWG WG - passes long wavelengths (red)
                # S31, cross strip WG - passes short wavelengths (blue)
                fig, ax = plt.subplots()
                ax.plot(
                    wavelength * 1e9,
                    10 * np.log10(np.abs(s_21) ** 2),
                    color="r",
                    label="S21",
                )
                ax.plot(
                    wavelength * 1e9,
                    10 * np.log10(np.abs(s_31) ** 2),
                    color="b",
                    label="S31",
                )

                ax.set_title("SWG Adiabatic Filter Spectrum (EME EMT model)")
                ax.set_xlabel("Wavelength [nm]", fontsize=14)
                ax.set_ylabel("Transmission [dB]")
                ax.legend()

                fig.show()

                # plot field monitors for high-pass case
                # Reshape the data
                E_field = np.array(E_field)
                Ey = E_field[1, :, :, 0, 0, 1]  # Extract Ey component only
                # Plotting
                fig, ax = plt.subplots()
                img = ax.imshow(
                    np.abs(Ey.T),
                    extent=[
                        y_domain.min(),
                        y_domain.max(),
                        x_domain.min(),
                        x_domain.max(),
                    ],
                    origin="lower",
                    aspect="auto",
                    cmap="viridis",
                )
                cbar = fig.colorbar(img, ax=ax, label="Ey")
                ax.set_xlabel("Y [µm]")  # X and Y are swapped
                ax.set_ylabel("X [µm]")  # X and Y are swapped
                ax.set_title(
                    f"Field Distribution Ey at wavelength = {wavl_start*1e9} nm"
                )
                plt.show()

                E_field = np.array(E_field)
                Ey = E_field[0, :, :, 0, 0, 1]  # Extract Ey component only
                # Plotting
                fig, ax = plt.subplots()
                img = ax.imshow(
                    np.abs(Ey.T),
                    extent=[
                        y_domain.min(),
                        y_domain.max(),
                        x_domain.min(),
                        x_domain.max(),
                    ],
                    origin="lower",
                    aspect="auto",
                    cmap="viridis",
                )
                cbar = fig.colorbar(img, ax=ax, label="Ey")
                ax.set_xlabel("Y [µm]")  # X and Y are swapped
                ax.set_ylabel("X [µm]")  # X and Y are swapped
                ax.set_title(
                    f"Field Distribution Ey at wavelength = {wavl_stop*1e9} nm"
                )
                plt.show()

    def FDTD_fullsim(
        self, mesh=2, wavl_start=1.4e-6, wavl_stop=1.7e-6, wavl_pts=301, visual=True, folder = False, simfile=False
    ):
        sim_buffer = 0.2e-6  # buffer between material and sim region
        pml_buffer = 4e-6  # distance buffer between waveguide and sim edge

        z_span = self.thickness_device + 5e-6
        z_span_sim = z_span - 2 * sim_buffer
        y_span_sim = self.width_wg1 + self.width_wg2 + self.gap_end + 2 * pml_buffer
        y_span = y_span_sim + 2 * sim_buffer
        y_span_port = 4 * self.width_wg_port
        y = 0  # y center point of the simulation, substate, and superstate
        y_dev = y + 1.5e-6

        x0 = 0
        x1 = x0 + self.length_swg_taper
        x2 = x1 + self.length_coupling
        x3 = x2 + self.length_bend
        x4 = x3 + self.length_bend_output
        x5 = x4 + self.length_swg_taper
        x_end = x5

        # Create Folder?
        if folder:
            folder_path="5_full_simulation/"        
            # Create Folder if doesn't exist
            isExist = os.path.exists(folder_path)
            if not isExist:
                os.makedirs(folder_path)
        else:
            folder_path=""

        with lumapi.FDTD(hide=False) as fdtd:
            # Step 1: build geometry
            fdtd.addrect()
            fdtd.set("name", "substrate")
            fdtd.set("material", self.mat_substrate)
            fdtd.set("x min", x0 - sim_buffer)
            fdtd.set("x max", x_end + sim_buffer)
            fdtd.set("y", y)
            fdtd.set("y span", y_span + sim_buffer)
            fdtd.set("z max", 0)
            fdtd.set("z min", -self.thickness_substrate)
            fdtd.set("override mesh order from material database", 1)
            fdtd.set("mesh order", 3)
            fdtd.set("alpha", 0.2)

            fdtd.addrect()
            fdtd.set("name", "superstrate")
            fdtd.set("material", self.mat_superstrate)
            fdtd.set("x min", x0 - sim_buffer)
            fdtd.set("x max", x_end + sim_buffer)
            fdtd.set("y", y)
            fdtd.set("y span", y_span + sim_buffer)
            fdtd.set("z min", 0)
            fdtd.set("z max", self.thickness_superstrate)
            fdtd.set("override mesh order from material database", 1)
            fdtd.set("mesh order", 3)
            fdtd.set("alpha", 0.2)

            wg1_y_center = y_dev + self.gap_start / 2 + self.width_wg1 / 2
            wg2_end_y_center = (
                wg1_y_center - self.width_wg1 / 2 - self.gap_end - self.width_wg2 / 2
            )
            wg2_out_y_center = wg1_y_center - self.output_pitch

            # wg1: SWG waveguide and tapers
            fdtd.addpoly()
            fdtd.set("name", "wg1_swg_taper1")
            fdtd.set("x", x0)
            fdtd.set("y", wg1_y_center)
            fdtd.set("z min", 0)
            fdtd.set("z max", self.thickness_device)

            fdtd.eval("M=matrix(6,2);")
            fdtd.eval(f"M(1,1:2)=[{x0-sim_buffer}, {self.width_wg1/2}];")
            fdtd.eval(f"M(2,1:2)=[{x0}, {self.width_wg1/2}];")
            fdtd.eval(f"M(3,1:2)=[{x1}, {self.width_wg1_taper_tip/2}];")
            fdtd.eval(f"M(4,1:2)=[{x1}, {-self.width_wg1_taper_tip/2}];")
            fdtd.eval(f"M(5,1:2)=[{x0}, {-self.width_wg1/2}];")
            fdtd.eval(f"M(6,1:2)=[{x0-sim_buffer}, {-self.width_wg1/2}];")
            fdtd.eval("set('vertices',M);")
            fdtd.set("material", self.mat_device)

            fdtd.addpoly()
            fdtd.set("name", "wg1_swg_taper2")
            fdtd.set("x", x0)
            fdtd.set("y", wg1_y_center)
            fdtd.set("z min", 0)
            fdtd.set("z max", self.thickness_device)

            fdtd.eval("M=matrix(6,2);")
            fdtd.eval(f"M(1,1:2)=[{x4}, {self.width_wg1_taper_tip/2}];")
            fdtd.eval(f"M(2,1:2)=[{x5}, {self.width_wg1/2}];")
            fdtd.eval(f"M(3,1:2)=[{x5+sim_buffer}, {self.width_wg1/2}];")
            fdtd.eval(f"M(4,1:2)=[{x5+sim_buffer}, {-self.width_wg1/2}];")
            fdtd.eval(f"M(5,1:2)=[{x5}, {-self.width_wg1/2}];")
            fdtd.eval(f"M(6,1:2)=[{x4}, {-self.width_wg1_taper_tip/2}];")
            fdtd.eval("set('vertices',M);")
            fdtd.set("material", self.mat_device)

            # Add wg1 SWGs by executing the loop in lumerical
            # Performing this operation through lumapi is very slow...
            cmds = f"""
            cursor = {x0};
            for(0; cursor <= ({x_end} - {x0}); 0) {{
                addrect;
                set('name', 'SWG');
                set('material','{self.mat_device}');
                set('x min', cursor);
                set('x max', cursor + {self.swg_pitch} * {self.swg_dc});
                set("y", {wg1_y_center});
                set("y span", {self.width_wg1});
                set("z min", 0);
                set("z max", {self.thickness_device});
                cursor = cursor + {self.swg_pitch};
                
            }}
            """
            fdtd.eval(cmds)

            # wg1: strip fishbone section
            if self.width_fishbone > 0:
                fdtd.addrect()
                fdtd.set("name", "wg1_fishbone")
                fdtd.set("material", self.mat_device)
                fdtd.set("x min", x4)
                fdtd.set("x max", x5 + sim_buffer)
                fdtd.set("y", wg1_y_center)
                fdtd.set("y span", self.width_fishbone)
                fdtd.set("z min", 0)
                fdtd.set("z max", self.thickness_device)

            fdtd.addpoly()
            fdtd.set("name", "wg2_output_taper")
            fdtd.set("x", x0)
            fdtd.set("y", wg2_out_y_center)
            fdtd.set("z min", 0)
            fdtd.set("z max", self.thickness_device)

            fdtd.eval("M=matrix(6,2);")
            fdtd.eval(f"M(1,1:2)=[{x4}, {self.width_wg2/2}];")
            fdtd.eval(f"M(2,1:2)=[{x5}, {self.width_wg_port/2}];")
            fdtd.eval(f"M(3,1:2)=[{x5+sim_buffer}, {self.width_wg_port/2}];")
            fdtd.eval(f"M(4,1:2)=[{x5+sim_buffer}, {-self.width_wg_port/2}];")
            fdtd.eval(f"M(5,1:2)=[{x5}, {-self.width_wg_port/2}];")
            fdtd.eval(f"M(6,1:2)=[{x4}, {-self.width_wg2/2}];")
            fdtd.eval("set('vertices',M);")
            fdtd.set("material", self.mat_device)

            # wg2: strip waveguide beginning taper section
            wg2_start_y_center = wg1_y_center - self.width_wg1 / 2
            fdtd.addpoly()
            fdtd.set("name", "wg2_start")
            fdtd.set("x", x0)
            fdtd.set("y", wg2_start_y_center)
            fdtd.set("z min", 0)
            fdtd.set("z max", self.thickness_device)

            fdtd.eval("M=matrix(4,2);")
            fdtd.eval(f"M(1,1:2)=[{x1}, {-self.gap_start}];")
            fdtd.eval(f"M(2,1:2)=[{x2}, {-self.gap_start}];")
            fdtd.eval(f"M(3,1:2)=[{x2}, {-self.gap_start-self.width_wg2}];")
            fdtd.eval(f"M(4,1:2)=[{x1}, {-self.gap_start-self.width_wg2_tip}];")
            fdtd.eval("set('vertices',M);")
            fdtd.set("material", self.mat_device)

            # wg2: strip waveguide s-bend section (gap_start to gap_end)
            y1 = wg1_y_center - self.width_wg1 / 2 - self.gap_start - self.width_wg2 / 2
            y2 = wg2_end_y_center
            sbend_height = -np.abs(y1 - y2)
            fdtd.addobject("s_bend_wg")
            fdtd.set("x", x2)
            fdtd.set("y", y1)
            fdtd.set("z", self.thickness_device / 2)
            fdtd.set("base width", self.width_wg2)
            fdtd.set("base height", self.thickness_device)
            fdtd.set("x span", self.length_bend)
            fdtd.set("y span", sbend_height)
            fdtd.set("material", self.mat_device)

            # wg2: strip waveguide s-bend section (gap_end to output_pitch)
            y3 = wg2_out_y_center
            y2 = wg2_end_y_center
            sbend_height = -np.abs(y3 - y2)
            fdtd.addobject("s_bend_wg")
            fdtd.set("x", x3)
            fdtd.set("y", y2)
            fdtd.set("z", self.thickness_device / 2)
            fdtd.set("base width", self.width_wg2)
            fdtd.set("base height", self.thickness_device)
            fdtd.set("x span", self.length_bend_output)
            fdtd.set("y span", sbend_height)
            fdtd.set("material", self.mat_device)

            # wg2: strip waveguide bend taper in
            radius = 2e-6
            fdtd.addobject("90_bend_wg")
            fdtd.set("start angle", 90)
            fdtd.set("radius", radius)
            fdtd.set("x", x1)
            fdtd.set(
                "y",
                wg1_y_center
                - self.width_wg1 / 2
                - self.gap_start
                - self.width_wg2_tip / 2
                - radius,
            )
            fdtd.set("z", self.thickness_device / 2)
            fdtd.set("base width", self.width_wg2_tip)
            fdtd.set("base height", self.thickness_device)
            fdtd.set("material", self.mat_device)

            # Build simulation
            # Add simulation region
            fdtd.addfdtd()
            fdtd.set("mesh accuracy", mesh)
            fdtd.set("x min", x0)
            fdtd.set("x max", x_end)
            fdtd.set("y", y)
            fdtd.set("y span", y_span_sim)
            fdtd.set("z", self.thickness_device / 2)
            fdtd.set("z span", z_span_sim)
            # Define simulation wavelength range
            fdtd.setglobalsource("Wavelength start", wavl_start)
            fdtd.setglobalsource("Wavelength stop", wavl_stop)
            # Estimate the required simulation time assume ng=4
            sim_time = 2 * (x_end - x0) / (3e8 / 4)  # factor of 2 for round-trip
            fdtd.set("simulation time", sim_time)
            # Define boundary conditions
            if self.mat_substrate == self.mat_superstrate:
                fdtd.set("z min bc", "Symmetric")
            else:
                fdtd.set("z min bc", "PML")
            fdtd.set("y max bc", "PML")
            fdtd.set("y min bc", "PML")
            fdtd.set("y max bc", "PML")
            fdtd.set("x min bc", "PML")
            fdtd.set("x max bc", "PML")

            # Add ports
            fdtd.addport()
            fdtd.set("name", "opt1")
            fdtd.set("x", x0 + sim_buffer)
            fdtd.set("y", wg1_y_center)
            fdtd.set("y span", y_span_port)
            fdtd.set("z", self.thickness_device / 2)
            fdtd.set("z span", z_span_sim)

            fdtd.addport()
            fdtd.set("name", "opt2")
            fdtd.set("direction", "Backward")
            fdtd.set("x", x_end - sim_buffer)
            fdtd.set("y", wg1_y_center)
            fdtd.set("y span", y_span_port)
            fdtd.set("z", self.thickness_device / 2)
            fdtd.set("z span", z_span_sim)

            fdtd.addport()
            fdtd.set("name", "opt3")
            fdtd.set("direction", "Backward")
            fdtd.set("x", x_end - sim_buffer)
            fdtd.set("y", wg2_out_y_center)
            fdtd.set("y span", y_span_port)
            fdtd.set("z", self.thickness_device / 2)
            fdtd.set("z span", z_span_sim)

            fdtd.select("FDTD::ports")
            fdtd.set("Monitor frequency points", wavl_pts)
            # Add field monitors (optional)
            swg_wg_id = (
                f"swg_width{int(1e9*self.width_wg1)}nm_pitch{int(1e9*self.swg_pitch)}nm"
            )
            if simfile:
                fdtd.save(folder_path+f"{swg_wg_id}")
        return

    def init_custom_material(self, lumerical):
        """
        Add material models to Lumerical solvers.

        Parameters
        ----------
        lumerical : lumapi lumerical instance.
            Lumerical instance to add the material into.

        Returns
        -------
        None.

        """
        matname = "Air (1)"
        newmaterial = lumerical.addmaterial("Dielectric")
        lumerical.setmaterial(newmaterial, "name", matname)
        lumerical.setmaterial(matname, "Refractive Index", 1)

        matname = "Si (Silicon) - Dispersive & Lossless"
        newmaterial = lumerical.addmaterial("Lorentz")
        lumerical.setmaterial(newmaterial, "name", matname)
        lumerical.setmaterial(matname, "Permittivity", 7.98737492)
        lumerical.setmaterial(matname, "Lorentz Linewidth", 1e8)
        lumerical.setmaterial(matname, "Lorentz Permittivity", 3.68799143)

        matname = "SiO2 (Glass) - Dispersive & Lossless"
        newmaterial = lumerical.addmaterial("Lorentz")
        lumerical.setmaterial(newmaterial, "name", matname)
        lumerical.setmaterial(matname, "Permittivity", 2.119881)
        lumerical.setmaterial(matname, "Lorentz Linewidth", 1e10)
        lumerical.setmaterial(matname, "Lorentz Resonance", 3.309238e13)
        lumerical.setmaterial(matname, "Lorentz Permittivity", 49.43721)
        lumerical.eval(f"setmaterial('{matname}', 'color', [0.85; 0.27; 0.27; 1]);") 

        matname = "SiO2 (Glass) - Const"
        newmaterial = lumerical.addmaterial("Dielectric")
        lumerical.setmaterial(newmaterial, "name", matname)
        lumerical.setmaterial(matname, "Permittivity", 1.444 * 1.444)

        matname = "SWG_strip"
        newmaterial = lumerical.addmaterial("Dielectric")
        lumerical.setmaterial(newmaterial, "name", matname)
        lumerical.setmaterial(matname, "Refractive Index", 2.73)
        lumerical.switchtolayout()

if __name__ == "__main__":
    swgaf = swg_adiabatic_filter(
        width_wg1=0.44e-6,
        width_wg1_taper_tip=0.06e-6,
        width_wg2_tip=0.1e-6,
        width_wg2=0.245e-6,
        
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
    
    #swgaf.Wg_analysis(swg_model='FDTD',folder=True,simfile=True)         # 1 WAVEGUIDE ANALYSIS - Calls 1a and, 1b(EMT) or 1c(FDTD)
    # swgaf.FDE_stripwg(folder=True,simfile=True)                          # 1a FDE Strip Waveguide Simulation    
    # swgaf.EMT_swg(simulate_lumerical=True,folder=True,simfile=True)      # 1b FDE EMT Material Wavgeuide Simulation
    # swgaf.FDTD_swg(folder=True,simfile=True)                             # 1c Optional: FDTD SWG Waveguide Simulation

    # swgaf.supermodes(folder=True,simfile=True)                           # 2 Optional: SUPERMODE ANALYSIS

    # swgaf.EMT_bandstructure(folder=True,simfile=True)                    # 3 FDTD EMT Bandstructure Simulation 
    
    # swgaf.optimize_device(folder=True,simfile=True)                      # 4 EME Optimize coupling lengths
    # swgaf.generate_spectrum(folder=True,simfile=True)                    # 4a Optional: EME Spectrum generation to get a rough gauge device is functioning as a filter, NOT VALID for analyzing crossing wavelength.
    
    # swgaf.FDTD_fullsim(folder=True,simfile=True)                         # 5 Final Verification

# %%

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
#import cupy as np
from scipy.spatial.transform import Rotation
import scipy.io
import os

class Base_gwf:
    """
    Basic tools used by classes inheriting from this class.
    """
    def plot_gwf(self, start_time=None, stop_time=None):
        fig, ax = plt.subplots()
        remove_rf_flag = False

        if hasattr(self, "t_180"):
            ax.axvline(x=(self.t_180+self.mid_180)*1e3, ls="--", color="black")
            #ax.axvline(x=(self.mid_180)*1e3, ls="--", color="black")

            if hasattr(self, "rf") == False:
                self.rf = np.ones(self.gwf.shape)
                self.rf[self.t >= self.t_180+self.w_180, :] = -1
                remove_rf_flag = True

        if hasattr(self, "rf") == False:
            self.rf = np.ones(self.gwf.shape)
            plot_title = "Absolute gradient"

            remove_rf_flag = True
        else:
            plot_title = "Effective gradient"

        ax.plot(self.t*1e3, self.gwf_x*self.rf[:,0]*1e3, label="x")
        ax.plot(self.t*1e3, self.gwf_y*self.rf[:,1]*1e3, label="y")
        ax.plot(self.t*1e3, self.gwf_z*self.rf[:,2]*1e3, label="z")

        ax.fill_between(self.t*1e3, self.gwf_x*self.rf[:,0]*1e3, alpha=0.2)
        ax.fill_between(self.t*1e3, self.gwf_y*self.rf[:,1]*1e3, alpha=0.2)
        ax.fill_between(self.t*1e3, self.gwf_z*self.rf[:,2]*1e3, alpha=0.2)

        if start_time:
            ax.axvline(x=start_time*1e3, color="black")
        if stop_time:
            ax.axvline(x=stop_time*1e3, color="black")
        

        ax.set_title(plot_title)
        ax.set_xlabel("Time [ms]")
        ax.set_ylabel("Gradient amplitude [mT/m]")
        ax.legend()

        if remove_rf_flag:
            delattr(self, "rf")

    def get_rf(self, optimize=False, start_time=None, x0=False, print_result=False):
        """Creates an rf array

        Args:
            optimize (bool, optional): Use an optimizer to get the proper switch from 1 to -1. Defaults to False.
            start_time (float, optional): start time of integration in [s]. Defaults to None.
            x0 (bool, optional): Initial guess for the time of the midpoint of the refocusing RF pulse. Defaults to False.
            print_result (bool, optional): Prints some results. Defaults to False.
        """

        # If this is true, we try to find the optimal mid_180 for 0 q at the end of slice selection
        if optimize:
            if start_time == None:
                start_time = self.t_excitation + 3100e-6

            if x0 == False:
                x0 = 5e-3

            result = minimize(self._get_optimal_rf_objective_function, x0=np.array([int(x0/self.dt)]), args=(start_time), method="Nelder-Mead")
            #result = minimize(self._get_optimal_rf_objective_function, x0=np.array([int((self.t_180+3e-3)/4e-6)]), args=(start_time), method="Nelder-Mead")
            self.mid_180 = result.x[0]*self.dt
            self.mid_180_idx = result.x[0]

            if print_result:
                print(f"Time mid 180: {self.mid_180+self.t_180*1e3:.1f} ms\t|q| = {result.fun:.1f}")

            self.rf = np.ones(self.gwf.shape)
            self.rf[self.t >= self.t_180+self.mid_180, :] = -1
        else:
            self.rf = np.ones(self.gwf.shape)
            self.rf[self.t >= self.t_180+self.mid_180, :] = -1
    
    def _get_optimal_rf_objective_function(self, mid_180_idx, start_time=None):
        rf = np.ones(self.gwf.shape)
        rf[self.t >= self.t_180+mid_180_idx*self.dt, :] = -1

        gamma = 2.6751e+08
        q = gamma*np.cumsum(self.gwf[self.t >= start_time,:]*rf[self.t >= start_time,:], axis=0)*self.dt

        return np.abs(q[-1,2])

    def get_optimal_TE(self, start_time=None, run_get_optimal_rf=True, initial_guess=90e-3):
        if run_get_optimal_rf:
            self.get_rf(optimize=True, start_time=start_time)

        result = minimize(self._get_optimal_TE_objective_function, x0=np.array([initial_guess]), args=(start_time), method="Nelder-Mead")
        TE = result.x[0]

        self.TE = TE
        return TE

    def _get_optimal_TE_objective_function(self, TE, start_time=None):
        start, stop = self._timing_info(start_time, TE)

        # The gyromagnetic constant
        gamma = 2.6751e+08

        q = gamma*np.cumsum(self.gwf[start:stop+1,:]*self.rf[start:stop+1,:], axis=0)*self.dt
        return np.abs(q[-1,1]) # We want q in the y-axis to be 0 at TE, as that is where the phase should be 0

    def get_optimal_excitation_time(self, initial_guess, stop_time):

        # This is best done on a b0 shot
        # Provide an initial guess where q_z is supposed to be zero after the excitation rewinder
        pass
        result = minimize(self._get_optimal_excitation_time_objective_function, x0=np.array([initial_guess]), args=(stop_time), method="Nelder-Mead")
        t_excitation = result.x[0]

        self.t_excitation = t_excitation
        return t_excitation

    def _get_optimal_excitation_time_objective_function(self, excitation_time, stop_time=None):
        start, stop = self._timing_info(excitation_time, stop_time)

        # The gyromagnetic constant
        gamma = 2.6751e+08

        q = gamma*np.cumsum(self.gwf[start:stop+1,:], axis=0)*self.dt
        return np.abs(q[-1,2]) # We want q in the z-axis to be 0 at stop time, as that is where the phase should be 0

    def get_q_old(self, start_time=None, stop_time=None):
        start, stop = self._timing_info(start_time, stop_time)

        # The gyromagnetic constant
        gamma = 2.6751e+08

        self.q = gamma*np.cumsum(self.gwf[start:stop+1,:]*self.rf[start:stop+1,:], axis=0)*self.dt

    def get_q(self, gwf=None, rf=None, start_time=None, stop_time=None, update_attribute=True):
        start, stop = self._timing_info(start_time, stop_time)

        # The gyromagnetic constant
        gamma = 2.6751e+08

        if gwf is None:
            gwf = self.gwf
        if rf is None:
            rf = self.rf

        #if stop > len(gwf)-2:
        #    stop = len(gwf)-2

        # Calculate q
        q = gamma*np.cumsum(gwf[start:stop+1]*rf[start:stop+1], axis=0)*self.dt
        
        if update_attribute:
            self.q = q

        return q

    def plot_q(self, start_time=None, stop_time=None, update_attribute=True):
        self.get_q(start_time=start_time, stop_time=stop_time, update_attribute=update_attribute)
        start, stop = self._timing_info(start_time, stop_time)

        fig, ax = plt.subplots()

        ax.plot(self.t[start:stop+1]*1e3, self.q[:,0], label="x")
        ax.plot(self.t[start:stop+1]*1e3, self.q[:,1], label="y")
        ax.plot(self.t[start:stop+1]*1e3, self.q[:,2], label="z")

        ax.fill_between(self.t[start:stop+1]*1e3, self.q[:,0], alpha=0.2)
        ax.fill_between(self.t[start:stop+1]*1e3, self.q[:,1], alpha=0.2)
        ax.fill_between(self.t[start:stop+1]*1e3, self.q[:,2], alpha=0.2)

        q_magnitude = np.linalg.norm(self.q, axis=1)
        ax.plot(self.t[start:stop+1]*1e3, q_magnitude, label="Magnitude", color="black")

        ax.set_title("q")
        ax.set_xlabel("Time [ms]")
        ax.set_ylabel("q")
        ax.legend()

    def get_m1(self, gwf=None, rf=None, start_time=None, stop_time=None, update_attribute=True):
        start, stop = self._timing_info(start_time, stop_time)

        # The gyromagnetic constant
        gamma = 2.6751e+08

        if gwf is None:
            gwf = self.gwf
        if rf is None:
            rf = self.rf

        q = self.get_q(gwf, rf, start_time, stop_time, update_attribute=update_attribute)
        m1 = -1*np.sum(q)*self.dt

        self.m1 = m1
        return m1

    def get_b_old(self, start_time=None, stop_time=None):
        """
            Stop should be TE or end of encoding time.
        """
        # Always recalculate q since we might have different start/stop times
        self.get_q(start_time, stop_time)

        start, stop = self._timing_info(start_time, stop_time)

        # Calculate M0 and b
        M0 = np.matmul(self.q[start:stop+1, :].T, self.q[start:stop+1, :])*self.dt
        self.b = np.trace(M0) 
        return self.b

    def get_b(self, gwf=None, rf=None, start_time=None, stop_time=None, return_b_tensor=False, update_attribute=True):
        """
            Stop should be TE or end of encoding time.
        """
        if gwf is None:
            gwf = self.gwf
        if rf is None:
            rf = self.rf

        # Always recalculate q since we might have different start/stop times
        q = self.get_q(gwf, rf, start_time, stop_time, update_attribute=update_attribute)

        start, stop = self._timing_info(start_time, stop_time)

        # Calculate M0 and b
        #M0 = np.matmul(q[start:stop+1].T, q[start:stop+1])*self.dt
        M0 = np.matmul(q.T, q)*self.dt

        #fig, ax = plt.subplots()
        #ax.plot(q)
        #print(start, stop)

        if M0.ndim > 1:
            b = np.trace(M0)
        else:
            b = M0

        if update_attribute:
            self.b = b

        if return_b_tensor:
            #b_tensor = np.array((M0[0,0], M0[1,1], M0[2,2], M0[0,1], M0[0,2], M0[1,2]))
            #b_tensor = np.array((M0[0,0], M0[1,1], M0[2,2], M0[0,1]*np.sqrt(2), M0[0,2]*np.sqrt(2), M0[1,2]*np.sqrt(2)))
            b_tensor = M0
            return b_tensor
        else:
            return b
        
    def get_m1(self, gwf=None, rf=None, start_time=None, stop_time=None, update_attribute=True):
        if gwf is None:
            gwf = self.gwf
        if rf is None:
            rf = self.rf

        # Always recalculate q since we might have different start/stop times
        q = self.get_q(gwf, rf, start_time, stop_time, update_attribute=update_attribute)
        alpha = -np.sum(q)*self.dt
        alpha = -np.cumsum(q)[-1]*self.dt

        return alpha

    def get_b_over_time(self, start_time=None, stop_time=None):
        # We wish to plot the b-value over time
        # We need to calculate the b-value for each time point

        start, stop = self._timing_info(start_time, stop_time)

        # Create the arrays
        self.b_over_time = np.zeros((stop-start+1))
        
        for time_point in range(self.b_over_time.shape[0]):
            self.b_over_time[time_point] = self.get_b(start_time=start_time, stop_time=(self.t[start]+time_point*self.dt))

    def plot_b_over_time(self, start_time=None, stop_time=None):
        self.get_b_over_time(start_time, stop_time)

        start, stop = self._timing_info(start_time, stop_time)

        fig, ax = plt.subplots()
        ax.plot(self.t[start:stop+1]*1e3, self.b_over_time*1e-6)

        ax.set_title("b-value over time")
        ax.set_xlabel("Time [ms]")
        ax.set_ylabel("b-value [s/mm2]")

    def export_gwf_rf_dt(self, path, fname="dMRItools_gwf_rf_dt", start_time=None, stop_time=None):
        """
        Exports the gwf, rf, and dt variables to .mat files for use in other dMRI toolboxes such as fwf_sequence_tools.

        Args:
            path (string): Path to folder
            fname (str, optional): Filename without extension. Defaults to "dMRItools_gwf_rf_dt".
        """
        start, stop = self._timing_info(start_time, stop_time)

        gwf_rf_dt = {"gwf" : self.gwf[start:stop+1],
               "rf" : self.rf[start:stop+1,0],
               "dt" : self.dt}

        fname_with_extension = f"{fname}.mat"
        file_path = os.path.join(path, fname_with_extension)
        scipy.io.savemat(file_path, gwf_rf_dt, oned_as="column")

    def _timing_info(self, start_time, stop_time):
        """
        Translates time from seconds to array indices based on self.t
        """
        # Timing info
        if start_time == None:
            start_time = self.t[0]
        if stop_time == None:
            stop_time = self.t[-1]

        start_idx = np.where(self.t >= start_time)[0][0]
        stop_idx = np.where(self.t <= stop_time)[0][-1]

        return start_idx, stop_idx

    def _separate_gwf_axes(self):
        """
        Creates separate x, y, and z arrays. Used in e.g. plotting. Important to call whenever self.gwf is changed.
        """
        # Set variables with the separated axes
        self.gwf_x = self.gwf[:,0]
        self.gwf_y = self.gwf[:,1]
        self.gwf_z = self.gwf[:,2]





class Gwf(Base_gwf):
    """
    Object that performs calculations of diffusion waveforms. Can essentially be used on any waveforms, including entire pulse sequences.
    """
    def __init__(self, dt, gwf=None, rf=None, duration180=None, pre180=None, post180=None, t180=None):
        # Time resolution
        self.dt = dt


        if gwf is not None and rf is not None:
            self.rf = np.hstack((rf, rf, rf))
            self.gwf = gwf
        else:

            # RF 180 info
            self.duration180 = duration180
            self.rf180 = pre180[:int(self.duration180/self.dt)]*0 # We do this to make sure they have the same xyz dimensions

            # Gradient info
            self.pre180 = pre180
            self.post180 = post180

            # Also store the lengths of the original arrays
            self.npre180 = len(self.pre180)
            self.npost180 = len(self.post180)

            # Stitch together the components
            if post180 is None:
                # In this case, we assume there is just one component
                self.gwf = pre180
            else:
                self.gwf = np.concatenate((self.pre180, self.rf180, self.post180), axis=0)

            # Make 3-dimentional if not already.
            if self.gwf.ndim != 2:
                new_gwf = np.zeros((self.gwf.shape[0], 3))
                gwf_axis_to_add = np.zeros((self.gwf.shape[0]))

                # We have a 1D gwf, make it 3D
                new_gwf[:,0] = self.gwf
                new_gwf[:,1] = gwf_axis_to_add
                new_gwf[:,2] = gwf_axis_to_add

                self.gwf = new_gwf

            else: # gwf might be 2D, make it 3D
                if self.gwf.shape[1] == 2: # We have a 2D gwf, make it 3D
                    new_gwf = np.zeros((self.gwf.shape[0], 3))
                    gwf_axis_to_add = np.zeros((self.gwf.shape[0]))
                    new_gwf[:,0] = self.gwf[:,0]
                    new_gwf[:,1] = self.gwf[:,1]*0.3
                    new_gwf[:,2] = gwf_axis_to_add

                    self.gwf = new_gwf

            # Get the time axis
            self.t = np.linspace(0, self.gwf.shape[0]*self.dt, self.gwf.shape[0])

            # Create the RF array (1 or -1 depending on which side we are of the rf180)
            self.rf = np.ones(self.gwf.shape)
            if t180 is None:
                self.rf[self.t >= self.pre180.shape[0]*self.dt+self.duration180/2] = -1
            else:
                self.rf[self.t >= t180+self.duration180/2] = -1

        # Set variables with the separated axes
        self.gwf_x = self.gwf[:,0]
        self.gwf_y = self.gwf[:,1]
        self.gwf_z = self.gwf[:,2]

        # Get the time axis
        self.t = np.linspace(0, self.gwf.shape[0]*self.dt, self.gwf.shape[0])

    
    def get_pre180_and_post180(self):
        self.gwf_pre180 = self.gwf[:self.npre180, :]
        self.gwf_post180 = self.gwf[-self.npost180:, :]

    #def get_q(self, gwf=None, start_time=None, stop_time=None):
        #start, stop = self._timing_info(start_time, stop_time)

        ## The gyromagnetic constant
        #gamma = 2.6751e+08

        #if gwf is None:
            #gwf = self.gwf

        #self.q = gamma*np.cumsum(gwf[start:stop+1,:]*self.rf[start:stop+1,:], axis=0)*self.dt


    def get_b_old(self, gwf=None, start_time=None, stop_time=None, update_attribute=True):
        """
            Stop should be TE or end of encoding time.
        """
        if gwf is None:
            gwf = self.gwf

        # Always recalculate q since we might have different start/stop times
        self.get_q(gwf, start_time, stop_time)

        start, stop = self._timing_info(start_time, stop_time)

        # Calculate M0 and b
        M0 = np.matmul(self.q[start:stop+1, :].T, self.q[start:stop+1, :])*self.dt
        b = np.trace(M0)

        if update_attribute:
            self.b = b

        return b

    def get_b_over_time(self, start_time=None, stop_time=None):
        # We wish to plot the b-value over time
        # We need to calculate the b-value for each time point

        start, stop = self._timing_info(start_time, stop_time)

        # Create the arrays
        self.b_over_time = np.zeros((stop-start+1))
        
        for time_point in range(self.b_over_time.shape[0]):
            self.b_over_time[time_point] = self.get_b(start_time=start_time, stop_time=(self.t[start]+time_point*self.dt))

    def set_b_by_scaling_amplitude(self, b_desired, print_results=False):
        """
            Uses an optimizer to scale the amplitude to achieve a certain bvalue.

            b_desired = float. The desired b-value in SI units. E.g. [s/mm2]*1e6, [ms/um2]*1e9
        """

        results = minimize(self._set_b_by_scaling_amplitude_objective_function, x0=np.array([1]), args=(b_desired), method="Nelder-Mead")

        # Update the gwf by scaling with the optimized factor
        if results.x < 0:
            factor = -1
        else:
            factor = 1

        self.gwf *= results.x*factor
        self.pre180 *= results.x*factor
        self.post180 *= results.x*factor

        if print_results:
            print(f"Scale factor: {results.x}")
            print(f"b-value: {self.get_b()*1e-6:.1f} s/mm2")

    def _set_b_by_scaling_amplitude_objective_function(self, factor, b_desired=0):
        """
        Objective function for set_b_by_scaling_amplitude. Checks the absolute difference between the desired and the optimized b-value.
        """
        if factor < 0:
            factor *= -1
        gwf = self.gwf*factor
        b_optimized = self.get_b(gwf=gwf, update_attribute=False)

        return np.abs(b_desired - b_optimized)

    def plot_q(self, start_time=None, stop_time=None):
        self.get_q(start_time, stop_time)
        start, stop = self._timing_info(start_time, stop_time)

        fig, ax = plt.subplots()

        ax.plot(self.t[start:stop+1]*1e3, self.q[:,0], label="x")
        ax.plot(self.t[start:stop+1]*1e3, self.q[:,1], label="y")
        ax.plot(self.t[start:stop+1]*1e3, self.q[:,2], label="z")

        ax.fill_between(self.t[start:stop+1]*1e3, self.q[:,0], alpha=0.2)
        ax.fill_between(self.t[start:stop+1]*1e3, self.q[:,1], alpha=0.2)
        ax.fill_between(self.t[start:stop+1]*1e3, self.q[:,2], alpha=0.2)

        ax.set_title("q-vector")
        ax.set_xlabel("Time [ms]")
        ax.set_ylabel("q")
        ax.legend()

    def plot_b_over_time(self, start_time=None, stop_time=None):
        self.get_b_over_time(start_time, stop_time)

        start, stop = self._timing_info(start_time, stop_time)

        fig, ax = plt.subplots()
        ax.plot(self.t[start:stop+1]*1e3, self.b_over_time*1e-6)

        ax.set_title("b-value over time")
        ax.set_xlabel("Time [ms]")
        ax.set_ylabel("b-value [s/mm2]")

    def rotate_encoding(self, angle_x=0, angle_y=0, angle_z=0, inverse=False):

        # rotation vectors
        rx = Rotation.from_rotvec(angle_x * np.array([1, 0, 0]))
        ry = Rotation.from_rotvec(angle_y * np.array([0, 1, 0]))
        rz = Rotation.from_rotvec(angle_z * np.array([0, 0, 1]))

        # Apply rotation
        if inverse:
            self.gwf = rz.apply(self.gwf, inverse=inverse)
            self.gwf = ry.apply(self.gwf, inverse=inverse)
            self.gwf = rx.apply(self.gwf, inverse=inverse)
        else:
            self.gwf = rx.apply(self.gwf, inverse=inverse)
            self.gwf = ry.apply(self.gwf, inverse=inverse)
            self.gwf = rz.apply(self.gwf, inverse=inverse)

        # Update gwf axes
        self._separate_gwf_axes()

    def rotate_encoding_with_uvec(self, uvec, rescale_b=False):
        """
        Rotates a [1,1,1] diffusion encoding to specified direction with unit vectors. Input is an array containing the unit vector.
        """
        # Get the current b-value
        if rescale_b:
            current_b = self.get_b()

        # Re-scale the diffusion encoding with the uvec to change the direction
        self.gwf *= uvec

        if rescale_b:
            self.set_b_by_scaling_amplitude(current_b)

        # Update gwf axes 
        self._separate_gwf_axes()

    def get_uvec(self):
        """
        Returns the uvec of the encoding from the B-tensor. Is therefore not compatible with negative directions.
        """

        # Get the B-tensor
        B_tensor = self.get_b(return_b_tensor=True)

        # Get the diagonal of the B-tensor
        B_tensor_diagonal = np.diagonal(B_tensor)

        # Take the square root of the diagonal
        B_tensor_diagonal_sqrt = np.sqrt(B_tensor_diagonal)

        # Normalize
        uvec = B_tensor_diagonal_sqrt/np.linalg.norm(B_tensor_diagonal_sqrt)

        return uvec


class Sequence_base(Base_gwf):
    def __init__(self, gwf=None, dt=None, t=None, start_time=None, stop_time=None):
        self.dt = dt

        if t is None:
            self.t = np.linspace(0, len(gwf)*dt, len(gwf))
        else:
            self.t = t

        start, stop = self._timing_info(start_time, stop_time)

        self.gwf = gwf[start:stop+1, :]
        self.t = self.t[start:stop+1]

        # Set variables with separated axes
        self._separate_gwf_axes()

    def segment_sequence(self, start_time, t_excitation, w_excitation, t_diffusion_pre180, w_diffusion_pre180, t_180, w_180, t_diffusion_post180, w_diffusion_post180, t_readout, w_readout):
        """
        Segments the pulse sequence into its different parts using timinng information from e.g. plotter output.
        The idea is to make all of the timing info attributes of the object, and to preserve all of the timing info as we change things in the sequence.
        """


        # Create attributes
        self.start_time = start_time
        self.t_excitation = t_excitation
        self.t_diffusion_pre180 = t_diffusion_pre180
        self.t_180 = t_180
        self.t_diffusion_post180 = t_diffusion_post180
        self.t_readout = t_readout

        # Duration (width)
        self.w_excitation = w_excitation
        self.w_diffusion_pre180 = w_diffusion_pre180
        self.w_180 = w_180
        self.w_diffusion_post180 = w_diffusion_post180
        self.w_readout = w_readout

        # Create empty gwf array
        gwf_stitched = np.zeros((len(self.t), 3))

        start_excitation, stop_excitation = self._timing_info(self.t_excitation, self.t_excitation + self.w_excitation)
        start_diffusion_pre180, stop_diffusion_pre180 = self._timing_info(self.t_diffusion_pre180, self.t_diffusion_pre180 + self.w_diffusion_pre180)
        start_180, stop_180 = self._timing_info(self.t_180, self.t_180 + self.w_180)
        start_diffusion_post180, stop_diffusion_post180 = self._timing_info(self.t_diffusion_post180, self.t_diffusion_post180 + self.w_diffusion_post180)
        start_readout, stop_readout = self._timing_info(self.t_readout, self.t_readout + self.w_readout)

        # Create the segmented arrays
        # I had to add the final index with 2 in order to capture the full segments for some reason.
        self.gwf_excitation = self.gwf[start_excitation:stop_excitation+2, :]
        self.gwf_diffusion_pre180 = self.gwf[start_diffusion_pre180:stop_diffusion_pre180+2, :]
        self.gwf_180 = self.gwf[start_180:stop_180+2, :]
        self.gwf_diffusion_post180 = self.gwf[start_diffusion_post180:stop_diffusion_post180+2, :]
        self.gwf_readout = self.gwf[start_readout:stop_readout+2, :]

        #print(self.gwf_excitation[-10:,2])
        #print(self.gwf_180[-10:, 2])
        #print(self.gwf_readout[-10:, 1])
        """
        # Merge the components into a gwf
        self.gwf = np.concatenate((self.gwf_excitation, self.gwf_diffusion_pre180, self.gwf_180, self.gwf_diffusion_post180, self.gwf_readout))
        self.t = np.linspace(self.t_excitation, len(self.gwf)*self.dt+self.t_excitation, len(self.gwf))
        self._separate_gwf_axes()
        """

        gwf_stitched[start_excitation:stop_excitation+2, :] = self.gwf_excitation
        gwf_stitched[start_diffusion_pre180:stop_diffusion_pre180+2, :] = self.gwf_diffusion_pre180
        gwf_stitched[start_180:stop_180+2, :] = self.gwf_180
        gwf_stitched[start_diffusion_post180:stop_diffusion_post180+2, :] = self.gwf_diffusion_post180
        gwf_stitched[start_readout:stop_readout+2, :] = self.gwf_readout

        self.gwf = gwf_stitched
        #self._separate_gwf_axes()
        self.update_gwf_array()

    def generic_sequence(self, xres, yres, zres, transmit_bandwidth=600, gwf_excitation=None, gwf_diffusion_pre180=None, gwf_180=None, gwf_diffusion_post180=None, gwf_readout=None, optimal=True, all_crushers=False, crushers=True, only_crush_when_needed=False, nominal_bvalue=800e6, qc=False):
        """
        This function builds a generic pulse sequence from the ground up, assuming infinite slew rates.
        The resolutions are given in mm. Triangular crushers are placed around the slice selection for the 180-pulse.

        If no gradient components are given, standard gradients are created.
        90-pulse slice selection duration 6 ms, amplitude given by zres.
        180-pulse slice selection duration 7 ms, amplitude given by zres.
        Diffusion gradients amplitude for bvalue 800 s/mm2, duration 20 ms each. Placed in x direction.
        Readout as a single zero.

        delta = 20 ms
        DELTA = 20 + 10 + 7 = 37 ms
        10 ms after 1st lobe. 7 ms 180-pulse.
        
        
        For an optimal pulse sequence, an excitation rewinder is created and placed right after the excitation pulse.
        Crushers are placed only on one axis, with the lowest resolution for crushing efficiency.

        For the non-optimal pulse sequence, the excitation rewinder is merged with the first crusher in the slice direction.
        Crushers are placed on all axes.

        xres : int. Resolution in m.
        yres : int. Resolution in m.
        zres : int. Resolution in m.
        transmit_bandwidth : int. The transmit bandwidth in Hz. Typically around 600-800 Hz (I checked two vendor implementations).
        gwf_* : Numpy array. Gradient time series. Shape (n_time_points, 3).
        optimal : Bool. Whether to create an optimally designed pulse sequence or not. Read description above.
        """
        # Constants
        gamma = 2.6751e+08

        if qc:
            qc_factor = 0
        else:
            qc_factor = 1

        ### Excitation gradient ###
        if gwf_excitation is None:
            # Calculate the pulse amplitude from the bandwidth and zres
            excitation_amplitude = 2*np.pi/gamma * transmit_bandwidth / zres

            gwf_excitation = self._create_gwf_component(delta=6e-3, slew_rate=1e6, amplitude=excitation_amplitude, axis=2)
            if optimal:
                #gwf_excitation = self.excitation_create_rewinder(gwf=gwf_excitation, update_attribute=False, start_time=3e-3, print_results=False)
                rewinder_duration = 3e-3
                gwf_rewinder = self._create_gwf_component(delta=rewinder_duration, slew_rate=excitation_amplitude/rewinder_duration*4, amplitude=-excitation_amplitude*3e-3*2/rewinder_duration, axis=2)

                gwf_excitation = np.concatenate((gwf_excitation, gwf_rewinder, np.zeros((int(3e-3/self.dt), 3))))*qc_factor
            else:
                gwf_excitation = np.concatenate((gwf_excitation, np.zeros((int(3e-3/self.dt), 3))))*qc_factor
            self.start_time = 3e-3

        ### Diffusion gradients ###
            diffusion_amplitude = np.sqrt(nominal_bvalue/(gamma**2 * (20e-3)**2 * (40e-3 - 20e-3 / 3)))
        #if gwf_diffusion_pre180 is None:
            ## We keep the timing parameters static, and only adjust amplitude for different b-values
            ## Calculate the amplitude needed for the nominal b-value using delta = 20e-3 and DELTA = 20e-3 + 10e-3 + 7e-3 + 3e-3 = 40e-3
            #diffusion_amplitude = np.sqrt(nominal_bvalue/(gamma**2 * (20e-3)**2 * (40e-3 - 20e-3 / 3)))

            #gwf_diffusion_pre180 = self._create_gwf_component(delta=20e-3, slew_rate=1e6, amplitude=diffusion_amplitude, axis=0)
            #gwf_diffusion_pre180 = np.concatenate((gwf_diffusion_pre180, np.zeros((int(10e-3/self.dt), 3))))
        ##else: 
            ##gwf_diffusion_pre180 = self.gwf_diffusion_pre180
        
        #if gwf_diffusion_post180 is None:
            ## We keep the timing parameters static, and only adjust amplitude for different b-values
            ## Calculate the amplitude needed for the nominal b-value using delta = 20e-3 and DELTA = 20e-3 + 10e-3 + 7e-3 + 3e-3 = 40e-3
            #diffusion_amplitude = np.sqrt(nominal_bvalue/(gamma**2 * (20e-3)**2 * (40e-3 - 20e-3 / 3)))

            #gwf_diffusion_post180 = self._create_gwf_component(delta=20e-3, slew_rate=1e6, amplitude=diffusion_amplitude, axis=0)
        ##else: 
            ##gwf_diffusion_post180 = self.gwf_diffusion_post180

        
        def check_crusher_duration(crusher_duration, crusher_area, crusher_amplitude):
            while crusher_duration < 2e-3:
                crusher_amplitude -= .1e-3
                crusher_duration = crusher_area/crusher_amplitude
            return crusher_duration, crusher_amplitude


        # Determine excitation pulse amplitude based on zres
        #excitation_amplitude = 10e-3

        crusher_amplitude_x = 30e-3
        crusher_amplitude_y = 30e-3
        crusher_amplitude_z = 30e-3

        ### Create crusher gradients
        # We aim to create phase dispersions of 4*pi
        # Find out the crusher area for each axis
        crushing_factor = 3 # Should be at least 2
        if all_crushers:
            number_of_axes_factor = 3
            number_of_axes_factor = 1
        else:
            number_of_axes_factor = 1

        crusher_area_x = crushing_factor*2*np.pi/(gamma*xres)*2/number_of_axes_factor # We double the area as we are planning to use triangular crusher pulses
        crusher_area_y = crushing_factor*2*np.pi/(gamma*yres)*2/number_of_axes_factor
        crusher_area_z = crushing_factor*2*np.pi/(gamma*zres)*2/number_of_axes_factor

        ### TEST
        #if all_crushers:
            #crusher_area_x = crushing_factor*2*np.pi/(gamma*np.sqrt(xres**2+yres**2+zres**2))
            #crusher_area_y = crushing_factor*2*np.pi/(gamma*np.sqrt(xres**2+yres**2+zres**2))
            #crusher_area_z = crushing_factor*2*np.pi/(gamma*np.sqrt(xres**2+yres**2+zres**2))

        # Check if we have enough crushing from diffusion gradients
        if only_crush_when_needed:
            #gwf_diffusion_post180_area = np.abs(np.sum(gwf_diffusion_post180[1:-1], axis=0))*self.dt
            gwf_diffusion_post180_area = diffusion_amplitude*20e-3

            if gwf_diffusion_post180_area >= crusher_area_x/2 or gwf_diffusion_post180_area >= crusher_area_y/2 or gwf_diffusion_post180_area >= crusher_area_z/2:
                # Turn off crushers
                crushers = False


        if crushers is False:
            crusher_area_x = 0
            crusher_area_y = 0
            crusher_area_z = 0

        # We set the crusher amplitude to be 30 mT/m. Calculate the crusher duration
        crusher_duration_x = crusher_area_x/crusher_amplitude_x
        crusher_duration_y = crusher_area_y/crusher_amplitude_y
        crusher_duration_z = crusher_area_z/crusher_amplitude_z

        # Make sure all crushers are at least 2 ms, for some realism...
        if crushers:
            crusher_duration_x, crusher_amplitude_x = check_crusher_duration(crusher_duration_x, crusher_area_x, crusher_amplitude_x)
            crusher_duration_y, crusher_amplitude_y = check_crusher_duration(crusher_duration_y, crusher_area_y, crusher_amplitude_y)
            crusher_duration_z, crusher_amplitude_z = check_crusher_duration(crusher_duration_z, crusher_area_z, crusher_amplitude_z)

            # Create the crusher components
            crusher_x = self._create_gwf_component(crusher_duration_x, slew_rate=2*crusher_amplitude_x/crusher_duration_x, amplitude=crusher_amplitude_x, axis=0)
            crusher_y = self._create_gwf_component(crusher_duration_y, slew_rate=2*crusher_amplitude_x/crusher_duration_y, amplitude=crusher_amplitude_x, axis=1)
            crusher_z2 = self._create_gwf_component(crusher_duration_z, slew_rate=2*crusher_amplitude_z/crusher_duration_z, amplitude=crusher_amplitude_z, axis=2)
            crusher_z1 = crusher_z2
        else:
            crusher_x = self._create_gwf_component(self.dt, slew_rate=10, amplitude=1e-3, axis=0)*0
            crusher_y = self._create_gwf_component(self.dt, slew_rate=10, amplitude=1e-3, axis=1)*0
            crusher_z2 = self._create_gwf_component(self.dt, slew_rate=10, amplitude=1e-3, axis=2)*0
            crusher_z1 = crusher_z2
        if optimal:
            crusher_z1 = crusher_z2
        else:
            # Should generate a new crusher_z1 here with area adjusted for refocusing slice selection
            excitation_area = 3e-3*excitation_amplitude

            crusher_area_z1 = (crusher_area_z/2 - excitation_area)*2 # We turn the crusher area into a square again before subtracting the excitation area, and then revert to a triangle

            # Determine the crusher duration. Make sure it is at least 12 us long and does not become too long.


            crusher_amplitude_z = 30e-3
            crusher_duration_z1 = crusher_area_z1/crusher_amplitude_z
            if crushers is False:
                crusher_duration_z = 5e-3
            while crusher_duration_z1 <= 1.8e-3 or crusher_duration_z1 > crusher_duration_z:#(crusher_duration_z1 > crusher_duration_x or crusher_duration_z1 > crusher_duration_y):
                crusher_amplitude_z -= .03e-3 # We change the amplitude, as it may need to be negative to refocus both the excitation and the crusher post 180.
                crusher_duration_z1 = crusher_area_z1/crusher_amplitude_z

            crusher_z1 = self._create_gwf_component(crusher_duration_z1, slew_rate=2*np.abs(crusher_amplitude_z)/crusher_duration_z1, amplitude=crusher_amplitude_z, axis=2)

        if all_crushers:
            # Find the axis with the longest crusher in duration
            dur_longest_elements = 0
            if crusher_x.shape[0] > dur_longest_elements:
                dur_longest_elements = crusher_x.shape[0]
            if crusher_y.shape[0] > dur_longest_elements:
                dur_longest_elements = crusher_y.shape[0]
            if crusher_z2.shape[0] > dur_longest_elements:
                dur_longest_elements = crusher_z2.shape[0]
            if crusher_z1.shape[0] > dur_longest_elements:
                dur_longest_elements = crusher_z1.shape[0]

            # Now pad the crusher gradients until they are all of equal length
            if crusher_x.shape[0] < dur_longest_elements:
                crusher_x = np.concatenate((crusher_x, np.zeros((int(dur_longest_elements-crusher_x.shape[0]), 3))))
            if crusher_y.shape[0] < dur_longest_elements:
                crusher_y = np.concatenate((crusher_y, np.zeros((int(dur_longest_elements-crusher_y.shape[0]), 3))))
            if crusher_z2.shape[0] < dur_longest_elements:
                crusher_z2 = np.concatenate((crusher_z2, np.zeros((int(dur_longest_elements-crusher_z2.shape[0]), 3))))
            if crusher_z1.shape[0] < dur_longest_elements:
                crusher_z1 = np.concatenate((crusher_z1, np.zeros((int(dur_longest_elements-crusher_z1.shape[0]), 3))))

            crusher_all1 = np.vstack((np.flip(crusher_x[:,0]), np.flip(crusher_y[:,1]), np.flip(crusher_z1[:,2])))
            crusher_all1 = np.transpose(crusher_all1)

            crusher_all2 = np.vstack((crusher_x[:,0], crusher_y[:,1], crusher_z2[:,2]))
            crusher_all2 = np.transpose(crusher_all2)

        
        ### Recreate Diffusion gradients ###
        if gwf_diffusion_pre180 is None:
            # We keep the timing parameters static, and only adjust amplitude for different b-values
            # Calculate the amplitude needed for the nominal b-value using delta = 20e-3 and DELTA = 20e-3 + 10e-3 + 7e-3 + 3e-3 = 40e-3
            diffusion_amplitude = np.sqrt(nominal_bvalue/(gamma**2 * (20e-3)**2 * (20e-3 + 10e-3 + 7e-3 + crusher_x.shape[0]*self.dt + 3e-3 - 20e-3 / 3)))

            gwf_diffusion_pre180 = self._create_gwf_component(delta=20e-3, slew_rate=1e6, amplitude=diffusion_amplitude, axis=0)
            gwf_diffusion_pre180 = np.concatenate((gwf_diffusion_pre180, np.zeros((int(10e-3/self.dt), 3))))
        #else: 
            #gwf_diffusion_pre180 = self.gwf_diffusion_pre180
        
        if gwf_diffusion_post180 is None:
            # We keep the timing parameters static, and only adjust amplitude for different b-values
            # Calculate the amplitude needed for the nominal b-value using delta = 20e-3 and DELTA = 20e-3 + 10e-3 + 7e-3 + 3e-3 = 40e-3
            diffusion_amplitude = np.sqrt(nominal_bvalue/(gamma**2 * (20e-3)**2 * (20e-3 + 10e-3 + 7e-3 + crusher_x.shape[0]*self.dt + 3e-3 - 20e-3 / 3)))

            gwf_diffusion_post180 = self._create_gwf_component(delta=20e-3, slew_rate=1e6, amplitude=diffusion_amplitude, axis=0)
            gwf_diffusion_post180 = np.concatenate((np.zeros((int(3e-3/self.dt), 3)), gwf_diffusion_post180))

        if gwf_180 is None:
            # Calculate the pulse amplitude from the bandwidth and zres
            gwf180_amplitude = 2*np.pi/gamma * transmit_bandwidth / zres

            gwf_180 = self._create_gwf_component(delta=7e-3, slew_rate=1e6, amplitude=gwf180_amplitude, axis=2)

            # Now decide on which axes we add crushers on
            #if crushers:
            if all_crushers:
                gwf_180 = np.concatenate((crusher_all1[:-1]*qc_factor, gwf_180[1:-1], crusher_all2[1:]*qc_factor))*qc_factor
            else:
                gwf_180 = np.concatenate((crusher_z1[:-1]*qc_factor, gwf_180[1:-1], crusher_z2[1:]*qc_factor))*qc_factor
            
            #gwf_180 = np.concatenate((gwf_180, np.zeros((int(3e-3/self.dt), 3))))

        
        if gwf_readout is None:
            gwf_readout = np.array([0,0,0])


        # Assign the gwfs to the attributes
        self.gwf_excitation = gwf_excitation
        self.gwf_diffusion_pre180 = gwf_diffusion_pre180
        self.gwf_180 = gwf_180
        self.gwf_diffusion_post180 = gwf_diffusion_post180
        self.gwf_readout = gwf_readout

        #fig, ax = plt.subplots()
        #ax.plot(self.gwf_180)
        #print(self.gwf_180.shape)

        # Some necessary timing info
        self.t_excitation = 0
        self.t = np.linspace(0, 150e-3, int(150e-3/self.dt))

        self.update_gwf_array()

    def generic_sequence_crusher_check(self, xres, yres, zres, transmit_bandwidth=600, gwf_180=None, optimal=True, all_crushers=False, crushers=True, only_crush_when_needed=False, qc=False):
        # Constants
        gamma = 2.6751e+08
        crushers = True
        
        excitation_amplitude = 2*np.pi/gamma * transmit_bandwidth / zres

        def check_crusher_duration(crusher_duration, crusher_area, crusher_amplitude):
            while crusher_duration < 2e-3:
                crusher_amplitude -= .1e-3
                crusher_duration = crusher_area/crusher_amplitude
            return crusher_duration, crusher_amplitude

        ### Create crusher gradients
        # We aim to create phase dispersions of 4*pi
        # Find out the crusher area for each axis
        crushing_factor = 3 # Should be at least 2
        number_of_axes_factor = 1

        crusher_area_x = crushing_factor*2*np.pi/(gamma*xres)*2/number_of_axes_factor # We double the area as we are planning to use triangular crusher pulses
        crusher_area_y = crushing_factor*2*np.pi/(gamma*yres)*2/number_of_axes_factor
        crusher_area_z = crushing_factor*2*np.pi/(gamma*zres)*2/number_of_axes_factor
        gwf_diffusion_post180_area = np.abs(np.sum(self.gwf_diffusion_post180[1:-1], axis=0))*self.dt

        if gwf_diffusion_post180_area[0] >= crusher_area_x/2 or gwf_diffusion_post180_area[1] >= crusher_area_y/2 or gwf_diffusion_post180_area[2] >= crusher_area_z/2:
            # Turn off crushers
            crushers = False

        if crushers is False:
            crusher_area_x = 0
            crusher_area_y = 0
            crusher_area_z = 0

        crusher_amplitude_x = 30e-3
        crusher_amplitude_y = 30e-3
        crusher_amplitude_z = 30e-3

        # We set the crusher amplitude to be 30 mT/m. Calculate the crusher duration
        crusher_duration_x = crusher_area_x/crusher_amplitude_x
        crusher_duration_y = crusher_area_y/crusher_amplitude_y
        crusher_duration_z = crusher_area_z/crusher_amplitude_z

        # Make sure all crushers are at least 2 ms, for some realism...
        if crushers:
            crusher_duration_x, crusher_amplitude_x = check_crusher_duration(crusher_duration_x, crusher_area_x, crusher_amplitude_x)
            crusher_duration_y, crusher_amplitude_y = check_crusher_duration(crusher_duration_y, crusher_area_y, crusher_amplitude_y)
            crusher_duration_z, crusher_amplitude_z = check_crusher_duration(crusher_duration_z, crusher_area_z, crusher_amplitude_z)

            # Create the crusher components
            crusher_x = self._create_gwf_component(crusher_duration_x, slew_rate=2*crusher_amplitude_x/crusher_duration_x, amplitude=crusher_amplitude_x, axis=0)
            crusher_y = self._create_gwf_component(crusher_duration_y, slew_rate=2*crusher_amplitude_x/crusher_duration_y, amplitude=crusher_amplitude_x, axis=1)
            crusher_z2 = self._create_gwf_component(crusher_duration_z, slew_rate=2*crusher_amplitude_z/crusher_duration_z, amplitude=crusher_amplitude_z, axis=2)
            crusher_z1 = crusher_z2
        else:
            crusher_x = self._create_gwf_component(self.dt, slew_rate=10, amplitude=1e-3, axis=0)*0
            crusher_y = self._create_gwf_component(self.dt, slew_rate=10, amplitude=1e-3, axis=1)*0
            crusher_z2 = self._create_gwf_component(self.dt, slew_rate=10, amplitude=1e-3, axis=2)*0
            crusher_z1 = crusher_z2
        if optimal:
            crusher_z1 = crusher_z2
        else:
            # Should generate a new crusher_z1 here with area adjusted for refocusing slice selection
            excitation_area = 3e-3*excitation_amplitude

            crusher_area_z1 = (crusher_area_z/2 - excitation_area)*2 # We turn the crusher area into a square again before subtracting the excitation area, and then revert to a triangle

            # Determine the crusher duration. Make sure it is at least 12 us long and does not become too long.


            crusher_amplitude_z = 30e-3
            crusher_duration_z1 = crusher_area_z1/crusher_amplitude_z
            if crushers is False:
                crusher_duration_z = 5e-3
            while crusher_duration_z1 <= 1.8e-3 or crusher_duration_z1 > crusher_duration_z:#(crusher_duration_z1 > crusher_duration_x or crusher_duration_z1 > crusher_duration_y):
                crusher_amplitude_z -= .03e-3 # We change the amplitude, as it may need to be negative to refocus both the excitation and the crusher post 180.
                crusher_duration_z1 = crusher_area_z1/crusher_amplitude_z

            crusher_z1 = self._create_gwf_component(crusher_duration_z1, slew_rate=2*np.abs(crusher_amplitude_z)/crusher_duration_z1, amplitude=crusher_amplitude_z, axis=2)

        if all_crushers:
            # Find the axis with the longest crusher in duration
            dur_longest_elements = 0
            if crusher_x.shape[0] > dur_longest_elements:
                dur_longest_elements = crusher_x.shape[0]
            if crusher_y.shape[0] > dur_longest_elements:
                dur_longest_elements = crusher_y.shape[0]
            if crusher_z2.shape[0] > dur_longest_elements:
                dur_longest_elements = crusher_z2.shape[0]
            if crusher_z1.shape[0] > dur_longest_elements:
                dur_longest_elements = crusher_z1.shape[0]

            # Now pad the crusher gradients until they are all of equal length
            if crusher_x.shape[0] < dur_longest_elements:
                crusher_x = np.concatenate((crusher_x, np.zeros((int(dur_longest_elements-crusher_x.shape[0]), 3))))
            if crusher_y.shape[0] < dur_longest_elements:
                crusher_y = np.concatenate((crusher_y, np.zeros((int(dur_longest_elements-crusher_y.shape[0]), 3))))
            if crusher_z2.shape[0] < dur_longest_elements:
                crusher_z2 = np.concatenate((crusher_z2, np.zeros((int(dur_longest_elements-crusher_z2.shape[0]), 3))))
            if crusher_z1.shape[0] < dur_longest_elements:
                crusher_z1 = np.concatenate((crusher_z1, np.zeros((int(dur_longest_elements-crusher_z1.shape[0]), 3))))

            crusher_all1 = np.vstack((np.flip(crusher_x[:,0]), np.flip(crusher_y[:,1]), np.flip(crusher_z1[:,2])))
            crusher_all1 = np.transpose(crusher_all1)

            crusher_all2 = np.vstack((crusher_x[:,0], crusher_y[:,1], crusher_z2[:,2]))
            crusher_all2 = np.transpose(crusher_all2)

        if gwf_180 is None:
            # Calculate the pulse amplitude from the bandwidth and zres
            gwf180_amplitude = 2*np.pi/gamma * transmit_bandwidth / zres

            gwf_180 = self._create_gwf_component(delta=7e-3, slew_rate=1e6, amplitude=gwf180_amplitude, axis=2)

            # Now decide on which axes we add crushers on
            #if crushers:
            if all_crushers:
                gwf_180 = np.concatenate((crusher_all1[:-1], gwf_180[1:-1], crusher_all2[1:]))
            else:
                gwf_180 = np.concatenate((crusher_z1[:-1], gwf_180[1:-1], crusher_z2[1:]))

        # Assign the gwfs to the attributes
        self.gwf_180 = gwf_180

        #fig, ax = plt.subplots()
        #ax.plot(self.gwf_180)
        #print(self.gwf_180.shape)

        # Some necessary timing info
        #self.t_excitation = 0
        #self.t = np.linspace(0, 150e-3, int(150e-3/self.dt))

        self.update_gwf_array()

    def _create_gwf_component(self, delta, slew_rate, amplitude, axis):
        """
        Creates trapezoids in specified direction (x, y, or z). Given by axis = 0, 1, or 2.
        """

        gradient_trap = trapezoid(delta=delta, slew_rate=slew_rate, amplitude=amplitude, gradient_update_rate=self.dt, return_time_axis=False)
        gradient_zeros = np.zeros(gradient_trap.shape)

        if axis == 0:
            gwf_component = np.vstack((gradient_trap, gradient_zeros, gradient_zeros))
        elif axis == 1:
            gwf_component = np.vstack((gradient_zeros, gradient_trap, gradient_zeros))
        elif axis == 2:
            gwf_component = np.vstack((gradient_zeros, gradient_zeros, gradient_trap))

        gwf_component = np.transpose(gwf_component)

        return gwf_component

    def excitation_create_rewinder(self, gwf=None, start_time=None, stop_time=None, delta=2e-3, print_results=False, update_attribute=True):
        """
        Creates a rewinding pulse for the assigned start and stop times.
        """
        if gwf is None:
            gwf = self.gwf_excitation

        result = minimize(self._excitation_trapezoid_objective_function, x0=np.array([-10e-3]), args=(gwf, start_time, stop_time, delta), method="Nelder-Mead", options={"maxiter":10000})
        amplitude = result.x[0]
        fun_value = result.fun

        if print_results:
            print(f"Rewinder amplitude: {amplitude}")
            print(f"Optimization objective function value: {fun_value}")

        rewinder = trapezoid(delta=delta, slew_rate=50e6, amplitude=amplitude, gradient_update_rate=self.dt, return_time_axis=False)

        rewinder_xy = np.zeros(rewinder.shape[0])

        rewinder_xyz = np.zeros((rewinder.shape[0], 3))
        rewinder_xyz[:,0] = rewinder_xy
        rewinder_xyz[:,1] = rewinder_xy
        rewinder_xyz[:,2] = rewinder

        # Set the new self.gwf_excitation and update self.gwf and self.t, and all other timing info
        gwf_excitation = np.concatenate((gwf, rewinder_xyz))

        if update_attribute:
            self.gwf_excitation = gwf_excitation

            self.update_gwf_array()

        else:
            return gwf_excitation

        #self.gwf = np.concatenate((self.gwf_excitation, self.gwf_diffusion_pre180, self.gwf_180, self.gwf_diffusion_post180, self.gwf_readout))
        #self.t = np.linspace(self.t_excitation, len(self.gwf)*self.dt+self.t_excitation, len(self.gwf))
        #self._separate_gwf_axes()

        ## Update the timing with the rewinder width, which is the delta variable in this function
        #self.w_excitation += delta
        #self.t_diffusion_pre180 += delta
        #self.t_180 += delta
        #self.t_diffusion_post180 += delta
        #self.t_readout += delta

    def _excitation_trapezoid_objective_function(self, amplitude, gwf=None, start_time=None, stop_time=None, delta=1e-3):
        # Create trapezoid
        # Free variable should be amplitude as the duration has fixed steps
        trap = trapezoid(delta=delta, slew_rate=50e6, amplitude=amplitude, gradient_update_rate=self.dt, return_time_axis=False)

        start, stop = self._timing_info(start_time, stop_time)

        # Calculate q
        gamma = 2.6751e+08
        q_trapezoid = gamma*np.sum(trap)*self.dt

        # Calculate q we want to refocus
        q_sequence = gamma*np.sum(gwf[start:])*self.dt

        #rf = np.ones(gwf.shape)
        #q_sequence = self.get_q(gwf=gwf, rf=rf, start_time=start_time, stop_time=stop_time, update_attribute=False)[-1,2]

        return np.abs(q_sequence + q_trapezoid)

    def gwf180_mirror_slice_selection(self, start_time=None, stop_time=None):
        """
        Mirrors the slice selection using the 2nd half of the slice selection gradient
        """
        if hasattr(self, "residual") == False:
            self.residual = 0

        # Get the 2nd half of the slice selection gradient, flip it, and concatenate the two components
        start, stop = self._timing_info(self.mid_180, self.mid_180+10e-3)

        second_half_of_gwf180 = self.gwf_180[round(self.mid_180/self.dt):, :]
        new_first_half_of_gwf180 = np.flip(second_half_of_gwf180, axis=0)
        self.gwf_180 = np.concatenate((new_first_half_of_gwf180, second_half_of_gwf180), axis=0)

        # And now the timing info...
        new_length = len(self.gwf_180)*self.dt
        diff = new_length - self.w_180
        self.w_180 = new_length
        self.t_diffusion_post180 += diff
        self.t_readout += diff
        self.mid_180 = len(self.gwf_180)*self.dt/2 + self.residual

        # Update the gwf and t arrays
        self.gwf = np.concatenate((self.gwf_excitation, self.gwf_diffusion_pre180, self.gwf_180, self.gwf_diffusion_post180, self.gwf_readout))
        self.t = np.linspace(self.t_excitation, len(self.gwf)*self.dt+self.t_excitation, len(self.gwf))
        self._separate_gwf_axes()

    def get_b_separate(self, include_imaging=False, include_cross_terms=False):
        B_diff = 0
        B_im = 0
        B_ct = 0

        encoding_array_list = [self.gwf_diffusion_pre180, self.gwf_diffusion_post180]

        # Create an array containing the two diffusion encoding lobes, with a zero-array with the same length as the 180-pulse gradients
        zero180_array = np.zeros(self.gwf_180.shape)
        encoding_array = np.concatenate((self.gwf_diffusion_pre180, zero180_array, self.gwf_diffusion_post180))

        # create corresponding rf array
        rf_start, rf_stop = self._timing_info(start_time=self.t_diffusion_pre180, stop_time=self.t_diffusion_pre180+(len(encoding_array)-1)*self.dt)
        encoding_rf = self.rf[rf_start:rf_stop+2]

        diff, im = self.get_diff_and_im_components()
        rf = self.rf

        B_diff = self.get_b(gwf=diff, rf=rf, update_attribute=False, return_b_tensor=True, start_time=self.start_time, stop_time=self.TE)

        if include_imaging:
            B_im = self.get_b(gwf=im, rf=rf, update_attribute=False, return_b_tensor=True, start_time=self.start_time, stop_time=self.TE)
        if include_cross_terms:
            q_diff = self.get_q(gwf=diff, rf=rf, update_attribute=False, start_time=self.start_time, stop_time=self.TE)

            # Get q for imaging gradients
            q_im = self.get_q(gwf=im, rf=rf, update_attribute=False, start_time=self.start_time, stop_time=self.TE)

            # Get the two cross term components
            cross_term_1 = np.matmul(q_im.T, q_diff)*self.dt
            cross_term_2 = np.matmul(q_diff.T, q_im)*self.dt

            # Sum to get B_ct
            B_ct = cross_term_1 + cross_term_2

        B = B_diff + B_im + B_ct
        b = np.trace(B)

        return b


    def set_b_by_scaling_amplitude(self, b_desired, print_results=False, include_imaging=False, include_cross_terms=False):
        """
            Uses an optimizer to scale the amplitude to achieve a certain bvalue.

            b_desired = float. The desired b-value in SI units. E.g. [s/mm2]*1e6, [ms/um2]*1e9

            When inclding imaging or cross terms, the b-value is only calculated up to the end of the diffusion encoding. Readout is not included.
        """

        encoding_array_list = [self.gwf_diffusion_pre180, self.gwf_diffusion_post180]

        # Create an array containing the two diffusion encoding lobes, with a zero-array with the same length as the 180-pulse gradients
        zero180_array = np.zeros(self.gwf_180.shape)
        encoding_array = np.concatenate((self.gwf_diffusion_pre180, zero180_array, self.gwf_diffusion_post180))

        # create corresponding rf array
        rf_start, rf_stop = self._timing_info(start_time=self.t_diffusion_pre180, stop_time=self.t_diffusion_pre180+(len(encoding_array)-1)*self.dt)
        encoding_rf = self.rf[rf_start:rf_stop+2]

        if print_results:
            print(f"Starting b of encoding: {self.get_b(gwf=encoding_array, rf=encoding_rf)*1e-6:.1f}")
            print(f"Starting amplitude: {encoding_array[1000]}")

        results = minimize(self._set_b_by_scaling_amplitude_objective_function, x0=np.array([1]), args=(b_desired, encoding_array_list, include_imaging, include_cross_terms), method="Nelder-Mead")

        # Update the encoding gwf by scaling with the optimized factor
        if results.x < 0:
            factor = -1
        else:
            factor = 1
        self.gwf_diffusion_pre180 *= results.x*factor
        self.gwf_diffusion_post180 *= results.x*factor

        # Update the main gwf array
        self.update_gwf_array()

        ## Plotting for debugging
        #fig, ax = plt.subplots()
        #ax.plot(encoding_array*encoding_rf)
        #ax.plot(encoding_rf*0.02, ls="--")
        #ax.plot(self.gwf[rf_start:rf_stop+2]*encoding_rf)
        #self.plot_gwf()

        #analytical_b = ((2.6751e8)**2 * (self.gwf_diffusion_pre180[1000])**2 * (self.t_readout - self.t_diffusion_post180 )**2 *((self.t_diffusion_post180-self.t_diffusion_pre180)-(self.t_readout - self.t_diffusion_post180)/3))*1e-6

        #print_results=True
        if print_results:
            print(f"Scale factor: {results.x}")
            print(f"Objective function value: {results.fun}")
            print(f"b-value of new encoding: {self.get_b(gwf=encoding_array*results.x, rf=encoding_rf)*1e-6:.1f} s/mm2")
            #print(f"Analytical b of new encoding: {analytical_b} s/mm2")
            print(f"End amplitude: {self.gwf_diffusion_pre180[1000]}\n")
    
    def _set_b_by_scaling_amplitude_objective_function(self, factor, b_desired=0, encoding_array_list=None, include_imaging=False, include_cross_terms=False):
        """
        Objective function for set_b_by_scaling_amplitude. Checks the absolute difference between the desired and the optimized b-value.
        """
        B_diff = 0
        B_im = 0
        B_ct = 0

        if factor < 0:
            factor *= -1

        # Gör om funktionen så den summerar b_diff + b_im + b_c
        # Input ska endast vara diff-gradienter. RF och imaging beräknas senare.

        # Så 1) Beräkna alltid b_diff
        # Om imclude_imaging=True, beräkna b_im och sätt b_optimized = b_diff + b_im
        # Om include_cross_terms=True, beräkna b_c och addera det till b_optimized

        test_diff, test_im = self.get_diff_and_im_components()
        test_rf = self.rf

        ### Calculate B_diff
        # Construct the gwf
        gwf_diff = np.concatenate((encoding_array_list[0]*factor, np.zeros(self.gwf_180.shape), encoding_array_list[1]*factor))

        # Construct the rf
        #rf_diff = np.ones(gwf_diff.shape)
        #rf_diff[encoding_array_list[0].shape[0]-1+int(self.gwf_180.shape[0]/2):] = -1
        rf_start, rf_stop = self._timing_info(start_time=self.t_diffusion_pre180, stop_time=self.t_diffusion_post180)
        rf_diff = self.rf[rf_start:rf_stop]
        rf_diff = np.concatenate((rf_diff, np.ones((gwf_diff.shape[0]-rf_diff.shape[0], 3))*-1))

        # Get B_diff
        #B_diff = self.get_b(gwf=gwf_diff, rf=rf_diff, update_attribute=False, return_b_tensor=True)
        B_diff = self.get_b(gwf=test_diff*factor, rf=test_rf, update_attribute=False, return_b_tensor=True, start_time=self.start_time, stop_time=self.TE)

        if include_imaging:
            ### Calculate B_im
            # Construct the gwf
            gwf_im = np.concatenate((self.gwf_excitation, np.zeros(encoding_array_list[0].shape), self.gwf_180, np.zeros(encoding_array_list[1].shape)))
            gwf_im = gwf_im[int((self.start_time-self.t_excitation)/self.dt):]

            # Construct the rf
            rf_start, rf_stop = self._timing_info(start_time=self.t_excitation, stop_time=self.t_diffusion_post180)
            rf_im = self.rf[rf_start:rf_stop]
            rf_im = rf_im[int((self.start_time-self.t_excitation)/self.dt):]
            rf_im = np.concatenate((rf_im, np.ones((gwf_im.shape[0]-rf_im.shape[0], 3))*-1))

            ### Debugging
            #q = self.get_q(gwf=gwf_im, rf=rf_im)
            #fig, ax = plt.subplots()
            #ax.plot(q)

            # Get B_im
            #B_im = self.get_b(gwf=gwf_im, rf=rf_im, update_attribute=False, return_b_tensor=True)
            B_im = self.get_b(gwf=test_im, rf=test_rf, update_attribute=False, return_b_tensor=True, start_time=self.start_time, stop_time=self.TE)
        
        if include_cross_terms:
            # Get q for diffusion gradients
            gwf_diff2 = np.concatenate((np.zeros(self.gwf_excitation.shape), encoding_array_list[0]*factor, np.zeros(self.gwf_180.shape), encoding_array_list[1]*factor))
            gwf_diff2 = gwf_diff2[int((self.start_time-self.t_excitation)/self.dt):]

            #q_diff = self.get_q(gwf=gwf_diff2, rf=rf_im, update_attribute=False)
            q_diff = self.get_q(gwf=test_diff*factor, rf=test_rf, update_attribute=False, start_time=self.start_time, stop_time=self.TE)

            # Get q for imaging gradients
            #q_im = self.get_q(gwf=gwf_im, rf=rf_im, update_attribute=False)
            q_im = self.get_q(gwf=test_im, rf=test_rf, update_attribute=False, start_time=self.start_time, stop_time=self.TE)

            # Get the two cross term components
            cross_term_1 = np.matmul(q_im.T, q_diff)*self.dt
            cross_term_2 = np.matmul(q_diff.T, q_im)*self.dt

            # Sum to get B_ct
            B_ct = cross_term_1 + cross_term_2

            #B_ct = np.array((B_ct[0,0], B_ct[1,1], B_ct[2,2], B_ct[0,1]*np.sqrt(2), B_ct[0,2]*np.sqrt(2), B_ct[1,2]*np.sqrt(2)))
            #fig, ax = plt.subplots()
            #ax.plot(q_im/np.max(q_im))
            #ax.plot(q_diff/np.max(q_diff))
            #ax.plot(gwf_im)
            #ax.plot(gwf_diff2)



        B_optimized = B_diff + B_im + B_ct
        b_optimized = np.trace(B_optimized)
        #b_optimized = np.sum(B_optimized[0:3])

        #print(np.trace(B_diff)*1e-6, np.trace(B_im)*1e-6, np.trace(B_ct)*1e-6)

        return np.abs(b_desired - b_optimized)
    
    def rotate_encoding(self, angle_x=0, angle_y=0, angle_z=0, inverse=False):
        """
        Rotates the diffusion encoding. The anglex_<axis> denote the rotation angles in radians around the speicified rotation axis.
        """

        # rotation vectors
        rx = Rotation.from_rotvec(angle_x * np.array([1, 0, 0]))
        ry = Rotation.from_rotvec(angle_y * np.array([0, 1, 0]))
        rz = Rotation.from_rotvec(angle_z * np.array([0, 0, 1]))

        # Apply rotation to diffusion pre and post 180-pulse
        if inverse:
            self.gwf_diffusion_pre180 = rz.apply(self.gwf_diffusion_pre180, inverse=True)
            self.gwf_diffusion_pre180 = ry.apply(self.gwf_diffusion_pre180, inverse=True)
            self.gwf_diffusion_pre180 = rx.apply(self.gwf_diffusion_pre180, inverse=True)

            self.gwf_diffusion_post180 = rz.apply(self.gwf_diffusion_post180, inverse=True)
            self.gwf_diffusion_post180 = ry.apply(self.gwf_diffusion_post180, inverse=True)
            self.gwf_diffusion_post180 = rx.apply(self.gwf_diffusion_post180, inverse=True)
        else:
            self.gwf_diffusion_pre180 = rx.apply(self.gwf_diffusion_pre180)
            self.gwf_diffusion_pre180 = ry.apply(self.gwf_diffusion_pre180)
            self.gwf_diffusion_pre180 = rz.apply(self.gwf_diffusion_pre180)

            self.gwf_diffusion_post180 = rx.apply(self.gwf_diffusion_post180)
            self.gwf_diffusion_post180 = ry.apply(self.gwf_diffusion_post180)
            self.gwf_diffusion_post180 = rz.apply(self.gwf_diffusion_post180)

        # Update the gwf array
        self.update_gwf_array()

    def rotate_encoding_with_uvec(self, uvec, rescale_b=False):
        """
        Rotates a [1,1,1] diffusion encoding to specified direction with unit vectors. Input is an array containing the unit vector.
        """
        # Get the current b-value
        if rescale_b:
            current_b = self.get_b()

        # Re-scale the diffusion encoding with the uvec to change the direction
        self.gwf_diffusion_pre180 *= uvec
        self.gwf_diffusion_post180 *= uvec

        if rescale_b:
            self.set_b_by_scaling_amplitude(current_b)
        
        # Update gwf array
        self.update_gwf_array()

    def update_gwf_array(self):
        # Update the timing info
        # excitation
        self.w_excitation = np.around((len(self.gwf_excitation)-1)*self.dt, 7)
        #self.w_excitation = self.t_diffusion_pre180 - self.t_excitation
        # Diffusion pre 180
        self.t_diffusion_pre180 = np.around(self.t_excitation + self.w_excitation, 7)
        self.w_diffusion_pre180 = np.around((len(self.gwf_diffusion_pre180)-1)*self.dt, 7)
        # 180
        self.t_180 = np.around(self.t_diffusion_pre180 + self.w_diffusion_pre180, 7)
        self.w_180 = np.around((len(self.gwf_180)-1)*self.dt, 7)
        # Diffusion post 180
        self.t_diffusion_post180 = np.around(self.t_180 + self.w_180, 7)
        self.w_diffusion_post180 = np.around((len(self.gwf_diffusion_post180)-1)*self.dt, 7)
        # Readout
        self.t_readout = np.around(self.t_diffusion_post180 + self.w_diffusion_post180, 7)
        self.w_readout = np.around((len(self.gwf_readout)-1)*self.dt, 7)

        # Get indices from timing info
        start_excitation, stop_excitation = self._timing_info(self.t_excitation, self.t_excitation + self.w_excitation)
        start_diffusion_pre180, stop_diffusion_pre180 = self._timing_info(self.t_diffusion_pre180, self.t_diffusion_pre180 + self.w_diffusion_pre180)
        start_180, stop_180 = self._timing_info(self.t_180, self.t_180 + self.w_180)
        start_diffusion_post180, stop_diffusion_post180 = self._timing_info(self.t_diffusion_post180, self.t_diffusion_post180 + self.w_diffusion_post180)
        start_readout, stop_readout = self._timing_info(self.t_readout, self.t_readout + self.w_readout)

        #print(f"Update gwf array: {stop_excitation} {start_diffusion_pre180} {stop_diffusion_pre180} {start_180} {stop_180} {start_diffusion_post180} {stop_diffusion_post180}, {start_readout}")

        gwf_stitched = np.zeros((self.t.shape[0],3))

        gwf_stitched[start_excitation:stop_excitation+2, :] = self.gwf_excitation
        gwf_stitched[start_diffusion_pre180:stop_diffusion_pre180+2, :] = self.gwf_diffusion_pre180
        gwf_stitched[start_180:stop_180+2, :] = self.gwf_180
        gwf_stitched[start_diffusion_post180:stop_diffusion_post180+2, :] = self.gwf_diffusion_post180
        gwf_stitched[start_readout:stop_readout+2, :] = self.gwf_readout

        self.gwf = gwf_stitched
        self.t = np.linspace(self.t_excitation, len(self.gwf)*self.dt+self.t_excitation, len(self.gwf))
        self.t = np.linspace(0, len(self.gwf)*self.dt, len(self.gwf))

        # Update the separated axes
        self._separate_gwf_axes()

    def get_diff_and_im_components(self):
        # Get indices from timing info
        start_excitation, stop_excitation = self._timing_info(self.t_excitation, self.t_excitation + self.w_excitation)
        start_diffusion_pre180, stop_diffusion_pre180 = self._timing_info(self.t_diffusion_pre180, self.t_diffusion_pre180 + self.w_diffusion_pre180)
        start_180, stop_180 = self._timing_info(self.t_180, self.t_180 + self.w_180)
        start_diffusion_post180, stop_diffusion_post180 = self._timing_info(self.t_diffusion_post180, self.t_diffusion_post180 + self.w_diffusion_post180)
        start_readout, stop_readout = self._timing_info(self.t_readout, self.t_readout + self.w_readout)

        # Imaging gradients
        im_stitched = np.zeros(self.gwf.shape)
        im_stitched[start_excitation:stop_excitation+2, :] = self.gwf_excitation
        im_stitched[start_diffusion_pre180:stop_diffusion_pre180+2, :] = self.gwf_diffusion_pre180*0
        im_stitched[start_180:stop_180+2, :] = self.gwf_180
        im_stitched[start_diffusion_post180:stop_diffusion_post180+2, :] = self.gwf_diffusion_post180*0
        im_stitched[start_readout:stop_readout+2, :] = self.gwf_readout

        # Diffusion gradients
        diff_stitched = np.zeros(self.gwf.shape)
        diff_stitched[start_excitation:stop_excitation+2, :] = self.gwf_excitation*0
        diff_stitched[start_diffusion_pre180:stop_diffusion_pre180+2, :] = self.gwf_diffusion_pre180
        diff_stitched[start_180:stop_180+2, :] = self.gwf_180*0
        diff_stitched[start_diffusion_post180:stop_diffusion_post180+2, :] = self.gwf_diffusion_post180
        diff_stitched[start_readout:stop_readout+2, :] = self.gwf_readout*0

        return diff_stitched, im_stitched

    def _get_indices(self, time, width):
        """
        Gets the gwf array indices for the given pulse time and width.
        """
        indices = np.where(self.t >= time and self.t < time + width)[0]

        return indices







def trapezoid(delta, slew_rate, amplitude=None, gradient_update_rate=4e-6, return_time_axis=False):
    gradient_update_rate = gradient_update_rate

    # Create the array
    gradient = np.zeros((int(np.round(delta/gradient_update_rate, 0))))
    t = np.linspace(0, delta, len(gradient))
    
    # Ramp up from 0 to 1 and mirror it for ramp-down
    slew_rate_normalized = slew_rate/np.abs(amplitude)
    for i in range(len(gradient)):
        if t[i]*slew_rate_normalized < 1:
        
            gradient[i] = t[i]*slew_rate_normalized
            gradient[-(i+1)] = gradient[i]
            
    # Fill constant portion
    gradient[gradient == 0] = 1
    gradient[0] = 0
    gradient[-1] = 0
    
    if amplitude:
        # Re-scale to amplitude
        gradient *= amplitude

    if return_time_axis:
        return gradient, t
    else:
        return gradient
    
def double_diffusion_encoding(trapezoid, gradient_update_rate=4e-6):
    trapezoid_neg = -trapezoid[1:] # Remove first index as it should be shared with the positiva waveform
    dde_fc = np.concatenate((trapezoid, trapezoid_neg))
    t = np.linspace(0, len(dde_fc)*gradient_update_rate, len(dde_fc))
    return t, dde_fc

def build_encoding_pre_and_post_180(gwf_pre, gwf_post, t180, sinc=False, gradient_update_rate=4e-6, return_time_axis=False):
    # Create the gwf for the 180-pulse. For now it is just zeros
    gwf_180 = np.zeros((int(t180/gradient_update_rate)))

    # Concatenate the arrays
    gradient = np.concatenate((gwf_pre, gwf_180, gwf_post))

    if return_time_axis:
        t = np.linspace(0, len(gradient)*gradient_update_rate, len(gradient))
        return gradient, t
    else:
        return gradient
    
def generate_sinc(t180, amplitude, t_start=0, gradient_update_rate=4e-6, return_time_axis=False):
    # Generate empty array
    rf180 = np.zeros((int(t180/gradient_update_rate)))

    # Generate the sinc
    x = np.linspace(-4,4,int(t180/gradient_update_rate))
    rf180 = np.sinc(x)*amplitude


    if return_time_axis:
        t = np.linspace(0, t_start+t180, int((t_start+t180)/gradient_update_rate))

        # move sinc to the correct position
        new_rf180 = np.zeros(len(t))
        new_rf180[t>=t_start] = rf180
        return new_rf180, t
    else:
        return rf180
    
def pad_arrays(t_extension, gwf, t, before=False, after=False, gradient_update_rate=4e-6):
    # Create the extension array
    extension = np.zeros((int(t_extension/gradient_update_rate)))

    # Append the extension to the gwf
    if before:
        gwf = np.concatenate((extension, gwf))
    
    if after:
        gwf = np.concatenate((gwf, extension))

    # Create new t array
    t = np.linspace(0, len(gwf)*gradient_update_rate, len(gwf))

    return gwf, t

def encoding_sequence_fc(t_grad, t_pre180, t_post180, gradient, gradient_update_rate=4e-6):
    t_180 = (t_pre180+t_post180)
    grad_180 = np.zeros((int(np.round(t_180/gradient_update_rate, 0))))
    
    encoding_fc = np.concatenate((gradient, grad_180, gradient))
    encoding_nc = np.concatenate((gradient, grad_180, -gradient))
    
    t = np.linspace(0, t_grad+t_pre180+t_post180, len(encoding_fc))
    return t, encoding_fc, encoding_nc
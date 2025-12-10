import numpy as np
from nbodykit.lab import *
import matplotlib.pyplot as plt
from matplotlib import colors, cm
import matplotlib
import os
import time
import pandas as pd
from math import comb
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
from sklearn.gaussian_process import GaussianProcessRegressor
from Pk_tools import Fourier, smooth_field, get_Pk
import jax
import jax.numpy as jnp
# again, this only works on startup!
from jax import config
config.update("jax_enable_x64", True)
config.update("jax_enable_x64", True)

from numpy import sin as sin
from numpy import cos as cos
from numpy import arcsin as asin
from numpy import arccos as acos
ngrid=256


class Mark(object):
    def __init__(self, angles, nodes=None, kmax=0.3, David_fom=False, real_cov=None, kmin=0.01, fom_type='both', set_delta_range=False, delta_range=[-1.0, 5.0], lbox=700., ngrid=256,
                 n_nodes=4, use_white_mark=False, w_thr=1E-7, R=10, set_length_scale=False, length_scale=0.75, prefix='', kmax_bin=None, white_param=None,
                 sim_data_dir='/mnt/extraspace/jesscowell/MarkedCorr/Data/Sim_arrays/'):
        self.angles_array = angles
        self.sim_data_dir = sim_data_dir  # Directory for simulation data
        self.use_white = use_white_mark
        self.david_fom = David_fom
        self.white_param = white_param
        if nodes is None:
            self.nodes = self.convert_from_angle(self.angles_array)
        else:
            self.nodes = nodes
        self.real_cov = real_cov
        self.set_delta_range = set_delta_range
        self.delta_range = delta_range
        self.w_thr = w_thr
        self.smoothing_scale = R
        self.kmin = kmin
        self.kmax = kmax
        self.kmax_bin = kmax_bin
        self.fom = fom_type
        self.dpar = {'Om': 0.02, 's8': 0.02}
        self.sims = ["fid", "Om_m", "Om_p", "s8_m", "s8_p"]
        self.mark_names = ['m1']
        self.length_scale = length_scale
        self.set_length_scale = set_length_scale
        if self.set_length_scale == False:
            self.kernel = 20 * RBF(length_scale=self.length_scale)
        elif self.set_length_scale == True:
            self.kernel = 20 * RBF(length_scale=self.length_scale, length_scale_bounds='fixed')
        self.delta_s8 = 0.02
        self.delta_Om = 0.02
        self.lbox = lbox
        self.ngrid = ngrid
        # Data and analysis attributes are not loaded at init
        self.ktot = None
        self.good_k = None
        self.Pks = None
        self.fields = None
        self.k_fields = None
        self.smoothed_fields = None
        self.nmodes_pk = None
        self.Pk_fisher = None
        self.Pk_cov = None
        self.names = None
        self.mark_dict = None

    def load_data(self):
        """Load simulation data and run analysis. Call this when you need the data."""
        print("Loading simulation data...")
        self.ktot, self.good_k, self.Pks, self.fields, self.k_fields, self.smoothed_fields, self.nmodes_pk = self.load_sims(R=self.smoothing_scale)
        self.k_fields['fft_d_fiducial'] = self.k_fields['fiducial']
        self.Pk_fisher, self.Pk_cov = self.get_Pk_only_fisher()
        self.names = ['fiducial', 'Om_m', 'Om_p', 's8_m', 's8_p']
        self.mark_dict = self.get_mark_from_nodes()
        
    def white_mark(self, p, b, delta_R):
        '''calculates the White Mark function with smoothing scale of R MPc
        Parameters:
        p
        b
        delta_R : smoothed field
        R: smoothing scale, default 10Mpc
        '''
        m = (1 + (delta_R)/(1+b))**(-p)
        return(m)

    def update_angles(self, angles):
        ''''''
        '''change mark angle values'''
        self.angles_array=angles
        self.nodes = self.convert_from_angle(self.angles_array)
        
    def update_nodes(self, nodes):
        '''change node values'''
        self.nodes = nodes
        
    def plot_mark(self):
        '''plot mark function'''
        mark_nodes= self.nodes
        self.ensure_data_loaded()


        self.marks_dict=self.get_mark_from_nodes()
        delta_range =self.delta_range
        delta_R_train = self.delta_R_train
        kernel=self.kernel
        gpr = GaussianProcessRegressor(kernel=kernel, random_state=1)
        gpr.fit(delta_R_train, mark_nodes)
        
        delta_plotting = np.linspace(np.min(delta_range), np.max(delta_range) ,1000)
        mark_fid,_ = gpr.predict(delta_plotting.reshape(-1, 1) ,return_std=True)
        plt.plot(delta_plotting, mark_fid, ls='-' ,color='cornflowerblue',  linewidth=5)
        plt.plot(delta_R_train, mark_nodes, ls='' ,marker='.',color='red', markersize=20, linewidth=5)
        plt.xlabel('$\delta_R$')
        plt.ylim(-1,1)
        plt.ylabel('M($\delta_R$)')
        plt.title(f'fomtype = {self.fom}, R={self.smoothing_scale}')
    def smooth_field(self, k, field, R):
        '''
        Smooth a field by a given radius R(Mpc).
        Field supplied should be in k space

        '''
        W =  jnp.exp(-0.5*(k*R)**2)
        smoothed_field = (W* field)
        return smoothed_field
    
    def ensure_data_loaded(self):
        """Automatically load data if not already loaded."""
        if self.smoothed_fields is None:
            self.load_data()

    def get_error_improvement(self):
        self.ensure_data_loaded()
        self.marks_dict=self.get_mark_from_nodes()
        self.fom =self.get_fom(self.mark_names, self.fom)
        og_fish = self.get_Pk_only_fisher()[0]
        og_error = np.sqrt(self.my_pinv(og_fish))
        self.og_error=og_error
        print('Pk only error', og_error)
        print('MPk error', self.error)
        print('improvement factors in $S_8$ and $\Omega_m$ respectively:', np.diag(og_error/self.error).round(2), )
        return np.diag(og_error/self.error)
        
    def Fourier(self, field, Nmesh = 256, lbox=700, inverse=False):
        '''Return FFT of field
        Parameters:
        field - the 3D field array 
        lbox - Length of simulation in Mpc/h
        Nmesh - size of mesh
        kmin - min wave number to calculate Pk at
        nbins - number of k bins to use
        
        Returns:
        k-centers - grid of k values
        k-field - FFT of real field  
        '''
        # cell width
        d = lbox / Nmesh
        # Take the Fourier Transform of the 3D box
        if inverse:
            complex_field = jnp.fft.irfftn(field, )
        else:
            complex_field = jnp.fft.rfftn(field, )
        # natural wavemodes 
        kx = ky = jnp.fft.fftfreq(Nmesh, d=d)* 2. * np.pi # h/Mpc
        kz = jnp.fft.rfftfreq(Nmesh, d=d)* 2. * np.pi # h/Mpc
        nx, ny, nz = complex_field.shape #shape of ft is symmetric 
        # Compute the total wave number
        ktot = jnp.sqrt(kx[:,None, None]**2 + ky[None, :, None]**2+kz[None,None,:]**2)[:nx, :ny, :nz]
        if np.isnan(complex_field).any():
            print('fourier transform is nan!')
            quit()
        return ktot, complex_field
    def load_sims(self, R=10):
        """
        Loads and processes simulation data for a given smoothing scale R (Mpc).
        Steps:
        1. Loads raw density fields from disk
        2. Computes overdensity fields
        3. Applies Fourier transform
        4. Smooths fields in k-space and real space
        5. Calculates power spectra and applies scale cuts
        Returns:
            ktot: k-space grid
            good_k: mask for valid k values
            Pks: dictionary of power spectra
            fields: overdensity fields
            k_fields: Fourier-transformed fields
            smoothed_fields: smoothed overdensity fields
            nmodes_pk: number of k-modes per Pk
        """
        names = ['fiducial', 'Om_m', 'Om_p', 's8_m', 's8_p']
        seed = '101'
        snapnum = '05'
        Nmesh = 256
        kmax = self.kmax

        # Step 1: Load raw density fields
        rho_fields = self._load_density_fields(names, snapnum)

        # Step 2: Compute overdensity fields
        fields = self._compute_overdensity_fields(rho_fields)

        # Step 3: Fourier transform
        ktot, k_fields = self._fourier_transform_fields(fields, Nmesh)

        # Step 4: Smooth fields in k-space and real space
        smoothed_kfields, smoothed_fields = self._smooth_fields(k_fields, ktot, R)

        # Step 5: Compute power spectra and apply scale cuts
        good_k, Pks, nmodes_pk = self._compute_power_spectra(k_fields, ktot, kmax, self.kmax_bin)

        return ktot, good_k, Pks, fields, k_fields, smoothed_fields, nmodes_pk

    def _load_density_fields(self, names, snapnum):
        """Loads raw density fields from disk using the class's sim_data_dir."""
        rho_fields = {}
        for name in names:
            path = os.path.join(self.sim_data_dir, f'Snap_{snapnum}/{name}_256.npy')
            rho_fields[name] = np.load(path)
        return rho_fields

    def _compute_overdensity_fields(self, rho_fields):
        """Computes overdensity fields from raw density fields."""
        fields = {}
        for name, rho in rho_fields.items():
            fields[name] = rho / np.mean(rho) - 1
        return fields

    def _fourier_transform_fields(self, fields, Nmesh):
        """Applies Fourier transform to overdensity fields."""
        k_fields = {}
        ktot = None
        for name, field in fields.items():
            ktot, k_fields[name] = self.Fourier(field, Nmesh=Nmesh)
            k_fields[f'fft_d_{name}'] = k_fields[name]
        return ktot, k_fields

    def _smooth_fields(self, k_fields, ktot, R):
        """Smooths fields in k-space and real space."""
        smoothed_kfields = {}
        smoothed_fields = {}
        for name, k_field in k_fields.items():
            smoothed_kfields[name] = self.smooth_field(ktot, k_field, R)
            _, smoothed_fields[name] = self.Fourier(smoothed_kfields[name], inverse=True)
        return smoothed_kfields, smoothed_fields

    def _compute_power_spectra(self, k_fields, ktot, kmax, kmax_bin):
        """Calculates power spectra and applies scale cuts."""
        Pks = {}
        nmodes_pk = None
        good_k = None
        for name, k_field in k_fields.items():
            k, nmodes, pk = get_Pk(k_field, ktot, kmax=kmax_bin)
            mask = (0.01 < k) & (k < kmax)
            k = k[mask]
            Pks[name] = pk[mask]
            nmodes = nmodes[mask]
            # Save good_k and nmodes_pk from the first field (assume all same shape)
            if good_k is None:
                good_k = mask
                nmodes_pk = nmodes
                self.ks = k
        return good_k, Pks, nmodes_pk
    
    def get_Pk_only_fisher(self,):
        '''Fisher matrix from Pk only (NO MARKED PK)'''
        Pks=self.Pks
        deriv_s8 = (Pks['s8_p'] - Pks['s8_m'])/(2*self.delta_s8)
        self.deriv_s8_dd = deriv_s8
        deriv_Om = (Pks['Om_p'] - Pks['Om_m'])/(2*self.delta_Om)
        self.deriv_Om_dd = deriv_Om
        
        cov = np.diag(2*(Pks['fiducial']**2)/self.nmodes_pk)
        icov = self.my_pinv(cov)
        fisher_cov_so = jnp.dot(deriv_s8.T,np.dot( icov, deriv_Om))
        fisher_cov_os = jnp.dot(deriv_Om.T,np.dot( icov, deriv_s8))
        fisher_cov_ss = jnp.dot(deriv_s8.T,np.dot( icov, deriv_s8))
        fisher_cov_oo = jnp.dot(deriv_Om.T,np.dot( icov, deriv_Om))
        
        fisher_cov = np.array([[fisher_cov_ss, fisher_cov_so],
                            [fisher_cov_os, fisher_cov_oo]])
        
        return(fisher_cov, cov)
    
    def finite_diff_pair(self,):
        ''''ix, f1, f2, obs, for derivative calculation'''
        ipk = 0
        map_names =  ['d']+ self.mark_names #this is just m1 for now
        names = ['Om', 's8']
        for i1, n1 in enumerate(map_names):
            for n2 in map_names[i1:]: 
                for n3 in names:
                    yield ipk, n1, n2, n3
                    ipk += 1

    
    def all_field_sim_combinations(self, sim_names):
            '''
            Generates all possible (field1, field2, simulation) combinations for power spectrum calculations.
            field1, field2: from ['d'] + self.mark_names
            sim_names: list of simulation names (e.g., ['fiducial', 'Om_m', ...])
            Yields: (ipk, field1, field2, sim_name)
            '''
            ipk = 0
            map_names = ['d'] + self.mark_names
            for i1, field1 in enumerate(map_names):
                for field2 in map_names:
                    for sim_name in sim_names:
                        yield ipk, field1, field2, sim_name
                        ipk += 1


    def iterate_pairs(self, ):
        '''
        Generates index and field pairings for power spectrum calculations.
        Yields: (ipk, field1, field2) where field1 and field2 are combinations
        of 'd' (density field) and mark function names (e.g., 'm1').
        Examples: ('d','d'), ('d','m1'), ('m1','m1')
        '''

        map_names = ['d'] + self.mark_names
        ipk = 0
        for i1, field1 in enumerate(map_names):
            for field2 in map_names[i1:]: 
                yield ipk, field1, field2
                ipk += 1
        


        
    def my_pinv(self, cov, w_thr=1E-7):
        '''Calculate pseudo inverse of a covariance matrix.
        Parameters:
        cov: covariance matrix
        w_thr: threshold of lowest acceptable eigenvalue RATIO, n.b. we updated this '''
        w, v = np.linalg.eigh(cov) #cants use jax here because of = statement below
    #     cond_n = np.max(w)/w
        inv_cond = w/np.max(w)
        badw = inv_cond < w_thr
        w_inv = 1./w
        w_inv[badw] = 0.
#         print('number cut', np.sum(badw))
#         print('badw', badw)
#         print('max condition number',np.max(1/inv_cond))
    #     print(‘final condition number’,np.max(1/inv_cond[~badw]))
        pinv = jnp.dot(v, np.dot(np.diag(w_inv), v.T))
        return pinv

    def convert_from_angle(self, angles):
        '''convert from 3 angles to 4 points in delta_R space  i.e. cartesian 4D on surface of sphere'''
        a,b,c = angles
                    
        w = sin(a)*sin(b)*sin(c)
        x = sin(a)*sin(b)*cos(c)
        y =sin(a)*cos(b)
        z = cos(a)
        return(np.array([w,x,y,z]))
        
    def get_mark_from_nodes(self,):
        #takes array of marks
        grid=Nmesh=256
        kernel=self.kernel     
        nmodes=n_nodes=len(self.nodes) #numbers of nodes/points of mark function to connect 
        delta_R_fid = self.smoothed_fields['fiducial']

        if self.set_delta_range==True:
            delta_range = self.delta_range 
            print('set delta_range=true', delta_range)
            
        else:
            delta_range = [np.min(delta_R_fid), np.max(delta_R_fid)]
            self.delta_range=delta_range
            print('No set delta range provided, using default ', f'range:  {np.min(delta_R_fid)},{np.max(delta_R_fid)}')
            
            
        delta_R_train = ((np.arange(n_nodes)+0.5)/n_nodes*(delta_range[1]-delta_range[0])+delta_range[0]).reshape(-1,1) #this is the nodes for the GP!
        print('Nodes placed at', delta_R_train)
        self.delta_R_train = delta_R_train
                                
        marks_dict = {}
        i = 1  # only one mark supported currently
        if self.use_white:
            delta_R = self.smoothed_fields
            p, b = self.white_param
            mark_fid = self.white_mark(p, b, self.smoothed_fields['fiducial'])
            mark_Omp = self.white_mark(p, b, self.smoothed_fields['Om_p'])
            mark_Omm = self.white_mark(p, b, self.smoothed_fields['Om_m'])
            mark_s8p = self.white_mark(p, b, self.smoothed_fields['s8_p'])
            mark_s8m = self.white_mark(p, b, self.smoothed_fields['s8_m'])
            marks_dict[f'mark_fid_{i}'] = mark_fid
            marks_dict[f'mark_Omp{i}'] = mark_Omp
            marks_dict[f'mark_Omm{i}'] = mark_Omm
            marks_dict[f'mark_s8_p{i}'] = mark_s8p
            marks_dict[f'mark_s8_m{i}'] = mark_s8m
        else:
            # Use Gaussian Process to create marks
            gpr = GaussianProcessRegressor(kernel=self.kernel, random_state=1)
            gpr.fit(self.delta_R_train, self.nodes)
            mark_fid, _ = gpr.predict(self.smoothed_fields['fiducial'].flatten().reshape(-1, 1), return_std=True)
            self.delta_R_flat = self.smoothed_fields['fiducial'].flatten()
            self.mark_fid = mark_fid
            mark_Omp, _ = gpr.predict(self.smoothed_fields['Om_p'].flatten().reshape(-1, 1), return_std=True)
            mark_Omm, _ = gpr.predict(self.smoothed_fields['Om_m'].flatten().reshape(-1, 1), return_std=True)
            mark_s8p, _ = gpr.predict(self.smoothed_fields['s8_p'].flatten().reshape(-1, 1), return_std=True)
            mark_s8m, _ = gpr.predict(self.smoothed_fields['s8_m'].flatten().reshape(-1, 1), return_std=True)
            marks_dict[f'mark_fid_{i}'] = mark_fid.reshape([ngrid, ngrid, ngrid])
            marks_dict[f'mark_Omp{i}'] = mark_Omp.reshape([ngrid, ngrid, ngrid])
            marks_dict[f'mark_Omm{i}'] = mark_Omm.reshape([ngrid, ngrid, ngrid])
            marks_dict[f'mark_s8_p{i}'] = mark_s8p.reshape([ngrid, ngrid, ngrid])
            marks_dict[f'mark_s8_m{i}'] = mark_s8m.reshape([ngrid, ngrid, ngrid])
            self.marks_dict = marks_dict

        # Calculate the marked field in real space
        self.marked_field_fid = (self.fields['fiducial'] + 1) * marks_dict[f'mark_fid_{i}']
        self.marked_field_omp = (self.fields['Om_p'] + 1) * marks_dict[f'mark_Omp{i}']
        self.marked_field_omm = (self.fields['Om_m'] + 1) * marks_dict[f'mark_Omm{i}']
        self.marked_field_s8p = (self.fields['s8_p'] + 1) * marks_dict[f'mark_s8_p{i}']
        self.marked_field_s8m = (self.fields['s8_m'] + 1) * marks_dict[f'mark_s8_m{i}']

        self.marked_field_fid -= np.mean(self.marked_field_fid)
        self.marked_field_omp -= np.mean(self.marked_field_omp)
        self.marked_field_omm -= np.mean(self.marked_field_omm)
        self.marked_field_s8p -= np.mean(self.marked_field_s8p)
        self.marked_field_s8m -= np.mean(self.marked_field_s8m)

        # FFT marked field
        _, self.k_fields[f'fft_m{i}_fiducial'] = Fourier(self.marked_field_fid, Nmesh=ngrid)
        _, self.k_fields[f'fft_m{i}_Om_p'] = Fourier(self.marked_field_omp, Nmesh=ngrid)
        _, self.k_fields[f'fft_m{i}_Om_m'] = Fourier(self.marked_field_omm, Nmesh=ngrid)
        _, self.k_fields[f'fft_m{i}_s8_p'] = Fourier(self.marked_field_s8p, Nmesh=ngrid)
        _, self.k_fields[f'fft_m{i}_s8_m'] = Fourier(self.marked_field_s8m, Nmesh=ngrid)

        return marks_dict

        # self.marked_field_fid  -= np.mean(self.marked_field_fid )
        # self.marked_field_omp  -= np.mean(self.marked_field_omp )
        # self.marked_field_omm  -= np.mean(self.marked_field_omm )
        # self.marked_field_s8p  -= np.mean(self.marked_field_s8p )
        # self.marked_field_s8m  -= np.mean(self.marked_field_s8m )


        # #fft marked field
        # _, self.k_fields[f'fft_m{i}_fiducial'] = Fourier(self.marked_field_fid, Nmesh=Nmesh)
        # _, self.k_fields[f'fft_m{i}_Om_p'] = Fourier(self.marked_field_omp, Nmesh=Nmesh)
        # _, self.k_fields[f'fft_m{i}_Om_m'] = Fourier(self.marked_field_omm, Nmesh=Nmesh)
        # _, self.k_fields[f'fft_m{i}_s8_p'] = Fourier(self.marked_field_s8p, Nmesh=Nmesh)
        # _, self.k_fields[f'fft_m{i}_s8_m'] = Fourier(self.marked_field_s8m, Nmesh=Nmesh)



        # return(marks_dict)
    

    def get_all_Pks(self):
        Pks = {}
        for ix, f1, f2, f3 in self.all_field_sim_combinations(self.names):
            # calculate all Pk for different fields m1dm, d2d etc.
            _, _, Pks[f'Pk_{f1}{f2}_{f3}'] = get_Pk(
                self.k_fields[f'fft_{f1}_{f3}'],
                self.ktot,
                second=self.k_fields[f'fft_{f2}_{f3}'],
                kmax=self.kmax_bin
            )
            # scale cuts
            Pks[f'Pk_{f1}{f2}_{f3}'] = Pks[f'Pk_{f1}{f2}_{f3}'][self.good_k]
        return Pks





    def get_fom(self, mark_names, fom_type,):
        '''get fom from mark function'''
        #calculate Pks
        Pks = self.get_all_Pks()
        #calculate derivatives
        derivs = self.get_derivs(Pks, mark_names)
        self.derivs=derivs
        #calculate theoretical covariance
           
        length=len(derivs['deriv_s8'])
        cov = self.get_cov(Pks, mark_names, length)
        self.cov=cov
        #inverse cov
        self.icov = self.my_pinv(cov, w_thr=1E-7) #inverse covariance
        #fisher matrix 
        deriv_s8 = derivs['deriv_s8']; deriv_Om = derivs['deriv_Om']
        

        fisher_cov_so = jnp.dot(deriv_s8.T,np.dot( self.icov, deriv_Om))
        fisher_cov_os = jnp.dot(deriv_Om.T,np.dot( self.icov, deriv_s8))
        fisher_cov_ss = jnp.dot(deriv_s8.T,np.dot( self.icov, deriv_s8))
        fisher_cov_oo = jnp.dot(deriv_Om.T,np.dot( self.icov, deriv_Om))

        fisher_cov = np.array([[fisher_cov_ss, fisher_cov_so],
                            [fisher_cov_os, fisher_cov_oo]])
        self.fisher_cov=fisher_cov
        
        error = np.sqrt(self.my_pinv(fisher_cov))
        self.error=error
        self.plot_mark()
        
            
            
        if self.david_fom==True:
            if self.fom == 'both':
                f = np.linalg.det(fisher_cov)
            elif self.fom == 'om':
                f = fisher_cov[0, 0]
            elif self.fom == 's8':
                
                f = fisher_cov[1, 1]
            f = np.log10(f)
            return -f
    
        elif self.david_fom==False:
            if fom_type =='both':
                fom = np.linalg.det(fisher_cov)
            elif fom_type =='s8':
                fom = fisher_cov[0,0]
            elif fom_type =='om':
                fom = fisher_cov[1,1]
            return(-np.log10(fom))
            

        else:
            print('invalid FOM type!!!!')
            raise Exception


    def get_derivs(self, Pks, mark_names):
        derivs={}
        derivs['deriv_s8']= derivs['deriv_Om']=[]
        for ix, f1, f2, obs in self.finite_diff_pair():
            derivs[f'{obs}_{f1}{f2}_finite_diff'] = (Pks[f'Pk_{f1}{f2}_{obs}_p'] - Pks[f'Pk_{f1}{f2}_{obs}_m'])/(2*self.delta_Om)
            derivs[f'deriv_{obs}'] = (np.hstack([ derivs[f'deriv_{obs}'],derivs[f'{obs}_{f1}{f2}_finite_diff'] ]))
        return derivs

    def get_cov(self, Pks, mark_names,  length,marked=True,):
        if self.real_cov is not None:
            print('using provided covariance')
            return self.real_cov
        else:
            print('Calculating analytical covariance approximation...')
            length = comb(len(mark_names)+2,2)
            ndata = length*len(self.nmodes_pk)
            idx=[]

            indices={}
            for ix, field1, field2, in self.iterate_pairs():
                idx.append(f'id_{field1}{field2}' )
                indices[f'{field1}{field2}'] =  np.arange(ndata).reshape([length, ndata//length])[ix]

            cov = np.zeros([ndata, ndata])


            for _, field1_a, field2_a in self.iterate_pairs(): #field is either original or marked field
                id_a = indices[f'{field1_a}{field2_a}']
                for _, field1_b, field2_b in self.iterate_pairs():
                    id_b = indices[f'{field1_b}{field2_b}']
                    pk_field1a_field1b = Pks[f'Pk_{field1_a}{field1_b}_fiducial']
                    pk_field2a_field2b = Pks[f'Pk_{field2_a}{field2_b}_fiducial']
                    pk_field1a_field2b = Pks[f'Pk_{field1_a}{field2_b}_fiducial']
                    pk_field2a_field1b = Pks[f'Pk_{field2_a}{field1_b}_fiducial']
                    # print(len(pk_field2a_field1b))



                    cov[np.ix_(id_a, id_b)] = np.diag((pk_field1a_field1b*pk_field2a_field2b +
                                                    pk_field1a_field2b*pk_field2a_field1b)/self.nmodes_pk)

            path = '/mnt/extraspace/jesscowell/MarkedCorr/COLA_PKs_FINAL/'
            # print(len(cov))
            # np.save(path+'theoretical_cov', cov)
            return cov

if __name__ == "__main__":
    # Example usage: only runs when script is executed directly, not on import
    # You can customize this block for testing or running analysis
    angles = np.array([0, 0, 0])
    mark = Mark(angles)
    mark.load_data()  # Load data and run analysis
    print("Data loaded. You can now use mark's analysis methods.")
    # ... add more demo or test code here as needed ...






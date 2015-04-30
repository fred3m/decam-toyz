from __future__ import print_function, division
import os
import pandas
import numpy as np
from sqlalchemy import create_engine
import subprocess
import copy
from collections import OrderedDict
import logging as log
import astropy.io.fits as pyfits

from toyz.utils.errors import ToyzError
from astrotoyz.astromatic import api

"""
Improvements to make before pipeline is finished
1) Organize stacks by obj, night, filter, exptime
2) Test positions
3) Generatre PSF and test that it is position dependent
4) Test photometry
5) Convert to DECam-Toyz
6) Create .ahead with FILTER, (create) AIRMASS, EQUINOX
"""

def get_fits_name(expnum, prodtype):
    if prodtype=='image':
        fits_name = '{0}.fits'.format(expnum)
    else:
        fits_name = "{0}.{1}.fits".format(expnum, prodtype)
    return fits_name

def funpack_file(filename, fits_path):
    """
    Funpack a compressed fits file and copy it to a new path
    """
    if not os.path.isfile(fits_path):
        print("funpacking '{0}' to '{1}'\n".format(filename, fits_path))
        subprocess.call('funpack '+filename, shell=True)
        funpacked_name = os.path.basename(filename[:-3])
        cmd = 'mv '+os.path.join(os.path.dirname(filename), funpacked_name)
        cmd += ' '+fits_path
        subprocess.call(cmd, shell=True)

def check_path(pathname, path):
    if not os.path.exists(path):
        import toyz.utils.core as core
        if core.get_bool("{0} '{1}' does not exist, create (y/n)?".format(pathname, path)):
            core.create_paths(path)
        else:
            raise PipelineError("{0} does not exist".format(pathname))

class PipelineError(ToyzError):
    pass

class Pipeline:
    def __init__(self, img_path, idx_connect_str, temp_path, cat_path, 
            result_path=None, resamp_path=None, stack_path=None,
            config_path=None, build_paths={}, create_idx = None,
            default_kwargs={}):
        """
        Parameters
        ----------
        img_path: str
            path to decam images
        idx_connect_str: str
            sqlalchemy connection string to decam index database
        temp_path: str
            path to store temporary files
        cat_path: str
            path to save final catalogs
        result_path: str
            path to save resampled and stacked images
        config_path: str
            path to check for astromatic config files
                * defaults to decamtoyz/defaults
        build_paths: dict
            paths to astromatic builds
                * Not needed if the codes were installed system wide (ex. 'sex' runs SExtractor)
                * Keys are commands for astromatic packages ('sex', 'scamp', 'swarp', 'psfex')
                * Values are the paths to the build for the given key
        create_idx: bool
            By default, if the decam index DB cannot be found, the user will
            be prompted to create the index. Setting `create_idx` to `True` overrides 
            this behavior and automatically creates the index. Setting `create_idx` ``False``
            will raise an error
        """
        if idx_connect_str.startswith('sqlite'):
            if not os.path.isfile(idx_connect_str[10:]):
                print('path', idx_connect_str[10:])
                if create_idx is not None:
                    if create_idx:
                        import toyz.utils.core as core
                        if not core.get_bool(
                                "DECam file index does not exist, create it now? ('y'/'n')"):
                            raise PipelineError("Unable to locate DECam file index")
                    else:
                         raise PipelineError("Unable to locate DECam file index")
                import decamtoyz.index as index
                import toyz.utils.core as core
                recursive = core.get_bool("Search '{0}' recursively for images? ('y'/'n')")
                index.build_idx(img_path, idx_connect_str, True, recursive, True)
        self.idx = create_engine(idx_connect_str)
        self.idx_connect_str = idx_connect_str
        self.img_path = img_path
        self.temp_path = temp_path
        self.cat_path = cat_path
        self.result_path = result_path
        self.resamp_path = resamp_path
        self.stack_path = stack_path
        self.build_paths = build_paths
        self.default_kwargs = default_kwargs
        # IF the user doesn't specify a path for config files, use the default decam config files
        if config_path is None:
            from decam import root
            self.config_path = os.path.join(root, 'default')
        else:
            self.config_path = config_path
        
        # If any of the specified paths don't exist, give the user the option to create them
        check_path('temp_path', self.temp_path)
        check_path('cat_path', self.cat_path)
        check_path('stack_path', self.stack_path)
    
    def old_get_obs_tree(self, obj, proctype='InstCal', **kwargs):
        """
        Organize a set of observations for a given object (as well as the option of
        specifying nights, filters, and exptimes in kwargs).
        
        Then Copy funpacked (uncompressed) file versions into the temp directory.
        """
        where = "object like '{0}%'".format(obj)
        if 'nights' not in kwargs:
            sql = "select distinct cal_date from decam_obs where "+where
            df = pandas.read_sql(sql, self.idx)
            nights = df['cal_date'].values.tolist()
        else:
            nights = kwargs['nights']
        obs = OrderedDict()
        total_obs=0
        for night in nights:
            obs[night] = OrderedDict()
            night_where = where + " and cal_date='{0}'".format(night)
            if 'filters' not in kwargs:
                sql = "select distinct filter from decam_obs where "+night_where
                df = pandas.read_sql(sql, self.idx)
                filters = [f[0] for f in df['filter'].values.tolist()]
            else:
                filters = kwargs['filters']
            for f in filters:
                obs[night][f] = OrderedDict()
                f_where = night_where + " and filter like '{0}%'".format(f)
                if 'exptimes' not in kwargs:
                    sql = "select distinct exptime from decam_obs where "+f_where
                    df = pandas.read_sql(sql, self.idx)
                    exptimes = df['exptime'].values.tolist()
                else:
                    exptimes = kwargs['exptimes']
                for exptime in exptimes:
                    obs[night][f][exptime] = OrderedDict()
                    exp_where = f_where + " and exptime={0}".format(exptime)
                    sql = "select distinct expnum from decam_obs where "+exp_where
                    df = pandas.read_sql(sql, self.idx)
                    exps = df['expnum'].values.tolist()
                    for expnum in exps:
                        obs[night][f][exptime][expnum] = OrderedDict()
                        sql = ("select * from decam_files where EXPNUM={0} and "+
                            "PROCTYPE='{1}'").format(expnum, proctype)
                        df = pandas.read_sql(sql, self.idx)
                        total_obs+=1
                        for row in df.iterrows():
                            fits_name = get_fits_name(expnum, row[1]['PRODTYPE'])
                            fits_path = os.path.join(self.temp_path, fits_name)
                            #_funpack_file(row[1]['filename'], fits_path)
                            obs[night][f][exptime][expnum][row[1]['PRODTYPE']] = fits_path
        print('Observations:\n', obs)
        print('Total Observations for {0}: {1}'.format(obj, total_obs))
        return obs
    
    def run_sex(self, files, kwargs={}, frames=None, show_all_cmds=False):
        """
        Run SExtractor with a specified set of parameters.
        
        Parameters
        ----------
        files: dict
            Dict of filenames for fits files to use in sextractor. Possible keys are:
                * image: filename of the fits image (required)
                * dqmask: filename of a bad mixel mask for the given image (optional)
                * wtmap: filename of a weight map for the given image (optional)
        kwargs: dict
            Keyword arguements to pass to ``atrotoyz.Astromatic.run`` or
            ``astrotoyz.Astromatic.run_sex_frames``
        frames: str (optional)
            Only run sextractor on a specific set of frames. This should either be an 
            integer string, or a string of csv's
        show_all_cmds: bool (optional)
            Whether or not to show each command use to execute sextractor (defaults to
            ``False`` and is only used when running sextractor on multiple frames)
        """
        if 'code' not in kwargs:
            kwargs['code'] = 'SExtractor'
        if 'cmd' not in kwargs and 'SExtractor' in self.build_paths:
            kwargs['cmd'] = self.build_paths['SExtractor']
        if 'temp_path' not in kwargs:
            kwargs['temp_path'] = self.temp_path
        if 'config' not in kwargs:
            kwargs['config'] = {}
        if 'CATALOG_NAME' not in kwargs['config']:
            kwargs['config']['CATALOG_NAME'] = files['image'].replace('.fits', '.cat')
        if 'FLAG_IMAGE' not in kwargs['config'] and 'dqmask' in files:
            kwargs['config']['FLAG_IMAGE'] = files['dqmask']
        if 'WEIGHT_IMAGE' not in kwargs['config'] and 'wtmap' in files:
            kwargs['config']['WEIGHT_IMAGE'] = files['wtmap']
        sex = api.Astromatic(**kwargs)
        if frames is None:
            sex.run(files['image'])
        else:
            sex.run_sex_frames(files['image'], frames, show_all_cmds)
    
    def run_psfex(self, catalogs, kwargs={}):
        """
        Run PSFEx with a specified set of parameters.
        
        Parameters
        ----------
        catalogs: str or list
            catalog filename (or list of catalog filenames) to use
        kwargs: dict
            Keyword arguements to pass to PSFEx
        """
        if 'code' not in kwargs:
            kwargs['code'] = 'PSFEx'
        if 'cmd' not in kwargs and 'PSFEx' in self.build_paths:
            kwargs['cmd'] = self.build_paths['PSFEx']
        if 'temp_path' not in kwargs:
            kwargs['temp_path'] = self.temp_path
        psfex = api.Astromatic(**kwargs)
        psfex.run(catalogs)
    
    def run_scamp(self, catalogs, kwargs={}, save_catalog=None):
        """
        Run SCAMP with a specified set of parameters
        
        Parameters
        ----------
        catalogs: list
            List of catalog names used to generate astrometric solution
        kwargs: dict
            Dictionary of keyword arguments used to run SCAMP
        save_catalog: str (optional)
            If ``save_catalog`` is specified, the reference catalog used to generate the
            solution will be save to the path ``save_catalog``.
        """
        if 'code' not in kwargs:
            kwargs['code'] = 'SCAMP'
        if 'cmd' not in kwargs and 'SCAMP' in self.build_paths:
            kwargs['cmd'] = self.build_paths['SCAMP']
        if 'temp_path' not in kwargs:
            kwargs['temp_path'] = self.temp_path
        if 'config' not in kwargs:
            kwargs['config'] = {}
        if save_catalog is not None:
            kwargs['config']['SAVE_REFCATALOG'] = 'Y'
            kwargs['config']['REFOUT_CATPATH'] = save_catalog
        scamp = api.Astromatic(**kwargs)
        scamp.run(catalogs)
    
    def run_swarp(self, filenames, stack_filename=None, api_kwargs={}, 
            frames=None, run_type='both'):
        """
        Run SWARP with a specified set of parameters
        
        Parameters
        ----------
        filenames: list
            List of filenames that are stacked together
        stack_filename: str (optional)
            Name of final stacked image. If the user is only resampling but not stacking
            (``run_type='resample'``), this variable is ignored.
        api_kwargs: dict
            Keyword arguments used to run SWARP
        frames: list (optional)
            Subset of frames to stack. Default value is ``None``, which stacks all of the
            image frames for each file
        run_type: str
            How SCAMP will be run. Can be ``resample`` or ``stack`` or ``both``, which
            resamples and stacks the images.
        """
        if 'code' not in api_kwargs:
            api_kwargs['code'] = 'SWarp'
        if 'cmd' not in api_kwargs and 'SWARP' in self.build_paths:
            api_kwargs['cmd'] = self.build_paths['SWARP']
        if 'temp_path' not in api_kwargs:
            api_kwargs['temp_path'] = self.temp_path
        if 'config' not in api_kwargs:
            api_kwargs['config'] = {}
        if 'RESAMPLE_DIR' not in api_kwargs['config']:
            api_kwargs['config']['RESAMPLE_DIR'] = api_kwargs['temp_path']
        if run_type=='both' or run_type=='resample':
            # Resample images as specified by WCS keywords in their headers
            log.info("Create resampled images")
            kwargs = copy.deepcopy(api_kwargs)
            kwargs['config']['COMBINE'] = 'N'
            swarp = api.Astromatic(**kwargs)
            swarp.run(filenames)
        if run_type=='both' or run_type=='stack':
            log.info('Creating stack for each CCD')
            if stack_filename is None:
                raise PipelineError("Must include a stack_filename to stack a set of images")
            kwargs = copy.deepcopy(api_kwargs)
            kwargs['config']['RESAMPLE'] = 'N'
            if frames is None:
                hdulist = pyfits.open(filenames[0])
                frames = range(1,len(hdulist))
                hdulist.close()
            # Temporarily create a stack for each frame
            for frame in frames:
                resamp_names = [f.replace('.fits', '.{0:04d}.resamp.fits'.format(frame)) 
                    for f in filenames]
                stack_frame = os.path.join(kwargs['temp_path'], 
                    os.path.basename(stack_filename).replace('.fits', 
                    '-{0:04d}.stack.fits'.format(frame)))
                kwargs['config']['IMAGEOUT_NAME'] = stack_frame
                kwargs['config']['WEIGHTOUT_NAME'] = stack_frame.replace('.fits', '.weight.fits')
                swarp = api.Astromatic(**kwargs)
                swarp.run(resamp_names)
            
            # Combine the frame stacks into a single stacked image
            log.info("Combining into single stack")
            primary = pyfits.PrimaryHDU()
            stack = [primary]
            weights = [primary]
            for frame in frames:
                stack_frame = os.path.join(kwargs['temp_path'], 
                    os.path.basename(stack_filename).replace('.fits', 
                    '-{0:04d}.stack.fits'.format(frame)))
                weight_frame = stack_frame.replace('.fits', '.weight.fits')
                hdulist = pyfits.open(stack_frame)
                stack.append(pyfits.ImageHDU(hdulist[0].data,hdulist[0].header))
                hdulist.close()
                hdulist = pyfits.open(weight_frame)
                weights.append(pyfits.ImageHDU(hdulist[0].data,hdulist[0].header))
            weight_name = stack_filename.replace('.fits', '.wtmap.fits')
            stack = pyfits.HDUList(stack)
            stack.writeto(stack_filename, clobber=True)
            stack.close()
            weights = pyfits.HDUList(weights)
            weights.writeto(weight_name, clobber=True)
            weights.close()
    
    def get_obs(self, sql):
        return pandas.read_sql(sql, self.idx)
    
    def run(self, steps):
        """
        Run the pipeline given a list of PipelineSteps
        """
        for step in pipeline_steps:
            if step.prefunc is not None:
                step.prefunc(step, exposures)
            if step.code == 'SExtractor:':
                self.run_sex(step.files, step.api_kwargs, **step.kwargs)
            if step.postfunc is not None:
                step.postfunc(step, exposures)

class PipelineStep:
    def __init__(self, files, api_kwargs, pre_func=None, post_func=None,
            **kwargs):
        self.api_kwargs = copy.deepcopy(api_kwargs)
        self.code = api_kwargs['code']
        if self.code not in api.codes:
            raise PipelineError("Code must be one of "+','.join(api.codes.keys()))
        self.files
        self.pre_func = pre_func
        self.post_func = post_func
        self.kwargs = kwargs
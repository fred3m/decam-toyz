from __future__ import print_function, division
import os
import pandas
import numpy as np
from sqlalchemy import create_engine
import subprocess
import copy
from collections import OrderedDict
import astropy.io.fits as pyfits

from toyz.utils.errors import ToyzError
import astrotoyz.astromatic.api as api

"""
Improvements to make before pipeline is finished
1) Organize stacks by obj, night, filter, exptime
2) Test positions
3) Generatre PSF and test that it is position dependent
4) Test photometry
5) Convert to DECam-Toyz
6) Create .ahead with FILTER, (create) AIRMASS, EQUINOX
"""

pipeline_steps = ['sex_nopsf', 'scamp', 'resample', 'stack', 'sex_for_psf', 'psfex', 'sex_psf']

def _get_fits_name(expnum, prodtype):
    if prodtype=='image':
        fits_name = '{0}.fits'.format(expnum)
    else:
        fits_name = "{0}.{1}.fits".format(expnum, prodtype)
    return fits_name

def _funpack_file(filename, fits_path):
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
            raise DecamPipeError("{0} does not exist".format(pathname))

class PipelineError(ToyzError):
    pass

class Pipeline:
    def __init__(self, img_path, idx_connect_str, temp_path, cat_path, 
            result_path=None, resamp_path=None, stack_path=None,
            config_path=None, build_paths={}, create_idx = False):
        """
        - img_path: path to decam images
        - idx_connect_str: sqlalchemy connection string to decam index database
        - temp_path: path to store temporary files
        - cat_path: path to save final catalogs
        - result_path: path to save resampled and stacked images
        - config_path: path to check for astromatic config files
            * defaults to decamtoyz/defaults
        - build_paths: dictionary of paths to astromatic builds
            * Not needed if the codes were installed system wide (ex. 'sex' runs SExtractor)
            * Keys are commands for astromatic packages ('sex', 'scamp', 'swarp', 'psfex')
            * Values are the paths to the build for the given key
        - create_idx: By default, if the decam index DB cannot be found, the user will
          be prompted to create the index. Setting `create_idx` to `True` overrides this behavior
          and automatically creates the index
        """
        if idx_connect_str.startswith('sqlite'):
            if not os.path.isfile(idx_connect_str.lstrip('sqlite:///')):
                if not create_idx:
                    import toyz.utils.core as core
                    if not core.get_bool(
                            "DECam file index does not exist, create it now? ('y'/'n')"):
                        raise DecamPipeError("Unable to locate DECam file index")
                import decamtoyz.index as index
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
                            fits_name = _get_fits_name(expnum, row[1]['PRODTYPE'])
                            fits_path = os.path.join(self.temp_path, fits_name)
                            #_funpack_file(row[1]['filename'], fits_path)
                            obs[night][f][exptime][expnum][row[1]['PRODTYPE']] = fits_path
        print('Observations:\n', obs)
        print('Total Observations for {0}: {1}'.format(obj, total_obs))
        return obs
    
    def run_sex(self, exposures, frames=None):
        pass
    
    def run_psfex(self, exposures):
        pass
    
    def run_scamp(self, exposures, groupby=[]):
        pass
    
    def run_swarp(self, exposures, frames=None, groupby=[]):
        pass
    
    def get_obs(self, sql):
        return pandas.read_sql(sql, self.idx)
    
    def run(self, steps=pipeline_steps, exposures=None, sql=None):
        # If no dataframe of exposures is passed to the function, load them from 
        # the index
        if exposures is None:
            # If no sql to query the index is specified, load all fields in the index
            if sql is None:
                sql = "select * from decam_obs"
            exposures = pandas.read_sql("select * from decam_obs", self.idx)
        objects = exposures['object'].str.split('-').apply(pandas.Series).sort(0)[0].unique()
        print('Reducing fields:\n', objects)
        obs_tree = self.fill_obs_tree(obs_tree)
            
        for obj in objects:
            print('OBJECT', obj)
            obs_tree = get_obs_tree(engine, temp_path, obj, **obj_kwargs)
    
            # TODO: remove the following testing line
            obs = {'i':obs['i']}
    
            # run SExtractor to get positions used for astrometric solutions
            if 'sex_nopsf' in steps:
                print('Finding sources to use in astrometric solution\n')
                for f in obs: # filter is a python keyword, so we use f
                    for expnum, files in obs[f].items():
                        kwargs = copy.deepcopy(sex_kwargs)
                        kwargs['config']['CATALOG_NAME'] = files['image'].replace(
                            '.fits', '.cat')
                        kwargs['config']['FLAG_IMAGE'] = files['dqmask']
                        kwargs['config']['WEIGHT_IMAGE'] = files['wtmap']
                        kwargs['config']['PARAMETERS_NAME'] = os.path.join(
                            config_path, 'decam_nopsf.param')
                        kwargs['config_file'] = os.path.join(config_path, 'decam_nopsf.sex')
                        #print('\n\n\n', sex_nopsf_kwargs)
                        sex = api.Astromatic(**kwargs)
                        #sex.run_sex_frames(files['image'], '1', True)
                        sex.run(files['image'])
            # Run SCAMP to get astrometric solution
            if 'scamp' in steps:
                print('Calculating astrometric solution\n')
                for f in obs: # filter is a python keyword, so we use f
                    cat_paths = [
                        os.path.join(temp_path, '{0}.cat'.format(expnum)) for expnum in sorted(obs[f])]
                    scamp_kwargs['config']['REFOUT_CATPATH'] = cat_path
                    #print('catalog paths', cat_paths)
                    #print('merged_name', scamp_kwargs['config']['MERGEDOUTCAT_NAME'])
                    scamp = api.Astromatic(**scamp_kwargs)
                    scamp.run(cat_paths)
            # Run SWarp and create stacks
            if 'stack' in steps or 'resample' in steps:
                print('Stacking images')
                """
                for f in obs: # filter is a python keyword, so we use f
                    images = [files['image'] for expnum, files in obs[f].items()]
                    stack_name = os.path.join(stack_path, '{0}-{1}.fits'.format(obj, f))
                    swarp_kwargs['config']['IMAGEOUT_NAME'] = stack_name
                    swarp_kwargs['config']['WEIGHTOUT_NAME'] = os.path.join(stack_path,
                        '{0}-{1}.wtmap.fits'.format(obj, f))
                    if 'resample' not in steps:
                        swarp_kwargs['config']['RESAMPLE'] = 'N'
                    if 'stack' not in steps:
                        swarp_kwargs['config']['COMBINE'] = 'N'
                    #print(swarp_kwargs)
                    swarp = api.Astromatic(**swarp_kwargs)
                    swarp.run(images)
                """
                for f in obs:
                    if 'resample' in steps:
                        kwargs = copy.deepcopy(swarp_kwargs)
                        images = [files['image'] for expnum, files in obs[f].items()]
                        stack_name = os.path.join(stack_path, '{0}-{1}.fits'.format(obj, f))
                        kwargs['config']['COMBINE'] = 'N'
                        #print(kwargs)
                        swarp = api.Astromatic(**kwargs)
                        swarp.run(images)
                
                    if 'stack' in steps:
                        kwargs = copy.deepcopy(swarp_kwargs)
                        exps = obs[f].keys()
                        hdulist = pyfits.open(obs[f][exps[0]]['image'])
                        frames = len(hdulist)-1
                        hdulist.close()
                        for frame in range(1,frames+1):
                            images = [os.path.join(temp_path, '{0}.{1:04d}.resamp.fits'.format(
                                expnum, frame)) for expnum in exps]
                            stack_name = os.path.join(temp_path, '{0}-{1}-{2:04d}.stack.fits'.format(
                                obj, f, frame))
                            kwargs['config']['RESAMPLE'] = 'N'
                            kwargs['config']['IMAGEOUT_NAME'] = stack_name
                            kwargs['config']['WEIGHTOUT_NAME'] = stack_name.replace(
                                '.fits', '.wtmap.fits')
                            kwargs['config']['WEIGHT_SUFFIX'] = '.weight.fits'
                            #print(kwargs)
                            swarp = api.Astromatic(**kwargs)
                            swarp.run(images)
                        primary = pyfits.PrimaryHDU()
                        stack = [primary]
                        weights = [primary]
                        print('Combining stack frames')
                        for frame in range(1,frames+1):
                            stack_name = os.path.join(temp_path, '{0}-{1}-{2:04d}.stack.fits'.format(
                                obj, f, frame))
                            weight_name = stack_name.replace('.fits', '.wtmap.fits')
                            hdulist = pyfits.open(stack_name)
                            stack.append(pyfits.ImageHDU(hdulist[0].data,hdulist[0].header))
                            hdulist.close()
                            hdulist = pyfits.open(weight_name)
                            weights.append(pyfits.ImageHDU(hdulist[0].data,hdulist[0].header))
                        stack_name = os.path.join(stack_path, '{0}-{1}.fits'.format(obj, f))
                        weight_name = stack_name.replace('.fits', '.wtmap.fits')
                        stack = pyfits.HDUList(stack)
                        stack.writeto(stack_name, clobber=True)
                        stack.close()
                        weights = pyfits.HDUList(weights)
                        weights.writeto(weight_name, clobber=True)
                        weights.close()
    
            # Run SExtractor to get positions and vignettes for psfex
            if 'sex_for_psf' in steps:
                for f in obs:
                    stack_name = os.path.join(stack_path, '{0}-{1}.fits'.format(obj, f))
                    cat_name = os.path.join(temp_path, 
                        os.path.basename(stack_name.replace('.fits', '.cat')))
                    kwargs = copy.deepcopy(sex_kwargs)
                    kwargs['config']['CATALOG_NAME'] = cat_name
                    kwargs['config']['WEIGHT_IMAGE'] = stack_name.replace('.fits', '.wtmap.fits')
                    kwargs['config']['PARAMETERS_NAME'] = os.path.join(
                        config_path, 'decam_for_psf.param')
                    kwargs['config_file'] = os.path.join(config_path, 'decam_for_psf.sex')
                    #print('\n\n\n', kwargs)
                    sex = api.Astromatic(**kwargs)
                    sex.run(stack_name)
                    #sex.run_sex_frames(stack_name, '1', True)
            # Run PSFeX to generate PSF
            if 'psfex' in steps:
                for f in obs:
                    cat_name = os.path.join(temp_path, '{0}-{1}.cat'.format(obj, f))
                    #print(psfex_kwargs)
                    psfex = api.Astromatic(**psfex_kwargs)
                    psfex.run(cat_name)
            # Run SExtractor with PSF to get improved photometry
            if 'sex_psf' in steps:
                for f in obs:
                    stack_name = os.path.join(stack_path, '{0}-{1}.fits'.format(obj, f))
                    cat_name = os.path.join(cat_path, '{0}-{1}.cat.fits'.format(obj, f))
                    kwargs = copy.deepcopy(sex_kwargs)
                    kwargs['config']['CATALOG_NAME'] = cat_name
                    kwargs['config']['WEIGHT_IMAGE'] = stack_name.replace('.fits', '.wtmap.fits')
                    kwargs['config']['PARAMETERS_NAME'] = os.path.join(
                        config_path, 'decam_psf.param')
                    kwargs['config']['PSF_NAME'] = os.path.join(temp_path, 
                        os.path.basename(stack_name.replace('.fits', '.psf')))
                    kwargs['config_file'] = os.path.join(config_path, 'decam_psf.sex')
                    #print('\n\n\n', kwargs)
                    sex = api.Astromatic(**kwargs)
                    sex.run(stack_name)

class PipelineStep:
    def __init__(self, code, sql=None, exposures=None, pre_func=None, post_func=None):
        if code not in api.codes:
            raise PipelineError("Code must be one of "+','.join(api.codes.keys()))
        self.code = code
        self.exposures = exposures
        self.sql = sql
        self.pre_func = pre_func
        self.post_func = post_func

# variables used to test pipeline (TODO: remove these once the pipeline is working)
obj = 'SDSSJ1442'
frames = '1'
obj_kwargs = {
    'nights': ['2013-05-29'],
    
}

# Set defaults
dec_path = '/media/data-beta/users/fmooleka/decam'
connection = 'sqlite:////media/data-beta/users/fmooleka/decam/2013A-0723/2013A-0723_idx.db'
temp_path = os.path.join(dec_path, 'temp')
build_path = '/media/data-beta/users/fmooleka/astromatic/build'
config_path = os.path.join(dec_path, 'config')
cat_path = os.path.join(dec_path, 'catalogs')
stack_path = os.path.join(dec_path, '2013A-0723', 'mystacks')

# Sextractor parameters
sex_cmd = os.path.join(build_path, 'sextractor/bin/sex')
sex_path = os.path.join(dec_path, 'sex')
# SCAMP parameters
scamp_cmd = sex_cmd.replace('sextractor', 'scamp').replace('sex', 'scamp')
scamp_path = os.path.join(dec_path, 'scamp')
# SWarp parameters
swarp_cmd = scamp_cmd.replace('scamp', 'swarp')
swarp_path = os.path.join(dec_path, 'swarp')
# PSFex parameters
psfex_cmd = scamp_cmd.replace('scamp', 'psfex')
psfex_path = os.path.join(dec_path, 'psfex')

# connect to database
engine = create_engine(connection)

objects = pandas.read_sql("select * from decam_obs", engine)
o = objects['object'].str.split('-').apply(pandas.Series)
print('All fields:\n', o.sort(0)[0].unique())
o = o[(o[0].str.startswith('F')) | (o[0].str.startswith('SDSS'))].sort(0)[0].unique()

# Temporary fix to only run a single objct
o = [obj]
sex_kwargs = {
    'code': 'SExtractor',
    'temp_path': temp_path,
    'config': {
        #'CATALOG_TYPE': 'FITS_1.0', # Temporary, to check output catalog
        'FILTER_NAME': os.path.join(config_path, 'gauss_5.0_9x9.conv')
    },
    'cmd': sex_cmd
}
scamp_kwargs = {
    'cmd': scamp_cmd,
    'code': 'SCAMP',
    'temp_path': temp_path,
    'config': {},
    'config_file': os.path.join(config_path, 'decam.scamp')
}
swarp_kwargs = {
    'cmd': swarp_cmd,
    'code': 'SWarp',
    'temp_path': temp_path,
    'config': {
        'RESAMPLE_DIR': temp_path
    },
    'config_file': os.path.join(config_path, 'decam.swarp')
}
psfex_kwargs = {
    'cmd': psfex_cmd,
    'code': 'PSFEx',
    'temp_path': temp_path,
    'config': {},
    'config_file': os.path.join(config_path, 'decam.psfex')
}
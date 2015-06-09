# Copyright 2015 Fred Moolekamp
# BSD 3-clause license
import os
import pandas
import numpy as np
from sqlalchemy import create_engine
import subprocess
import copy
from collections import OrderedDict
import logging
from six import string_types
import datetime
from astropy.io import fits

from toyz.utils.errors import ToyzError
import astromatic_wrapper as aw
import decamtoyz.utils as utils

logger = logging.getLogger('decamtoyz.pipeline')

class PipelineError(ToyzError):
    pass

class Pipeline(aw.pipeline.Pipeline):
    def __init__(self, idx_connect_str, temp_path, cat_path, 
            stack_path=None, resample_path=None, 
            config_path=None, build_paths={}, log_path=None, create_idx = None,
            default_kwargs={}, steps=[], pipeline_name='', create_paths=False,
            **kwargs):
        """
        Parameters
        ----------
        idx_connect_str: str
            sqlalchemy connection string to decam index database
        temp_path: str
            path to store temporary files
        cat_path: str
            path to save final catalogs
        stack_path: str
            path to save stacked images
        resample_path: str
            path to save resampled images
        config_path: str
            path to check for astromatic config files
                * defaults to decamtoyz/defaults
        build_paths: dict
            paths to astromatic builds
                * Not needed if the codes were installed system wide (ex. 'sex' runs SExtractor)
                * Keys are commands for astromatic packages ('sex', 'scamp', 'swarp', 'psfex')
                * Values are the paths to the build for the given key
        log_path: str
            path to save astromatic xml log files
        create_idx: bool
            By default, if the decam index DB cannot be found, the user will
            be prompted to create the index. Setting `create_idx` to `True` overrides 
            this behavior and automatically creates the index. Setting `create_idx` ``False``
            will raise an error
        create_paths: bool (optional)
            Whether or not to automatically create paths that do not exist.
        pipeline_name: str
            Name of the pipeline
        kwargs: dict
            Additional keyword arguments that might be used in a custom pipeline.
        """
        if idx_connect_str.startswith('sqlite'):
            if not os.path.isfile(idx_connect_str[10:]):
                logger.info('path', idx_connect_str[10:])
                if create_idx is not None:
                    if create_idx:
                        if not aw.utils.utils.get_bool(
                                "DECam file index does not exist, create it now? ('y'/'n')"):
                            raise PipelineError("Unable to locate DECam file index")
                    else:
                         raise PipelineError("Unable to locate DECam file index")
                import decamtoyz.index as index
                recursive = aw.utils.utils.get_bool(
                    "Search '{0}' recursively for images? ('y'/'n')")
                index.build_idx(img_path, idx_connect_str, True, recursive, True)
        self.idx = create_engine(idx_connect_str)
        self.idx_connect_str = idx_connect_str
        self.temp_path = temp_path
        self.cat_path = cat_path
        self.stack_path = stack_path
        self.resample_path = resample_path
        self.build_paths = build_paths
        self.default_kwargs = default_kwargs
        # IF the user doesn't specify a path for config files, use the default decam config files
        if config_path is None:
            from decam import root
            self.config_path = os.path.join(root, 'default')
        else:
            self.config_path = config_path
        
        # If any of the specified paths don't exist, give the user the option to create them
        utils.check_path('temp_path', self.temp_path)
        utils.check_path('cat_path', self.cat_path)
        utils.check_path('stack_path', self.stack_path)
        
        # If the user specified a set of steps for the pipeline, add them here
        self.steps = steps
        self.next_id = 0
        
        # Set the time that the pipeline was created and create a directory
        # for log files
        self.name = pipeline_name
        self.run_date = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.log_path = log_path
        aw.utils.utils.create_paths([self.log_path])
    
    def run_swarp_old(self, step_id, filenames, stack_filename=None, api_kwargs={}, 
            frames=None, run_type='both'):
        """
        Run SWARP with a specified set of parameters
        
        Parameters
        ----------
        step_id: str
            Unique identifier for the current step in the pipeline
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
        logger.info('filenames:', filenames)
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
        if 'XML_NAME' in api_kwargs['config']:
            xml_name = api_kwargs['config']['XML_NAME']
        else:
            xml_name = None
        if run_type=='both' or run_type=='resample':
            # Resample images as specified by WCS keywords in their headers
            logger.info("Create resampled images")
            kwargs = copy.deepcopy(api_kwargs)
            kwargs['config']['COMBINE'] = 'N'
            if xml_name is not None:
                kwargs['config']['XML_NAME'] = xml_name.replace('.xml', '-resamp.xml')
            swarp = api.Astromatic(**kwargs)
            result = swarp.run(filenames)
            if result['status']!='success':
                raise PipelineError("Error running SWARP")
        if run_type=='both' or run_type=='stack':
            logger.info('Creating stack for each CCD')
            if stack_filename is None:
                raise PipelineError("Must include a stack_filename to stack a set of images")
            kwargs = copy.deepcopy(api_kwargs)
            kwargs['config']['RESAMPLE'] = 'N'
            if frames is None:
                hdulist = fits.open(filenames[0])
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
                kwargs['config']['WEIGHT_SUFFIX'] = '.weight.fits'
                if xml_name is not None:
                    kwargs['config']['XML_NAME'] = xml_name.replace(
                        '.xml', '-stack-{0}.xml'.format(frame))
                swarp = api.Astromatic(**kwargs)
                result = swarp.run(resamp_names)
                if result['status']!='success':
                    raise PipelineError("Error running SWARP")
            
            # Combine the frame stacks into a single stacked image
            logger.info("Combining into single stack")
            primary = fits.PrimaryHDU()
            stack = [primary]
            weights = [primary]
            for frame in frames:
                stack_frame = os.path.join(kwargs['temp_path'], 
                    os.path.basename(stack_filename).replace('.fits', 
                    '-{0:04d}.stack.fits'.format(frame)))
                weight_frame = stack_frame.replace('.fits', '.weight.fits')
                hdulist = fits.open(stack_frame)
                stack.append(fits.ImageHDU(hdulist[0].data,hdulist[0].header))
                hdulist.close()
                hdulist = fits.open(weight_frame)
                weights.append(fits.ImageHDU(hdulist[0].data,hdulist[0].header))
            weight_name = stack_filename.replace('.fits', '.wtmap.fits')
            stack = fits.HDUList(stack)
            stack.writeto(stack_filename, clobber=True)
            stack.close()
            weights = fits.HDUList(weights)
            weights.writeto(weight_name, clobber=True)
            weights.close()
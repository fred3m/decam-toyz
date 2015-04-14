"""
Configuration file required for all new Toyz
"""
from __future__ import division,print_function
import os

##################
# Custom Methods #
##################

# Root directory of the config.py file
root = os.path.abspath(os.path.join(os.path.dirname(__file__)))

####################
# Required methods #
####################

# List of static and template paths
static_paths = [os.path.join(root, 'static')]
template_paths = [os.path.join(root, 'templates')]

# Tiles added by your toy
workspace_tiles = {}

# Urls to add to the 'Toyz' tab on the home page. The keys are the text that will appear and the
# values are the urls
toyz_urls = {}

# If any tornado.RequestHandler templates have parameters, special functions to render
# the page must be defined here
render_functions = {}
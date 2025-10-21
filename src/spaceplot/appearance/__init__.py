import importlib.resources as resources

from ..resources.font_loader import register_fonts

# register fonts supplied with package
path = resources.files('spaceplot.resources.fonts')
for cont in path.iterdir():
    register_fonts(cont)

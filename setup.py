from distutils.core import setup

setup(name='faps',
      version='2.1.0',
      description=' Inference of paternity and sibling relationships accounting for uncertainty in genealogy',
      url='http://github.com/ellisztamas/faps',
      author='Tom Ellis',
      author_email='thomas.ellis@gmi.oeaw.ac.at',
      license='MIT',
      packages=['faps'],
      install_requires=['numpy','fastcluster','scipy','ipywidgets', 'pandas'],
      include_package_data=True,
      zip_safe=False)

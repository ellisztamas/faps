from distutils.core import setup

setup(name='faps',
      version='1.2.4',
      description=' Inference of paternity and sibling relationships accounting for uncertainty in genealogy',
      url='http://github.com/ellisztamas/faps',
      author='Tom Ellis',
      author_email='thomas.ellis@ebc.uu.se',
      license='MIT',
      packages=['faps'],
      install_requires=['numpy','pandas','fastcluster','scipy'],
      zip_safe=False)

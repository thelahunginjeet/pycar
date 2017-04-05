#!/usr/bin/env python

from distutils.core import setup,Command

class PyTest(Command):
    user_options =[]
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass
    def run(self):
        import sys,subprocess
        errno = subprocess.call([sys.executable,'tests/runtests.py'])
        raise SystemExit(errno)


setup(name='pycar',
      version='1.0',
      description='Python package for the RAICAR and BICAR algorithms',
      author='Kevin Brown',
      author_email='kevin.s.brown@uconn.edu',
      url='https://github.com/thelahunginjeet/pycar',
      packages=['pycar'],
      package_dir={'pycar': ''},
      package_data={'pycar' : ['tests/runtests.py','tests/test_raicar_pytest.py','tests/test_bicar_pytest.py','tests/bicartestdata.db','tests/icatestsignals.db']},
      install_requires = ['pyica'],
      dependency_links = ['https://github.com/thelahunginjeet/pyica'],
      cmdclass = {'test': PyTest},
      license='BSD-3',
      classifiers = [
          'License :: OSI Approved :: BSD-3 License',
          'Intended Audience :: Developers',
          'Intended Audience :: Scientists',
          'Progamming Language :: Python',
        ],
     )

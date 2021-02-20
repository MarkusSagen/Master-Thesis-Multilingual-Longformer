from setuptools import setup

setup(
   name='e3k',
   version='0.2',
   description='A scaffolding for data science projects.',
   author='Sebastian Callh',
   author_email='sebastian.callh@peltarion.com',
   packages=[
       'lib',
       'tracking'
   ],
   install_requires=[
       'torch >= 1.5, < 2.0'
   ]
)

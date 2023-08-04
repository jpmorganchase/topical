from setuptools import setup
from setuptools import find_packages
import subprocess
import os

setup(
    name='senatus-code-s4',
    version='0.1',
    packages=find_packages(exclude=(['test*', 'tmp*'])),
    url='https://bitbucketdc-cluster07.jpmchase.net/projects/ML4CODE/repos/senatus-code-s4',
    license='',
    author='N751360',
    author_email='shaltiel.eloul@jpmchase.com',
    description='senatus-code-s4',
    include_package_data=True
)
print(find_packages(exclude=(['test*', 'tmp*'])))
# Downloading all the models by the differents tasks using a proxy
subprocess.run("git config --global http.proxy http://proxy.jpmchase.net:10443",shell=True)
subprocess.run('git lfs install',shell=True)  # mac users need to install using homebrew.

# Downloading Microsoft's GraphCodeBERT
st=os.path.normpath(os.path.join('Topical'))
subprocess.run("git -C {st} clone https://huggingface.co/microsoft/graphcodebert-base".format(st=st),shell=True)

st=os.path.join('Topical', 'graphcodebert-base', '.git')

subprocess.run("rm -rf {st}".format(st=st),shell=True)
# Downloading DistilBERT english base
st=os.path.normpath(os.path.join('Topical'))
subprocess.run('git -C {st} clone https://huggingface.co/distilbert-base-uncased'.format(st=st),shell=True)
st=os.path.join('Topical','distilbert-base-uncased', '.git')
subprocess.run("rm -rf {st}".format(st=st),shell=True)
subprocess.run("git config --global --unset-all http.proxy",shell=True)
st=os.path.join('examples', 'dataset.zip')
subprocess.run("tar -xvzf {st}".format(st=st),shell=True)

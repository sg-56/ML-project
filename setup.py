from setuptools import setup,find_packages
from typing import List
def get_requirements(file_path:str) -> List[str]:
    '''
    This function will return the list of requiremtnts
    '''
    hypen_e = '-e .'
    requiremnents = []
    with open('requirements.txt') as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("/n","") for req in requirements]
        if hypen_e in requirements:
            requirements.remove(hypen_e)

    return requirements    





  
setup(
    name='ML-Project',
    version='0.1',
    description='Setup Package File for ML project',
    author='Shiva Shankar',
    author_email='shivashankar944@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
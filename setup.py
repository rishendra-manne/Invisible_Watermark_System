from setuptools import find_packages,setup

def get_requirements(file_path):
    requirements=[]
    remover='-e.'
    with open(file_path,'r') as file:
        requirements=file.readlines()
        requirements=[req.replace("/n","") for req in requirements]
        if remover in requirements:
            requirements.remove(remover)
    return requirements

setup(
    name='Invisible_Watermark_System',
    version='0.0.1',
    author='rishi',
    author_email='mrishe6@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')

)
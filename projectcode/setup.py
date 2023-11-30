from setuptools import find_packages, setup
from glob import glob

package_name = 'projectcode'

#data_files=[
    #    ('share/ament_index/resource_index/packages',
    #        ['resource/' + package_name]),
    #    ('share/' + package_name, ['package.xml']),
    #    ('share/' + package_name + '/launch', glob('launch/*')),
    #    ('share/' + package_name + '/urdf',   glob('urdf/*')),
    #],

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*')),
        ('share/' + package_name + '/urdf',   glob('urdf/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='robot',
    maintainer_email='robot@todo.todo',
    description='The 133a Project Code',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'proj = projectcode.proj:main',
            'proj_right_legs_move = projectcode.proj_right_legs_move:main',
            'proj_toe_on_ground = projectcode.proj_toe_on_ground:main',
        ],
    },
)

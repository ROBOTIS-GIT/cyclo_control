"""Setup metadata for the motion_controller_ros_py package."""

from setuptools import find_packages, setup

package_name = 'motion_controller_ros_py'
authors_info = [
    ('Hyunwoo Nam', 'nhw@robotis.com'),
]
authors = ', '.join(author for author, _ in authors_info)
author_emails = ', '.join(email for _, email in authors_info)
setup(
    name=package_name,
    version='0.0.2',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Pyo',
    maintainer_email='pyo@robotis.com',
    description='Motion Controller ROS 2 Python package',
    license='Apache 2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'retargeting_teleop = scripts.teleop_retargeting:main',
        ],
    },
)

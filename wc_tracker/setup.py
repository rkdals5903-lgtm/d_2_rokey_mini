from setuptools import find_packages, setup

package_name = 'wc_tracker'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sangcheol',
    maintainer_email='tkdcjf3805@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
        	'yolo_webcam = wc_tracker.yolo_webcam:main',
            'webcam_to_map_viewer = wc_tracker.webcam_to_map_viewer:main',
        ],
    },
)

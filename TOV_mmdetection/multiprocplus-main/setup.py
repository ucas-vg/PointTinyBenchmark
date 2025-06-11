from __future__ import print_function
import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="multiprocplus",
    version="0.2",
    author="hui",
    author_email="y19941010@126.com",
    description="a drop-in and easy-use enhancement for multiprcocessing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords='replace for loop by multi processing',
    license='BSD',

    url="https://github.com/yinglang/multiprocplus",
    project_urls={
        'Bug Tracker': 'https://github.com/yinglang/multiprocplus/issues',
        'Source Code': 'https://github.com/yinglang/multiprocplus',
    },

    platforms=['unix', 'linux', 'osx', 'cygwin', 'win32'],
    # package_dir={"multiprocessingplus": "src"},
    packages=setuptools.find_packages(),  # automatic find package imported in project
    # package_data={"multiprocessingplus": ["src/*.py"]},
    py_modules=['multi_process'],
    # meta data
    classifiers=[  # https://pypi.org/classifiers/
        'Development Status :: 4 - Beta ',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        "Operating System :: OS Independent",
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Utilities',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
    # dependency
    install_requires=[
    ],
    python_requires='>=3',
    data_files=[],

    # use_scm_version=True,
    # setup_requires=['setuptools_scm'],
)

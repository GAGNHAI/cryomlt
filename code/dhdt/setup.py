from setuptools import setup, find_packages

setup(
    name = "dhdtPy",
    packages = find_packages(),
    include_package_data = True,
    install_requires = [
        "pandas",
        "geopandas",
        "netCDF4",
        "configobj",
        "scipy",
        "shapely",
        "fiona",
        "progress",
    ],
    data_files=[
        ('share/dhdtPy', ['qsub/qsub_dhdt.sh','qsub/qsub_dhdt_mpi.sh',]),
    ],
    entry_points={
        'console_scripts': [
            'createOutDirs = dhdt.createOutDirs:main',
            'mergeData = dhdt.mergeData:main',
            'processData = dhdt.processData:main',
            'processDataTF = dhdt.processDataTF:main',
            'readData = dhdt.readData:main',
            'readDataTF = dhdt.readDataTF:main',
            'checkData = dhdt.checkDataApp:main',
            'dhdt = dhdt.dhdtApp:main',
            'plotdhdtMem = dhdt.plotMem:main',
        ],
    },
    author = "Magnus Hagdorn",
    description = "process satellite data",
)

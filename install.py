import os
import sys

print("\n ===============================================================\n")

# get operating system

os_name = sys.platform

# check python version

versionLong = sys.version.split(' ')
version = versionLong[0].split('.')

print(" Python version detected     %10s : " % (versionLong[0]), end=' ')

if int(version[0]) == 3 and int(version[1]) >= 6:
    print("   OK")
elif int(version[0]) == 2:
    print(" Please note that pyfem has been migrated to Python 3.x\n")
    print("   Install Pyhon 3.x\n")
else:
    print(" Not OK\n\n   Please install Python 2.6.x or 2.7.x\n")

# check numpy version

try:
    import numpy

    versionLong = numpy.__version__
    version = versionLong.split('.')

    print(" Numpy version detected      %10s : " % (versionLong), end=' ')

    if int(version[0]) == 1 and int(version[1]) >= 6:
        print("   OK")
    else:
        print(" Not OK\n\n   Please install Numpy 1.6.x or higher\n")
except ImportError:
    print(" NumPy not detected                      : Not OK")
    print("\n   Please install install Numpy 1.6.x or higher\n")

# check scipy version

try:
    import scipy

    versionLong = scipy.__version__
    version = versionLong.split('.')

    print(" Scipy version detected      %10s : " % (versionLong), end=' ')

    if int(version[0]) == 0 and int(version[1]) >= 9:
        print("   OK")
    elif int(version[0]) >= 1 and int(version[1]) >= 0:
        print("   OK")
    else:
        print(" Not OK\n\n   Please install Scipy 0.9.x or higher\n")
except ImportError:
    print(" SciPy not detected                     : Not OK")
    print("\n   Please install install Scipy 0.9.x or higher\n")

# check matplotlib

try:
    import matplotlib

    versionLong = matplotlib.__version__
    version = versionLong.split('.')

    print(" Matplotlib version detected %10s : " % (versionLong), end=' ')

    if int(version[0]) >= 1 and int(version[1]) >= 0:
        print("   OK")
    else:
        print(" Not OK\n\n   Please install Matplotlib 1.0.x or higher\n")
except ImportError:
    print(" matplotlib not detected                : Not OK")
    print("\n   Please install Matplotlib 1.0.x or higher\n")

# check meshio version

try:
    import meshio

    versionLong = meshio.__version__
    version = versionLong.split('.')

    print(" Meshio version detected     %10s : " % (versionLong), end=' ')

    if int(version[0]) <= 3:
        print(" Not OK\n\n  Please install Meshio 4.0.0\n")
        print("   pip install meshio==4.0.0\n")
    else:
        print("   OK")
except ImportError:
    print(" Meshio not detected                    : Not OK")
    print("\n   You cannot use gmsh input files!\n")
    print("\n   Please install Meshio 4.0.x or higher")
    print("   or run pyfem with limited functionality. \n")

# check pickle version

try:
    import pickle

    versionLong = pickle.format_version
    version = versionLong.split('.')

    print(" Pickle version detected     %10s : " % (versionLong), end=' ')

    if int(version[0]) >= 4:
        print("   OK")
except ImportError:
    print(" pickle not detected                    : Not OK")
    print("\n   Please install pickle or run ")
    print("   pyfem with limited functionality.\n")

# check h5py version

try:
    import h5py

    versionLong = h5py.__version__
    version = versionLong.split('.')

    print(" H5py version detected       %10s : " % (versionLong), end=' ')

    if int(version[0]) >= 2:
        print("   OK")
except ImportError:
    print(" h5py not detected                    : Not OK")
    print("\n   Please install h5py or run ")
    print("   pyfem with limited functionality.\n")

# get current path

path = os.getcwd()

if os_name[:5] == "linux":

    print("\n LINUX INSTALLATION")
    print(" ===============================================================\n")
    print(" When using a bash shell, add the following line")
    print(" to ~/.bashrc :\n")
    print("   alias  pyfem='python3 " + path + "/pyfem.py'\n")
    print(" When using csh or tcsh add the following lines to")
    print(" ~/.cshrc or ~/.tcshrc :\n")
    print("   alias  pyfem 'python3 " + path + "/pyfem.py'\n")
    print(" ===============================================================\n")
    print("  Installation successful!")
    print("  See the user manual for further instructions.\n\n")

elif os_name[:6] == "darwin":

    print("\n MAC-OS INSTALLATION")
    print(" ===============================================================\n")
    print(" Add the following line to ~/.bashrc :\n")
    # print('   export PYTHONPATH="'+path+'"')
    print("    alias  pyfem='python3 " + path + "/pyfem.py'\n")
    print(" ===============================================================\n")
    print("  Installation successful!")
    print("  See the user manual for further instructions.\n\n")

elif os_name[:3] == "win":

    batfile = open('pyfem.bat', 'w')

    fexec = sys.executable

    if fexec[-5:] == "w.exe":
        fexec = fexec[:-5] + ".exe"

    print(fexec)
    batfile.write(fexec + ' ' + path + '\\pyfem.py %*')

    batfile.close()

    print("\n WINDOWS INSTALLATION")
    print(" ===============================================================\n")
    print(" ===============================================================\n")
    print("  Installation successful!")
    print("  See the user manual for instructions.\n\n")

else:
    print("Operating system ", os_name, " not known.")

input("  Press Enter to continue...")

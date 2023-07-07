# -*- coding: mbcs -*-
#
# Abaqus/CAE Release 2020 replay file
# Internal Version: 2019_09_14-01.49.31 163176
# Run by SunJi on Fri Jul  7 13:19:06 2023
#

# from driverUtils import executeOnCaeGraphicsStartup
# executeOnCaeGraphicsStartup()
#: Executing "onCaeGraphicsStartup()" in the site directory ...
from abaqus import *
from abaqusConstants import *
session.Viewport(name='Viewport: 1', origin=(0.0, 0.0), width=210.815628051758, 
    height=200.66667175293)
session.viewports['Viewport: 1'].makeCurrent()
session.viewports['Viewport: 1'].maximize()
from caeModules import *
from driverUtils import executeOnCaeStartup
executeOnCaeStartup()
openMdb('rectangle_crack.cae')
#: The model database "F:\Github\pyfem\examples\mechanical_phase\rectange_crack\rectangle_crack.cae" has been opened.
session.viewports['Viewport: 1'].setValues(displayedObject=None)
session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(
    referenceRepresentation=ON)
p = mdb.models['Model-1'].parts['Part-1']
session.viewports['Viewport: 1'].setValues(displayedObject=p)
a = mdb.models['Model-1'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(
    optimizationTasks=OFF, geometricRestrictions=OFF, stopConditions=OFF)
p1 = mdb.models['Model-1'].parts['Part-1']
session.viewports['Viewport: 1'].setValues(displayedObject=p1)
p1 = mdb.models['Model-1'].parts['Part-1']
session.viewports['Viewport: 1'].setValues(displayedObject=p1)
del mdb.models['Model-1'].parts['Part-1'].sets['Set-X0Y0']
session.viewports['Viewport: 1'].partDisplay.setValues(mesh=ON)
p = mdb.models['Model-1'].parts['Part-1']
n = p.nodes
nodes = n.getSequenceFromMask(mask=('[#8 ]', ), )
p.Set(nodes=nodes, name='Set-X0Y0')
#: The set 'Set-X0Y0' has been created (1 node).
session.viewports['Viewport: 1'].partDisplay.setValues(mesh=OFF)
a = mdb.models['Model-1'].rootAssembly
a.regenerate()
a = mdb.models['Model-1'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
mdb.jobs['rectangle_crack'].writeInput(consistencyChecking=OFF)
#: The job input file has been written to "rectangle_crack.inp".
mdb.save()
#: The model database has been saved to "F:\Github\pyfem\examples\mechanical_phase\rectange_crack\rectangle_crack.cae".

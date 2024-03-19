import logging
import os
from typing import Annotated, Optional

import vtk
import qt
import ctk

import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)

from packaging import version

from slicer import vtkMRMLScalarVolumeNode

import csv
import numpy as np

import pickle
import json


import copy

from pathlib import Path

import manifolds

from StatsModel import LinearizedGeodesicPolynomialRegression_Kendall3D,\
                         MultivariateLinearizedGeodesicPolynomialRegression_Intercept_Kendall3D, \
                         MultivariateLinearizedGeodesicPolynomialRegression_Slope_Kendall3D \
                         
def _setSectionResizeMode(header, *args, **kwargs):
    """ To be compatible with Qt4 and Qt5 """
    if version.parse(qt.Qt.qVersion()) < version.parse("5.0.0"):
        header.setResizeMode(*args, **kwargs)
    else:
        header.setSectionResizeMode(*args, **kwargs)

#
# HGMComputation
#

class HGMComputation(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("HGMComputation")  # TODO: make this more human readable by adding spaces
        # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Shape Analysis")]
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["James Fishbaugh, Ye Han (Kitware Inc.)"]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        # _() function marks text as translatable to other languages
        self.parent.helpText = _("""
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#HGMComputation">module documentation</a>.
""")
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = _("""

""")

        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", registerSampleData)


#
# Register sample data sets in Sample Data module
#

def registerSampleData():
    """
    Add data sets to Sample Data module.
    """
    # It is always recommended to provide sample data for users to make it easy to try the module,
    # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

    import SampleData
    iconsPath = os.path.join(os.path.dirname(__file__), 'Resources/Icons')

    # To ensure that the source code repository remains small (can be downloaded and installed quickly)
    # it is recommended to store data sets that are larger than a few MB in a Github release.


#
# HGMComputationParameterNode
#

@parameterNodeWrapper
class HGMComputationParameterNode:
    """
    The parameters needed by module.
    """

#
# HGMComputationWidget
#

class HGMComputationWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None) -> None:
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None

    def setup(self) -> None:
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath('UI/HGMComputation.ui'))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = HGMComputationLogic()

        # By default input/output to the home directory 
        homePath = str(Path.home())
        self.ui.inputDirectoryButton.directory = homePath
        self.ui.outDirButton.directory = homePath

        self.shapeTableLoaded = False
        
        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)
        
        # Buttons
        #self.ui.inputDirectoryButton.connect('validInputChanged(bool)', self.inputDirectoryChanged)
        self.ui.csvPathLineEdit.connect('validInputChanged(bool)', self.onCSVPathLineEdit)
        self.ui.applyButton.connect('clicked(bool)', self.onApplyButton)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

    def cleanup(self) -> None:
        """
        Called when the application closes and the module widget is destroyed.
        """
        self.removeObservers()

    def enter(self) -> None:
        """
        Called each time the user opens this module.
        """
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self) -> None:
        """
        Called each time the user opens a different module.
        """
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)

    def onSceneStartClose(self, caller, event) -> None:
        """
        Called just before the scene is closed.
        """
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        """
        Called just after the scene is closed.
        """
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        """
        Ensure parameter node exists and observed.
        """
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

    def setParameterNode(self, inputParameterNode: Optional[HGMComputationParameterNode]) -> None:
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

    def onCSVPathLineEdit(self) -> None:
        self.readCSVFile(self.ui.inputDirectoryButton.directory, self.ui.csvPathLineEdit.currentPath)
        self.ui.applyButton.enabled = True

    def onApplyButton(self) -> None:
        """
        Run processing when user clicks "Apply" button.
        """
        with slicer.util.tryWithErrorDisplay(_("Failed to compute results."), waitCursor=True):

            # Compute output
            self.logic.process(self.subjIDs, self.shapePaths, self.timepts, self.covariates, self.ui.outDirButton.directory, self.ui.experimentNameLineEdit.text)

    def readCSVFile(self, inputDirectory, pathToCSV):
        
        self.shapePaths = []
        self.timepts = []
        self.covariates = []
        self.subjIDs = []

        with open(pathToCSV) as csvfile:
            allRows = list(csv.reader(csvfile, delimiter=',', quotechar='|'))
            headers = allRows[0]

            # Set the table headers
            self.ui.inputShapesTable.setColumnCount(len(headers))
            self.ui.inputShapesTable.setHorizontalHeaderLabels(headers)

            horizontalHeader = self.ui.inputShapesTable.horizontalHeader()
            horizontalHeader.setStretchLastSection(False)

            _setSectionResizeMode(horizontalHeader, 0, qt.QHeaderView.Stretch)

            nrows = len(allRows)
            self.ui.inputShapesTable.setRowCount(nrows)

            # The number of covariates are the additional headers beyond Input Shape, subject Index, and Time Point
            numCovariates = len(headers)-3
            
            curSubjIndex = 0
            curSubjShapes = []
            curSubjTimepts = []
            curSubjCovariates = []
            
            for i in range(1, nrows):

                curRow = allRows[i]

                # Todo: check to make sure file exists at path

                ### First, handle the data structures 

                # Subject index
                subjectIndex = int(curRow[1])

                # If this is another observation from the same subject
                if (subjectIndex == curSubjIndex):
                    curSubjShapes.append(os.path.join(inputDirectory, curRow[0]))
                    curSubjTimepts.append(float(curRow[2]))
                    curSubjCovariates.append(int(curRow[3]))

                    curID = os.path.splitext(os.path.basename(curRow[0]))[0][0:11]
                    if curID not in self.subjIDs:
                        self.subjIDs.append(curID)
                    
                # Else this is a new subject index, so lets add the previous subject information to the list of shapes
                else:

                    self.shapePaths.append(curSubjShapes)
                    curSubjShapes = []
                    curSubjShapes.append(os.path.join(inputDirectory, curRow[0]))

                    self.timepts.append(curSubjTimepts)
                    curSubjTimepts = []
                    curSubjTimepts.append(float(curRow[2]))

                    self.covariates.append([curSubjCovariates[-1]])
                    curSubjCovariates = []
                    curSubjCovariates.append(int(curRow[3]))

                    curSubjIndex += 1

                ### Next, handle populating the UI table
                
                self.ui.inputShapesTable.setRowCount(i)

                # Vtk shape file
                rootname = os.path.splitext(os.path.basename(curRow[0]))[0]
                labelVTKFile = qt.QLabel(rootname)
                self.ui.inputShapesTable.setCellWidget(i-1, 0, labelVTKFile)
                
                # Subject index
                labelSubjectIndex = qt.QLabel(curRow[1])
                labelSubjectIndex.setAlignment(0x84)
                self.ui.inputShapesTable.setCellWidget(i-1, 1, labelSubjectIndex)

                # Time point
                timepoint_2digits = '%0.2f' %(float(curRow[2]))
                labelTimePoint = qt.QLabel(str(timepoint_2digits))
                labelTimePoint.setAlignment(0x84)
                self.ui.inputShapesTable.setCellWidget(i-1, 2, labelTimePoint)

                # The rest are covariates
                for j in range(3, len(curRow)):

                    # Covariate label
                    labelCovariate = qt.QLabel(curRow[j])
                    labelCovariate.setAlignment(0x84)
                    self.ui.inputShapesTable.setCellWidget(i-1, j, labelCovariate)

        self.shapePaths.append(curSubjShapes)
        self.timepts.append(curSubjTimepts)
        self.covariates.append([curSubjCovariates[-1]])

        print(self.shapePaths)


    #
# HGMComputationLogic
#

class HGMComputationLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self) -> None:
        """
        Called when the logic class is instantiated. Can be used for initializing member variables.
        """
        ScriptedLoadableModuleLogic.__init__(self)

    def getParameterNode(self):
        return HGMComputationParameterNode(super().getParameterNode())

    def pickleParameter(self, parameter, filename):

        with open(filename, "wb") as fp:
            pickle.dump(parameter, fp)

    def process(self, subjIDs, shapePaths, timepts, covariates, outDir, experimentName) -> None:
        """
        Run the process
        :param 
        """

        # Maybe do some error checking here?        

        import time
        startTime = time.time()
        logging.info('Processing started')

        # Run the algorithm here

        polydataGroup = vtk.vtkMultiBlockDataGroupFilter()

        # Loop over the number of subjects
        for curSubj in range(0, len(shapePaths)):

            curPaths = shapePaths[curSubj]
            curT = timepts[curSubj]
        
            # Loop over the time points
            for t in range(0, len(curT)): 

                filename = curPaths[t]

                reader = vtk.vtkPolyDataReader()
                reader.SetFileName(filename)
                reader.Update()
                polydata = reader.GetOutput()

                polydataGroup.AddInputData(polydata)

        #########################################
        # Procrustes alignment for all subjects #
        #########################################
        polydataGroup.Update()
        procrustesFilter = vtk.vtkProcrustesAlignmentFilter()
        procrustesFilter.SetInputData(polydataGroup.GetOutput())
        procrustesFilter.GetLandmarkTransform().SetModeToSimilarity()
        procrustesFilter.Update()  
        
        # Per subject reference for triangles so we can connect back up points after model fitting
        referencePolyForTris = []
        # List of lists holding the polydata for each subject and each time point
        allPolyList = []
        # List of lists holding the pts matrix for each subject and each time point
        allPtsList = []

        expDir = os.path.join(outDir, experimentName)
        if (not os.path.exists(expDir)):
            os.makedirs(expDir)

        shapeOutDir = os.path.join(expDir, "shapes")
        if (not os.path.exists(shapeOutDir)):
            os.makedirs(shapeOutDir)

        # Loop over the number of subjects
        for curSubj in range(0, len(shapePaths)):

            curPaths = shapePaths[curSubj]
            curT = timepts[curSubj]

            curPtsList = []
            curPolyList = []

            for t in range(0, len(curT)):
            
                polydataT = vtk.vtkPolyData()
                polydataT.DeepCopy(procrustesFilter.GetOutput().GetBlock(curSubj * len(curT) + t))
                nPoint = polydataT.GetNumberOfPoints()
                pointMatrixT = np.zeros([3, nPoint])
                
                for k in range(nPoint):
                
                    point = polydataT.GetPoint(k)
                    pointMatrixT[0, k] = point[0]
                    pointMatrixT[1, k] = point[1]
                    pointMatrixT[2, k] = point[2]
                
                # Add this to the the current poly list
                curPolyList.append(polydataT)
                # And add it to the current points list
                curPtsList.append(pointMatrixT)
                
                # Save reference polydata for tris once per subject
                if (t==0):
                    referencePolyForTris.append(polydataT)
                
                writer = vtk.vtkPolyDataWriter()
                writer.SetInputData(polydataT)
                outFilename = '%s/%s' %(shapeOutDir, os.path.basename(curPaths[t]))
                writer.SetFileName(outFilename)
                writer.Write()
            
            allPolyList.append(curPolyList)
            allPtsList.append(curPtsList)

        ###########################################################
        # Polynomial regression for individual subjects (level 1) #
        ###########################################################

        # Holds the subject wise intercepts
        p0_list= []
        # Holds the subject wise slopes
        v_list = []
        # Order of polynomial regression
        polynomial_order = 1

        # Loop over the number of subjects
        for curSubj in range(0, len(allPtsList)):

            curT = timepts[curSubj]
            cur_pts_list = allPtsList[curSubj]
            
            kendall_shape_list = []

            for t in range(0, len(curT)):
                
                cur_pts = cur_pts_list[t]
                [dim, num_pts] = cur_pts.shape
                
                kendall_shape_t = manifolds.kendall3D(num_pts)
                kendall_shape_t.SetPoint(copy.deepcopy(cur_pts))
                        
                kendall_shape_list.append(kendall_shape_t)
                
            p0_i, v_i = LinearizedGeodesicPolynomialRegression_Kendall3D(np.array(curT), kendall_shape_list, order=polynomial_order, useFrechetMeanAnchor=False)
            
            p0_list.append(p0_i)
            v_list.append(v_i)

        # Make the output directory
        
        parameterDir = os.path.join(expDir, "model_parameters")

        if (not os.path.exists(parameterDir)):
            os.makedirs(parameterDir)

        # Save the regression parameters
        out_p0_list = "%s/%s_p0_list" %(parameterDir, experimentName)
        out_v_list = "%s/%s_v_list" %(parameterDir, experimentName)

        self.pickleParameter(p0_list, out_p0_list)
        self.pickleParameter(v_list, out_v_list)

        population_p0, population_v, covariates_intercepts = MultivariateLinearizedGeodesicPolynomialRegression_Intercept_Kendall3D(covariates, p0_list, order=1)

        tangent_slope_arr, covariates_slopes = MultivariateLinearizedGeodesicPolynomialRegression_Slope_Kendall3D(
         covariates, v_list, population_p0, p0_list, population_v, covariates_intercepts, level2_order=1)
        
        # Save the group parameters
        out_population_p0 = "%s/%s_population_p0" %(parameterDir, experimentName)
        out_population_v = "%s/%s_population_v" %(parameterDir, experimentName)
        out_tangent_slope_arr = "%s/%s_tangent_slope_arr" %(parameterDir, experimentName)
        out_covariates_slopes = "%s/%s_covariates_slopes" %(parameterDir, experimentName)        

        self.pickleParameter(population_p0, out_population_p0)
        self.pickleParameter(population_v, out_population_v)
        self.pickleParameter(tangent_slope_arr, out_tangent_slope_arr)
        self.pickleParameter(covariates_slopes, out_covariates_slopes)

        # For now, let's save the subject specific trajectories so there is something to visualize 
        regressionOutDir = os.path.join(expDir, "subject_level_regression")
        if (not os.path.exists(regressionOutDir)):
            os.makedirs(regressionOutDir)

        # Read a vtk file for reference polydata (they all have the same connectivity)
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(shapePaths[0][0])
        reader.Update()
        referencePolydata = reader.GetOutput()

        for i in range(0, len(subjIDs)):

            # Output a regression sequence for each subject
            curRegOutDir = os.path.join(regressionOutDir, subjIDs[i])
            
            if (not os.path.exists(curRegOutDir)):
                os.makedirs(curRegOutDir)

            cur_t = timepts[i]

            estimated_time_discretization = np.linspace(np.min(cur_t), np.max(cur_t), 50)
            
            # Slopes of a polynomial model
            slopes = v_list[i]
            intercept = p0_list[i]

            curPolydata = vtk.vtkPolyData()
            curPolydata.DeepCopy(referencePolydata)

            for t in range(0, len(estimated_time_discretization)):
  
                t_t = estimated_time_discretization[t]
                
                # The overall slope
                overall_slope = np.zeros(slopes[0].tVector.shape)
                for s in range(len(slopes)):
                    overall_slope += slopes[s].ScalarMultiply(t_t**(s+1)).tVector
                
                # Create a tangent vector object starting from the intercept
                v_tangent = manifolds.kendall3D_tVec(intercept.nPt)
                # Set the tangent vector to our overall slope
                v_tangent.SetTangentVector(overall_slope)

                # Shoot along the geodesic
                pi = intercept.ExponentialMap(v_tangent)
                
                pts_mat = pi.GetPoint()
                [dim, num_pts] = pts_mat.shape
                
                vtk_points_t = vtk.vtkPoints()
                
                for n in range(0, num_pts):
                
                    vtk_points_t.InsertNextPoint(pts_mat[0,n], pts_mat[1,n], pts_mat[2,n])
                
                curPolydata.SetPoints(vtk_points_t)
                
                writer = vtk.vtkPolyDataWriter()
                writer.SetInputData(curPolydata)
                reg_out_filename = '%s/%s_regression_%0.2d.vtk' %(curRegOutDir, subjIDs[i], t)
                writer.SetFileName(reg_out_filename)
                writer.Write()
 

        # Let's summarize the HGM with a json file
        experiment_dict = {
            "experiment_name": experimentName,
            "shape_directory": shapeOutDir,
            "model_parameter_directory": parameterDir
        }

        experiment_json_out = '%s/%s.json' %(expDir, experimentName)

        with open(experiment_json_out, "w") as jsonof:
            json.dump(experiment_dict, jsonof)

        # # Let's try showing a chart of the ages
        # num_subjects = len(shapePaths)
        # all_tables = []
        
        # for i in range(0, num_subjects):

        #     curTable = vtk.vtkTable()
            
        #     curT = timepts[curSubj]

        #     num_samples = len(curT)

        #     xAxisVTK = vtk.vtkIntArray()
        #     xAxisVTK.SetName("Age")
        #     curTable.AddColumn(xAxisVTK)
        #     yAxisVTK = vtk.vtkIntArray()
        #     yAxisVTK.SetName("Subject index")
        #     curTable.AddColumn(yAxisVTK)

        #     curTable.SetNumberOfRows(num_samples)

        #     for j in range(0, num_samples):
        #         curTable.SetValue(i, 0, curT[j])
        #         curTable.SetValue(i, 1, i)

        stopTime = time.time()
        logging.info(f'Processing completed in {stopTime-startTime:.2f} seconds')


#
# HGMComputationTest
#

class HGMComputationTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """ Do whatever is needed to reset the state - typically a scene clear will be enough.
        """
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here.
        """
        self.setUp()
        self.test_HGMComputation1()

    def test_HGMComputation1(self):
        """ Ideally you should have several levels of tests.  At the lowest level
        tests should exercise the functionality of the logic with different inputs
        (both valid and invalid).  At higher levels your tests should emulate the
        way the user would interact with your code and confirm that it still works
        the way you intended.
        One of the most important features of the tests is that it should alert other
        developers when their changes will have an impact on the behavior of your
        module.  For example, if a developer removes a feature that you depend on,
        your test should break so they know that the feature is needed.
        """

        self.delayDisplay("Starting the test")
       

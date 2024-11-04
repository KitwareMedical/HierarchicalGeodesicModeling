import logging
import os
import pathlib
from typing import Annotated, Optional
from multiprocessing import Pool

import vtk
import qt
import ctk

import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin

from packaging import version

import csv
import numpy as np

import pickle
import json


import copy

from pathlib import Path

import HGMComputationLib.manifolds as manifolds

from HGMComputationLib.StatsModel import FrechetMean_Kendall3D,\
    LinearizedGeodesicPolynomialRegression_Kendall3D,\
    MultivariateLinearizedGeodesicPolynomialRegression_Intercept_Kendall3D,\
    MultivariateLinearizedGeodesicPolynomialRegression_Slope_Kendall3D

#
# HGMComputation
#
class HGMComputation(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("Hierarchical Geodesic Modeling")
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Shape Analysis")]
        self.parent.dependencies = []
        self.parent.contributors = ["Ye Han, James Fishbaugh (Kitware)"]
        # _() function marks text as translatable to other languages
        self.parent.helpText = _(
            """
            Please see the documentation and example data in the <a href="https://github.com/KitwareMedical/HierarchicalGeodesicModeling">module's github page</a>.
            For more information about our geodesic modeling methodology, please see <a href="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10323213/">our IPMI paper</a>.
            """)
        self.parent.acknowledgementText = _(
            """
            This work was supported by NIH NIBIB awards R01EB021391(PI. Beatriz Paniagua, 
            Shape Analysis Toolbox for Medical Image Computing Projects)
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

    iconsPath = os.path.join(os.path.dirname(__file__), 'Resources/Icons')

    # To ensure that the source code repository remains small (can be downloaded and installed quickly)
    # it is recommended to store data sets that are larger than a few MB in a Github release.

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

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # UI
        self.ui.groupBox_computation.setDisabled(True)
        self.ui.spinBox_subjectModelDegree.setRange(1, 2)
        self.ui.comboBox_filenames.setDisabled(True)
        self.ui.comboBox_subjectIndex.setDisabled(True)
        self.ui.comboBox_timeVariables.setDisabled(True)
        self.ui.comboBox_subjectIdType.addItems(['Prefix', 'Suffix', 'Index'])
        self.ui.spinBox_idLength.setMinimum(1)
        self.ui.spinBox_idLength.setValue(11)
        self.ui.listWidget_covariatesSelection.setSelectionMode(qt.QAbstractItemView.MultiSelection)
        self.ui.spinBox_populationModelDegree.setRange(1, 1)  # todo: higher order for population level model
        self.ui.pushButton_compute.setDisabled(True)

        self.ui.groupBox_export.setDisabled(True)
        self.ui.spinBox_SPV.setMinimum(2)
        self.ui.spinBox_SPV.setMaximum(101)
        self.ui.pushButton_hgmSPVAdd.setEnabled(False)
        self.ui.pushButton_hgmSPVRemove.setEnabled(False)
        self.ui.pushButton_hgmSPVClear.setEnabled(False)
        self.ui.tableWidget_hgmSPV.setEnabled(False)

        # Connections
        self.ui.pushButton_loadData.connect('clicked()', self.onLoadData)
        self.ui.pushButton_compute.connect('clicked()', self.onCompute)
        self.ui.pushButton_export.connect('clicked()', self.onExport)
        self.ui.pushButton_exportSPV.connect('clicked()', self.onExportSPV)
        self.ui.checkBox_hgmSPV.connect('stateChanged(int)', self.onHgmSPVChanged)
        self.ui.pushButton_hgmSPVAdd.connect('clicked()', self.onAddHGM)
        self.ui.pushButton_hgmSPVRemove.connect('clicked()', self.onRemoveHGM)
        self.ui.pushButton_hgmSPVClear.connect('clicked()', self.onClearHGM)
        self.ui.pushButton_loadModel.connect('clicked()', self.onLoadExistingModel)

    def cleanup(self) -> None:
        """
        Called when the application closes and the module widget is destroyed.
        """
        self.removeObservers()
        self.ui.tableWidget_inputShapes.clear()

    def enter(self) -> None:
        """
        Called each time the user opens this module.
        """


    def exit(self) -> None:
        """
        Called each time the user opens a different module.
        """

    def onSceneStartClose(self, caller, event) -> None:
        """
        Called just before the scene is closed.
        """

    def onSceneEndClose(self, caller, event) -> None:
        """
        Called just after the scene is closed.
        """

    def onLoadData(self):
        input_directory = self.ui.DirectoryButton_inputDirectory.directory
        demographics_path = self.ui.PathLineEdit_demographics.currentPath
        max_covariates_time_index = self.logic.readCSVFile(input_directory, demographics_path, self.ui.tableWidget_inputShapes)
        if max_covariates_time_index != 0:
            table_widget = self.ui.tableWidget_inputShapes
            self.ui.comboBox_filenames.addItem(table_widget.horizontalHeaderItem(0).text())
            self.ui.comboBox_subjectIndex.addItem(table_widget.horizontalHeaderItem(1).text())
            self.ui.comboBox_timeVariables.addItem(table_widget.horizontalHeaderItem(2).text())
            self.ui.groupBox_computation.setEnabled(True)
            self.ui.listWidget_covariatesSelection.clear()
            for i in range(3, self.ui.tableWidget_inputShapes.columnCount):
                self.ui.listWidget_covariatesSelection.addItem(table_widget.horizontalHeaderItem(i).text())
            self.ui.spinBox_covariatesTimeIndex.setRange(0, max_covariates_time_index)
            self.ui.pushButton_compute.setDisabled(False)
        else:
            return  # todo: pop window for failing to load file.

    def onCompute(self) -> None:
        """
        Run processing when user clicks "Compute" button.
        """
        with slicer.util.tryWithErrorDisplay(_("Failed to compute results."), waitCursor=True):
            # compute output
            selected_covariates_indices = []
            selected_covariates_names = []
            for item in self.ui.listWidget_covariatesSelection.selectedItems():  # todo: double check index ordering
                selected_covariates_indices.append(self.ui.listWidget_covariatesSelection.row(item))
                selected_covariates_names.append(item.text())

            if self.logic.compute(selected_covariates_indices, selected_covariates_names, self.ui):
                self.ui.groupBox_visualization.setEnabled(True)
                self.ui.groupBox_export.setEnabled(True)

                self.ui.tableWidget_hgmSPV.setColumnCount(len(selected_covariates_names))
                self.ui.tableWidget_hgmSPV.setHorizontalHeaderLabels(selected_covariates_names)
                horizontal_header = self.ui.tableWidget_hgmSPV.horizontalHeader()
                horizontal_header.setStretchLastSection(True)
                self.ui.tableWidget_hgmSPV.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
                self.ui.tableWidget_hgmSPV.setSelectionMode(qt.QAbstractItemView.SingleSelection)

    def onExport(self):
        self.logic.export(self.ui.DirectoryButton_export.directory, self.ui.lineEdit_experimentName.text)

    def onExportSPV(self):
        self.logic.exportSPV(self.ui.DirectoryButton_export.directory,
                             self.ui.lineEdit_experimentName.text,
                             self.ui.spinBox_SPV.value,
                             self.ui.checkBox_meanSPV.isChecked(),
                             self.ui.checkBox_subjectSPV.isChecked(),
                             self.ui.checkBox_hgmSPV.isChecked(),
                             self.ui.tableWidget_hgmSPV)
    def onHgmSPVChanged(self):
        if self.ui.checkBox_hgmSPV.isChecked() is True:
            self.ui.pushButton_hgmSPVAdd.setEnabled(True)
            self.ui.pushButton_hgmSPVRemove.setEnabled(True)
            self.ui.pushButton_hgmSPVClear.setEnabled(True)
            self.ui.tableWidget_hgmSPV.setEnabled(True)
        else:
            self.ui.pushButton_hgmSPVAdd.setEnabled(False)
            self.ui.pushButton_hgmSPVRemove.setEnabled(False)
            self.ui.pushButton_hgmSPVClear.setEnabled(False)
            self.ui.tableWidget_hgmSPV.setEnabled(False)

    def onAddHGM(self):
        self.logic.addHGM(self.ui.tableWidget_hgmSPV)

    def onRemoveHGM(self):
        self.logic.removeHGM(self.ui.tableWidget_hgmSPV)

    def onClearHGM(self):
        self.logic.clearHGM(self.ui.tableWidget_hgmSPV)

    def onLoadExistingModel(self):
        self.logic.loadExistingModel(self.ui.PathLineEdit_existingModel.currentPath,
                                     self.ui.gridLayout_visualizationSliders,
                                     self.ui.comboBox_visualizeModel)

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

        # input
        self.shape_paths = []
        self.subject_indices = []
        self.filenames = []
        self.subject_ids = []
        self.time_points = []
        self.minimum_time_point = None
        self.maximum_time_point = None
        self.covariates = []
        self.polydatas = []

        # computation
        self.p0_list = []
        self.v_list = []
        self.mean_p0 = None
        self.mean_v = []
        self.subject_model_order = None
        self.population_model_order = None
        self.population_p0 = None
        self.population_v = []
        self.tangent_slope_arr = None
        self.maximum_covariates = []
        self.minimum_covariates = []

        # visualization
        self.visualization_nodes = []
        self.visualization_grid_layout = None
        self.visualization_labels = []
        self.visualization_sliders = []
        self.visualization_grid_layout = None
        self.selected_covariates_indices = []
        self.selected_covariates_names = []
        self.visualization_index = None

    def pickleParameter(self, parameter, filename):
        with open(filename, "wb") as fp:
            pickle.dump(parameter, fp)

    def pickleLoad(self, filename):
        with open(filename, 'rb') as fp:
            return pickle.load(fp)

    def cleanup(self):
        # input
        self.shape_paths = []
        self.subject_indices = []
        self.filenames = []
        self.subject_ids = []
        self.time_points = []
        self.minimum_time_point = None
        self.maximum_time_point = None
        self.covariates = []
        self.polydatas = []

        # computation
        self.p0_list = []
        self.v_list = []
        self.mean_p0 = None
        self.mean_v = []
        self.subject_model_order = None
        self.population_model_order = None
        self.population_p0 = None
        self.population_v = None
        self.tangent_slope_arr = None
        self.maximum_covariates = []
        self.minimum_covariates = []

        # visualization
        self.visualization_nodes = []
        self.visualization_grid_layout = None
        self.visualization_labels = []
        self.visualization_sliders = []
        self.visualization_grid_layout = None
        self.selected_covariates_indices = []
        self.selected_covariates_names = []
        self.visualization_index = None

    def readCSVFile(self, input_directory, demographics_path, table_widget):
        self.cleanup()
        with open(demographics_path) as csvfile:
            all_rows = list(csv.reader(csvfile, delimiter=',', quotechar='|'))
            num_rows = len(all_rows)

            current_subject_index = int(all_rows[1][1])
            # todo: currently assuming subject index starts at 0
            self.subject_indices.append(current_subject_index)
            current_subject_shapes = []
            current_subject_time_points = []
            current_subject_covariates = []
            current_subject_filenames = []

            # The number of covariates are the additional headers beyond Input Shape, subject Index, and Time Point
            num_covariates = len(all_rows[0]) - 3

            # Read inputs
            for i in range(1, num_rows):
                row = all_rows[i]
                row_file_path = os.path.join(input_directory, row[0])
                if not os.path.exists(row_file_path):
                    logging.error("File path does not exist: " + row_file_path)
                    return 0

                row_time_point = float(row[2])
                if self.minimum_time_point is None or self.maximum_time_point is None:
                    self.minimum_time_point = row_time_point
                    self.maximum_time_point = row_time_point
                else:
                    if row_time_point < self.minimum_time_point:
                        self.minimum_time_point = row_time_point
                    if row_time_point > self.maximum_time_point:
                        self.maximum_time_point = row_time_point

                row_subject_index = int(row[1])
                # Another observation from the current subject
                if row_subject_index == current_subject_index:
                    current_subject_filenames.append(row[0])
                    current_subject_shapes.append(row_file_path)
                    current_subject_time_points.append(row_time_point)
                    row_covariates = []
                    for j in range(3, 3 + num_covariates):
                        row_covariates.append(float(row[j]))
                    current_subject_covariates.append(row_covariates)

                # New subject
                else:
                    self.filenames.append(current_subject_filenames)
                    self.shape_paths.append(current_subject_shapes)
                    self.time_points.append(current_subject_time_points)
                    self.covariates.append(current_subject_covariates)

                    current_subject_filenames = [row[0]]
                    current_subject_shapes = [row_file_path]
                    current_subject_time_points = [row_time_point]
                    row_covariates = []
                    for j in range(3, 3 + num_covariates):
                        row_covariates.append(float(row[j]))
                    current_subject_covariates = [row_covariates]
                    current_subject_index = row_subject_index
                    self.subject_indices.append(current_subject_index)

            self.filenames.append(current_subject_filenames)
            self.shape_paths.append(current_subject_shapes)
            self.time_points.append(current_subject_time_points)
            self.covariates.append(current_subject_covariates)

            # Populate UI table
            headers = all_rows[0]
            table_widget.setColumnCount(len(headers))
            table_widget.setHorizontalHeaderLabels(headers)
            horizontal_header = table_widget.horizontalHeader()
            horizontal_header.setStretchLastSection(True)

            table_widget.setRowCount(num_rows - 1)
            for i in range(1, num_rows):
                row = all_rows[i]

                # Vtk shape file
                labelVTKFile = qt.QLabel(row[0])
                table_widget.setCellWidget(i - 1, 0, labelVTKFile)

                # subject index
                labelSubjectIndex = qt.QLabel(row[1])
                labelSubjectIndex.setAlignment(qt.Qt.AlignCenter)
                table_widget.setCellWidget(i - 1, 1, labelSubjectIndex)

                # time point
                timepoint_2digits = '%0.2f' % (float(row[2]))
                labelTimePoint = qt.QLabel(str(timepoint_2digits))
                labelTimePoint.setAlignment(qt.Qt.AlignCenter)
                table_widget.setCellWidget(i - 1, 2, labelTimePoint)

                # The rest are covariates
                for j in range(3, len(row)):
                    # covariate label
                    labelCovariate = qt.QLabel(row[j])
                    labelCovariate.setAlignment(qt.Qt.AlignCenter)
                    table_widget.setCellWidget(i - 1, j, labelCovariate)

        table_widget.resizeColumnsToContents()
        return len(self.filenames[0])

    def compute(self, selected_covariates_indices, selected_covariates_names, ui):
        """
        Run the process
        :param 
        """
        import time
        startTime = time.time()
        logging.info('Processing started')

        subject_model_order = ui.spinBox_subjectModelDegree.value
        population_model_order = ui.spinBox_populationModelDegree.value
        grid_layout = ui.gridLayout_visualizationSliders
        visualize_model = ui.comboBox_visualizeModel
        subject_id_type = ui.comboBox_subjectIdType
        id_length = ui.spinBox_idLength.value
        covariates_time_index = ui.spinBox_covariatesTimeIndex.value

        ### preprocessing using partial procrustes alignment for all subjects
        polydataGroup = vtk.vtkMultiBlockDataGroupFilter()
        for subject_index in self.subject_indices:

            current_paths = self.shape_paths[subject_index]
            current_time = self.time_points[subject_index]

            for t in range(0, len(current_time)):

                filename = current_paths[t]
                reader = vtk.vtkPolyDataReader()
                reader.SetFileName(filename)
                reader.Update()
                polydata = reader.GetOutput()
                polydataGroup.AddInputData(polydata)

        polydataGroup.Update()
        procrustesFilter = vtk.vtkProcrustesAlignmentFilter()
        procrustesFilter.SetInputData(polydataGroup.GetOutput())
        procrustesFilter.GetLandmarkTransform().SetModeToSimilarity()
        procrustesFilter.Update()  

        # Per subject reference for triangles so we can connect back up points after model fitting
        referencePolyForTris = []
        # List of lists holding the pts matrix for each subject and each time point
        allPtsList = []

        # Loop over the number of subjects
        for subject_index in self.subject_indices:
            current_time = self.time_points[subject_index]
            current_points_list = []
            current_polydata_list = []

            for t in range(0, len(current_time)):
                polydata_t = vtk.vtkPolyData()
                # note: currently assuming all subject had same number of time points
                polydata_t.DeepCopy(procrustesFilter.GetOutput().GetBlock(subject_index * len(current_time) + t))
                nPoint = polydata_t.GetNumberOfPoints()
                pointMatrixT = np.zeros([3, nPoint])
                
                for k in range(nPoint):
                    point = polydata_t.GetPoint(k)
                    pointMatrixT[0, k] = point[0]
                    pointMatrixT[1, k] = point[1]
                    pointMatrixT[2, k] = point[2]

                current_polydata_list.append(polydata_t)
                current_points_list.append(pointMatrixT)

                # Save reference polydata for tris once per subject
                if (t==0):
                    referencePolyForTris.append(polydata_t)
            allPtsList.append(current_points_list)

        # subject level longitudinal model
        self.p0_list = []
        self.v_list = []
        self.subject_model_order = subject_model_order
        self.population_model_order = population_model_order
        # todo: multiprocessing subject-wise trajectory computation
        for subject_index in self.subject_indices:

            current_time = self.time_points[subject_index]
            current_points_list = allPtsList[subject_index]
            kendall_shape_list = []

            for t in range(0, len(current_time)):
                
                cur_pts = current_points_list[t]
                [dim, num_pts] = cur_pts.shape
                kendall_shape_t = manifolds.kendall3D(num_pts)
                kendall_shape_t.SetPoint(copy.deepcopy(cur_pts))
                kendall_shape_list.append(kendall_shape_t)

            p0_i, v_i = LinearizedGeodesicPolynomialRegression_Kendall3D(np.array(current_time),
                                                                         kendall_shape_list,
                                                                         order=subject_model_order,
                                                                         useFrechetMeanAnchor=False)
            self.p0_list.append(p0_i)
            self.v_list.append(v_i)

        # mean trajectory
        self.mean_p0 = FrechetMean_Kendall3D(self.p0_list)
        num_slopes = len(self.v_list[0])
        [dim, num_pts] = self.mean_p0.GetPoint().shape
        self.mean_v = []
        for i in range(num_slopes):
            self.mean_v.append(manifolds.kendall3D_tVec(num_pts))
        for subject_index in self.subject_indices:
            for i in range(num_slopes):
                self.mean_v[i].tVector += self.p0_list[subject_index].ParallelTranslateAtoB(self.p0_list[subject_index], self.mean_p0, self.v_list[subject_index][i]).tVector
        for i in range(num_slopes):
            self.mean_v[i] = self.mean_v[i].ScalarMultiply(1/len(self.subject_indices))

        # population level covariates model
        if len(selected_covariates_indices) != 0:
            selected_covariates = []
            self.minimum_covariates = []
            self.maximum_covariates = []
            for subject_index in self.subject_indices:
                current_subject_covartiates = []
                for covariates_index in selected_covariates_indices:
                    current_subject_covartiates.append(
                        self.covariates[subject_index][covariates_time_index][covariates_index])
                # record max/min covariates value
                if subject_index == 0:
                    self.minimum_covariates = current_subject_covartiates.copy()
                    self.maximum_covariates = current_subject_covartiates.copy()
                else:
                    for i in range(len(current_subject_covartiates)):
                        if current_subject_covartiates[i] < self.minimum_covariates[i]:
                            self.minimum_covariates[i] = current_subject_covartiates[i]
                        if current_subject_covartiates[i] > self.maximum_covariates[i]:
                            self.maximum_covariates[i] = current_subject_covartiates[i]
                selected_covariates.append(current_subject_covartiates)

            self.population_p0, self.population_v, covariates_intercepts = MultivariateLinearizedGeodesicPolynomialRegression_Intercept_Kendall3D(
                selected_covariates, self.p0_list, order=1)
            self.tangent_slope_arr, covariates_slopes = MultivariateLinearizedGeodesicPolynomialRegression_Slope_Kendall3D(
                selected_covariates, self.v_list, self.population_p0, self.p0_list, self.population_v,
                covariates_intercepts, level2_order=population_model_order)
        else:
            self.population_p0 = None
            self.population_v = None
            self.tangent_slope_arr = None
            logging.info("No covariate is selected to compute hierarchical geodesic model")

        # visualization
        slicer.app.processEvents()
        self.visualization_grid_layout = grid_layout
        self.selected_covariates_indices = selected_covariates_indices
        self.selected_covariates_names = selected_covariates_names

        visualize_model.addItem("Mean")

        if len(selected_covariates_indices) != 0:
            visualize_model.addItem("HGM")

        # todo: extension removal
        self.subject_ids = []
        if subject_id_type.currentText == 'Prefix':
            for subject_filenames in self.filenames:
                if id_length > len(subject_filenames[0]):
                    self.subject_ids.append(subject_filenames[0])
                else:
                    self.subject_ids.append(subject_filenames[0][:id_length])
                visualize_model.addItem(self.subject_ids[-1])
        elif subject_id_type.currentText == 'Suffix':
            for subject_filenames in self.filenames:
                if id_length > len(subject_filenames[0]):
                    self.subject_ids.append(subject_filenames[0])
                else:
                    self.subject_ids.append(subject_filenames[0][(len(subject_filenames[0]) - id_length):])
                visualize_model.addItem(self.subject_ids[-1])
        elif subject_id_type.currentText == 'Index':
            for subject_index in self.subject_indices:
                self.subject_ids.append("subject " + str(subject_index))
                visualize_model.addItem(self.subject_ids[-1])
        else:
            logging.error('Wrong subject id type.')
        visualize_model.currentIndexChanged.connect(self.visualizationSubjectChanged)

        # mean shape
        label = qt.QLabel()
        label.setText("time")
        label.setAlignment(qt.Qt.AlignCenter)
        self.visualization_labels.append(label)
        self.visualization_grid_layout.addWidget(label, 1, 0)

        slider = ctk.ctkSliderWidget()
        slider.minimum = self.minimum_time_point
        slider.maximum = self.maximum_time_point
        slider.value = self.minimum_time_point
        # todo: step size based on input time scale
        slider.decimals = 2
        slider.singleStep = 0.01
        slider.pageStep = 0.2
        self.visualization_sliders.append(slider)
        self.visualization_grid_layout.addWidget(slider, 1, 1)
        slider.valueChanged.connect(self.visualizationSliderChanged)

        # visualization node
        self.visualization_index = 0

        # mean shape
        mean_polydata = vtk.vtkPolyData()
        mean_polydata.DeepCopy(referencePolyForTris[0])
        pts_mat = self.mean_p0.GetPoint()
        [dim, num_pts] = pts_mat.shape

        vtk_points_t = vtk.vtkPoints()
        for n in range(0, num_pts):
            vtk_points_t.InsertNextPoint(pts_mat[0, n], pts_mat[1, n], pts_mat[2, n])
        mean_polydata.SetPoints(vtk_points_t)
        self.polydatas.append(mean_polydata)

        mean_shape_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "mean shape")
        mean_shape_node.CreateDefaultDisplayNodes()
        mean_shape_node.SetAndObservePolyData(mean_polydata)
        self.visualization_nodes.append(mean_shape_node)

        # HGM shape
        if len(selected_covariates_indices) != 0:
            hgm_polydata = vtk.vtkPolyData()
            hgm_polydata.DeepCopy(referencePolyForTris[0])
            pts_mat = self.population_p0.GetPoint()
            [dim, num_pts] = pts_mat.shape

            vtk_points_t = vtk.vtkPoints()
            for n in range(0, num_pts):
                vtk_points_t.InsertNextPoint(pts_mat[0, n], pts_mat[1, n], pts_mat[2, n])
            hgm_polydata.SetPoints(vtk_points_t)
            self.polydatas.append(hgm_polydata)

            hgm_shape_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "hgm shape")
            hgm_shape_node.CreateDefaultDisplayNodes()
            hgm_shape_node.SetAndObservePolyData(hgm_polydata)
            hgm_shape_node.GetDisplayNode().SetVisibility(0)
            self.visualization_nodes.append(hgm_shape_node)

        # subject shapes
        for i in range(len(referencePolyForTris)):
            subject_i_polydata = vtk.vtkPolyData()
            subject_i_polydata.DeepCopy(referencePolyForTris[i])
            self.polydatas.append(subject_i_polydata)

            subject_i_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", self.subject_ids[i])
            subject_i_node.CreateDefaultDisplayNodes()
            subject_i_node.SetAndObservePolyData(subject_i_polydata)
            subject_i_node.GetDisplayNode().SetVisibility(0)
            self.visualization_nodes.append(subject_i_node)

        stopTime = time.time()
        logging.info(f'Processing completed in {stopTime-startTime:.2f} seconds')

        slicer.util.resetThreeDViews()
        return True

    def visualizationSliderChanged(self):
        if self.visualization_index == 1 and self.population_p0 is not None:
            # hgm
            population_p0 = self.population_p0
            population_v = self.population_v
            tangent_slope_arr = self.tangent_slope_arr
            polydata = self.polydatas[self.visualization_index]
            nManifoldDim = population_p0.nPt

            population_v_final = manifolds.kendall3D_tVec(nManifoldDim)
            population_tVector_final = np.zeros(population_v_final.tVector.shape)
            for i in range(len(self.selected_covariates_names)):
                population_tVector_final += population_v[i].ScalarMultiply(self.visualization_sliders[i+1].value).tVector
            population_v_final.SetTangentVector(population_tVector_final)
            population_p0_final = population_p0.ExponentialMap(population_v_final)

            vo_beta0 = manifolds.kendall3D_tVec(nManifoldDim)
            v0_list = []
            for o in range(self.population_model_order):
                tangent_slope_arr_final = np.zeros(population_v_final.tVector.shape)
                tangent_slope_arr_final += tangent_slope_arr[o][-1].tVector
                for i in range(len(self.selected_covariates_names)):
                    tangent_slope_arr_final += tangent_slope_arr[o][i].ScalarMultiply(self.visualization_sliders[i+1].value).tVector
                vo_beta0.SetTangentVector(tangent_slope_arr_final)
                v0_list.append(population_p0.ParallelTranslateToA(population_p0_final, vo_beta0))

            tVec = np.zeros(population_v_final.tVector.shape)
            for o in range(self.population_model_order):
                tVec += v0_list[o].ScalarMultiply(self.visualization_sliders[0].value**(o+1)).tVector
            v0_normal = manifolds.kendall3D_tVec(nManifoldDim)
            v0_normal.SetTangentVector(tVec)
            p_i = population_p0_final.ExponentialMap(v0_normal)

            pts_mat = p_i.GetPoint()
            [dim, num_pts] = pts_mat.shape

            vtk_points_t = vtk.vtkPoints()
            for n in range(0, num_pts):
                vtk_points_t.InsertNextPoint(pts_mat[0, n], pts_mat[1, n], pts_mat[2, n])
            polydata.SetPoints(vtk_points_t)
        else:
            # mean or subject i
            if self.visualization_index == 0:
                slopes = self.mean_v
                intercept = self.mean_p0
            else:
                if self.population_p0 is not None:
                    slopes = self.v_list[self.visualization_index - 2]
                    intercept = self.p0_list[self.visualization_index - 2]
                    logging.info(self.visualization_index - 2)
                else:
                    slopes = self.v_list[self.visualization_index - 1]
                    intercept = self.p0_list[self.visualization_index - 1]
                    logging.info(self.visualization_index - 1)
            polydata = self.polydatas[self.visualization_index]

            # The overall slope
            overall_slope = np.zeros(slopes[0].tVector.shape)
            t = self.visualization_sliders[0].value
            for s in range(len(slopes)):
                overall_slope += slopes[s].ScalarMultiply(t**(s+1)).tVector

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
                vtk_points_t.InsertNextPoint(pts_mat[0, n], pts_mat[1, n], pts_mat[2, n])
            polydata.SetPoints(vtk_points_t)

    def visualizationSubjectChanged(self, index):

        # module sliders update
        for label in self.visualization_labels:
            self.visualization_grid_layout.removeWidget(label)
            label.deleteLater()
        self.visualization_labels = []
        for slider in self.visualization_sliders:
            self.visualization_grid_layout.removeWidget(slider)
            slider.deleteLater()
        self.visualization_sliders = []

        # visualization node visibility
        for node in self.visualization_nodes:
            node.GetDisplayNode().SetVisibility(0)
        self.visualization_nodes[index].GetDisplayNode().SetVisibility(1)

        label = qt.QLabel()
        label.setText("Time")
        label.setAlignment(qt.Qt.AlignCenter)
        self.visualization_labels.append(label)
        self.visualization_grid_layout.addWidget(label, 1, 0)

        slider = ctk.ctkSliderWidget()
        slider.minimum = self.minimum_time_point
        slider.maximum = self.maximum_time_point
        slider.value = self.minimum_time_point
        slider.decimals = 2
        slider.singleStep = 0.01
        slider.pageStep = 0.2
        self.visualization_sliders.append(slider)
        self.visualization_grid_layout.addWidget(slider, 1, 1)
        slider.valueChanged.connect(self.visualizationSliderChanged)

        if index == 1 and self.population_p0 is not None:
            for i in range(len(self.selected_covariates_names)):
                # Create the variance ratio label
                label = qt.QLabel()
                label.setText(self.selected_covariates_names[i])
                label.setAlignment(qt.Qt.AlignCenter)
                self.visualization_labels.append(label)
                self.visualization_grid_layout.addWidget(label, i + 2, 0)

                # Create the slider
                slider = ctk.ctkSliderWidget()
                # todo: step size based on covariate range
                slider.minimum = self.minimum_covariates[i]
                slider.maximum = self.maximum_covariates[i]
                slider.value = self.minimum_covariates[i]
                slider.decimals = 2
                slider.singleStep = 0.01
                slider.pageStep = 0.2
                self.visualization_sliders.append(slider)
                self.visualization_grid_layout.addWidget(slider, i + 2, 1)

                # Connect
                slider.valueChanged.connect(self.visualizationSliderChanged)

        self.visualization_index = index
        self.visualizationSliderChanged()

    def export(self, output_directory, experiment_name):

        experiment_directory = os.path.join(output_directory, experiment_name)
        if not os.path.exists(experiment_directory):
            os.makedirs(experiment_directory)

        # reference polydatas
        out_polydata = "%s/%s_polydata.vtk" % (experiment_directory, experiment_name)
        writer = vtk.vtkPolyDataWriter()
        writer.SetInputData(self.polydatas[0])
        writer.SetFileName(out_polydata)
        writer.Write()

        # subject model parameters
        out_p0_list = "%s/%s_p0_list" % (experiment_directory, experiment_name)
        out_v_list = "%s/%s_v_list" % (experiment_directory, experiment_name)
        self.pickleParameter(self.p0_list, out_p0_list)
        self.pickleParameter(self.v_list, out_v_list)

        # mean shape parameters
        out_mean_p0 = "%s/%s_mean_p0" % (experiment_directory, experiment_name)
        out_mean_v = "%s/%s_mean_v" % (experiment_directory, experiment_name)
        self.pickleParameter(self.mean_p0, out_mean_p0)
        self.pickleParameter(self.mean_v, out_mean_v)

        # HGM parameters
        if self.population_p0 is not None:
            out_population_p0 = "%s/%s_population_p0" % (experiment_directory, experiment_name)
            out_population_v = "%s/%s_population_v" % (experiment_directory, experiment_name)
            out_tangent_slope_arr = "%s/%s_tangent_slope_arr" % (experiment_directory, experiment_name)

            self.pickleParameter(self.population_p0, out_population_p0)
            self.pickleParameter(self.population_v, out_population_v)
            self.pickleParameter(self.tangent_slope_arr, out_tangent_slope_arr)

        # summarize information all with a json file
        experiment_dict = {
            "experiment_name": experiment_name,
            "include_HGM": self.population_p0 is not None,
            "subject_ids": self.subject_ids,
            "minimum_time_point": self.minimum_time_point,
            "maximum_time_point": self.maximum_time_point,
            "selected_covariate_names": self.selected_covariates_names,
            "minimum_covariates": self.minimum_covariates,
            "maximum_covariates": self.maximum_covariates,
            "subject_model_order": self.subject_model_order,
            "population_model_order": self.population_model_order
        }
        experiment_json_out = '%s/%s.json' % (experiment_directory, experiment_name)
        with open(experiment_json_out, "w") as jsonof:
            json.dump(experiment_dict, jsonof)

    def exportSPV(self, output_directory, experiment_name, spinbox_spv, mean_spv, subject_spv, hgm_spv, hgm_table):
        experiment_directory = os.path.join(output_directory, experiment_name)
        if not os.path.exists(experiment_directory):
            os.makedirs(experiment_directory)

        time_interval = (self.maximum_time_point - self.minimum_time_point) / (spinbox_spv - 1)
        csv_list = []

        csv_header = []
        for step in range(spinbox_spv):
            csv_header.append("time %.2f" % (self.minimum_time_point + time_interval * step))
        csv_list.append(csv_header)

        if mean_spv is True:
            csv_list_mean = []
            slopes = self.mean_v
            intercept = self.mean_p0
            for step in range(spinbox_spv):
                time_point = self.minimum_time_point + time_interval * step
                mean_polydata = vtk.vtkPolyData()
                mean_polydata.DeepCopy(self.polydatas[0])

                overall_slope = np.zeros(slopes[0].tVector.shape)
                for s in range(len(slopes)):
                    overall_slope += slopes[s].ScalarMultiply(time_point ** (s + 1)).tVector
                v_tangent = manifolds.kendall3D_tVec(intercept.nPt)
                v_tangent.SetTangentVector(overall_slope)
                pi = intercept.ExponentialMap(v_tangent)
                pts_mat = pi.GetPoint()
                [dim, num_pts] = pts_mat.shape
                vtk_points_t = vtk.vtkPoints()
                for n in range(0, num_pts):
                    vtk_points_t.InsertNextPoint(pts_mat[0, n], pts_mat[1, n], pts_mat[2, n])
                mean_polydata.SetPoints(vtk_points_t)

                if not os.path.exists("%s/mean/" % experiment_directory):
                    os.makedirs("%s/mean/" % experiment_directory)
                out_polydata = "%s/mean/mean_t%.2f.vtk" % (experiment_directory, time_point)
                writer = vtk.vtkPolyDataWriter()
                writer.SetInputData(mean_polydata)
                writer.SetFileName(out_polydata)
                writer.Write()

                csv_list_mean.append(out_polydata)

            csv_list.append(csv_list_mean)

        if subject_spv is True:

            for i in range(len(self.subject_ids)):
                slopes = self.v_list[i]
                intercept = self.p0_list[i]
                csv_list_subject = []

                for step in range(spinbox_spv):
                    time_point = self.minimum_time_point + time_interval * step
                    subject_polydata = vtk.vtkPolyData()
                    subject_polydata.DeepCopy(self.polydatas[0])

                    overall_slope = np.zeros(slopes[0].tVector.shape)
                    for s in range(len(slopes)):
                        overall_slope += slopes[s].ScalarMultiply(time_point ** (s + 1)).tVector
                    v_tangent = manifolds.kendall3D_tVec(intercept.nPt)
                    v_tangent.SetTangentVector(overall_slope)
                    pi = intercept.ExponentialMap(v_tangent)
                    pts_mat = pi.GetPoint()
                    [dim, num_pts] = pts_mat.shape
                    vtk_points_t = vtk.vtkPoints()
                    for n in range(0, num_pts):
                        vtk_points_t.InsertNextPoint(pts_mat[0, n], pts_mat[1, n], pts_mat[2, n])
                    subject_polydata.SetPoints(vtk_points_t)

                    if not os.path.exists("%s/%s/" % (experiment_directory, self.subject_ids[i])):
                        os.makedirs("%s/%s/" % (experiment_directory, self.subject_ids[i]))
                    out_polydata = "%s/%s/%s_t%.2f.vtk" % (experiment_directory, self.subject_ids[i], self.subject_ids[i], time_point)
                    writer = vtk.vtkPolyDataWriter()
                    writer.SetInputData(subject_polydata)
                    writer.SetFileName(out_polydata)
                    writer.Write()

                    csv_list_subject.append(out_polydata)

                csv_list.append(csv_list_subject)

        if hgm_spv is True:

            population_p0 = self.population_p0
            population_v = self.population_v
            tangent_slope_arr = self.tangent_slope_arr
            nManifoldDim = population_p0.nPt

            for row in range(hgm_table.rowCount):
                csv_list_hgm = []
                covariates_list = []
                for column in range(hgm_table.columnCount):
                    covariates_list.append(hgm_table.cellWidget(row, column).value)

                population_v_final = manifolds.kendall3D_tVec(nManifoldDim)
                population_tVector_final = np.zeros(population_v_final.tVector.shape)
                for i in range(len(self.selected_covariates_names)):
                    population_tVector_final += population_v[i].ScalarMultiply(covariates_list[i]).tVector
                population_v_final.SetTangentVector(population_tVector_final)
                population_p0_final = population_p0.ExponentialMap(population_v_final)

                vo_beta0 = manifolds.kendall3D_tVec(nManifoldDim)
                v0_list = []
                for o in range(self.population_model_order):
                    tangent_slope_arr_final = np.zeros(population_v_final.tVector.shape)
                    tangent_slope_arr_final += tangent_slope_arr[o][-1].tVector
                    for i in range(len(self.selected_covariates_names)):
                        tangent_slope_arr_final += tangent_slope_arr[o][i].ScalarMultiply(covariates_list[i]).tVector
                    vo_beta0.SetTangentVector(tangent_slope_arr_final)
                    v0_list.append(population_p0.ParallelTranslateToA(population_p0_final, vo_beta0))

                for step in range(spinbox_spv):
                    hgm_polydata = vtk.vtkPolyData()
                    hgm_polydata.DeepCopy(self.polydatas[0])
                    time_point = self.minimum_time_point + time_interval * step
                    tVec = np.zeros(population_v_final.tVector.shape)
                    for o in range(self.population_model_order):
                        tVec += v0_list[o].ScalarMultiply(time_point ** (o + 1)).tVector
                    v0_normal = manifolds.kendall3D_tVec(nManifoldDim)
                    v0_normal.SetTangentVector(tVec)
                    p_i = population_p0_final.ExponentialMap(v0_normal)

                    pts_mat = p_i.GetPoint()
                    [dim, num_pts] = pts_mat.shape

                    vtk_points_t = vtk.vtkPoints()
                    for n in range(0, num_pts):
                        vtk_points_t.InsertNextPoint(pts_mat[0, n], pts_mat[1, n], pts_mat[2, n])
                    hgm_polydata.SetPoints(vtk_points_t)

                    covariates_str = ""
                    for i in range(len(self.selected_covariates_names)):
                        covariates_str += "_%s%.2f" % (self.selected_covariates_names[i], covariates_list[i])

                    if not os.path.exists("%s/hgm%s/" % (experiment_directory, covariates_str)):
                        os.makedirs("%s/hgm%s/" % (experiment_directory, covariates_str))
                    out_polydata = "%s/hgm%s/hgm%s_t%.2f.vtk" % (experiment_directory, covariates_str, covariates_str, time_point)
                    writer = vtk.vtkPolyDataWriter()
                    writer.SetInputData(hgm_polydata)
                    writer.SetFileName(out_polydata)
                    writer.Write()

                    csv_list_hgm.append(out_polydata)

                csv_list.append(csv_list_hgm)

        csv_path = "%s/%s_SPV.csv" % (experiment_directory, experiment_name)
        with open(csv_path, "w") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerows(csv_list)

    def addHGM(self, tablewidget):
        n_covariates = len(self.selected_covariates_names)
        tablewidget.insertRow(tablewidget.rowCount)
        for i in range(n_covariates):
            spinbox = qt.QDoubleSpinBox()
            spinbox.setRange(self.minimum_covariates[i], self.maximum_covariates[i])
            tablewidget.setCellWidget(tablewidget.rowCount - 1, i, spinbox)

    def removeHGM(self, tablewidget):
        if tablewidget.selectionModel().hasSelection:
            tablewidget.removeRow(tablewidget.selectionModel().selectedRows()[0].row())

    def clearHGM(self, tablewidget):
        tablewidget.clearContents()
        tablewidget.setRowCount(0)

    def loadExistingModel(self, experiment_json, grid_layout, visualize_model):
        with open(experiment_json, "r") as jsonof:
            data = json.load(jsonof)
            experiment_name = data["experiment_name"]
            include_HGM = data["include_HGM"]
            self.subject_ids = data["subject_ids"]
            self.minimum_time_point = data["minimum_time_point"]
            self.maximum_time_point = data["maximum_time_point"]
            self.selected_covariates_names = data["selected_covariate_names"]
            self.minimum_covariates = data["minimum_covariates"]
            self.maximum_covariates = data["maximum_covariates"]
            self.subject_model_order = data["subject_model_order"]
            self.population_model_order = data["population_model_order"]

            experiment_directory = os.path.dirname(experiment_json)

            # reference polydatas
            in_polydata = "%s/%s_polydata.vtk" % (experiment_directory, experiment_name)
            reader = vtk.vtkPolyDataReader()
            reader.SetFileName(in_polydata)
            reader.Update()
            reference_polydata = reader.GetOutput()

            # subject model parameters
            in_p0_list = "%s/%s_p0_list" % (experiment_directory, experiment_name)
            in_v_list = "%s/%s_v_list" % (experiment_directory, experiment_name)
            self.p0_list = self.pickleLoad(in_p0_list)
            self.v_list = self.pickleLoad(in_v_list)

            # mean shape parameters
            in_mean_p0 = "%s/%s_mean_p0" % (experiment_directory, experiment_name)
            in_mean_v = "%s/%s_mean_v" % (experiment_directory, experiment_name)
            self.mean_p0 = self.pickleLoad(in_mean_p0)
            self.mean_v = self.pickleLoad(in_mean_v)

            # HGM parameters
            if include_HGM is True:
                in_population_p0 = "%s/%s_population_p0" % (experiment_directory, experiment_name)
                in_population_v = "%s/%s_population_v" % (experiment_directory, experiment_name)
                in_tangent_slope_arr = "%s/%s_tangent_slope_arr" % (experiment_directory, experiment_name)
                self.population_p0 = self.pickleLoad(in_population_p0)
                self.population_v = self.pickleLoad(in_population_v)
                self.tangent_slope_arr = self.pickleLoad(in_tangent_slope_arr)

            # visualization
            # visualize model combobox
            visualize_model.clear()
            visualize_model.addItem("Mean")
            if include_HGM:
                visualize_model.addItem("HGM")
            for subject_id in self.subject_ids:
                visualize_model.addItem(subject_id)
            visualize_model.currentIndexChanged.connect(self.visualizationSubjectChanged)

            # module slider
            self.visualization_grid_layout = grid_layout
            for label in self.visualization_labels:
                self.visualization_grid_layout.removeWidget(label)
                label.deleteLater()
            self.visualization_labels = []
            for slider in self.visualization_sliders:
                self.visualization_grid_layout.removeWidget(slider)
                slider.deleteLater()
            self.visualization_sliders = []

            # mean shape
            label = qt.QLabel()
            label.setText("time")
            label.setAlignment(qt.Qt.AlignCenter)
            self.visualization_labels.append(label)
            self.visualization_grid_layout.addWidget(label, 1, 0)

            slider = ctk.ctkSliderWidget()
            slider.minimum = self.minimum_time_point
            slider.maximum = self.maximum_time_point
            slider.value = self.minimum_time_point
            # todo: step size based on input time scale
            slider.decimals = 2
            slider.singleStep = 0.01
            slider.pageStep = 0.2
            self.visualization_sliders.append(slider)
            self.visualization_grid_layout.addWidget(slider, 1, 1)
            slider.valueChanged.connect(self.visualizationSliderChanged)

            # visualization node
            self.visualization_index = 0

            # mean shape
            mean_polydata = vtk.vtkPolyData()
            mean_polydata.DeepCopy(reference_polydata)
            pts_mat = self.mean_p0.GetPoint()
            [dim, num_pts] = pts_mat.shape

            vtk_points_t = vtk.vtkPoints()
            for n in range(0, num_pts):
                vtk_points_t.InsertNextPoint(pts_mat[0, n], pts_mat[1, n], pts_mat[2, n])
            mean_polydata.SetPoints(vtk_points_t)
            self.polydatas.append(mean_polydata)

            mean_shape_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "mean shape")
            mean_shape_node.CreateDefaultDisplayNodes()
            mean_shape_node.SetAndObservePolyData(mean_polydata)
            self.visualization_nodes.append(mean_shape_node)

            # HGM shape
            if include_HGM:
                hgm_polydata = vtk.vtkPolyData()
                hgm_polydata.DeepCopy(reference_polydata)
                pts_mat = self.population_p0.GetPoint()
                [dim, num_pts] = pts_mat.shape

                vtk_points_t = vtk.vtkPoints()
                for n in range(0, num_pts):
                    vtk_points_t.InsertNextPoint(pts_mat[0, n], pts_mat[1, n], pts_mat[2, n])
                hgm_polydata.SetPoints(vtk_points_t)
                self.polydatas.append(hgm_polydata)

                hgm_shape_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "hgm shape")
                hgm_shape_node.CreateDefaultDisplayNodes()
                hgm_shape_node.SetAndObservePolyData(hgm_polydata)
                hgm_shape_node.GetDisplayNode().SetVisibility(0)
                self.visualization_nodes.append(hgm_shape_node)

            # subject shapes
            for i in range(len(self.subject_ids)):
                subject_i_polydata = vtk.vtkPolyData()
                subject_i_polydata.DeepCopy(reference_polydata)
                self.polydatas.append(subject_i_polydata)

                subject_i_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", self.subject_ids[i])
                subject_i_node.CreateDefaultDisplayNodes()
                subject_i_node.SetAndObservePolyData(subject_i_polydata)
                subject_i_node.GetDisplayNode().SetVisibility(0)
                self.visualization_nodes.append(subject_i_node)

            slicer.util.resetThreeDViews()


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
       

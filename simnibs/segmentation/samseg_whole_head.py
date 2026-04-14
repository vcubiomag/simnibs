import numpy as np
import os
import pickle
from samseg.Samseg import Samseg
from samseg.SamsegUtility import logTransform, writeImage, getOptimizationOptions
from simnibs.segmentation.simnibs_segmentation_utils import (
    readCroppedImages,
    maskOutBackground,
    getModelSpecificationsWholeHead,
)
from samseg.Affine import Affine
from samseg.ProbabilisticAtlas import ProbabilisticAtlas
from samseg.utilities import Specification
import samseg.gems as gems
import logging
from samseg.figures import initVisualizer


eps = np.finfo(float).eps


class SamsegWholeHead(Samseg):
    def __init__(
        self,
        imageFileNames,
        atlasDir,
        savePath,
        transformedTemplateFileName,
        userModelSpecifications={},
        userOptimizationOptions={},
        imageToImageTransformMatrix=None,
        visualizer=None,
        saveHistory=None,
        savePosteriors=False,
        saveWarp=None,
        saveMesh=None,
        threshold=None,
        thresholdSearchString=None,
        targetIntensity=None,
        targetSearchStrings=None,
        modeNames=None,
        pallidumAsWM=True,
        saveModelProbabilities=False,
        gmmFileName=None,
        ignoreUnknownPriors=False,
        dissectionPhoto=None,
        nthreads=1,
    ):
        # super().__init__(
        # imageFileNames,
        # atlasDir,
        # savePath,
        # userModelSpecifications,
        # userOptimizationOptions,
        # imageToImageTransformMatrix,
        # visualizer,
        # saveHistory,
        # savePosteriors,
        # saveWarp,
        # saveMesh,
        # threshold,
        # thresholdSearchString,
        # targetIntensity,
        # targetSearchStrings,
        # modeNames,
        # pallidumAsWM,
        # saveModelProbabilities,
        # gmmFileName,
        # ignoreUnknownPriors,
        # dissectionPhoto,
        # nthreads)

        # def __init__(self, imageFileNames, atlasDir, savePath, userModelSpecifications=None, userOptimizationOptions=None,
        #      transformedTemplateFileName=None, visualizer=None, saveHistory=None, savePosteriors=None,
        #      saveWarp=None, saveMesh=None, threshold=None, thresholdSearchString=None,
        #      targetIntensity=None, targetSearchStrings=None):

        # Store input parameters as class variables
        self.imageFileNames = imageFileNames
        self.savePath = savePath
        self.atlasDir = atlasDir
        self.threshold = threshold
        self.thresholdSearchString = thresholdSearchString
        self.targetIntensity = targetIntensity
        self.targetSearchStrings = targetSearchStrings

        # Initialize some objects
        self.affine = Affine(
            imageFileName=self.imageFileNames[0],
            meshCollectionFileName=os.path.join(
                self.atlasDir, "atlasForAffineRegistration.txt.gz"
            ),
            templateFileName=os.path.join(self.atlasDir, "template.nii.gz"),
        )
        self.probabilisticAtlas = ProbabilisticAtlas()

        # Get full model specifications and optimization options (using default unless overridden by user)
        self.modelSpecifications = getModelSpecificationsWholeHead(
            atlasDir, userModelSpecifications
        )
        self.optimizationOptions = getOptimizationOptions(
            atlasDir, userOptimizationOptions
        )

        # Get transformed template, if any
        self.transformedTemplateFileName = transformedTemplateFileName

        logger = logging.getLogger(__name__)
        # Print specifications
        logger.info("##----------------------------------------------")
        logger.info("              Samsegment Options")
        logger.info("##----------------------------------------------")
        logger.info("output directory:" + savePath)
        logger.info("input images: {}".format(imageFileNames))
        if self.transformedTemplateFileName is not None:
            logger.info("transformed template:" + self.transformedTemplateFileName)
        logger.info("modelSpecifications:" + str(self.modelSpecifications))
        logger.info("optimizationOptions:" + str(self.optimizationOptions))

        # Convert modelSpecifications from dictionary into something more convenient to access
        self.modelSpecifications = Specification(self.modelSpecifications)

        # Setup a null visualizer if necessary
        if visualizer is None:
            self.visualizer = initVisualizer(False, False)
        else:
            self.visualizer = visualizer

        self.saveHistory = saveHistory
        self.savePosteriors = savePosteriors
        self.saveWarp = saveWarp
        self.saveMesh = saveMesh

        # Make sure we can write in the target/results directory
        os.makedirs(savePath, exist_ok=True)

        # Class variables that will be used later
        self.biasField = None
        self.gmm = None
        self.imageBuffers = None
        self.mask = None
        self.classFractions = None
        self.cropping = None
        self.transform = None
        self.voxelSpacing = None
        self.optimizationSummary = None
        self.optimizationHistory = None
        self.deformation = None
        self.deformationAtlasFileName = None

        # Samseg class variables not used in SimNIBS
        self.modeNames = modeNames
        self.saveModelProbabilities = saveModelProbabilities
        self.ignoreUnknownPriors = ignoreUnknownPriors
        self.dissectionPhoto = dissectionPhoto
        self.nthreads = nthreads

        self.transformedTemplateFileName = transformedTemplateFileName

    def preProcessWholeHead(self):
        # =======================================================================================
        #
        # Preprocessing (reading and masking of data)
        #
        # =======================================================================================

        # Read the image data from disk. At the same time, construct a 3-D affine transformation (i.e.,
        # translation, rotation, scaling, and skewing) as well - this transformation will later be used
        # to initially transform the location of the atlas mesh's nodes into the coordinate system of the image.
        self.imageBuffers, self.transform, self.voxelSpacing, self.cropping = (
            readCroppedImages(self.imageFileNames, self.transformedTemplateFileName)
        )

        # Background masking: simply setting intensity values outside of a very rough brain mask to zero
        # ensures that they'll be skipped in all subsequent computations
        self.imageBuffers, self.mask = maskOutBackground(
            self.imageBuffers,
            self.imageFileNames,
            self.modelSpecifications.atlasFileName,
            self.transform,
            self.modelSpecifications.brainMaskingSmoothingSigma,
            self.modelSpecifications.brainMaskingThreshold,
            self.probabilisticAtlas,
            upsampled=False,
        )

        # Let's prepare for the bias field correction that is part of the imaging model. It assumes
        # an additive effect, whereas the MR physics indicate it's a multiplicative one - so we log
        # transform the data first.
        self.imageBuffers = logTransform(self.imageBuffers, self.mask)

        # Visualize some stuff
        if hasattr(self.visualizer, "show_flag"):
            self.visualizer.show(
                mesh=self.probabilisticAtlas.getMesh(
                    self.modelSpecifications.atlasFileName, self.transform
                ),
                shape=self.imageBuffers.shape,
                window_id="samsegment mesh",
                title="Mesh",
                names=self.modelSpecifications.names,
                legend_width=350,
            )
            self.visualizer.show(
                images=self.imageBuffers,
                window_id="samsegment images",
                title="Samsegment Masked and Log-Transformed Contrasts",
            )

    def standard_segmentation(self):
        posteriors, _, _, _, _ = self.segment()
        structureNumbers = np.array(np.argmax(posteriors, 1), dtype=np.uint32)

        freeSurferSegmentation = np.zeros(self.imageBuffers.shape[0:3], dtype=np.uint16)
        FreeSurferLabels = np.array(
            self.modelSpecifications.FreeSurferLabels, dtype=np.uint16
        )
        freeSurferSegmentation[self.mask] = FreeSurferLabels[structureNumbers]

        # Write out various images - segmentation first
        exampleImage = gems.KvlImage(self.imageFileNames[0])
        writeImage(
            os.path.join(self.savePath, "crispSegmentation.nii.gz"),
            freeSurferSegmentation,
            self.cropping,
            exampleImage,
        )

    def getOptimizationSummary(self):
        return self.optimizationSummary

    def writeMesh(self, output_name):
        mesh = self.probabilisticAtlas.getMesh(
            self.modelSpecifications.atlasFileName,
            self.transform,
            initialDeformation=self.deformation,
            initialDeformationMeshCollectionFileName=self.deformationAtlasFileName,
        )

        estimatedNodePositions = (
            self.probabilisticAtlas.mapPositionsFromSubjectToTemplateSpace(
                mesh.points, self.transform
            )
        )
        self.probabilisticAtlas.saveDeformedAtlas(
            self.modelSpecifications.atlasFileName, output_name, estimatedNodePositions
        )

    def saveHistory(self, output_path):
        history = {
            "input": {
                "imageFileNames": self.imageFileNames,
                "transformedTemplateFileName": self.transformedTemplateFileName,
                "modelSpecifications": self.modelSpecifications,
                "optimizationOptions": self.optimizationOptions,
                "savePath": self.savePath,
            },
            "imageBuffers": self.imageBuffers,
            "mask": self.mask,
            "historyWithinEachMultiResolutionLevel": self.optimizationHistory,
            "labels": self.modelSpecifications.FreeSurferLabels,
            "names": self.modelSpecifications.names,
            "optimizationSummary": self.optimizationSummary.append,
        }

        with open(os.path.join(output_path, "history.p"), "wb") as file:
            pickle.dump(history, file, protocol=pickle.HIGHEST_PROTOCOL)

    def saveParametersAndInput(self, filename=None):
        # Make sure the bias field basis functions are not downsampled
        # This might happen if the downsampling factor in the last
        # resolution level is >1
        self.biasField.downSampleBasisFunctions([1, 1, 1])
        parameters_and_inputs = {
            "GMMParameters": {
                "mixtureWeights": self.gmm.mixtureWeights,
                "means": self.gmm.means,
                "variances": self.gmm.variances,
            },
            "fractionsTable": self.classFractions,
            "gaussiansPerClass": self.gmm.numberOfGaussiansPerClass,
            "deformation": self.deformation,
            "modelSpecifications": self.modelSpecifications,
            "imageFileNames": self.imageFileNames,
            "deformationAtlas": self.deformationAtlasFileName,
            "imageBuffers": self.imageBuffers,
            "biasFields": self.biasField.getBiasFields(),
            "mask": self.mask,
            "cropping": self.cropping,
            "transform": self.transform.as_numpy_array,
            "names": self.modelSpecifications.names,
        }

        if filename is not None:
            p = parameters_and_inputs.copy()
            del p["imageBuffers"]
            del p["biasFields"]
            with open(filename, "wb") as file:
                pickle.dump(p, file, protocol=pickle.HIGHEST_PROTOCOL)

        return parameters_and_inputs

import pandas as pd
from pydicom import read_file
import re
import logging

class DicomTable:
    def __init__(self, dicom_path):
        self.dicom_path = dicom_path
        self.dicom_file = read_file( self.dicom_path )
        self.record_lookup = self.get_patient_data()
        self.record_id = self.get_record_id()

    def get_record_id(self):
        if self.dicom_file == None:
            return None

        return self.record_lookup.patient_id[0] + "_" + self.record_lookup.laterality[0] + "_" + \
               self.record_lookup.study_date[0] + "_" + str(self.record_lookup.series_number[0].astype(int))

    def get_oct_data(self):
        image_positions = []
        stack_positions = []
        x_scales = []
        y_scales = []
        x_starts = []
        y_starts = []
        x_ends = []
        y_ends = []
        for i in range( 0, len( self.dicom_file.PerFrameFunctionalGroupsSequence ) ):
            image_positions.append(
                self.dicom_file.PerFrameFunctionalGroupsSequence[i].PlanePositionSequence[0].ImagePositionPatient )
            stack_positions.append(
                self.dicom_file.PerFrameFunctionalGroupsSequence[i].FrameContentSequence[0].InStackPositionNumber )
            x_scales.append(
                self.dicom_file.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0].PixelSpacing[1] )
            y_scales.append(
                self.dicom_file.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0].PixelSpacing[0] )
            y_starts.append( self.dicom_file.PerFrameFunctionalGroupsSequence[i].OphthalmicFrameLocationSequence[
                                 0].ReferenceCoordinates[0] )
            x_starts.append( self.dicom_file.PerFrameFunctionalGroupsSequence[i].OphthalmicFrameLocationSequence[
                                 0].ReferenceCoordinates[1] )
            y_ends.append( self.dicom_file.PerFrameFunctionalGroupsSequence[i].OphthalmicFrameLocationSequence[
                               0].ReferenceCoordinates[2] )
            x_ends.append( self.dicom_file.PerFrameFunctionalGroupsSequence[i].OphthalmicFrameLocationSequence[
                               0].ReferenceCoordinates[3] )

        return image_positions, stack_positions, x_scales, y_scales, y_starts, x_starts, y_ends, x_ends

    def filter_dicom(self):
        manuf = self.dicom_file.Manufacturer
        study_descr = self.dicom_file.StudyDescription
        series_description = self.dicom_file.SeriesDescription
        pixel_shape = self.dicom_file.pixel_array.shape
        if (manuf == "Heidelberg Engineering") & (study_descr == 'Makula (OCT)') & (series_description == 'Volume IR') \
                & (pixel_shape[0] == 49):
            return self.dicom_file
        else:
            logging.info(
                "Dicom did not contain correct data, see values for mauf, study desc, series desc and pixel shape: "
                "{},{},{},{}".format( manuf, study_descr, series_description, pixel_shape ) )
            return (None)

    def get_patient_data(self):
        patient_dict = {}
        oct_dict = {}
        # load all dicom files and append to dicom list
        # try if dicom has all files
        self.dicom_file = self.filter_dicom()
        if self.dicom_file is None:
            return None
        if self.dicom_file is not None:
            # remove all non digits from string
            patient_dict["patient_id"] = re.findall( r'\d+', self.dicom_file.PatientID )
            patient_dict["laterality"] = self.dicom_file.ImageLaterality
            patient_dict["study_date"] = self.dicom_file.StudyDate
            patient_dict["series_number"] = self.dicom_file.SeriesNumber

            # get all oct data
            image_positions, stack_positions, x_scales, y_scales, y_starts, x_starts \
                , y_ends, x_ends = self.get_oct_data()

            oct_dict["image_positions"] = image_positions
            oct_dict["stack_positions"] = stack_positions
            oct_dict["x_scales"] = x_scales
            oct_dict["y_scales"] = y_scales
            oct_dict["y_starts"] = y_starts
            oct_dict["x_starts"] = x_starts
            oct_dict["y_ends"] = y_ends
            oct_dict["x_ends"] = x_ends

        # create dataframe with all relevant data from dicom
        patient_pd = pd.DataFrame.from_dict( patient_dict )
        oct_pd = pd.DataFrame.from_dict( oct_dict )

        patient_full_pd = pd.concat( (patient_pd, oct_pd), axis = 1 )
        return patient_full_pd

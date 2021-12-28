import SimpleITK as sitk
from pathlib import Path
from dcmrtstruct2nii.adapters.input.abstractinputadapter import AbstractInputAdapter
from dcmrtstruct2nii.exceptions import InvalidFileFormatException


class DcmInputAdapter(AbstractInputAdapter):
    def ingest(self, input_dir, series_id=None):
        """
        Load DICOMs from input_dir to a single 3D image and make sure axial
        direction is on third axis.
        :param input_dir: Input directory where the dicom files are located
        :return: multidimensional array with pixel data, metadata
        """
        dicom_reader = sitk.ImageSeriesReader()

        if series_id is not None:
            # checkers
            assert isinstance(input_dir, str), ValueError(
                "input_dir should be dir where .dcm files are stored"
            )

            assert series_id in dicom_reader.GetGDCMSeriesIDs(input_dir), ValueError(
                f"There is no DICOM files with Series ID: {series_id} in {input_dir}"
            )

            dicom_file_names = dicom_reader.GetGDCMSeriesFileNames(input_dir, series_id)

        # elif below is deprecated, above is a better solution
        elif isinstance(input_dir, (tuple, list)):
            dicom_file_names = dicom_reader.GetGDCMSeriesFileNames(
                str(Path(input_dir[0]).parent)
            )
            dicom_file_names = tuple(
                [i for i in list(dicom_file_names) if i in input_dir]
            )
        else:
            dicom_file_names = dicom_reader.GetGDCMSeriesFileNames(str(input_dir))

        if not dicom_file_names:
            raise InvalidFileFormatException(
                "Directory {} is not a dicom".format(input_dir)
            )

        dicom_reader.SetFileNames(dicom_file_names)

        dicom_image = dicom_reader.Execute()

        return dicom_image

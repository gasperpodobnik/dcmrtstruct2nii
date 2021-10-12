import numpy as np
from skimage import draw
import SimpleITK as sitk

from dcmrtstruct2nii.exceptions import ContourOutOfBoundsException

import logging

import pydicom


class DcmPatientCoords2Mask:
    def _poly2mask(self, coords_x, coords_y, shape):
        mask = draw.polygon2mask(
            tuple(reversed(shape)), np.column_stack((coords_y, coords_x))
        )

        return mask

    def convert(
        self,
        rtstruct_contours,
        dicom_image,
        mask_background,
        mask_foreground,
        transform_dicom=None,
    ):
        shape = dicom_image.GetSize()

        mask = sitk.Image(shape, sitk.sitkUInt8)
        mask.CopyInformation(dicom_image)

        np_mask = sitk.GetArrayFromImage(mask)
        np_mask.fill(mask_background)

        for contour in rtstruct_contours:
            if contour["type"].upper() not in ["CLOSED_PLANAR", "INTERPOLATED_PLANAR"]:
                if "name" in contour:
                    logging.info(
                        f'Skipping contour {contour["name"]}, unsupported type: {contour["type"]}'
                    )
                else:
                    logging.info(
                        f'Skipping unnamed contour, unsupported type: {contour["type"]}'
                    )
                continue

            coordinates = contour["points"]
            if transform_dicom is not None:
                coordinates_np = np.array(
                    [
                        np.array(coordinates["x"], dtype=float),
                        np.array(coordinates["y"], dtype=float),
                        np.array(coordinates["z"], dtype=float),
                    ]
                ).T
                coordinates_np = np.asarray(
                    [transform_dicom.TransformPoint(p) for p in coordinates_np]
                )
                coordinates = {
                    "x": coordinates_np[:, 0].astype(pydicom.valuerep.DSfloat).tolist(),
                    "y": coordinates_np[:, 1].astype(pydicom.valuerep.DSfloat).tolist(),
                    "z": coordinates_np[:, 2].astype(pydicom.valuerep.DSfloat).tolist(),
                }

            pts = np.zeros([len(coordinates["x"]), 3])

            for index in range(0, len(coordinates["x"])):
                # lets convert world coordinates to voxel coordinates
                world_coords = dicom_image.TransformPhysicalPointToContinuousIndex(
                    (
                        coordinates["x"][index],
                        coordinates["y"][index],
                        coordinates["z"][index],
                    )
                )
                pts[index, 0] = world_coords[0]
                pts[index, 1] = world_coords[1]
                pts[index, 2] = world_coords[2]

            z = int(round(pts[0, 2]))

            try:
                filled_poly = self._poly2mask(
                    pts[:, 0], pts[:, 1], [shape[0], shape[1]]
                )
                np_mask[z, filled_poly] = mask_foreground  # sitk is xyz, numpy is zyx
                mask = sitk.GetImageFromArray(np_mask)
            except IndexError:
                # if this is triggered the contour is out of bounds
                raise ContourOutOfBoundsException()
            except RuntimeError as e:
                # this error is sometimes thrown by SimpleITK if the index goes out of bounds
                if "index out of bounds" in str(e):
                    raise ContourOutOfBoundsException()
                raise e  # something serious is going on

        return mask

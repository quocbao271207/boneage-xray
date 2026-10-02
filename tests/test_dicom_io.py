import io
import zipfile

import numpy as np
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

from boneage.dicom_io import dicom_bytes_to_case, iter_zip_dicom_cases, normalize_sex


def make_dicom_bytes(*, sex: str = "F", photometric: str = "MONOCHROME2") -> bytes:
    pixels = np.tile(np.arange(128, dtype=np.uint16), (128, 1))
    metadata = FileMetaDataset()
    metadata.TransferSyntaxUID = ExplicitVRLittleEndian
    metadata.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    metadata.MediaStorageSOPInstanceUID = generate_uid()
    dataset = FileDataset(None, {}, file_meta=metadata, preamble=b"\0" * 128)
    dataset.SOPClassUID = SecondaryCaptureImageStorage
    dataset.SOPInstanceUID = metadata.MediaStorageSOPInstanceUID
    dataset.Modality = "DX"
    dataset.PatientSex = sex
    dataset.Rows, dataset.Columns = pixels.shape
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = photometric
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 0
    dataset.PixelData = pixels.tobytes()
    buffer = io.BytesIO()
    dataset.save_as(buffer, enforce_file_format=True)
    return buffer.getvalue()


@pytest.mark.parametrize(("source", "expected"), [("F", (0.0, "F")), ("male", (1.0, "M"))])
def test_normalize_sex(source: str, expected: tuple[float, str]) -> None:
    assert normalize_sex(source) == expected


def test_monochrome1_is_inverted() -> None:
    case = dicom_bytes_to_case(
        make_dicom_bytes(photometric="MONOCHROME1"),
        case_key="safe",
    )
    pixels = np.asarray(case.image)
    assert pixels[:, 0].mean() > pixels[:, -1].mean()


def test_zip_reader_uses_hashed_case_key_without_extracting_names(tmp_path) -> None:
    archive_path = tmp_path / "hospital.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("patient-name/patient-id.dcm", make_dicom_bytes())

    cases = list(iter_zip_dicom_cases(archive_path, max_cases=1))

    assert len(cases) == 1
    assert cases[0].case_key.startswith("case_")
    assert "patient" not in cases[0].case_key
    assert not (tmp_path / "patient-name").exists()


def test_unknown_sex_is_rejected() -> None:
    with pytest.raises(ValueError, match="PatientSex"):
        dicom_bytes_to_case(make_dicom_bytes(sex="O"), case_key="safe")

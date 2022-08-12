import SimpleITK
import json
import os
from pathlib import Path
import numpy as np
from scipy.ndimage import label, generate_binary_structure, binary_dilation, binary_fill_holes
from deepvoxnet2.utilities.transformations import move
from deepvoxnet2.components.mirc import Mirc as Mirc_, Dataset, Case, Record, NiftyMultiModality
from deepvoxnet2.components.sampler import MircSampler
from deepvoxnet2.components.model import DvnModel
from deepvoxnet2.utilities.conversions import sitk_to_nii


DEFAULT_INPUT_PATH = Path("/input")
DEFAULT_ALGORITHM_OUTPUT_IMAGES_PATH = Path("/output/images/")
DEFAULT_ALGORITHM_OUTPUT_FILE_PATH = Path("/output/results.json")


class Mirc():
    def __init__(self,
                 input_path: Path = DEFAULT_INPUT_PATH,
                 output_path: Path = DEFAULT_ALGORITHM_OUTPUT_IMAGES_PATH):

        self.debug = False  # False for running the docker!
        if self.debug:
            self._input_path = Path('/path-do-input-data/')
            self._output_path = Path('/path-to-output-dir/')
            self._algorithm_output_path = self._output_path / 'stroke-lesion-segmentation'
            self._output_file = self._output_path / 'results.json'
            self._case_results = []

        else:
            self._input_path = input_path
            self._output_path = output_path
            self._algorithm_output_path = self._output_path / 'stroke-lesion-segmentation'
            self._output_file = DEFAULT_ALGORITHM_OUTPUT_FILE_PATH
            self._case_results = []

    @staticmethod
    def correct_zero_lesions_volume(arr, orig_vol, max_vol=1e14):
        if orig_vol > np.sum(arr):
            arr[...] = orig_vol / arr.size

        if np.sum(arr) > max_vol:
            arr *= max_vol / np.sum(arr)

        return arr

    @staticmethod
    def keep_n_labels(arr, n_labels, label_arr):
        iso_struct = generate_binary_structure(rank=3, connectivity=1)
        sort_idx = np.argsort([np.sum(label_arr == i) for i in range(1, np.max(label_arr) + 1)]) + 1
        for i in range(1, n_labels + 1):
            lesion = label_arr == sort_idx[-i]
            lesion_ = binary_dilation(lesion, iso_struct)
            arr[np.logical_xor(lesion, lesion_)] = 0
            arr[np.logical_xor(binary_fill_holes(lesion_), lesion_)] = 0

        return arr

    @staticmethod
    def improve_prediction(y_pred, min_threshold=0.1, binary_threshold=None, dummy=1e-21, max_lesion_count=1e14, max_vol=1e14):
        assert y_pred.ndim == 3
        assert min_threshold is None or binary_threshold is None and not (min_threshold is None and binary_threshold is None)
        vol = np.sum(y_pred)
        lesions, lesion_count = label(y_pred > (binary_threshold or 0.5))
        y_pred_ = np.where(y_pred > (binary_threshold or min_threshold), 1 if binary_threshold else y_pred, dummy).astype(np.float32)
        y_pred_ = Mirc.keep_n_labels(y_pred_, min(max_lesion_count, lesion_count) - 1, lesions)
        if lesion_count == 0:
            Mirc.correct_zero_lesions_volume(y_pred_, vol, max_vol)

        # vol_ = np.sum(y_pred_)
        # _, lesion_count_ = label(y_pred_)
        # print("\t-->", (lesion_count, lesion_count_), (int(vol), int(vol_)), np.min(y_pred_), y_pred_.dtype)
        return y_pred_

    def predict(self, input_data):
        """
        Input   input_data, dict.
                The dictionary contains 3 images and 3 json files.
                keys:  'dwi_image' , 'adc_image', 'flair_image', 'dwi_json', 'adc_json', 'flair_json'

        Output  prediction, array.
                Binary mask encoding the lesion segmentation (0 background, 1 foreground).
        """
        # Get all image inputs.
        dwi_image, adc_image, flair_image = input_data['dwi_image'],\
                                            input_data['adc_image'],\
                                            input_data['flair_image']


        # Get all json inputs.
        dwi_json, adc_json, flair_json = input_data['dwi_json'],\
                                         input_data['adc_json'],\
                                         input_data['flair_json']

        ################################################################################################################
        #################################### Beginning of your prediction method. ######################################

        # print(SimpleITK.GetArrayFromImage(dwi_image).shape, SimpleITK.GetArrayFromImage(adc_image).shape, SimpleITK.GetArrayFromImage(flair_image).shape)
        dwi_image, adc_image, flair_image = sitk_to_nii(dwi_image), sitk_to_nii(adc_image), sitk_to_nii(flair_image)
        # print(dwi_image.shape, adc_image.shape, flair_image.shape)
        flair_image, _ = move(flair_image, fixed_nii=dwi_image, parameter_map="rigid")
        # print(dwi_image.shape, adc_image.shape, flair_image.shape)
        dataset = Dataset("_")
        case = Case("_")
        record = Record("_")
        record.add(NiftyMultiModality(
            "input",
            niftys=[
               flair_image,
               dwi_image,
               adc_image
            ]))
        case.add(record)
        dataset.add(case)
        mirc = Mirc_(dataset)
        sampler = MircSampler(mirc)
        predictions = []
        for i in range(5):
            dvn_model = DvnModel.load_model(f"/dvn_models/dvn_model_final_{i}")
            predictions.append(dvn_model.predict("full_test", sampler)[0][0][0][0, :, :, :, 0])

        prediction = np.mean(predictions, axis=0) > 0.5
        # prediction = Mirc.improve_prediction(prediction, min_threshold=None, binary_threshold=0.5, dummy=1e-21, max_lesion_count=15, max_vol=45000)
        #################################### End of your prediction method. ############################################
        ################################################################################################################

        return prediction.T.astype(np.float32)

    def process_isles_case(self, input_data, input_filename):
        # Get origin, spacing and direction from the DWI image.
        origin, spacing, direction = input_data['dwi_image'].GetOrigin(),\
                                     input_data['dwi_image'].GetSpacing(),\
                                     input_data['dwi_image'].GetDirection()

        # Segment images.
        prediction = self.predict(input_data) # function you need to update!

        # Build the itk object.
        output_image = SimpleITK.GetImageFromArray(prediction)
        output_image.SetOrigin(origin), output_image.SetSpacing(spacing), output_image.SetDirection(direction)

        # Write segmentation to output location.
        if not self._algorithm_output_path.exists():
            os.makedirs(str(self._algorithm_output_path))
        output_image_path = self._algorithm_output_path / input_filename
        SimpleITK.WriteImage(output_image, str(output_image_path))

        # Write segmentation file to json.
        if output_image_path.exists():
            json_result = {"outputs": [dict(type="Image", slug="stroke-lesion-segmentation",
                                                 filename=str(output_image_path.name))],
                           "inputs": [dict(type="Image", slug="dwi-brain-mri",
                                           filename=input_filename)]}

            self._case_results.append(json_result)
            self.save()

    def load_isles_case(self):
        """ Loads the 6 inputs of ISLES22 (3 MR images, 3 metadata json files accompanying each MR modality).
        Note: Cases missing the metadata will still have a json file, though their fields will be empty. """

        # Get MR data paths.
        dwi_image_path = self.get_file_path(slug='dwi-brain-mri', filetype='image')
        adc_image_path = self.get_file_path(slug='adc-brain-mri', filetype='image')
        flair_image_path = self.get_file_path(slug='flair-brain-mri', filetype='image')

        # Get MR metadata paths.
        dwi_json_path = self.get_file_path(slug='dwi-mri-acquisition-parameters', filetype='json')
        adc_json_path = self.get_file_path(slug='adc-mri-parameters', filetype='json')
        flair_json_path = self.get_file_path(slug='flair-mri-acquisition-parameters', filetype='json')

        input_data = {'dwi_image': SimpleITK.ReadImage(str(dwi_image_path)), 'dwi_json': json.load(open(dwi_json_path)),
                      'adc_image': SimpleITK.ReadImage(str(adc_image_path)), 'adc_json': json.load(open(adc_json_path)),
                      'flair_image': SimpleITK.ReadImage(str(flair_image_path)), 'flair_json': json.load(open(flair_json_path))}

        # Set input information.
        input_filename = str(dwi_image_path).split('/')[-1]
        return input_data, input_filename

    def get_file_path(self, slug, filetype='image'):
        """ Gets the path for each MR image/json file."""

        if filetype == 'image':
            file_list = list((self._input_path / "images" / slug).glob("*.mha"))
        elif filetype == 'json':
            file_list = list(self._input_path.glob("*{}.json".format(slug)))

        # Check that there is a single file to load.
        if len(file_list) != 1:
            print('Loading error')
        else:
            return file_list[0]

    def save(self):
        with open(str(self._output_file), "w") as f:
            json.dump(self._case_results, f)

    def process(self):
        input_data, input_filename = self.load_isles_case()
        self.process_isles_case(input_data, input_filename)


if __name__ == "__main__":
    Mirc().process()

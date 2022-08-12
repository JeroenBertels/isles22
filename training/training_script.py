"""
A high-resolution (1 mm x 1 mm x 1 mm) No New-Net siamese/contra-lateral implementation for ISLES challenge @ MICCAI 2022

The network that is used is the unet_generalized_v3 version, in which not only the U pathway is weight-shared, but also the common/fully-connected pathway.
Furthermore, this network outputs a prediction for each permutation of the inputs (concatenated according to this permutation right before the common/fully-connected pathway).

optional arguments:
- fold_i [0, 1, 2, 3, 4]: what fold to run according to the isles dataset creation
"""

import os
import pickle
import shutil
import argparse
import numpy as np
from deepvoxnet2.components.mirc import Mirc, Dataset, Case, Record, NiftyFileModality, NiftiFileMultiModality
from deepvoxnet2.components.sampler import MircSampler
from deepvoxnet2.components.transformers import RandomCrop, ElasticDeformation, WindowNormalize, Resample, Swap, Concat, Split, MircInput, Flip, Group, Put, KerasModel, Crop, AffineDeformation
from deepvoxnet2.components.model import DvnModel
from deepvoxnet2.keras.metrics import get_metric
from deepvoxnet2.keras.losses import get_loss
from deepvoxnet2.keras.optimizers import Adam
from deepvoxnet2.keras.callbacks import DvnModelEvaluator, LearningRateScheduler, DvnModelCheckpoint
from deepvoxnet2.factories.directory_structure import MircStructure
from training.unet_generalized_v3 import create_generalized_unet_v3_model


def isles(data="train", fold_i=0, nb_folds=5, base_dir="/usr/local/micapollo01/MIC/DATA/SHARED/STAFF/eheyle3/ISLES/ISLES22/dataset-ISLES22^public^unzipped^version"):
    assert data in ["train", "val"], "Unknown data request. Please choose from: 'train' or 'val'."
    data_dir = os.path.join(base_dir, "rawdata")
    subjects = [subject for subject in sorted(os.listdir(data_dir)) if subject.startswith("sub")]
    np.random.seed(0)
    np.random.shuffle(subjects)
    assert isinstance(nb_folds, int) and fold_i in list(range(nb_folds)), "Inconsistent fold_i/nb_folds request."
    max_nb_cases_per_fold = int(np.ceil(len(subjects) / nb_folds))
    subjects_val = subjects[fold_i * max_nb_cases_per_fold:(fold_i + 1) * max_nb_cases_per_fold]
    subjects = subjects_val if data == "val" else [subject for subject in subjects if subject not in subjects_val]
    dataset = Dataset("ISLES", dataset_dir=base_dir)
    for subject in subjects:
        case_dir = os.path.join(data_dir, subject)
        case = Case(subject)
        sessions = [session for session in sorted(os.listdir(case_dir)) if session.startswith("ses")]
        assert len(sessions) == 1
        for session in sessions:
            record_dir = os.path.join(case_dir, session)
            record = Record(session)
            record.add(NiftyFileModality("flair", os.path.join(record_dir, f"{subject}_{session}_flair_reg.nii.gz")))
            record.add(NiftyFileModality("dwi", os.path.join(record_dir, f"{subject}_{session}_dwi.nii.gz")))
            record.add(NiftyFileModality("adc", os.path.join(record_dir, f"{subject}_{session}_adc.nii.gz")))
            record.add(NiftyFileModality("lesion", os.path.join(base_dir, "derivatives", subject, session, f"{subject}_{session}_msk.nii.gz")))
            record.add(NiftiFileMultiModality(
                "input",
                [
                    os.path.join(record_dir, f"{subject}_{session}_flair_reg.nii.gz"),
                    os.path.join(record_dir, f"{subject}_{session}_dwi.nii.gz"),
                    os.path.join(record_dir, f"{subject}_{session}_adc.nii.gz")
                ]
            ))
            case.add(record)

        dataset.add(case)

    return dataset


def main(base_dir, run_name, experiment_name, round_i=None, fold_i=0):
    #########################################
    # SETTING UP INPUT AND OUTPUT STRUCTURE #
    #########################################
    # training
    train_mirc = Mirc(isles(data="train", fold_i=fold_i))
    train_sampler = MircSampler(train_mirc, mode="per_case", shuffle=True)
    # validation
    val_mirc = Mirc(isles(data="val", fold_i=fold_i))
    val_sampler = MircSampler(val_mirc, mode="per_record", shuffle=False)
    # creating output directories
    structure = MircStructure(
        base_dir=os.path.join(base_dir, "Runs"),
        run_name=run_name,
        experiment_name=experiment_name,
        round_i=round_i,
        fold_i=fold_i,
        validation_mirc=val_mirc
    )
    structure.create()
    shutil.copyfile(os.path.realpath(__file__), os.path.join(structure.fold_dir, "script.py"))

    #####################
    # KERAS MODEL SETUP #
    #####################
    output_size = (72, 72, 64)
    keras_model = create_generalized_unet_v3_model(
        number_input_features=3,
        subsample_factors_per_pathway=(
            (1, 1, 1),
            (2, 2, 2),
            (4, 4, 4)
        ),
        kernel_sizes_per_pathway=(
            (((3, 3, 3), (3, 3, 3)), ((3, 3, 3), (3, 3, 3))),
            (((3, 3, 3), (3, 3, 3)), ((3, 3, 3), (3, 3, 3))),
            (((3, 3, 3), (3, 3, 3)), ((3, 3, 3), (3, 3, 3)))
        ),
        number_features_per_pathway=(
            ((30, 30), (30, 30)),
            ((60, 60), (60, 30)),
            ((120, 120), (120, 60))
        ),
        kernel_sizes_common_pathway=((1, 1, 1),) * 3,
        number_features_common_pathway=(60, 60, 1),
        dropout_common_pathway=(0, 0.2, 0.2),
        extra_output_kernel_sizes=(((1, 1, 1),),),
        extra_output_number_features=((1,),),
        extra_output_dropout=((0.2,),),
        extra_output_at_common_pathway_layer=(1,),
        extra_output_activation_final_layer=("sigmoid",),
        output_size=output_size,
        dynamic_input_shapes=True,
        number_siam_pathways=2,
        instance_normalization=False,
        batch_normalization=True
    )
    keras_model_transformer = KerasModel(keras_model, output_to_input=[0, 0, 1, 1])

    ##########
    # INPUTS #
    ##########
    x_input = MircInput(["input"], output_shapes=[(1, None, None, None, 3)], n=1)
    y_input = MircInput(["lesion"], output_shapes=[(1, None, None, None, 1)], n=1)
    x_input_low = Resample((2, 2, 2), order=1, prefilter=False)(x_input)
    y_input_low = Resample((2, 2, 2), order=0, prefilter=False)(y_input)

    ########################
    # PATH TRAINING INPUTS #
    ########################
    x_path, y_path = x_input_low, y_input_low
    x_path = WindowNormalize()(x_path)
    x_path, y_path = AffineDeformation(
        x_path,
        rotation_window_width=(3.14 / 18, 3.14 / 18, 3.14 / 18),
        translation_window_width=(1, 1, 1),
        width_as_std=True,
        order=[1, 0])(x_path, y_path)
    x_path, y_path = ElasticDeformation(
        x_path,
        shift=(1, 1, 1),
        nsize=(28, 28, 20))(x_path, y_path)
    cl_path, cly_path = Flip((1, 0, 0))(x_path, y_path)
    x_path, y_path, cl_path, cly_path = Flip((0.5, 0, 0))(x_path, y_path, cl_path, cly_path)
    x_path, y_path, cl_path, cly_path = RandomCrop(
        y_path,
        segment_size=output_size,
        subsample_factors=(1, 1, 1),
        nonzero=0.95,
        n=1,
        default_value=0)(x_path, y_path, cl_path, cly_path)
    x_path, y_path = Group()([x_path, cl_path]), Group()([y_path, cly_path])
    x_path, y_path = Swap()(x_path, y_path)
    x_output_train = keras_model_transformer(x_path)
    y_output_train = Split(indices=(0, 0, 1, 1))(y_path)

    ##########################
    # PATH VALIDATION INPUTS #
    ##########################
    x_path, y_path = x_input_low, y_input_low
    x_path = WindowNormalize()(x_path)
    cl_path, cly_path = Flip((1, 0, 0))(x_path, y_path)
    x_path, y_path, cl_path, cly_path = Crop(
        x_path,
        segment_size=(112, 112, 80),
        subsample_factors=(1, 1, 1),
        default_value=0)(x_path, y_path, cl_path, cly_path)
    x_path = Group()([x_path, cl_path])
    y_path = Group()([y_path, y_path, cly_path, cly_path])
    x_output_val = keras_model_transformer(x_path)
    y_output_val = y_path

    ################################
    # PATH FULL VALIDATION OUTPUTS #
    ################################
    x_path = x_output_val
    x_path = Concat(axis=0)([Split(indices=(0, 1))(x_path), Split(indices=(2, 3))(x_path)])
    x_path = Put(Group()([x_input, x_input]), keep_counts=True)(x_path)
    x_output_full_val = x_path
    y_output_full_val = Group()([y_input, y_input])

    ################################
    # DVN MODEL SETUP AND TRAINING #
    ################################
    dvn_model = DvnModel({
        "train": [x_output_train, y_output_train],
        "val": [x_output_val, y_output_val],
        "full_val": [x_output_full_val, y_output_full_val],
        "full_test": [x_output_full_val]
    })
    volume_difference = get_metric("volume_error", voxel_volume="auto")
    abs_volume_difference = get_metric("absolute_volume_error", voxel_volume="auto")
    dice_score = get_metric("dice_coefficient", threshold=0.5)
    cross_entropy = get_loss("cross_entropy")
    dice_loss = get_loss("dice_loss")
    dice_loss_ = get_loss("dice_loss", reduce_along_batch=True)
    learning_rates = [1e-4 * 2] * 80 + [1e-4 * 2 / 5] * 80 + [1e-4 * 2 / 25] * 80
    dvn_model.compile("train", optimizer=Adam(learning_rate=learning_rates[0]), losses=[[cross_entropy], [dice_loss, dice_loss_], [cross_entropy], [dice_loss, dice_loss_]])
    dvn_model.compile("val", losses=[[cross_entropy], [dice_loss, dice_loss_], [cross_entropy], [dice_loss, dice_loss_]])
    dvn_model.compile("full_val", losses=[[cross_entropy], [dice_loss, dice_loss_]], metrics=[[cross_entropy, dice_score, volume_difference, abs_volume_difference], [cross_entropy, dice_score, volume_difference, abs_volume_difference]])
    callbacks = [
        LearningRateScheduler(lambda epoch, lr: learning_rates[epoch]),
        DvnModelEvaluator(dvn_model, "full_val", val_sampler, output_dirs=structure.val_images_output_dirs, freq=8, logs_dir=structure.logs_dir),
        DvnModelCheckpoint(dvn_model, structure.models_dir, freq=8),
    ]
    history = dvn_model.fit("train", train_sampler, epochs=len(learning_rates), batch_size=8, callbacks=callbacks, logs_dir=structure.logs_dir, shuffle_samples=False, steps_per_epoch=128, initial_epoch=0, num_parallel_calls=4, prefetch_size=64)
    with open(structure.history_path, "wb") as f:
        pickle.dump(history.history, f)

    dvn_model.save(os.path.join(structure.models_dir, "dvn_model_final"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='nnn-v3_ce-sdsd_cl_hlr')
    parser.add_argument('fold_i', type=int, default=0, nargs="?")
    args = parser.parse_args()
    main(
        base_dir="/usr/local/micapollo01/MIC/DATA/STAFF/jberte3/tmp/datasets",
        run_name="isles",
        experiment_name=f"nnn-v3_ce-sdsd_cl_hlr",
        fold_i=args.fold_i
    )

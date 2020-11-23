import os
import cv2
import utils
import random
import numpy as np
from preprocessing import utils
from preprocessing import Market_data_loading as Market
import datetime
from multiprocessing import Process, current_process
import sys
import time
# import matplotlib.pyplot as plt
# import matplotlib.image as plt_im
import multiprocessing



def generate_syntethic_images(Market_data, num_images_to_generate, viewpoint = 1, other_attrs = None,
                              constraint_functions = []):

    images_no_head_occlusions = set(Market.get_images_with_attib(Market_data, Market.attr_OcclusionUp, 0))
    target_images = set(Market.get_images_with_attib(Market_data, Market.attr_viewpoint, viewpoint))
    target_images = target_images.intersection(images_no_head_occlusions)

    if other_attrs is not None:
        for attr in other_attrs:
            images_with_attr = Market.get_images_with_attib(Market_data, attr, other_attrs[attr])
            if len(images_with_attr) > 0:
                target_images = target_images.intersection(set(images_with_attr))

    target_images = list(target_images)

    for idx in range(0, num_images_to_generate):
        cv2.destroyAllWindows()

        img_name1 = random.choice(target_images)
        img_name2 = random.choice(target_images)

        img_path1 = os.path.join(Market.Market_images_dir, img_name1)
        mask_path1 = os.path.join(Market.Market_masks_dir, img_name1)
        keypoints1 = Market_data[img_name1]['keypoints']
        attr1 = Market_data[img_name1]['attrs']
        img1 = cv2.imread(img_path1)
        mask1 = cv2.imread(mask_path1,cv2.IMREAD_GRAYSCALE)
        #mask1 = Market.load_crop_Market_mask(mask_path1)

        img_path2 = os.path.join(Market.Market_images_dir, img_name2)
        mask_path2 = os.path.join(Market.Market_masks_dir, img_name2)
        keypoints2 = Market_data[img_name2]['keypoints']
        attr2 = Market_data[img_name2]['attrs']
        img2 = cv2.imread(img_path2)
        mask2 = cv2.imread(mask_path2,cv2.IMREAD_GRAYSCALE)
        #mask2 = Market.load_crop_Market_mask(mask_path2)

        assert mask1.shape[0] == img1.shape[0] and mask2.shape[1] == img2.shape[1]

        for constraint_function in constraint_functions:
            if not constraint_function(img1, img2, mask1, mask2, keypoints1, keypoints2,
                                       attr1, attr2):
                idx -= 1
                continue
        # # display
        # img2_display = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        # concat_images = [img1, img2_display]
        # if generated_replaced_area is not None:
        #     concat_images.append(generated_replaced_area)
        # if generated_replaced_rect is not None:
        #     concat_images.append(generated_replaced_rect)
        # if generated_replaced_ht is not None:
        #     concat_images.append(generated_replaced_ht)
        #
        # display_img = cv2.hconcat(concat_images)
        # cv2.imshow('morphing', display_img)
        # cv2.waitKey()
    return



def foreground_removal(img, mask):
        kernel = np.ones((3, 3), np.uint8)
        mask_area_enlarged = cv2.dilate(mask, kernel, iterations=3)
        img = np.uint8(img)
        img_area_removed = cv2.inpaint(src = img, inpaintMask = mask_area_enlarged, inpaintRadius = 20 , flags = cv2.INPAINT_TELEA)
        return img_area_removed


def remove_foreground_of_all_the_dataset(ImagesPath, MasksPath, save_image, creating_one_numpy_from_all_the_dataset, DirToSave):
# the goal of this function is to process the whole dataset and either save the processed images on the disk (in training phase we will only load the images) OR create a numpy array to be used in the training phase.
    All_imgs_without_foreground = []
    Masks = os.listdir (MasksPath)
    for i, img_name in enumerate (Masks):
        img = cv2.imread(filename = os.path.join(ImagesPath, img_name))
        mask = cv2.imread(filename = os.path.join(MasksPath, img_name), flags= cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            continue
        else:
            assert mask.shape[0] == img.shape[0] and mask.shape[1] == img.shape[1]

        img_without_foreground = foreground_removal (img, mask)

        display_image = False
        if display_image:
            cv2.imshow('1_original_img', img)
            cv2.imshow('2_its_mask', mask)
            cv2.imshow('3_this_img_without_its_foreground', img_without_foreground)
            cv2.waitKey()

        if save_image:
            os.makedirs(DirToSave, exist_ok=True)
            name = os.path.join(DirToSave, "{}.jpg".format(img_name))
            cv2.imwrite(filename = name, img = img_without_foreground)
            # if i % 500 == 0:
            #     print("Processed Image is saved on the disk     {}/{}".format(i, len(Masks)))

        if creating_one_numpy_from_all_the_dataset:
            All_imgs_without_foreground.append(img_without_foreground)
            if i % 500 == 0:
                print("Removing the Foreground and creating one numpy array for all the -target- images     {}/{}".format(i, len(Masks)))


    if creating_one_numpy_from_all_the_dataset:
        All_imgs_without_foreground_numpy_array = np.stack(All_imgs_without_foreground)
        return All_imgs_without_foreground_numpy_array


def offline_replace_img1_back_with_others_back(name_img1, MaskDir1, ImgDir1, target_background_array, target_background_dir):

    split_name = name_img1.split("_")[0]
    name_img1 = os.path.join(split_name, name_img1)
    img1 = cv2.imread(filename = os.path.join(ImgDir1, name_img1))
    mask1 = cv2.imread(filename = os.path.join(MaskDir1, name_img1), flags = cv2.IMREAD_GRAYSCALE)

    # both img1 and mask1 should be read successfully, otherwise we return None.
    if img1 is None or mask1 is None:
        return None
    else:
        assert mask1.shape[0] == img1.shape[0] and mask1.shape[1] == img1.shape[1]



    if target_background_array is not None and target_background_dir is not None:
        raise ValueError("In function 'offline_replace_img1_back_with_others_back', either 'target_background_array' or 'target_background_dir' should be None")

    # if you want to read the image from the array
    if target_background_array is not None:
        # read background to past the foreground of the img1 on it.
        randint = random.randint(0, target_background_array.shape[0]-1)
        random_target_background = target_background_array[randint,:,:,:]

    # if you want to load the image from disk
    if  target_background_dir is not None:
        random_filename = random.choice([x for x in os.listdir(target_background_dir) if os.path.isfile(os.path.join(target_background_dir, x))])
        random_target_background = cv2.imread(filename = os.path.join(target_background_dir, random_filename))

    # mask_img1 = cv2.cvtColor(src=mask1, code=cv2.COLOR_GRAY2BGR)
    # mask_img1 = mask_img1.astype(np.bool)
    # np.copyto(dst=random_target_background, src=img1, where=mask_img1, casting='unsafe')

    assert img1.shape[0] == random_target_background.shape[0] and img1.shape[1] == random_target_background.shape[1]

    background_mask = cv2.cvtColor(255 - mask1, cv2.COLOR_GRAY2BGR)  # inverted mask1 (mask 1 is in gray scale)
    background_mask = cv2.blur(background_mask, (2,2))
    mask_img1 = cv2.cvtColor(src=mask1, code=cv2.COLOR_GRAY2BGR)  #

    mask_img1 = cv2.blur(mask_img1, (2, 2))
    masked_fg = (img1 * (1 / 255)) * (mask_img1 * (1 / 255))  #  smoothed forgeground of img1, with black background
    masked_bg = (random_target_background * (1 / 255)) * (background_mask * (1 / 255))  # First, I clear where I want to put the foreground of img1.
    img1fore_img2back = np.uint8(cv2.addWeighted(masked_fg, 255, masked_bg, 255, 0))  # images are overlay

    display2 = False
    if display2:
        print("visualization 2 ...")
        cv2.imshow('t_1', img1fore_img2back)
        cv2.imshow('i1_1', img1)
        cv2.waitKey()

    return img1fore_img2back


def replace_background(img1, mask1, img2, mask2):

    np.copy(img2)
    img2_background = foreground_removal(img = img2, mask= mask2)
    background_mask = cv2.cvtColor(255 - mask1, cv2.COLOR_GRAY2BGR)  # inverted mask1
    background_mask = cv2.blur(background_mask, (5, 5))
    mask_img1 = cv2.cvtColor(src=mask1, code=cv2.COLOR_GRAY2BGR)  #

    mask_img1 = cv2.blur(mask_img1, (20, 20))
    masked_fg = (img1 * (1 / 255)) * (mask_img1 * (1 / 255))  #  smoothed forgeground of img1, with black background
    masked_bg = (img2_background * (1 / 255)) * (background_mask * (1 / 255))  # First, I clear where I want to put the foreground of img1.
    result = np.uint8(cv2.addWeighted(masked_fg, 255, masked_bg, 255, 0))  # images are overlay
    #mask_img1 = mask_img1.astype(np.bool)
    #np.copyto(dst=img2_background, src=img1, where=mask_img1, casting='unsafe')

    display = False
    if display:
        cv2.imshow('blured_img1', mask_img1)
        cv2.imshow('fg', masked_fg)
        cv2.imshow('bg', masked_bg)
        cv2.imshow('i1', img1)
        cv2.imshow('m1', mask1)
        cv2.imshow('i2', img2)
        cv2.imshow('m2', mask2)
        cv2.imshow('r1', result)  # img1 foreground will be copied on the img2 background
        cv2.waitKey()

    return img2_background

# ---------------------------------------------------
# constrain functions
# ---------------------------------------------------
def is_compatible_area(img1, img2, mask1, mask2,
                       kp1, kp2, attr1, attr2, th_area = 0.8):
    area1 = cv2.countNonZero(mask1)
    area2 = cv2.countNonZero(mask2)

    if min(area1, area2)/max(area1, area2) < th_area:
        return False
    return True


def is_compatible_iou(img1, img2, mask1, mask2,
                      kp1, kp2, attr1, attr2, iou_threshold = 0.5):
    mask1_aligned, mask2_aligned, _, _ = utils.align_images_width(mask1, mask2)
    iou = utils.compute_mask_iou(mask1_aligned, mask2_aligned)
    # print('iou is ', iou)
    if iou < iou_threshold:
        return False
    return True

# ---------------------------------------------------
# end constraint functions
# ---------------------------------------------------

def online_replace_img1_back_with_others_back(name_img1, MaskDir, ImgDir, enable_constraints_ht= False):
    split_name = name_img1.split("_")[0]
    name_img1 = os.path.join(split_name, name_img1)
    img1 = cv2.imread(filename = os.path.join(ImgDir, name_img1))
    mask1 = cv2.imread(filename = os.path.join(MaskDir, name_img1), flags = cv2.IMREAD_GRAYSCALE)

    # both img1 and mask1 should be read successfully, otherwise we return None.
    if img1 is None or mask1 is None:
        return None
    else:
        assert mask1.shape[0] == img1.shape[0] and mask1.shape[1] == img1.shape[1]

    mask1[mask1 < 128] = 0
    mask1[mask1 > 128] = 255

    # try until we could read an image with its mask successfully. Then, we check to see if this image is a suitable image or not.
    while True:
        random_filename = random.choice([x for x in os.listdir(MaskDir) if os.path.isfile(os.path.join(MaskDir, x))])
        random_img = cv2.imread(filename = os.path.join(ImgDir, random_filename))
        its_mask = cv2.imread(filename = os.path.join(MaskDir, random_filename), flags = cv2.IMREAD_GRAYSCALE)

        if random_img is not None and its_mask is not None:

            assert its_mask.shape[0] == random_img.shape[0] and its_mask.shape[1] == random_img.shape[1]

            its_mask[its_mask < 128] = 0
            its_mask[its_mask > 128] = 255
            # Now, we check to see if this image has a suitable background and can be considered for generating new samples.
            # @todo:  can add here multiple constraint functions on masks (like orientation, center etc).
            if enable_constraints_ht:
                if is_compatible_iou(None, None, mask1, its_mask, None, None, None, None):
                     if is_compatible_area(None, None, mask1, its_mask, None, None, None, None, th_area=0.5):
                        break  # if both conditions are True, we break the loop,
                     else:     # otherwise we search for another image
                        continue

            # todo: resize img1_foreground to fit when pasting it on img2_background
            # x1, y1, w1, h1 = cv2.boundingRect(mask1)
            # x2, y2, w2, h2 = cv2.boundingRect(mask2)
            #
            # if w1*h1 > w2*h2 :  # the area of mask1 is larger than the mask2. So, we want to shrink mask1 to fit in.
            #     cv2.resize(src = mask1, dst = , Size(), 0.5, 0.5, interpolation = cv2.INTER_AREA)

            # it the suitable image is find, we break the loop and perform the background replacement (start image generation).
            break

    result_image = replace_background(img1 = img1, mask1 = mask1, img2 = random_img, mask2 = its_mask)

    return result_image

def f(name):
    print('{}: hello {} from {}'.format(
        datetime.datetime.now(), name, current_process().name))
    sys.stdout.flush()

if __name__ == '__main__':

    Generate_new_images = 10
    OriginalImage1 = "0000/0000_000_01_0303morning_0008_0.jpg"
    OriginalImage2 = "0001/0001_000_01_0303morning_0017_2.jpg"
    OriginalImage3 = "0018/0018_000_01_0303morning_0238_6.jpg"
    OriginalImage4 = "0083/0083_002_01_0303morning_1439_0.jpg"
    OriginalImage5 = "0114/0114_019_10_0303noon_0284_2_ex.jpg"
    OriginalImage6 = "0256/0256_000_13_0303afternoon_0956_1.jpg"

    ImgDir_init = "/media/ehsan/2cbae9f1-6394-4fa9-b963-5ef890eee044/B_DATASETS/MSMT17_V2/MSMT17_V2/mask_train_v2_256x128"
    MaskDir_init = "/media/ehsan/2cbae9f1-6394-4fa9-b963-5ef890eee044/B_DATASETS/MSMT17_V2/MSMT17_V2/binary_mask_train_v2_256x128"

    target_background_dir_inti = "/media/ehsan/2cbae9f1-6394-4fa9-b963-5ef890eee044/B_DATASETS/MSMT17_V2/MSMT17_V2/MSMT17_backgrounds_256x128"

    #### First Idea: First remove the foreground for all the images. This way, computational costs in the training phase will be reduced.
    ## Disadvantage is that it is you should script another code to consider IoU constrain to see if the background is suitable enough or not.
    ## Advantage: we only process the dataset onece to remove the foregrounds.
    ImgDir = os.listdir(ImgDir_init)
    MaskDir = os.listdir(MaskDir_init)
    for index, folder_name in enumerate(ImgDir):
        TargetImagesArray = remove_foreground_of_all_the_dataset(ImagesPath = os.path.join(ImgDir_init, folder_name),
                                                                 MasksPath = os.path.join(MaskDir_init, folder_name),
                                                                 creating_one_numpy_from_all_the_dataset = False,
                                                                 save_image = True,
                                                                 DirToSave = os.path.join(target_background_dir_inti, folder_name))
        print("removing the background and saving the images on the disk:\t{}/{}".format(index, len(ImgDir)))

    # result_image1 = offline_replace_img1_back_with_others_back (name_img1 = OriginalImage1,
    #                                                             MaskDir1 = MaskDir, ImgDir1 = ImgDir,
    #                                                             target_background_array = TargetImagesArray,
    #                                                             target_background_dir= target_background_dir)


    #### Second Idea: Remove the foreground of each image online (while training) only when it is a suitable candidate background.
    ## Disadvantage of this method is its high computational costs (while it will be repeated in each epoch)
    #result_image2 = online_replace_img1_back_with_others_back (name_img1 = OriginalImage1, MaskDir = MaskDir, ImgDir = ImgDir)

    print("Finished")




import os
import os.path as osp
from config import cfg
from .bases import BaseImageDataset
import random

class PRCC_Orig(BaseImageDataset):
    dataset_dir = cfg.DATASETS.ROOT_DIR

    def __init__(self,root='',is_train='', verbose=True, **kwargs):
        super(PRCC_Orig, self).__init__()
        self.is_train = is_train
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train')
        #  we should randomly select one image of each person in the A folder, as gallery
        #  source: https://ieeexplore.ieee.org/document/8936426
        self.gallery_dir = osp.join(self.dataset_dir, 'test', 'A')
        # Folders B and C are considered as prob. In folder B, persons have same clothes with gallery (i.e., folder A),
        # but in C persons have different clothes from the gallery.
        self.query_dir = osp.join(self.dataset_dir, 'test', 'B') # 'B' or 'C' for same clothes and different clothes, respectively

        self.list_train_path = os.listdir(self.train_dir)
        self.list_gallery_path = os.listdir(self.gallery_dir)
        self.list_query_path = os.listdir(self.query_dir)

        self._check_before_run()
        train = self.train_process_dir(self.train_dir)
        #val, num_val_pids, num_val_imgs = self._process_dir(self.train_dir, self.list_val_path)
        query = self.query_process_dir(self.query_dir)
        gallery = self.gallery_process_dir(self.gallery_dir)
        if verbose:
            print("=> the original dataset is loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        # if cfg.TRAIN_MODE == "orig_IDs":
        #     if not osp.exists(cfg.MODEL.NETWORK == "LTE_CNN"):
        #         # check to see if feat2 is available (feat2 is needed for our loss function)
        #         raise RuntimeError("'{}' is not available".format(cfg.Save_nonIDs_DIR))


    def train_process_dir(self, dir_path):

        if self.is_train: # new id will be assigned: new_id=pid+clothid
                dataset = []
                pid_container_new = set()
                pid_container_after = set()
                pid2label = {}
                person_ids = os.listdir(dir_path)
                ### important notice: When you want to relable the data with IDs from 0 to N, you must check that
                ###   the same person in gallery and query should receives an identical ID label

                # give new labels to images
                for indx, id_folder in enumerate(person_ids):
                    imgs_list = os.listdir(os.path.join(dir_path, id_folder))
                    for img_idx, img_info in enumerate(imgs_list):
                        camera = img_info.split('_')[0]
                        if camera == "A":  ## camid: A = 1, B=2, C=3  ## clothid: A = 4, B=4, C=5
                            clothid = 4
                        elif camera == "B":
                            clothid = 4
                        elif camera == "C":
                            if cfg.Train_on_ShortTerm_data:
                                continue
                            else:
                                clothid = 5
                        else:
                            raise ValueError("check the codes for a semantic error!")
                        new_id = id_folder + str(clothid)
                        pid_container_new.add(new_id)
                pid2label = {pid: label for label, pid in enumerate(pid_container_new)}
                print("*" * 50)
                print(dir_path)
                print(pid2label)
                print("*" * 50)

                # prepare the dataset
                for indx, id_folder in enumerate(person_ids):
                    imgs_list = os.listdir(os.path.join(dir_path, id_folder))
                    for img_idx, img_info in enumerate(imgs_list):
                        img_path = os.path.join(dir_path, id_folder, img_info)
                        camera = img_info.split('_')[0]
                        if camera == "A":  ## camid: A = 1, B=2, C=3  ## clothid: A = 4, B=4, C=5
                            clothid = 4
                            camid = 1
                        elif camera == "B":
                            clothid = 4
                            camid = 2
                        elif camera == "C":
                            if cfg.Train_on_ShortTerm_data:
                                continue
                            else:
                                clothid = 5
                                camid = 3
                        else:
                            raise ValueError("check the codes for a semantic error!")
                        new_id = id_folder + str(clothid)
                        person_id = pid2label[new_id]
                        if cfg.MODEL.NETWORK=='LTE_CNN':
                            feat2dir = cfg.DATASETS.feat2_DIR
                            feat2_dir = os.path.join(feat2dir, '{}.npy'.format(img_info))
                        elif cfg.MODEL.NETWORK=='STE_CNN' or  cfg.MODEL.NETWORK=='simple_baseline' or cfg.MODEL.NETWORK=='strong_baseline':
                            feat2_dir = None
                        dataset.append((img_path, person_id, camid, feat2_dir, clothid))
                        pid_container_after.add(person_id)
                # check if pid starts from 0 and increments with 1
                for idx, pid in enumerate(pid_container_after):
                    assert idx == pid, "See code comment for explanation"
                return dataset

        elif not self.is_train: # In test time, prepare the dataset with its original ids
            #print("In test phase, training data are not processed.")
            dataset = []
            pid_container_after = set()
            person_ids = os.listdir(dir_path)
            pid_container_orig = sorted(set(person_ids))

            # give ids from zero
            pid2label = {pid: label for label, pid in enumerate(pid_container_orig)}
            print("*" * 50)
            print(dir_path)
            print(pid2label)
            print("*" * 50)

            # prepare the dataset
            for indx, id_folder in enumerate(person_ids):
                imgs_list = os.listdir(os.path.join(dir_path, id_folder))
                for img_idx, img_info in enumerate(imgs_list):
                    img_path = os.path.join(dir_path, id_folder, img_info)
                    camera = img_info.split('_')[0]
                    if camera == "A":  ## camid: A = 1, B=2, C=3  ## clothid: A = 4, B=4, C=5
                        clothid = 4
                        camid = 1
                    elif camera == "B":
                        clothid = 4
                        camid = 2
                    elif camera == "C":
                        if cfg.Train_on_ShortTerm_data:
                            continue
                        else:
                            clothid = 5
                            camid = 3
                    else:
                        raise ValueError("check the codes for a semantic error!")
                    person_id = pid2label[id_folder]
                    if cfg.MODEL.NETWORK == 'LTE_CNN':
                        feat2dir = cfg.DATASETS.feat2_DIR
                        feat2_dir = os.path.join(feat2dir, '{}.npy'.format(img_info))
                    elif cfg.MODEL.NETWORK == 'STE_CNN' or cfg.MODEL.NETWORK == 'simple_baseline' or cfg.MODEL.NETWORK == 'strong_baseline':
                        feat2_dir = None
                    dataset.append((img_path, person_id, camid, feat2_dir, clothid))
                    pid_container_after.add(person_id)
            # check if pid starts from 0 and increments with 1
            for idx, pid in enumerate(pid_container_after):
                assert idx == pid, "See code comment for explanation"
            return dataset

        else:
            raise ValueError("Set is_train as True or False")

    def gallery_process_dir(self, dir_path):
        # gallery is taken from folder A ==> clothid=4 and camid=1

        if self.is_train:  # new id will be assigned: new_id=pid+clothid
            dataset = []
            person_ids = os.listdir(dir_path)

            new_ids = []
            for id in person_ids:  # new_id = pid + clothid
                id = id + '4' # A = 4, B=4, C=5
                new_ids.append(id)

            pid_container = sorted(set(new_ids))
            pid_container_after = set()
            pid2label = {pid: label for label, pid in enumerate(pid_container)}
            ### important notice: When you want to relable the data with IDs from 0 to N, you must check that
            ###   the same person in gallery and query should receives an identical ID label
            print("*" * 50)
            print(dir_path)
            print(pid2label)
            print("*" * 50)

            # prepare the dataset
            for indx, id_folder in enumerate(person_ids):
                imgs_list = os.listdir(os.path.join(dir_path, id_folder))
                random_img_of_this_id = random.choice(imgs_list)
                img_path = os.path.join(dir_path, id_folder, random_img_of_this_id)
                pid = id_folder + '4' # A = 4, B=4, C=5
                person_id = pid2label[pid]  # train ids must be relabelled from zero
                camid = 1 # A = 1, B=2, C=3
                if cfg.MODEL.NETWORK == 'LTE_CNN':
                    feat2dir = cfg.DATASETS.feat2_DIR
                    feat2_dir = os.path.join(feat2dir, '{}.npy'.format(random_img_of_this_id))
                elif cfg.MODEL.NETWORK == 'STE_CNN' or cfg.MODEL.NETWORK == 'simple_baseline' or cfg.MODEL.NETWORK == 'strong_baseline':
                    feat2_dir = None
                clothid = 4 # A = 4, B=4, C=5
                dataset.append((img_path, person_id, camid, feat2_dir, clothid))
                pid_container_after.add(person_id)

            # check if pid starts from 0 and increments with 1
            for idx, pid in enumerate(pid_container_after):
                assert idx == pid, "See code comment for explanation"
            return dataset

        elif not self.is_train:  # In test time, prepare the dataset with its original ids
            dataset = []
            pid_container_after = set()
            person_ids = os.listdir(dir_path)
            pid_container_orig = sorted(set(person_ids))

            # give ids from zero
            pid2label = {pid: label for label, pid in enumerate(pid_container_orig)}
            print("*" * 50)
            print(dir_path)
            print(pid2label)
            print("*" * 50)

            # prepare the dataset
            for indx, id_folder in enumerate(person_ids):
                imgs_list = os.listdir(os.path.join(dir_path, id_folder))
                random_img_of_this_id = random.choice(imgs_list)
                img_path = os.path.join(dir_path, id_folder, random_img_of_this_id)
                pid = id_folder # in test time, we use the original ids
                person_id = pid2label[pid]  # train ids must be relabelled from zero
                camid = 1  # A = 1, B=2, C=3
                if cfg.MODEL.NETWORK == 'LTE_CNN':
                    feat2dir = cfg.DATASETS.feat2_DIR
                    feat2_dir = os.path.join(feat2dir, '{}.npy'.format(random_img_of_this_id))
                elif cfg.MODEL.NETWORK == 'STE_CNN' or cfg.MODEL.NETWORK == 'simple_baseline' or cfg.MODEL.NETWORK == 'strong_baseline':
                    feat2_dir = None
                clothid = 4  # A = 4, B=4, C=5
                dataset.append((img_path, person_id, camid, feat2_dir, clothid))
                pid_container_after.add(person_id)

            # check if pid starts from 0 and increments with 1
            for idx, pid in enumerate(pid_container_after):
                assert idx == pid, "See code comment for explanation"
            return dataset

        else:
            raise ValueError("Set is_train as True or False")

    def query_process_dir(self, dir_path):
        # query is taken from folder B (so, clothid=4 and camid=2) or it is taken folder C (so, clothid=5 and camid=3)
        if cfg.Train_on_ShortTerm_data:
            folder = dir_path.split("/")[-1]
            if folder != "C":
                raise ValueError("Train has been done on short term data. so, test should be done on long term data. Please use Folder 'C' as query.")

        if self.is_train:  # new ids will be assigned: new_id=pid+clothid
            folder = dir_path.split("/")[-1]
            dataset = []
            person_ids = os.listdir(dir_path)
            new_ids = []

            # assign new IDs
            for id in person_ids:  # new_id = pid + clothid
                if folder == "B":
                    id = id + '4'  # A = 4, B=4, C=5
                elif folder == "C":
                    id = id + '5'  # A = 4, B=4, C=5
                else:
                    raise ValueError("check the code semanticly")
                new_ids.append(id)
            pid_container = sorted(set(new_ids))
            pid_container_after = set()
            pid2label = {pid: label for label, pid in enumerate(pid_container)}
            print("*" * 50)
            print(dir_path)
            print(pid2label)
            print("*" * 50)

            # prepare the dataset
            for indx, id_folder in enumerate(person_ids):
                imgs_list = os.listdir(os.path.join(dir_path, id_folder))
                for img_idx, img_info in enumerate(imgs_list):
                    img_path = os.path.join(dir_path, id_folder, img_info)

                    if folder == "B":
                        camid = 2
                        clothid = 4
                    elif folder == "C":
                        camid = 3
                        clothid = 5
                    else:
                        raise ValueError("check the codes for a semantic error!")

                    if cfg.MODEL.NETWORK == 'LTE_CNN':
                        feat2dir = cfg.DATASETS.feat2_DIR
                        feat2_dir = os.path.join(feat2dir, '{}.npy'.format(img_info))
                    elif cfg.MODEL.NETWORK == 'STE_CNN' or cfg.MODEL.NETWORK == 'simple_baseline' or cfg.MODEL.NETWORK == 'strong_baseline':
                        feat2_dir = None
                    pid = id_folder + str(clothid) # In train time, new Ids are assigned to each image of the person with different clothes
                    person_id = pid2label[pid]
                    dataset.append((img_path, person_id, camid, feat2_dir, clothid))
                    pid_container_after.add(person_id)

            # check if pid starts from 0 and increments with 1
            for idx, pid in enumerate(pid_container_after):
                assert idx == pid, "See code comment for explanation"
            return dataset

        elif not self.is_train:  # In test time, prepare the dataset with its original ids
            folder = dir_path.split("/")[-1]
            dataset = []
            pid_container_after = set()
            person_ids = os.listdir(dir_path)
            pid_container_orig = sorted(set(person_ids))

            # give ids from zero
            pid2label = {pid: label for label, pid in enumerate(pid_container_orig)}
            print("*" * 50)
            print(dir_path)
            print(pid2label)
            print("*" * 50)

            # prepare the dataset
            for indx, id_folder in enumerate(person_ids):
                imgs_list = os.listdir(os.path.join(dir_path, id_folder))
                for img_idx, img_info in enumerate(imgs_list):
                    img_path = os.path.join(dir_path, id_folder, img_info)

                    if folder == "B":
                        camid = 2
                        clothid = 4
                    elif folder == "C":
                        camid = 3
                        clothid = 5
                    else:
                        raise ValueError("check the codes for a semantic error!")

                    if cfg.MODEL.NETWORK == 'LTE_CNN':
                        feat2dir = cfg.DATASETS.feat2_DIR
                        feat2_dir = os.path.join(feat2dir, '{}.npy'.format(img_info))
                    elif cfg.MODEL.NETWORK == 'STE_CNN' or cfg.MODEL.NETWORK == 'simple_baseline' or cfg.MODEL.NETWORK == 'strong_baseline':
                        feat2_dir = None
                    pid = id_folder # In test time, original Ids are used
                    person_id = pid2label[pid]
                    dataset.append((img_path, person_id, camid, feat2_dir, clothid))
                    pid_container_after.add(person_id)

            # check if pid starts from 0 and increments with 1
            for idx, pid in enumerate(pid_container_after):
                assert idx == pid, "See code comment for explanation"
            return dataset

        else:
            raise ValueError("Set is_train as True or False")



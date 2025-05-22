import glob
import json
import logging
import time
import traceback
import warnings
from collections import defaultdict
from typing import Any, Mapping

import pandas as pd
import torch
from Bio.PDB import PDBParser
from protenix.data.rnafm_featurizer import RNAFMEmbedder
import numpy as np
import warnings
from Bio import BiopythonWarning
from Bio.PDB.PDBExceptions import PDBConstructionWarning
warnings.simplefilter('ignore', BiopythonWarning)
warnings.simplefilter('ignore', PDBConstructionWarning)

from biotite.structure import AtomArray
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from protenix.data.data_pipeline import DataPipeline
from protenix.data.json_to_feature import SampleDictToFeatures
from protenix.data.msa_featurizer import InferenceMSAFeaturizer
from protenix.data.utils import data_type_transform, make_dummy_feature
from protenix.utils.distributed import DIST_WRAPPER
from protenix.utils.torch_utils import dict_to_tensor

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", module="biotite")


# def get_rna_dataloader(configs: Any) -> DataLoader:
def get_rna_dataloader(configs: Any, split: str = "train") -> DataLoader:
    """
    Creates and returns a DataLoader for inference using the SimpleRNADataset.

    Args:
        configs: A configuration object containing the necessary parameters for the DataLoader.

    Returns:
        A DataLoader object configured for inference.
    """
    crop_size = 420
    data_config = configs.data
    for train_name in data_config.train_sets:
        config_dict = data_config[train_name].to_dict()
        crop_size = config_dict['cropping_configs']['crop_size']
    print(f"cropping size is {crop_size}")
    #exit(0)
    inference_dataset = SimpleRNADataset(
        input_json_path=configs.input_json_path,
        dump_dir=configs.dump_dir,
        use_msa=configs.use_msa,
        crop_size = crop_size,
        split=split,
    )
    sampler = DistributedSampler(
        dataset=inference_dataset,
        num_replicas=DIST_WRAPPER.world_size,
        rank=DIST_WRAPPER.rank,
        shuffle=True,
    )
    dataloader = DataLoader(
        dataset=inference_dataset,
        batch_size=1,
        sampler=sampler,
        collate_fn=lambda batch: batch,
        num_workers=configs.num_workers,
    )
    return dataloader


class SimpleRNADataset(Dataset):
    def __init__(
        self,
        input_json_path: str,
        dump_dir: str,
        use_msa: bool = True,
        crop_size: int = 420,
        split: str = "train",
    ) -> None:
        data_dir = '/home/ubuntu/shafi_workspace/Protenix-RNA-Kaggle/data/'
        # label_fn = data_dir + 'train_labels_filtered.csv'
        # label_dict = self.parse_labels(label_fn)

        # pick CSVs based on split
        if split == "train":
            seq_fn   = data_dir + 'train_sequences.csv'
            label_fn = data_dir + 'train_labels.csv'
            is_val   = False
        elif split == "val":
            seq_fn   = data_dir + 'validation_sequences.csv'
            label_fn = data_dir + 'validation_labels.csv'
            is_val   = True
        else:
            raise ValueError(f"Unknown split={split!r}; must be 'train' or 'val'.")

        label_dict = self.parse_labels(label_fn, is_valid=is_val)

        self.input_json_path = input_json_path
        self.dump_dir = dump_dir
        print('use_msa: ', use_msa)
        #exit(0)
        self.use_msa = use_msa
        
        df = pd.read_csv(seq_fn)
        if split == "train":
            df['temporal_cutoff'] = pd.to_datetime(df['temporal_cutoff'], dayfirst=True)

            # 3) Define the cutoff threshold
            cutoff_date = pd.Timestamp('2024-09-18')

            # 4) Filter rows where cutoff is before September 18, 2024
            df = df[df['temporal_cutoff'] < cutoff_date]

        self.inputs = []
        for _, row in df.iterrows():
            target_id = row['target_id']
            sequence = row['sequence']
            assert sequence==label_dict[target_id]['seq']
            if len(sequence) > crop_size:
                continue
            if '-' in sequence:
                print('skip: - sequence ')
                continue
            self.inputs.append({
                "sequences": [{
                 "rnaSequence": {
                    "sequence": sequence,
                    "count": 1,
                     "msa": {
                           "precomputed_msa_dir": "/home/ubuntu/shafi_workspace/Protenix-RNA-Kaggle/data/MSA",
                           "pairing_db": ""
                          },
                  },

                 }],
                "name": target_id
               })
        
        
        print('data samples: ', len(self.inputs))
        self.name_to_xyz = {}
        for name in label_dict.keys():
            self.name_to_xyz[name] = label_dict[name]['xyz'].transpose(1,0,2)
            

        self.crop_size = crop_size
        self.rnafm = RNAFMEmbedder()

    def parse_labels(self,csv_file, is_valid=False):
        pass
        df = pd.read_csv(csv_file)
        df.fillna(0, inplace=True)

        # extract to {pdb_name: {"seq": xxx,  xyz: np.array[n, seq_len, 3] } } if x_i, y_i, z_i not -1e18
        structure_data = defaultdict(lambda: {"seq": "", "xyz": []})

        for _, row in df.iterrows():
            full_id = row["ID"]  # e.g., R1107_1
            resname = row["resname"]  # A, U, G, C
            target_id = "_".join(full_id.split("_")[:-1])  # 提取如 2FY1_B

            structure_data[target_id]["seq"] += resname

            # max_n = 41 if is_valid else  2
            # xyzs = []
            # for i in range(1, max_n):
            #     x, y, z = row[f"x_{i}"], row[f"y_{i}"], row[f"z_{i}"]
            #     if x == -1e18 and y == -1e18 and z == -1e18:
            #         break
            #     xyzs.append([x, y, z])

            xyzs = []
            x, y, z = row["x_1"], row["y_1"], row["z_1"]
            xyzs.append([x, y, z])
            structure_data[target_id]["xyz"].append(xyzs)

        # 转为 NumPy 数组
        for k in structure_data.keys():
            structure_data[k]["xyz"] = np.array(structure_data[k]["xyz"])
        return structure_data


    def parse_pdb_to_xyz(self, pdb_file):
        parser = PDBParser()

        structure = parser.get_structure('', pdb_file)
        assert (len(structure) == 1)
        model = structure[0]
        model = list(model)
        assert (len(model) == 1)
        chain = model[0]

        xyz = []
        seq = []
        pre_resid = 0
        for residue in chain:
            # print(residue)
            if residue.get_resname() in ['A', 'U', 'G', 'C']:
                # Check if the residue has a C1' atom
                if 'C1\'' in residue:
                #if True:
                    atom = residue['C1\'']
                    coord = atom.get_coord()
                    resname = residue.get_resname()
                    resid = residue.get_id()[1]

                    assert (resid == pre_resid + 1)
                    pre_resid = resid

                    xyz.append(coord)
                    seq.append(resname)

        xyz = np.array(xyz)  ##print(f"Residue {resname} {resid}, Atom: {atom.get_name()}, xyz: {xyz}")
        seq = ''.join(seq)
        return xyz, seq

    def process_one(
        self,
        single_sample_dict: Mapping[str, Any],
    ) -> tuple[dict[str, torch.Tensor], AtomArray, dict[str, float]]:
        """
        Processes a single sample from the input JSON to generate features and statistics.

        Args:
            single_sample_dict: A dictionary containing the sample data.

        Returns:
            A tuple containing:
                - A dictionary of features.
                - An AtomArray object.
                - A dictionary of time tracking statistics.
        """
        t0 = time.time()
        xyz = self.name_to_xyz[single_sample_dict["name"]]
        seq = single_sample_dict["sequences"][0]['rnaSequence']['sequence']
        assert  len(seq) == xyz.shape[1]


        if len(seq) > self.crop_size:
            print("crop seq and xyz: ", len(seq))
            # random crop  seq and xyz
            start = np.random.randint(0, len(seq)-self.crop_size)
            end = start + self.crop_size
            seq = seq[start:end]
            xyz = xyz[:, start:end, :]
            single_sample_dict["sequences"][0]['rnaSequence']['sequence'] = seq


        # now only take the top1
        only_top1 = True
        if only_top1:
            xyz = xyz[0]

        sample2feat = SampleDictToFeatures(
            single_sample_dict,
        )
        features_dict, atom_array, token_array = sample2feat.get_feature_dict()
        # ─── RNA-FM embedding ───
        try:
            rnafm_emb = self.rnafm.embed(seq)   # (L,640) np.ndarray
            features_dict["rnafm_embed"] = torch.from_numpy(rnafm_emb)
        except Exception:
            logger.exception("RNA-FM embedding failed for %s", single_sample_dict["name"])


        features_dict["distogram_rep_atom_mask"] = torch.Tensor(
            atom_array.distogram_rep_atom_mask
        ).long()
        entity_poly_type = sample2feat.entity_poly_type

        ##
        coordinate_list = []
        coordinate_mask_list = []

        if only_top1:
            idx = 0
            for atom in atom_array:
                if atom.atom_name == 'C1\'':
                    coordinate_list.append(xyz[idx])
                    if xyz[idx][0] <= -1e8:
                        coordinate_mask_list.append(0)
                    else:
                        coordinate_mask_list.append(1)
                    idx += 1
                else:
                    coordinate_list.append([0, 0, 0])
                    coordinate_mask_list.append(0)
            # print("xyz.shape, idx: ", xyz.shape, idx)
            assert idx == len(xyz)
        else:
            n_gt = len(xyz)
            print('n_gt: ',  n_gt)
            coordinate_zero =[ [0, 0, 0] ] * n_gt

            idx = 0
            for atom in atom_array:
                if atom.atom_name == 'C1\'':
                    coordinate_list.append(xyz[:, idx].tolist() )
                    if xyz[0, idx][0] <= -1e8:
                        coordinate_mask_list.append(0)
                    else:
                        coordinate_mask_list.append(1)
                    idx += 1
                else:
                    coordinate_list.append(coordinate_zero)
                    coordinate_mask_list.append(0)
            assert idx == xyz.shape[1]

        t1 = time.time()

        # Msa features
        entity_to_asym_id = DataPipeline.get_label_entity_id_to_asym_id_int(atom_array)
        msa_features = (
            InferenceMSAFeaturizer.make_msa_feature(
                pdb_name = single_sample_dict["name"],
                bioassembly=single_sample_dict["sequences"],
                entity_to_asym_id=entity_to_asym_id,
                token_array=token_array,
                atom_array=atom_array,
            )
            if self.use_msa
            else {}
        )

        # Make dummy features for not implemented features
        dummy_feats = ["template"]
        if len(msa_features) == 0:
            dummy_feats.append("msa")
        else:
            msa_features = dict_to_tensor(msa_features)
            features_dict.update(msa_features)
        features_dict = make_dummy_feature(
            features_dict=features_dict,
            dummy_feats=dummy_feats,
        )

        # Transform to right data type
        feat = data_type_transform(feat_or_label_dict=features_dict)

        t2 = time.time()

        data = {}
        data["input_feature_dict"] = feat

        # Add dimension related items
        N_token = feat["token_index"].shape[0]
        N_atom = feat["atom_to_token_idx"].shape[0]
        N_msa = feat["msa"].shape[0]

        stats = {}
        for mol_type in ["ligand", "protein", "dna", "rna"]:
            mol_type_mask = feat[f"is_{mol_type}"].bool()
            stats[f"{mol_type}/atom"] = int(mol_type_mask.sum(dim=-1).item())
            stats[f"{mol_type}/token"] = len(
                torch.unique(feat["atom_to_token_idx"][mol_type_mask])
            )

        N_asym = len(torch.unique(data["input_feature_dict"]["asym_id"]))
        data.update(
            {
                "N_asym": torch.tensor([N_asym]),
                "N_token": torch.tensor([N_token]),
                "N_atom": torch.tensor([N_atom]),
                "N_msa": torch.tensor([N_msa]),
            }
        )

        def formatted_key(key):
            type_, unit = key.split("/")
            if type_ == "protein":
                type_ = "prot"
            elif type_ == "ligand":
                type_ = "lig"
            else:
                pass
            return f"N_{type_}_{unit}"

        data.update(
            {
                formatted_key(k): torch.tensor([stats[k]])
                for k in [
                    "protein/atom",
                    "ligand/atom",
                    "dna/atom",
                    "rna/atom",
                    "protein/token",
                    "ligand/token",
                    "dna/token",
                    "rna/token",
                ]
            }
        )
        data.update({"entity_poly_type": entity_poly_type})
        t3 = time.time()
        time_tracker = {
            "crop": t1 - t0,
            "featurizer": t2 - t1,
            "added_feature": t3 - t2,
        }
        if only_top1:
            data['coordinate'] = torch.from_numpy(np.array(coordinate_list)).float()
            data['coordinate_mask'] = torch.from_numpy(np.array(coordinate_mask_list)).long()
        else:
            # seg_len, 5, 3 --> 5, seg_len, 3
            c = np.ascontiguousarray(np.array(coordinate_list,
                                              dtype=np.float32).transpose(1, 0, 2))
            data['coordinate_multi'] = torch.from_numpy(c).float()
            data['coordinate_mask'] = torch.from_numpy(np.array(coordinate_mask_list)).float()
            data['coordinate'] = data['coordinate_multi'][0]

        return data, atom_array, time_tracker

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int):
        single_sample_dict = self.inputs[index].copy()
        data, atom_array, _ = self.process_one(
            single_sample_dict=single_sample_dict
        )
        # data["sample_name"] = single_sample_dict["name"]
        # data["sample_index"] = index
        # return data#, atom_array, error_message
        # for trainer._evaluate() we need batch["basic"]["pdb_id"]
        data["basic"] = {
            "pdb_id": single_sample_dict["name"]
        }
        data["sample_name"]  = single_sample_dict["name"]
        data["sample_index"] = index
        return data

if __name__ == '__main__':
    dset = SimpleRNADataset("", "")

# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import glob
import json
import logging
import time
import traceback
import warnings
from typing import Any, Mapping

import torch
from Bio.PDB import PDBParser
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


def get_inference_dataloader(configs: Any) -> DataLoader:
    """
    Creates and returns a DataLoader for inference using the InferenceDataset.

    Args:
        configs: A configuration object containing the necessary parameters for the DataLoader.

    Returns:
        A DataLoader object configured for inference.
    """
    inference_dataset = InferenceDataset(
        input_json_path=configs.input_json_path,
        dump_dir=configs.dump_dir,
        use_msa=configs.use_msa,
    )
    sampler = DistributedSampler(
        dataset=inference_dataset,
        num_replicas=DIST_WRAPPER.world_size,
        rank=DIST_WRAPPER.rank,
        shuffle=False,
    )
    dataloader = DataLoader(
        dataset=inference_dataset,
        batch_size=1,
        sampler=sampler,
        collate_fn=lambda batch: batch,
        num_workers=configs.num_workers,
    )
    return dataloader


class InferenceDataset(Dataset):
    def __init__(
        self,
        input_json_path: str,
        dump_dir: str,
        use_msa: bool = True,
    ) -> None:

        self.input_json_path = input_json_path
        self.dump_dir = dump_dir
        self.use_msa = use_msa
        with open(self.input_json_path, "r") as f:
            self.inputs = json.load(f)
            #self.inputs =  self.inputs[:23]
        print(self.inputs[0])
        #exit(0)
#         casp_vfold_result_dir = '/home/lhw/work/rna2025/data/casp16/vfold-prediction/'
#         self.name_to_xyz = {}
#         for sample in self.inputs:
#             name = sample["name"]
#             fn = casp_vfold_result_dir+name
#             names = glob.glob(fn+"/*_1.pdb")
#             assert len(names) == 1
#             fn = names[0]
#             xyz, _ = self.parse_pdb_to_xyz(fn)
#             print(xyz.shape)
#             self.name_to_xyz[name] = xyz
        #exit(0)

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
        # general features
        t0 = time.time()
        sample2feat = SampleDictToFeatures(
            single_sample_dict,
        )
        features_dict, atom_array, token_array = sample2feat.get_feature_dict()
        features_dict["distogram_rep_atom_mask"] = torch.Tensor(
            atom_array.distogram_rep_atom_mask
        ).long()
        entity_poly_type = sample2feat.entity_poly_type

#         ##
#         coordinate_list = []
#         coordinate_mask_list = []
#         xyz = self.name_to_xyz[single_sample_dict["name"]]


#         # extra_features["atom_to_tokatom_idx"] = torch.Tensor(
#         #     self.cropped_atom_array.tokatom_idx
#         # ).long()
#         idx = 0
#         for atom in atom_array:
#             # print('res_name: ', atom.res_name)
#             # print('atom.atom_name: ', atom.atom_name)
#             if atom.atom_name == 'C1\'':
#                 coordinate_list.append(xyz[idx])
#                 coordinate_mask_list.append(1)
#                 idx += 1
#             else:
#                 coordinate_list.append([0,0,0])
#                 coordinate_mask_list.append(0)
#         #print("xyz.shape, idx: ", xyz.shape, idx)
#         assert idx == len(xyz)


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
        # print(msa_features)
        # exit(0)
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
#         data['coordinate'] = torch.from_numpy(np.array(coordinate_list)).float()
#         data['coordinate_mask'] = torch.from_numpy(np.array(coordinate_mask_list)).long()

        return data, atom_array, time_tracker

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int) -> tuple[dict[str, torch.Tensor], AtomArray, str]:
        try:
            single_sample_dict = self.inputs[index]
            #sample_name = single_sample_dict["name"]
            #logger.info(f"Featurizing {sample_name}...")

            data, atom_array, _ = self.process_one(
                single_sample_dict=single_sample_dict
            )
            error_message = ""
        except Exception as e:
            data, atom_array = {}, None
            error_message = f"{e}:\n{traceback.format_exc()}"
        data["sample_name"] = single_sample_dict["name"]
        data["sample_index"] = index
        return data, atom_array, error_message

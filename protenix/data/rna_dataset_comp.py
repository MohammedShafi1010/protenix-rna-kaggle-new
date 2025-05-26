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
from Bio.PDB import PDBParser, MMCIFParser
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


def get_rna_dataloader(configs: Any, split: str = "train") -> DataLoader:
    """
    Creates and returns a DataLoader for inference using the SimpleRNADataset.

    Args:
        configs: A configuration object containing the necessary parameters for the DataLoader.
        split: Dataset split ('train' or 'val')

    Returns:
        A DataLoader object configured for inference.
    """
    crop_size = 420
    data_config = configs.data
    for train_name in data_config.train_sets:
        config_dict = data_config[train_name].to_dict()
        crop_size = config_dict['cropping_configs']['crop_size']
    print(f"cropping size is {crop_size}")
    
    inference_dataset = SimpleRNADataset(
        input_json_path=configs.input_json_path,
        dump_dir=configs.dump_dir,
        use_msa=configs.use_msa,
        crop_size=crop_size,
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
        self.input_json_path = input_json_path
        self.dump_dir = dump_dir
        self.use_msa = use_msa
        self.crop_size = crop_size
        self.split = split

        if split == "train":
            self.cif_dir = "/home/ubuntu/shafi_workspace/Protenix-RNA-Kaggle/data/Train_dataset/"
        if split == "val":
            self.cif_dir = "/home/ubuntu/shafi_workspace/Protenix-RNA-Kaggle/data/Test_dataset/"
        
        print('use_msa: ', use_msa)
        
        # Get all CIF files in the directory
        cif_files = glob.glob(f"{self.cif_dir}/*.cif")
        print(f"Found {len(cif_files)} CIF files")
        
        self.inputs = []
        self.name_to_data = {}
        
        # Process each CIF file
        for cif_file in cif_files:
            try:
                structure_name = self.extract_structure_name(cif_file)
                sequence, coordinates, atom_names = self.parse_cif_file(cif_file)
                
                if len(sequence) == 0:
                    print(f"Skipping {cif_file}: No RNA sequence found")
                    continue
                    
                if '-' in sequence:
                    print(f'Skipping {cif_file}: Contains gap characters')
                    continue
                # Skip if sequence is too long
                if len(sequence) > crop_size:
                    print(f'Skipping {cif_file}: Sequence too long ({len(sequence)} > {crop_size})')
                    continue
                # Store the data
                self.name_to_data[structure_name] = {
                    'sequence': sequence,
                    'coordinates': coordinates,  # Shape: (n_atoms, 3)
                    'atom_names': atom_names,    # List of atom names
                }
                
                # Create input entry
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
                    "name": structure_name
                })
                
                print(f"Processed {structure_name}: {len(sequence)} residues, {len(coordinates)} atoms")
                
            except Exception as e:
                print(f"Error processing {cif_file}: {str(e)}")
                continue
        
        print(f'Total data samples: {len(self.inputs)}')
        self.rnafm = RNAFMEmbedder()

    def extract_structure_name(self, cif_file: str) -> str:
        """Extract structure name from CIF file path"""
        import os
        return os.path.splitext(os.path.basename(cif_file))[0]

    def parse_cif_file(self, cif_file: str):
        """
        Parse CIF file to extract RNA sequence and atom coordinates
        
        Returns:
            sequence: RNA sequence string
            coordinates: numpy array of shape (n_atoms, 3)
            atom_names: list of atom names
        """
        ATOM_INDEX_MAP = {
            "A": {"OP3":0,"P":1,"OP1":2,"OP2":3,"O5'":4,"C5'":5,"C4'":6,"O4'":7,
                "C3'":8,"O3'":9,"C2'":10,"O2'":11,"C1'":12,"N9":13,"C8":14,
                "N7":15,"C5":16,"C6":17,"N6":18,"N1":19,"C2":20,"N3":21,"C4":22},
            "C": {"OP3":0,"P":1,"OP1":2,"OP2":3,"O5'":4,"C5'":5,"C4'":6,"O4'":7,
                "C3'":8,"O3'":9,"C2'":10,"O2'":11,"C1'":12,"N1":13,"C2":14,
                "O2":15,"N3":16,"C4":17,"N4":18,"C5":19,"C6":20},
            "G": {"OP3":0,"P":1,"OP1":2,"OP2":3,"O5'":4,"C5'":5,"C4'":6,"O4'":7,
                "C3'":8,"O3'":9,"C2'":10,"O2'":11,"C1'":12,"N9":13,"C8":14,
                "N7":15,"C5":16,"C6":17,"O6":18,"N1":19,"C2":20,"N2":21,"N3":22,"C4":23},
            "U": {"OP3":0,"P":1,"OP1":2,"OP2":3,"O5'":4,"C5'":5,"C4'":6,"O4'":7,
                "C3'":8,"O3'":9,"C2'":10,"O2'":11,"C1'":12,"N1":13,"C2":14,
                "O2":15,"N3":16,"C4":17,"O4":18,"C5":19,"C6":20},
        }
        
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure('', cif_file)
        
        # Use only first model
        model = structure[0]
        
        sequence = ""
        coordinates = []
        atom_names = []
        residue_index = 0
        
        # Process all chains
        for chain in model:
            for residue in chain:
                resname = residue.get_resname()
                
                # Check if it's an RNA nucleotide
                if resname in ['A', 'U', 'G', 'C']:
                    sequence += resname
                    
                    # Get expected atoms for this nucleotide type
                    expected_atoms = ATOM_INDEX_MAP[resname]
                    
                    # Skip OP3 for all residues except the first one
                    if residue_index > 0 and "OP3" in expected_atoms:
                        expected_atoms_filtered = {k: v for k, v in expected_atoms.items() if k != "OP3"}
                    else:
                        expected_atoms_filtered = expected_atoms
                    
                    # Collect actual atoms from residue
                    actual_atoms = {}
                    for atom in residue:
                        atom_name = atom.get_name()
                        if atom_name in expected_atoms_filtered:
                            actual_atoms[atom_name] = atom.get_coord()
                    
                    # Add atoms in the expected order
                    for atom_name in sorted(expected_atoms_filtered.keys(), key=lambda x: expected_atoms_filtered[x]):
                        atom_names.append(atom_name)
                        if atom_name in actual_atoms:
                            coordinates.append(actual_atoms[atom_name])
                        else:
                            # Add empty coordinate for missing atoms
                            coordinates.append([9999, 9999, 9999])
                    
                    residue_index += 1
        
        return sequence, np.array(coordinates), atom_names


    def process_one(
        self,
        single_sample_dict: Mapping[str, Any],
    ) -> tuple[dict[str, torch.Tensor], AtomArray, dict[str, float]]:
        """
        Processes a single sample from the input to generate features and statistics.
        """
        t0 = time.time()

        # ——— LOG CURRENT SAMPLE INFO ———
        pdb_id = single_sample_dict["name"]
        seq = self.name_to_data[pdb_id]["sequence"]
        # fetch the precomputed_msa_dir from the input spec (if any)
        msa_info = single_sample_dict["sequences"][0]["rnaSequence"].get("msa", {})
        msa_dir = msa_info.get("precomputed_msa_dir", "N/A")
        logger.info(
            f"⟳ Processing sample {pdb_id} | seq_len={len(seq)} | msa_dir={msa_dir}"
        )
        # —————————————————————————————
        
        structure_name = single_sample_dict["name"]
        structure_data = self.name_to_data[structure_name]
        
        sequence = structure_data['sequence']
        coordinates = structure_data['coordinates']
        atom_names = structure_data['atom_names']
        # print(atom_names)
        # print(len(atom_names))
        # print(coordinates)
        
        sample2feat = SampleDictToFeatures(single_sample_dict)
        features_dict, atom_array, token_array = sample2feat.get_feature_dict()

        # RNA-FM embedding
        try:
            rnafm_emb = self.rnafm.embed(sequence)   # (L,640) np.ndarray
            features_dict["rnafm_embed"] = torch.from_numpy(rnafm_emb)
        except Exception:
            logger.exception("RNA-FM embedding failed for %s", structure_name)

        features_dict["distogram_rep_atom_mask"] = torch.Tensor(
            atom_array.distogram_rep_atom_mask
        ).long()
        entity_poly_type = sample2feat.entity_poly_type

        # Map coordinates to atom array
        coordinate_list = []
        coordinate_mask_list = []
        
        # define a threshold for “too large” values
        MAX_COORD = 1000

        coord_idx = 0
        for atom in atom_array:
            if coord_idx < len(coordinates):
                raw_coord = coordinates[coord_idx]
                valid = True
                try:
                    arr = np.array(raw_coord, dtype=float)
                    # must be length-3, finite, and not exceed threshold
                    if arr.shape != (3,) or not np.all(np.isfinite(arr)) or np.any(np.abs(arr) > MAX_COORD):
                        valid = False
                except Exception:
                    valid = False

                if valid:
                    coordinate_list.append(arr.tolist())
                    coordinate_mask_list.append(1)
                else:
                    # invalid → zero out
                    coordinate_list.append([0.0, 0.0, 0.0])
                    coordinate_mask_list.append(0)
                coord_idx += 1
            else:
                # ran out of real coordinates
                coordinate_list.append([0.0, 0.0, 0.0])
                coordinate_mask_list.append(0)

        t1 = time.time()

        # MSA features
        entity_to_asym_id = DataPipeline.get_label_entity_id_to_asym_id_int(atom_array)
        msa_features = (
            InferenceMSAFeaturizer.make_msa_feature(
                pdb_name=structure_name,
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
        logger.info(
            (
                f"N_asym {data['N_asym'].item()}, N_token {data['N_token'].item()}, "
                f"N_atom {data['N_atom'].item()}, N_msa {data['N_msa'].item()}"
            )
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
        
        data['coordinate'] = torch.from_numpy(np.array(coordinate_list)).float()
        data['coordinate_mask'] = torch.from_numpy(np.array(coordinate_mask_list)).long()
        return data, atom_array, time_tracker

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int):
        single_sample_dict = self.inputs[index].copy()
        data, atom_array, _ = self.process_one(
            single_sample_dict=single_sample_dict
        )
        
        # For trainer._evaluate() we need batch["basic"]["pdb_id"]
        data["basic"] = {
            "pdb_id": single_sample_dict["name"]
        }
        data["sample_name"] = single_sample_dict["name"]
        data["sample_index"] = index
        # print(data)
        return data


if __name__ == '__main__':
    dset = SimpleRNADataset("", "")
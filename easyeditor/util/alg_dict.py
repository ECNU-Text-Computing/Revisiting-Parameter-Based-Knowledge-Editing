"""Algorithm registry: maps method names to their apply functions and hyperparameter classes."""

from ..models.rome import ROMEHyperParams, apply_rome_to_model
from ..models.memit import MEMITHyperParams, apply_memit_to_model
from ..models.mend import MENDHyperParams, MendRewriteExecutor
from ..models.ft import FTHyperParams, apply_ft_to_model
from ..models.ike import IKEHyperParams, apply_ike_to_model
from ..models.lora import LoRAHyperParams, apply_lora_to_model
from ..models.grace import GraceHyperParams, apply_grace_to_model
from ..models.pmet import PMETHyperParams, apply_pmet_to_model
from ..models.wise import WISEHyperParams, apply_wise_to_model
from ..models.alphaedit import AlphaEditHyperParams, apply_AlphaEdit_to_model
from ..dataset import ZsreDataset, CounterFactDataset

ALG_DICT = {
    'ROME': apply_rome_to_model,
    'MEMIT': apply_memit_to_model,
    'FT': apply_ft_to_model,
    'MEND': MendRewriteExecutor().apply_to_model,
    'IKE': apply_ike_to_model,
    'LoRA': apply_lora_to_model,
    'GRACE': apply_grace_to_model,
    'PMET': apply_pmet_to_model,
    'WISE': apply_wise_to_model,
    'AlphaEdit': apply_AlphaEdit_to_model,
}

DS_DICT = {
    'cf': CounterFactDataset,
    'zsre': ZsreDataset,
}

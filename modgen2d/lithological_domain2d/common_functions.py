# This file is part of modgen2d a Python package for ...
# Copyright (C) XXXX Joseph P. Vantassel (joseph.p.vantassel@gmail.com)
#
# LICENSE

"""Define a two-dimensional domain that defines lithology."""
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
import numpy as np
import warnings
  
def _extract_lithological_fields(ld):
    """
    Extract lithological domain fields from either a ReadOnly object or a dict.

    Parameters
    ----------
    ld : LithologicalDomain2DReadOnly or dict
        Either a domain object with attributes (lm_type, domain,
        lithological_matrix, interface_config_revision_id)
        or a dict containing these keys.

    Returns
    -------
    tuple
        A 4-tuple: (lm_type, domain, lithological_matrix, interface_config_revision_id)

    Raises
    ------
    KeyError
        If `ld` is a dict and required keys are missing.
    """
    if isinstance(ld, dict):
        return (
            ld["lm_type"],
            ld["domain"],
            ld["lithological_matrix"],
            ld["interface_config_revision_id"]
        )
    else:
        return (
            ld.lm_type,
            ld.domain,
            ld.lithological_matrix,
            ld.interface_config_revision_id
        )
        
def _merge_lithological_domains(lithological_domain_A, lithological_domain_B, lit_id_none_list=['X']):
    """
    Merge two lithological domains (either ReadOnly objects or dicts).

    Priority rule: Values from Domain B overwrite values from Domain A,
    except when B contains a 'none' lithological ID (specified in `lit_id_none_list`).

    Parameters
    ----------
    lithological_domain_A : LithologicalDomain2DReadOnly or dict
        First lithological domain. May be a ReadOnly object or a dict with keys:
        'lm_type', 'domain', 'lithological_matrix', 'interface_config_revision_id'.

    lithological_domain_B : LithologicalDomain2DReadOnly or dict
        Second lithological domain. Priority is given to this domain during merge.
        Must match domain size, extents, and surface revision ID of A.

    lit_id_none_list : list of str, optional
        List of lithology IDs that represent "none" (i.e., no material).
        Defaults to ['X'].

    Returns
    -------
    merged_matrix : np.ndarray
        A numpy array representing the merged lithological matrix.

    overlap : bool
        True if both domains had non-none values at the same grid locations,
        indicating overlapping materials (useful for obstruction/utility detection).

    Notes
    -----
    - Supports both dict-based and ReadOnly object inputs.
    - Domain B overwrites A, unless its value is a 'none' ID.
    - Overlap detection checks where both A and B have valid (non-none) values.
    """

    lm_type_A, domain_A, matrixA, rev_id_A = _extract_lithological_fields(lithological_domain_A)
    lm_type_B, domain_B, matrixB, rev_id_B = _extract_lithological_fields(lithological_domain_B)

    # Checks
    assert lm_type_B != "NA", "lm_type cannot be NA"
    
    if lm_type_A == "NA":
        return matrixB, False

    if domain_A != domain_B:
        raise TypeError(
            f"Domains of A and B do not match. "
            f"A: spans={domain_A.spans}, dhs={domain_A.dhs}; "
            f"B: spans={domain_B.spans}, dhs={domain_B.dhs}"
        )

    if matrixA.shape != matrixB.shape:
        raise ValueError(
            f"Expected same lithological domain matrices size, got {matrixA.shape} != {matrixB.shape}"
        )

    if rev_id_A != rev_id_B:
        raise ValueError("Surface config revision IDs do not match. Both A and B must be for same interfaces configuration. Refresh if needed") ##TODO

    if not isinstance(lit_id_none_list, list) or not lit_id_none_list or not all(isinstance(item, str) for item in lit_id_none_list):
        raise ValueError("lit_id_none_list must be a non-empty list of strings.")

    # Masks to identify non-'none' values
    mask_A_non_none = ~np.isin(matrixA, lit_id_none_list)
    mask_B_non_none = ~np.isin(matrixB, lit_id_none_list)

    # B overwrites A 
    merged_lithological_matrix = np.where(mask_B_non_none, matrixB, matrixA)
    # (replace it with that of B even if merged already have values, i.e prioritize B)

    # Overlap detection - #Check if overlap with merged layers (useful for obstructions)
    overlap = np.sum(mask_A_non_none * mask_B_non_none)>0
    return merged_lithological_matrix, overlap

    # self.lithological_matrix = merged
    # self.overlap = overlap_check>0 #Overlap with merged layers (useful for utils)
    # self.utils_description = f'{self.utils_description} + {lithological_domain_B.utils_description}'

def _warn_if_changed(a, b, msg="On refreshing the lithological domain, lithological_matrix changed."):
    if a is None and b is None:
        changed = False
    elif a is None or b is None:
        changed = True
    elif np.isscalar(a) and np.isscalar(b):
        changed = a != b
    elif isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        changed = not np.array_equal(a, b)
    else:
        # one is scalar, the other is array → consider changed
        changed = True

    if changed:
        warnings.warn(msg)





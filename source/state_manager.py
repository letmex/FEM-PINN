from dataclasses import dataclass
from typing import Dict, Optional

import torch

from thermo_mech_model import element_to_nodal


@dataclass
class InstantaneousState:
    """Instantaneous fields/mechanics for one inner iteration."""

    T: torch.Tensor
    ux: torch.Tensor
    uy: torch.Tensor
    d_iter: torch.Tensor
    psi_I_elem: torch.Tensor
    psi_II_elem: torch.Tensor
    mech_state: Dict[str, torch.Tensor]


@dataclass
class CandidateState:
    """Candidate (not yet accepted) path-dependent state for current time step."""

    HI_elem: torch.Tensor
    HII_elem: torch.Tensor
    He_elem: torch.Tensor
    d_cand: torch.Tensor


@dataclass
class AcceptedState:
    """Accepted (step-frozen) state propagated between time steps."""

    step: int
    time: float
    T: torch.Tensor
    ux: torch.Tensor
    uy: torch.Tensor
    d: torch.Tensor
    HI_elem: torch.Tensor
    HII_elem: torch.Tensor
    He_elem: torch.Tensor


class PathDependentStateManager:
    """
    FE-style path-dependent state machine.

    State layers:
      1) accepted: step-frozen history variables
      2) current_ref: reference state used inside inner loop
      3) candidate: trial state used by phase solve and acceptance logic

    This explicitly decouples path-dependent updates from optimizer internals.
    """

    def __init__(self, thermo_model, T_conn, area_elem, n_nodes, history_update_mode="step_end"):
        self.thermo_model = thermo_model
        self.T_conn = T_conn
        self.area_elem = area_elem
        self.n_nodes = int(n_nodes)
        self.history_update_mode = str(history_update_mode).lower()
        if self.history_update_mode not in ("step_end", "inner_accumulate"):
            raise ValueError("history_update_mode must be 'step_end' or 'inner_accumulate'.")

        zeros = torch.zeros_like(area_elem)
        self._accepted_HI = zeros.clone()
        self._accepted_HII = zeros.clone()
        self._current_ref_HI = zeros.clone()
        self._current_ref_HII = zeros.clone()
        self._candidate_HI = zeros.clone()
        self._candidate_HII = zeros.clone()

    def load_from_tensors(self, HI_elem, HII_elem):
        self._accepted_HI = HI_elem.detach().clone()
        self._accepted_HII = HII_elem.detach().clone()
        self.step_begin()

    def step_begin(self):
        self._current_ref_HI = self._accepted_HI.detach().clone()
        self._current_ref_HII = self._accepted_HII.detach().clone()
        self._candidate_HI = self._accepted_HI.detach().clone()
        self._candidate_HII = self._accepted_HII.detach().clone()

    def build_candidate(self, psi_I_elem, psi_II_elem):
        psi_I = psi_I_elem.detach()
        psi_II = psi_II_elem.detach()

        HI_base = self._accepted_HI if self.history_update_mode == "step_end" else self._current_ref_HI
        HII_base = self._accepted_HII if self.history_update_mode == "step_end" else self._current_ref_HII

        self._candidate_HI = torch.maximum(HI_base, psi_I)
        self._candidate_HII = torch.maximum(HII_base, psi_II)

        if self.history_update_mode == "inner_accumulate":
            self._current_ref_HI = self._candidate_HI.detach().clone()
            self._current_ref_HII = self._candidate_HII.detach().clone()

        He_elem = self._candidate_HI + self.thermo_model.gc_ratio * self._candidate_HII
        return self._candidate_HI, self._candidate_HII, He_elem

    def accept_step(self, psi_I_elem, psi_II_elem):
        if self.history_update_mode == "step_end":
            self._accepted_HI = torch.maximum(self._accepted_HI, psi_I_elem.detach())
            self._accepted_HII = torch.maximum(self._accepted_HII, psi_II_elem.detach())
        else:
            self._accepted_HI = self._candidate_HI.detach().clone()
            self._accepted_HII = self._candidate_HII.detach().clone()
        self.step_begin()

    def element_state(self, use_candidate=True):
        if use_candidate:
            HI_elem = self._candidate_HI
            HII_elem = self._candidate_HII
        else:
            HI_elem = self._accepted_HI
            HII_elem = self._accepted_HII
        He_elem = HI_elem + self.thermo_model.gc_ratio * HII_elem
        return HI_elem, HII_elem, He_elem

    def nodal_state(self, use_candidate=True):
        HI_elem, HII_elem, He_elem = self.element_state(use_candidate=use_candidate)
        HI_nodes = element_to_nodal(HI_elem, self.T_conn, self.n_nodes, area_elem=self.area_elem)
        HII_nodes = element_to_nodal(HII_elem, self.T_conn, self.n_nodes, area_elem=self.area_elem)
        He_nodes = element_to_nodal(He_elem, self.T_conn, self.n_nodes, area_elem=self.area_elem)
        return HI_nodes, HII_nodes, He_nodes

    def checkpoint_payload(self) -> Dict[str, torch.Tensor]:
        return {
            "HI_elem_prev": self._accepted_HI.detach().cpu(),
            "HII_elem_prev": self._accepted_HII.detach().cpu(),
        }

    def summary(self) -> Dict[str, float]:
        HI_elem, HII_elem, He_elem = self.element_state(use_candidate=False)
        return {
            "max_HI_elem": float(torch.max(HI_elem).item()),
            "max_HII_elem": float(torch.max(HII_elem).item()),
            "max_He_elem": float(torch.max(He_elem).item()),
        }


# Backward-compatible alias used by legacy code paths.
HistoryPhaseStateManager = PathDependentStateManager

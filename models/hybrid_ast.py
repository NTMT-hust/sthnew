"""HGnnDTI-inspired isolate-antibiotic heterogeneous graph model."""

import torch
from torch import nn
from torch.nn import functional as F


ELEMENTS = ("C", "N", "O", "S", "F", "Cl", "Br", "P", "I", "B")
ATOM_DIM = len(ELEMENTS) + 7 + 5 + 2


def atom_features(atom):
    e = [float(atom.GetSymbol() == symbol) for symbol in ELEMENTS]
    if not any(e):
        e = [0.0] * len(ELEMENTS)
    degree = [float(atom.GetDegree() == d) for d in range(6)] + [float(atom.GetDegree() >= 6)]
    charge = [float(atom.GetFormalCharge() == c) for c in range(-2, 3)]
    return e + degree + charge + [float(atom.GetIsAromatic()), float(atom.IsInRing())]


def smiles_graph(smiles: str, Chem):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or molecule.GetNumAtoms() == 0:
        raise ValueError(f"Invalid SMILES: {smiles}")
    x = torch.tensor([atom_features(atom) for atom in molecule.GetAtoms()], dtype=torch.float32)
    adj = torch.eye(molecule.GetNumAtoms(), dtype=torch.float32)
    for bond in molecule.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adj[a, b] = adj[b, a] = 1.0
    degree = adj.sum(1).clamp_min(1).rsqrt()
    return x, degree[:, None] * adj * degree[None, :]


class MolecularGraphEncoder(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.atom_in = nn.Linear(ATOM_DIM, hidden)
        self.conv1 = nn.Linear(hidden, hidden)
        self.conv2 = nn.Linear(hidden, hidden)

    def forward(self, graphs, device):
        vectors = []
        for x, adj in graphs:
            x, adj = x.to(device), adj.to(device)
            h = F.relu(self.atom_in(x))
            h = F.relu(self.conv1(adj @ h))
            h = F.relu(self.conv2(adj @ h))
            vectors.append(h.mean(0))
        return torch.stack(vectors)


class HeterogeneousASTModel(nn.Module):
    def __init__(self, spectrum_dim: int, hidden: int = 128, n_classes: int = 2):
        super().__init__()
        self.drug_encoder = MolecularGraphEncoder(hidden)
        self.isolate_encoder = nn.Sequential(nn.Linear(spectrum_dim, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.ast_relation = nn.Embedding(n_classes, hidden)
        self.isolate_msg = nn.Linear(hidden, hidden, bias=False)
        self.isolate_fuse = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.drug_fuse = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.decoder = nn.Sequential(nn.Linear(hidden * 4, hidden), nn.GELU(), nn.Dropout(0.2),
                                     nn.Linear(hidden, n_classes))

    def encode_nodes(self, spectra, drug_graphs, edge_iso, edge_drug, edge_label,
                     excluded_edges, device):
        isolate = self.isolate_encoder(spectra)
        drug = self.drug_encoder(drug_graphs, device)
        keep = torch.ones(edge_iso.numel(), dtype=torch.bool, device=device)
        if excluded_edges.numel():
            keep[excluded_edges] = False
        src, dst = edge_iso[keep], edge_drug[keep]
        relation = self.ast_relation(edge_label[keep])
        iso_msg = torch.zeros_like(isolate)
        iso_msg.index_add_(0, src, drug[dst] + relation)
        iso_degree = torch.zeros(isolate.size(0), device=device)
        iso_degree.index_add_(0, src, torch.ones_like(src, dtype=torch.float32))
        iso_msg /= iso_degree.clamp_min(1).unsqueeze(-1)
        isolate = self.isolate_fuse(torch.cat((isolate, iso_msg), dim=-1))

        drug_msg = torch.zeros_like(drug)
        drug_msg.index_add_(0, dst, self.isolate_msg(isolate[src]) + relation)
        drug_degree = torch.zeros(drug.size(0), device=device)
        drug_degree.index_add_(0, dst, torch.ones_like(dst, dtype=torch.float32))
        drug_msg /= drug_degree.clamp_min(1).unsqueeze(-1)
        drug = self.drug_fuse(torch.cat((drug, drug_msg), dim=-1))
        return isolate, drug

    def classify_edges(self, isolate, drug, edge_iso, edge_drug):
        a, b = isolate[edge_iso], drug[edge_drug]
        return self.decoder(torch.cat((a, b, a * b, torch.abs(a - b)), dim=-1))

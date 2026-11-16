"""MaxSim over the packed bank must equal brute-force per-page MaxSim (needs torch; skipped without)."""


def test_maxsim_bank_matches_bruteforce():
    try:
        import torch
    except ImportError:
        print("  skip: torch not installed")
        return
    from src.retriever import maxsim_scores, pack_bank

    torch.manual_seed(0)
    norm = torch.nn.functional.normalize
    lengths = (3, 9, 5, 1, 7)
    embs = [norm(torch.randn(n, 8), dim=-1) for n in lengths]
    q = norm(torch.randn(4, 8), dim=-1)
    ref = torch.stack([(q @ e.T).amax(dim=1).sum() for e in embs])
    bank = pack_bank(embs, device="cpu", chunk_size=2)          # pages sorted by length, padded per chunk
    assert len(bank) == 5 and [c.shape[1] for c in bank.chunks] == [3, 7, 9]
    assert all(c.dtype == torch.float16 for c in bank.chunks)
    assert torch.allclose(maxsim_scores(q, bank, device="cpu"), ref, atol=1e-2)
    rows = torch.tensor([4, 0, 1])
    assert torch.allclose(maxsim_scores(q, bank, rows, device="cpu"), ref[rows], atol=1e-2)

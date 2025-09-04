import torch

from audio_metrics.data import AudioMetricsData


def test_incremental_stats():
    # test equivalence between incremental and non-incremental stats
    n_dim = 8

    x1 = torch.randn((1, n_dim))
    x2 = torch.randn((100, n_dim))
    x3 = torch.randn((1000, n_dim))

    x123 = torch.cat((x1, x2, x3))

    amd = AudioMetricsData(store_embeddings=False)
    # add x1, x2, x3 separately
    amd.add(x1)
    amd.add(x2)
    amd.add(x3)
    m_a = amd.mean
    c_a = amd.cov

    amd = AudioMetricsData(store_embeddings=False)
    # add x1, x2, x3 as a single tensor
    amd.add(x123)
    m_b = amd.mean
    c_b = amd.cov

    torch.testing.assert_close(m_a, m_b, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(c_a, c_b, rtol=1e-6, atol=1e-6)

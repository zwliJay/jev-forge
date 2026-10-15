"""Complete-question packing must account for shared padding across questions."""
from jevforge.train import batches_by_tokens


def test_shared_padding_budget_and_question_integrity():
    # Both individual questions fit 1024, but their combined padded tensor does not.
    short = {'id': 'short', 'leaves': [[1] * 64 for _ in range(10)]}
    long = {'id': 'long', 'leaves': [[1] * 512 for _ in range(2)]}
    batches = batches_by_tokens([short, long], 8, 1024)
    consumed = [item for _, micro in batches for item in micro]
    assert consumed == [short, long]
    assert len(batches) == 2
    for _, micro in batches:
        width = max(len(path) for item in micro for path in item['leaves'])
        assert ((width + 63) // 64 * 64) * sum(len(x['leaves']) for x in micro) <= 1024


def test_oversized_single_question_is_preserved_whole():
    question = {'leaves': [[1] * 1025 for _ in range(3)]}
    batches = batches_by_tokens([question], 8, 1024)
    assert len(batches) == 1
    assert batches[0][1] == [question]

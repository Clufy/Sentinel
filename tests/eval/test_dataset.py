from eval.dataset import EvalItem, load_dataset


def test_dataset_large_enough():
    items = load_dataset()
    assert len(items) >= 30
    assert all(isinstance(i, EvalItem) for i in items)


def test_at_least_five_unanswerable():
    assert len([i for i in load_dataset() if not i.answerable]) >= 5


def test_unanswerable_have_no_expected_ids():
    for i in load_dataset():
        if not i.answerable:
            assert i.expected_control_ids == []


def test_answerable_have_expected_ids_and_sketch():
    for i in load_dataset():
        if i.answerable:
            assert len(i.expected_control_ids) >= 1
            assert i.gold_answer_sketch.strip()


def test_expected_ids_canonical():
    frameworks = {"NIST", "ISO", "CIS"}
    for i in load_dataset():
        for cid in i.expected_control_ids:
            parts = cid.split(" ", 1)
            assert len(parts) == 2 and parts[0] in frameworks, cid

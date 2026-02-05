from structlens.utils.logging_config import get_logger, setup_logging

logger = get_logger("structlens.metrics_test")
setup_logging(log_level="DEBUG")


def test_ted_basic():
    from structlens.metrics import ted

    tree_label_expected_triplets = [
        (
            [0, 0, 0],
            [0, 0, 0],
            ["a", "b", "c"],
            ["a", "b", "d"],
            1,
        ),
        (
            [0, 0, 1],
            [0, 0, 0, 1],
            None,
            None,
            2,
        ),
        (
            [0, 0, 0, 1, 1, 3, 4],
            [0, 0, 0, 2, 2, 3, 4],
            None,
            None,
            3,
        ),
        (
            [0, 0, 0, 1, 3, 2, 5],
            [0, 0, 0, 2, 3, 1, 5],
            None,
            None,
            4,
        ),
    ]

    print(f"tree_label_expected_triplets # = {len(tree_label_expected_triplets)}")

    for t1, t2, l1, l2, expected in tree_label_expected_triplets:
        print(f"t1 = {t1}, t2 = {t2}, l1 = {l1}, l2 = {l2}")
        ted_value = ted(t1, t2, l1, l2)
        assert ted_value == expected, (
            f"ted_basic({t1}, {t2}, {l1}, {l2}) = {ted} != {expected}"
        )

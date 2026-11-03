from pathlib import Path

import pytest

from my_torch.tools.shuffle_dataset import (
    DEFAULT_OUTPUT,
    collect_files,
    main,
    read_entries,
    split,
)

CHECK = "rnbqkbnr/pppp2pp/8/4pp1Q/3P4/4P3/PPP2PPP/RNB1KBNR b KQkq - 1 3 Check White"
MATE = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3 Checkmate Black"
NOTHING = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 Nothing"


@pytest.fixture(autouse=True)
def in_tmp(tmp_path, monkeypatch):
    """DEFAULT_OUTPUT is relative: run every test from an empty folder."""
    monkeypatch.chdir(tmp_path)


def write(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{line}\n" for line in lines))
    return path


def test_folders_give_their_txt_files_in_order(tmp_path):
    write(tmp_path / "data/nothing/a.txt", [NOTHING])
    write(tmp_path / "data/check/b.txt", [CHECK])
    write(tmp_path / "data/notes.md", ["not a chess file"])
    assert collect_files(["data"], Path("out")) == [
        Path("data/check/b.txt"), Path("data/nothing/a.txt"),
    ]


def test_files_are_kept_as_given():
    assert collect_files(["b.txt", "a.txt"], Path("out")) == [Path("b.txt"), Path("a.txt")]


def test_the_output_folder_is_skipped(tmp_path):
    write(tmp_path / "data/check/a.txt", [CHECK])
    write(tmp_path / "data/out/train.txt", [CHECK])
    assert collect_files(["data"], Path("data/out")) == [Path("data/check/a.txt")]


def test_the_default_output_is_skipped_even_with_another_output(tmp_path):
    """An old dataset/mixed must not change the order before the shuffle."""
    write(tmp_path / "dataset/check/a.txt", [CHECK])
    write(tmp_path / DEFAULT_OUTPUT / "train.txt", [CHECK])
    assert collect_files(["dataset"], Path("elsewhere")) == [Path("dataset/check/a.txt")]


def test_entries_keep_the_line_and_its_basic_class(tmp_path):
    entries, duplicates = read_entries([write(tmp_path / "a.txt", [CHECK, MATE, NOTHING])])
    assert entries == [(CHECK, "Check"), (MATE, "Checkmate"), (NOTHING, "Nothing")]
    assert duplicates == 0


def test_duplicates_are_dropped_across_files(tmp_path):
    """Same placement and side to move, other counters: still the same board."""
    other_counters = CHECK.replace("- 1 3", "- 9 50")
    files = [write(tmp_path / "a.txt", [CHECK]), write(tmp_path / "b.txt", [other_counters])]
    entries, duplicates = read_entries(files)
    assert entries == [(CHECK, "Check")]
    assert duplicates == 1


def test_the_side_to_move_makes_another_board(tmp_path):
    black = NOTHING.replace(" w ", " b ")
    entries, _ = read_entries([write(tmp_path / "a.txt", [NOTHING, black])])
    assert len(entries) == 2


def test_spaces_and_blank_lines_are_cleaned(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text(f"\n  {NOTHING.replace(' ', '   ')}  \r\n\n")
    assert read_entries([path])[0] == [(NOTHING, "Nothing")]


@pytest.mark.parametrize("lines, message", [
    ([CHECK, "4k3/8/8/8/8/8/8/4K3 w - -"], "a.txt: line 2: missing the expected output"),
    (["4k3/8/8/8/8/8/8/4K3 w - - Chek"], "a.txt: unknown label 'Chek'"),
    (["4k3/8/8/8/8/8/8/4K3 x - - Check"], "a.txt: line 1: side to move"),
])
def test_a_bad_file_is_named(tmp_path, lines, message):
    with pytest.raises(ValueError, match=message):
        read_entries([write(tmp_path / "a.txt", lines)])


ENTRIES = [(f"line {i}", "Nothing") for i in range(100)]


def test_split_sizes():
    train, validation = split(ENTRIES, 0.2, seed=1)
    assert (len(train), len(validation)) == (80, 20)
    assert sorted(train + validation) == sorted(ENTRIES)


def test_split_is_reproducible():
    assert split(ENTRIES, 0.2, seed=1) == split(ENTRIES, 0.2, seed=1)
    assert split(ENTRIES, 0.2, seed=1) != split(ENTRIES, 0.2, seed=2)


def test_split_shuffles():
    train, validation = split(ENTRIES, 0.2, seed=1)
    assert validation + train != ENTRIES


def test_split_leaves_its_input_alone():
    entries = list(ENTRIES)
    split(entries, 0.2, seed=1)
    assert entries == ENTRIES


def test_limit_keeps_a_random_subset():
    train, validation = split(ENTRIES, 0.2, seed=1, limit=10)
    assert (len(train), len(validation)) == (8, 2)
    assert set(train + validation) <= set(ENTRIES)


def test_no_validation():
    train, validation = split(ENTRIES, 0.0, seed=1)
    assert (len(train), validation) == (100, [])


def test_main_writes_train_and_validation(tmp_path, capsys):
    write(tmp_path / "dataset/check/a.txt", [CHECK])
    write(tmp_path / "dataset/checkmate/a.txt", [MATE])
    write(tmp_path / "dataset/nothing/a.txt", [NOTHING, NOTHING.replace(" w ", " b ")])
    assert main(["--validation", "0.25"]) == 0
    train = (tmp_path / DEFAULT_OUTPUT / "train.txt").read_text().splitlines()
    validation = (tmp_path / DEFAULT_OUTPUT / "validation.txt").read_text().splitlines()
    assert (len(train), len(validation)) == (3, 1)
    out = capsys.readouterr().out
    assert out.startswith("4 boards read, 0 duplicates removed\n")
    assert "train.txt: 3 boards" in out


def test_main_twice_gives_the_same_files(tmp_path):
    write(tmp_path / "dataset/check/a.txt", [CHECK])
    write(tmp_path / "dataset/nothing/a.txt", [NOTHING, MATE])
    main([])
    first = (tmp_path / DEFAULT_OUTPUT / "train.txt").read_text()
    main([])
    assert (tmp_path / DEFAULT_OUTPUT / "train.txt").read_text() == first


def test_main_custom_output(tmp_path):
    write(tmp_path / "boards.txt", [CHECK, MATE, NOTHING])
    assert main(["boards.txt", "-o", "out", "--validation", "0"]) == 0
    assert len((tmp_path / "out/train.txt").read_text().splitlines()) == 3


@pytest.mark.parametrize("argv, message", [
    (["missing.txt"], "No such file"),
    (["empty"], "no board found"),
])
def test_main_errors_exit_84(tmp_path, capsys, argv, message):
    (tmp_path / "empty").mkdir()
    assert main(argv) == 84
    err = capsys.readouterr().err
    assert err.startswith("shuffle_dataset: ") and message in err


@pytest.mark.parametrize("argv", [["--validation", "1"], ["--validation", "-0.1"], ["--limit", "0"]])
def test_bad_option_values_are_rejected(argv):
    with pytest.raises(SystemExit):
        main(argv)

from core.paper import (AUTO, ORIGINAL, PAGE_SIZE_CHOICES, nearest_paper_size, normalize_page_size, paper_size_pt,
                        pixels)


def test_pixels_match_the_inch_table():
    assert pixels("A4", 300) == (2481, 3507)          # inch table rounded to 2 decimals, as in the old generator
    assert pixels("Letter", 150) == (1275, 1650)


def test_nearest_paper_size_ignores_orientation():
    assert nearest_paper_size(3508, 2481, 300)[0] == "A4"
    assert nearest_paper_size(1700, 2200, 200)[0] == "Letter"
    assert nearest_paper_size(2480, 3508, 300)[1] < 0.01


def test_paper_size_pt_and_orientation():
    assert paper_size_pt("Letter") == (612.0, 792.0)
    assert paper_size_pt("Letter", landscape=True) == (792.0, 612.0)


def test_choices_and_normalisation():
    assert PAGE_SIZE_CHOICES[:2] == (ORIGINAL, AUTO) and "A4" in PAGE_SIZE_CHOICES
    assert normalize_page_size("a4") == "A4" and normalize_page_size("AUTO") == AUTO
    assert normalize_page_size("nonsense") == ORIGINAL and normalize_page_size("") == ORIGINAL

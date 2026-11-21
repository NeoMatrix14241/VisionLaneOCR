from data.dpi_res_matrix import nearest_paper_size, pixels


def test_pixels_match_the_published_table():
    assert pixels("A4", 300) == (2481, 3507)  # inch table rounded to 2 decimals, as in the old generator
    assert pixels("Letter", 150) == (1275, 1650)


def test_nearest_paper_size_ignores_orientation():
    assert nearest_paper_size(3508, 2481, 300)[0] == "A4"

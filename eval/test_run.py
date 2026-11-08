from eval.run import norm, quote_hit


def test_norm_collapses_layout_and_quotes():
    assert norm("(6) ―aggregate turnover‖means  the\nvalue") == "(6) aggregate turnover means the value"


def test_quote_hit_contained_partial_and_miss():
    q = norm("within thirty days from the date on which he becomes liable to registration")
    assert quote_hit(q, norm("... shall apply WITHIN thirty days from the date on which\nhe becomes liable to registration, in such manner"))
    assert quote_hit(q, norm("shall apply within thirty days from the date on which he becomes"))  # chunk boundary cut
    assert not quote_hit(q, norm("refund within two years from the relevant date"))

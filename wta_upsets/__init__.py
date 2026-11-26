"""How often does the lower-ranked player win a WTA singles match?

The package is organised as a short pipeline:

``matches``    load season files, apply the eligibility checks, audit exclusions
``rates``      upset counts, rates and Wilson intervals by ranking gap and surface
``benchmark``  rank-ratio logistic model scored on a held-out later season
``figure``     the two-panel chart used in the README
``cli``        ``python -m wta_upsets``
"""

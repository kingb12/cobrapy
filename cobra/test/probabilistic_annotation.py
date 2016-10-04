from unittest import TestCase, TestLoader, TextTestRunner, skipIf

import sys
from os.path import join
from os import name
from json import load
from contextlib import contextmanager
import pickle
import re

from six import iteritems, StringIO

try:
    import numpy
except:
    numpy = None
try:
    import matplotlib
except:
    matplotlib = None
try:
    import pandas
except:
    pandas = None
try:
    import tabulate
except:
    tabulate = None

if __name__ == "__main__":
    sys.path.insert(0, "../..")
    from cobra.test import create_test_model, data_directory
    from cobra import Model, Reaction, Metabolite
    from cobra.manipulation import initialize_growth_medium
    from cobra.solvers import solver_dict, get_solver_name
    from cobra.flux_analysis import *
    sys.path.pop(0)
else:
    from . import create_test_model, data_directory
    from .. import Model, Reaction, Metabolite
    from ..manipulation import initialize_growth_medium
    from ..solvers import solver_dict, get_solver_name
    from ..flux_analysis import *


@contextmanager
def captured_output():
    """ A context manager to test the IO summary methods """
    new_out, new_err = StringIO(), StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout, sys.stderr = old_out, old_err


class TestProbabilisticAnnotation(TestCase):
    """Test the simulation functions of probabilistic annotation"""

    def setUp(self):
        pass

    def test_schlama(self):
        # Adapted from test_gapfilling in test.cobra.flux_analysis
        try:
            solver = get_solver_name(mip=True)
        except:
            self.skipTest("no MILP solver found")
        m = Model()
        m.add_metabolites(map(Metabolite, ["a", "b", "c"]))
        r = Reaction("EX_A")
        m.add_reaction(r)
        r.add_metabolites({m.metabolites.a: 1})
        r = Reaction("r1")
        m.add_reaction(r)
        r.add_metabolites({m.metabolites.b: -1, m.metabolites.c: 1})
        r = Reaction("DM_C")
        m.add_reaction(r)
        r.add_metabolites({m.metabolites.c: -1})
        r.objective_coefficient = 1

        U = Model()
        r = Reaction("a2b")
        U.add_reaction(r)
        r.build_reaction_from_string("a --> b", verbose=False)
        r = Reaction("a2d")
        U.add_reaction(r)
        r.build_reaction_from_string("a --> d", verbose=False)
        r = Reaction("d2b")
        U.add_reaction(r)
        r.build_reaction_from_string("d --> b", verbose=False)

        # TODO: ADD REACTION LIKELIHOODS CREATION HERE
    
        # GrowMatch
        result = gapfilling.growMatch(m, U)[0]
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].id, "a2d")
        self.assertEqual(result[1].id, "d2b")


# make a test suite to run all of the tests
loader = TestLoader()
suite = loader.loadTestsFromModule(sys.modules[__name__])


def test_all():
    TextTestRunner(verbosity=2).run(suite)

if __name__ == "__main__":
    test_all()
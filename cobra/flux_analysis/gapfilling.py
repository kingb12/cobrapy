from __future__ import print_function

import copy

from ..core import Model, Reaction, Metabolite, Object, DictList
from ..solvers import get_solver_name
from ..manipulation import modify


class SUXModelMILP(Model):
    """Model with additional Universal and Exchange reactions.
    Adds corresponding dummy reactions and dummy metabolites for each added
    reaction which are used to impose MILP constraints to minimize the
    total number of added reactions. See the figure for more
    information on the structure of the matrix.
    """

    def __init__(self, model, Universal=None, threshold=.05,
                 penalties=None, dm_rxns=True, ex_rxns=False):
        Model.__init__(self, "")
        # store parameters
        self.threshold = threshold
        if penalties is None:
            self.penalties = {"Universal": 1, "Exchange": 100, "Demand": 1}
        else:
            self.penalties = penalties
        # want to only operate on a copy of Universal so as not to mess up
        # is this necessary?
        if Universal is None:
            Universal = Model("Universal_Reactions")
        else:
            Universal = Universal.copy()

        modify.convert_to_irreversible(Universal)

        for rxn in Universal.reactions:
            rxn.notes["gapfilling_type"] = rxn.id if penalties is not None and rxn.id in penalties else "Universal"

        # SUX += Exchange (when exchange generator has been written)
        # For now, adding exchange reactions to Universal - could add to a new
        # model called exchange and allow their addition or not....
        if ex_rxns:
            for m in model.metabolites:
                rxn = Reaction('SMILEY_EX_' + m.id)
                rxn.lower_bound = 0
                rxn.upper_bound = 1000
                rxn.add_metabolites({m: 1.0})
                rxn.notes["gapfilling_type"] = "Exchange"
                Universal.add_reaction(rxn)

        if dm_rxns:
            # ADD DEMAND REACTIONS FOR ALL METABOLITES TO UNIVERSAL MODEL
            for m in model.metabolites:
                rxn = Reaction('SMILEY_DM_' + m.id)
                rxn.lower_bound = 0
                rxn.upper_bound = 1000
                rxn.add_metabolites({m: -1.0})
                rxn.notes["gapfilling_type"] = "Demand"
                Universal.add_reaction(rxn)

        Model.add_reactions(self, model.copy().reactions)
        Model.add_reactions(self, [r for r in Universal.reactions if not model.reactions.has_id(r.id)])

        # all reactions with an index < len(model.reactions) were original
        self.original_reactions = self.reactions[:len(model.reactions)]
        self.added_reactions = self.reactions[len(model.reactions):]

        # Add MILP indicator reactions
        indicators = []
        for reaction in self.added_reactions:
            dummy_metabolite = Metabolite("dummy_met_" + reaction.id)
            dummy_metabolite._constraint_sense = "L"
            reaction.add_metabolites({dummy_metabolite: 1})
            indicator_reaction = Reaction("indicator_" + reaction.id)
            indicator_reaction.add_metabolites(
                {dummy_metabolite: -1 * reaction.upper_bound})
            indicator_reaction.lower_bound = 0
            indicator_reaction.upper_bound = 1
            indicator_reaction.variable_kind = "integer"
            indicator_reaction.objective_coefficient = \
                self.penalties[reaction.notes["gapfilling_type"]]
            indicators.append(indicator_reaction)
        Model.add_reactions(self, indicators)

        # original reaction objectives need to be set to lower bounds
        self._update_objectives()

    def _update_objectives(self, added=True):
        """Update the metabolite which encodes the objective function
        with the objective coefficients for the reaction, and impose
        penalties for added reactions.
        """
        for reaction in self.original_reactions:
            if reaction.objective_coefficient > 0:
                reaction.lower_bound = max(
                    reaction.lower_bound,
                    reaction.objective_coefficient * self.threshold)
            reaction.objective_coefficient = 0

    def add_reactions(self, reactions):
        Model.add_reactions(self, reactions)
        self.original_reactions.extend(reactions)
        self._update_objectives()

    def solve(self, solver=None, iterations=1, debug=False, time_limit=100,
              **solver_parameters):
        """solve the MILP problem"""
        if solver is None:
            solver = get_solver_name(mip=True)
        used_reactions = [None] * iterations
        numeric_error_cutoff = 0.0001
        self._update_objectives()
        for i in range(iterations):
            used_reactions[i] = []
            self.optimize(objective_sense="minimize",
                          solver=solver, **solver_parameters)
            if debug:
                print("Iteration %d: Status is %s" % (i, self.solution.status))
            for reaction in self.added_reactions:
                # The dummy reaction should have a flux of either 0 or 1.
                # If it is 1 (nonzero), then the reaction was used in
                # the solution.
                ind = self.reactions.get_by_id("indicator_" + reaction.id)
                if ind.x > numeric_error_cutoff:
                    used_reactions[i].append(reaction)
                    ind.objective_coefficient += \
                        self.penalties[reaction.notes["gapfilling_type"]]
                    if debug:
                        print('    ', reaction, reaction.objective_coefficient)

        return used_reactions

class ReactionLikelihoods(Object):
    """
    a class for reaction likelihoods from Probabilistic Annotation
    """
    EXCHANGE_PENALTY = 25

    def __init__(self, reactions_dict=None):
        if reactions_dict is None:
            reactions_dict = dict()
        self.reactions = reactions_dict
        self._check_rep()

    def load(self, reaction_probs, universal):
        """
        Load reactions from probabilistic annotation output that are also in universal
        :param universal: a model holding a database of reactions for gapfilling
        :param reaction_probs: dict(reaction_id -> probability) to encode
        :return: None
        """
        for rxn_id in reaction_probs:
            if universal.reactions.has_id(rxn_id):
                reaction = universal.reactions.get_by_id(rxn_id)
                self.reactions[reaction] = reaction_probs[rxn_id]

    def get_penalties(self):
        penalties = dict()
        for rxn in self.reactions:
            penalties[rxn.id] = max(1 - self.reactions[rxn], 0) * (1 if len(rxn.check_mass_balance()) == 0 else ReactionLikelihoods.EXCHANGE_PENALTY)
        return penalties

    def put(self, reaction, value):
        if not isinstance(reaction, Reaction):
            raise TypeError("reaction must be of type Reaction")
        if float(value) > 1 or float(value) < 0:
            raise ValueError("value must be a probability [0,1]")
        self.reactions[reaction] = float(value)

    def remove(self, reaction):
        if not isinstance(reaction, Reaction):
            raise TypeError("reaction must be of type Reaction")
        del self.reactions[reaction]

    def get_likelihoods(self, reaction_list):
        return dict([(r, self.reactions[r] if r in self.reactions else None) for r in reaction_list])

    def _check_rep(self):
        for rxn in self.reactions:
            if not isinstance(rxn, Reaction):
                raise TypeError("reaction must be of type Reaction")
            value = self.reactions[rxn]
            if float(value) > 1 or float(value) < 0:
                raise ValueError("value must be a probability [0,1]")


def growMatch(model, Universal, dm_rxns=False, ex_rxns=False,
              penalties=None, **solver_parameters):
    """runs growMatch"""
    SUX = SUXModelMILP(model, Universal, dm_rxns=dm_rxns, ex_rxns=ex_rxns,
                       penalties=penalties)
    return SUX.solve(**solver_parameters)


def SMILEY(model, metabolite_id, Universal,
           dm_rxns=False, ex_rxns=False, penalties=None, **solver_parameters):
    """
    runs the SMILEY algorithm to determine which gaps should be
    filled in order for the model to create the metabolite with the
    given metabolite_id.

    This function is good for running the algorithm once. For more fine-
    grained control, create a SUXModelMILP object, add a demand reaction
    for the given metabolite_id, and call the solve function on the
    SUXModelMILP object.
    """
    SUX = SUXModelMILP(model, Universal, dm_rxns=dm_rxns, ex_rxns=ex_rxns,
                       penalties=penalties)
    # change the objective to be the metabolite
    for reaction in SUX.original_reactions:
        reaction.objective_coefficient = 0
    demand_name = "SMILEY_DM_" + metabolite_id
    if demand_name not in SUX.reactions:
        demand_reaction = Reaction(demand_name)
        demand_reaction.add_metabolites(
            {SUX.metabolites.get_by_id(metabolite_id): -1})
        SUX.add_reaction(demand_reaction)
    else:
        demand_reaction = SUX.reactions.get_by_id(demand_name)
    demand_reaction.lower_bound = SUX.threshold
    return SUX.solve(**solver_parameters)


def probabilistic(model, penalties, Universal, dm_rxns=False, ex_rxns=False, **solver_parameters):
    """runs a probabilistic gap-fill modeled similarly to growMatch with adjusted weights"""
    default_penalties = {"Universal": 1, "Exchange": 100, "Demand": 1}
    default_penalties.update(penalties)
    SUX = SUXModelMILP(model, Universal, dm_rxns=dm_rxns, ex_rxns=ex_rxns,
                       penalties=default_penalties)
    return SUX.solve(**solver_parameters)

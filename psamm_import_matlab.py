# This file is part of PSAMM.
#
# PSAMM is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PSAMM is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PSAMM.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright 2015  Jon Lund Steffensen <jon_steffensen@uri.edu>

import os
import logging
import scipy.io

from six import iteritems

from psamm.reaction import Reaction, Compound
from psamm_import.model import (
    Importer as BaseImporter, ModelLoadError, ParseError,
    CompoundEntry, ReactionEntry, MetabolicModel)

logger = logging.getLogger(__name__)


class Importer(BaseImporter):
    """Read metabolic model from Matlab (.mat) COBRA file."""

    name = 'matlab'
    title = 'COBRA matlab'
    generic = True

    def help(self):
        print('Source must contain the model definition in COBRA Matlab'
              ' format.\n'
              'Expected files in source directory:\n'
              '- *.mat')

    def _resolve_source(self, source):
        """Resolve source to file path if it is a directory."""
        if os.path.isdir(source):
            sources = glob.glob(os.path.join(source, '*.mat'))
            if len(sources) == 0:
                raise ModelLoadError('No .mat file found in source directory')
            elif len(sources) > 1:
                raise ModelLoadError(
                    'More than one .mat file found in source directory')
            return sources[0]
        return source

    def _import(self, f):
        mat = scipy.io.loadmat(f, struct_as_record=False)

        # Find key for model
        model_key = None
        for key in mat:
            if not key.startswith('__'):
                if model_key is not None:
                    raise ParseError(
                        'Multiple variables found in file: {}, {}'.format(
                            model_key, key))
                model_key = key

        logger.info('Reading model stored in "{}"'.format(model_key))
        model = mat[model_key][0, 0]

        if not hasattr(model, 'S'):
            raise ParseError('Model does not have field "S"!')

        compound_count, reaction_count = model.S.shape

        if not hasattr(model, 'rxns') or len(model.rxns) != reaction_count:
            raise ParseError('Mismatch between sizes of "rxns" and "S"!')

        if not hasattr(model, 'mets') or len(model.mets) != compound_count:
            raise ParseError('Mismatch between sizes of "mets" and "S"!')

        self._compounds = list(self._parse_compounds(model))
        self._reactions = list(self._parse_reactions(model))

        self._parse_compound_names(model)
        self._parse_compound_formulas(model)
        self._parse_compound_charge(model)

        self._parse_reaction_names(model)
        self._parse_reaction_equations(model)
        self._parse_reaction_subsystems(model)
        self._parse_flux_bounds(model)

        met_model = MetabolicModel(
            model_key, (CompoundEntry(**props) for props in self._compounds),
            (ReactionEntry(**props) for props in self._reactions))

        # Detect biomass reaction
        if hasattr(model, 'c'):
            biomass_reactions = set()
            for i, reaction in enumerate(self._reactions):
                if model.c[i, 0] != 0:
                    biomass_reactions.add(reaction['id'])

            if len(biomass_reactions) == 0:
                logger.warning('No objective reaction found in the model')
            elif len(biomass_reactions) > 1:
                logger.warning(
                    'More than one objective reaction found'
                    ' in the model: {}'.format(', '.join(biomass_reactions)))
            else:
                reaction = next(iter(biomass_reactions))
                met_model.biomass_reaction = reaction
                logger.info('Detected biomass reaction: {}'.format(reaction))

        return met_model

    def _parse_compounds(self, model):
        for i, id_array in enumerate(model.mets):
            compound_id = id_array[0][0]

            # Work around IDs that end with "[X]". These are currently
            # misinterpreted as compartments.
            if compound_id.endswith(']'):
                compound_id += '_'

            yield {'id': compound_id}

    def _parse_compound_names(self, model):
        if not hasattr(model, 'metNames'):
            logger.warning('No compound names defined in model')
            return

        for i, compound in enumerate(self._compounds):
            name = model.metNames[i, 0]
            if len(name) > 0:
                compound['name'] = name[0]

    def _parse_compound_formulas(self, model):
        if not hasattr(model, 'metFormulas'):
            logger.warning('No compound formulas defined in model')
            return

        for i, compound in enumerate(self._compounds):
            formula = model.metFormulas[i, 0]
            if len(formula) > 0:
                compound['formula'] = formula[0]

    def _parse_compound_charge(self, model):
        if not hasattr(model, 'metCharge'):
            logger.warning('No compound charge defined in model')
            return

        for i, compound in enumerate(self._compounds):
            charge = model.metCharge[i, 0]
            compound['charge'] = int(charge)

    def _parse_reactions(self, model):
        for i, id_array in enumerate(model.rxns):
            reaction_id = id_array[0][0]
            yield {'id': reaction_id}

    def _parse_reaction_names(self, model):
        if not hasattr(model, 'rxnNames'):
            logger.warning('No reaction names defined in model')
            return

        for i, reaction in enumerate(self._reactions):
            name = model.rxnNames[i, 0]
            if len(name) > 0:
                reaction['name'] = name[0]

    def _parse_reaction_equations(self, model):
        for i, reaction in enumerate(self._reactions):
            # Parse reaction equation
            compounds = []
            rows, _ = model.S[:, i].nonzero()
            for row in rows:
                compound_id = self._compounds[row]['id']
                value = float(model.S[row, i])
                if value % 1 == 0:
                    value = int(value)
                compounds.append((Compound(compound_id), value))

            direction = (
                Reaction.Bidir if bool(model.rev[i][0]) else Reaction.Right)
            equation = Reaction(
                direction,
                ((cpd, -value) for cpd, value in compounds if value < 0),
                ((cpd, value) for cpd, value in compounds if value > 0))

            reaction['equation'] = equation

    def _parse_reaction_subsystems(self, model):
        if not hasattr(model, 'subSystems'):
            logger.warning('No reaction subsystems defined in model')
            return

        for i, reaction in enumerate(self._reactions):
            subsystem = model.subSystems[i, 0]
            if len(subsystem) > 0:
                reaction['subsystem'] = subsystem[0]

    def _parse_flux_bounds(self, model):
        if not hasattr(model, 'lb') or not hasattr(model, 'ub'):
            logger.warning('No flux bounds defined in model')
            return

        for i, reaction in enumerate(self._reactions):
            reaction['lower_flux'] = int(model.lb[i, 0])
            reaction['upper_flux'] = int(model.ub[i, 0])

    def import_model(self, source):
        if not hasattr(source, 'read'):  # Not a File-like object
            with open(self._resolve_source(source), 'r') as f:
                return self._import(f)
        else:
            return self._import(source)

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
# Copyright 2015-2017  Jon Lund Steffensen <jon_steffensen@uri.edu>

import re
import os
import glob
import logging
import scipy.io

from six import text_type, itervalues

from psamm.datasource.entry import (DictCompoundEntry as CompoundEntry,
                                    DictReactionEntry as ReactionEntry)
from psamm.reaction import Reaction, Compound, Direction
from psamm_import.model import (
    Importer as BaseImporter, ModelLoadError, ParseError, MetabolicModel)

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
        self._model = mat[model_key][0, 0]

        if not hasattr(self._model, 'S'):
            raise ParseError('Model does not have field "S"!')

        compound_count, reaction_count = self._model.S.shape

        if (not hasattr(self._model, 'rxns') or
                len(self._model.rxns) != reaction_count):
            raise ParseError('Mismatch between sizes of "rxns" and "S"!')

        if (not hasattr(self._model, 'mets') or
                len(self._model.mets) != compound_count):
            raise ParseError('Mismatch between sizes of "mets" and "S"!')

        # Create lists so that we can later iterate over the entries in the
        # same order. This is needed because the data associated with each
        # compound/reaction is linked by being at the same index.
        compound_list = list(self._parse_compounds())
        reaction_list = list(self._parse_reactions())
        model = MetabolicModel(compound_list, reaction_list)
        model.name = model_key

        # Just keep the IDs in these lists. This makes it easier to swap out
        # entries for updated ones later.
        self._compound_list = [c.id for c in compound_list]
        self._reaction_list = [r.id for r in reaction_list]

        # Translate compound IDs and update compound list.
        compound_id_map = self._convert_compounds(model)
        self._compound_list = [
            compound_id_map.get(c.id, c.id) for c in compound_list]

        self._parse_compound_names(model)
        self._parse_compound_formulas(model)
        self._parse_compound_charge(model)

        self._parse_reaction_names(model)
        self._parse_reaction_equations(model)
        self._parse_reaction_subsystems(model)
        self._parse_reaction_gene_rules(model)
        self._parse_flux_bounds(model)

        # Detect biomass reaction
        if hasattr(self._model, 'c'):
            biomass_reactions = set()
            for i, reaction_id in enumerate(self._reaction_list):
                if self._model.c[i, 0] != 0:
                    biomass_reactions.add(reaction_id)

            if len(biomass_reactions) == 0:
                logger.warning('No objective reaction found in the model')
            elif len(biomass_reactions) > 1:
                logger.warning(
                    'More than one objective reaction found'
                    ' in the model: {}'.format(', '.join(biomass_reactions)))
            else:
                reaction = next(iter(biomass_reactions))
                model.biomass_reaction = text_type(reaction)
                logger.info('Detected biomass reaction: {}'.format(reaction))

        return model

    def _parse_compounds(self):
        """Yield entries for all compounds in model."""
        for i, id_array in enumerate(self._model.mets):
            compound_id = text_type(id_array[0][0])
            yield CompoundEntry({'id': compound_id})

    def _parse_reactions(self):
        """Yield entries for all reactions in model."""
        for i, id_array in enumerate(self._model.rxns):
            reaction_id = text_type(id_array[0][0])
            yield ReactionEntry({'id': reaction_id})

    def _convert_compounds(self, model):
        id_map = {}
        new_compounds = []
        for compound in itervalues(model.compounds):
            # Work around IDs that end with "[X]". Translate to "_X".
            m = re.match(r'^(.*)\[(\w+)\]$', compound.id)
            if m:
                new_id = '{}_{}'.format(m.group(1), m.group(2))
                id_map[compound.id] = new_id
                props = dict(compound.properties)
                props['id'] = new_id
                props['compartment'] = m.group(2)
                new_compounds.append(CompoundEntry(props))
            else:
                new_compounds.append(compound)

        model.compounds.clear()
        model.compounds.update((c.id, c) for c in new_compounds)

        return id_map

    def _parse_compound_names(self, model):
        """Parse and update names of model compounds."""
        if not hasattr(self._model, 'metNames'):
            logger.warning('No compound names defined in model')
            return

        for i, compound_id in enumerate(self._compound_list):
            name = self._model.metNames[i, 0]
            if len(name) > 0:
                model.compounds[compound_id].properties['name'] = (
                    text_type(name[0]))

    def _parse_compound_formulas(self, model):
        """Parse and update formula of model compounds."""
        if not hasattr(self._model, 'metFormulas'):
            logger.warning('No compound formulas defined in model')
            return

        for i, compound_id in enumerate(self._compound_list):
            formula = self._model.metFormulas[i, 0]
            if len(formula) > 0:
                model.compounds[compound_id].properties['formula'] = (
                    self._try_parse_formula(compound_id, formula[0]))

    def _parse_compound_charge(self, model):
        """Parse and update charge of model compounds."""
        if not hasattr(self._model, 'metCharge'):
            logger.warning('No compound charge defined in model')
            return

        for i, compound_id in enumerate(self._compound_list):
            charge = self._model.metCharge[i, 0]
            model.compounds[compound_id].properties['charge'] = int(charge)

    def _parse_reaction_names(self, model):
        """Parse and update names of model reactions."""
        if not hasattr(self._model, 'rxnNames'):
            logger.warning('No reaction names defined in model')
            return

        for i, reaction_id in enumerate(self._reaction_list):
            name = self._model.rxnNames[i, 0]
            if len(name) > 0:
                model.reactions[reaction_id].properties['name'] = (
                    text_type(name[0]))

    def _parse_reaction_equations(self, model):
        """Parse and update equations of model reactions."""
        for i, reaction_id in enumerate(self._reaction_list):
            # Parse reaction equation
            rows, _ = self._model.S[:, i].nonzero()

            def iter_compounds():
                for row in rows:
                    compound_id = self._compound_list[row]
                    compartment = model.compounds[compound_id].properties.get(
                        'compartment')

                    value = float(self._model.S[row, i])
                    if value % 1 == 0:
                        value = int(value)
                    yield Compound(compound_id, compartment), value

            direction = (Direction.Both if bool(self._model.rev[i][0])
                         else Direction.Forward)
            equation = Reaction(direction, iter_compounds())

            model.reactions[reaction_id].properties['equation'] = equation

    def _parse_reaction_subsystems(self, model):
        """Parse and update subsystem of model reactions."""
        if not hasattr(self._model, 'subSystems'):
            logger.warning('No reaction subsystems defined in model')
            return

        for i, reaction_id in enumerate(self._reaction_list):
            subsystem = self._model.subSystems[i, 0]
            if len(subsystem) > 0:
                model.reactions[reaction_id].properties['subsystem'] = (
                    text_type(subsystem[0]))

    def _parse_reaction_gene_rules(self, model):
        """Parse and update genes of model reactions."""
        if not hasattr(self._model, 'genes'):
            logger.warning('No genes defined in model')
            return

        if not hasattr(self._model, 'rules'):
            logger.warning('No gene association rules in model')
            return

        gene_count = self._model.genes.shape[0]

        var_p = re.compile(r'x\((\d+)\)')

        def gene_repl(match):
            gene_id = int(match.group(1))
            if not 1 <= gene_id <= gene_count:
                raise ParseError('Invalid gene index {}'.format(gene_id))

            gene = self._model.genes[gene_id - 1, 0]
            if len(gene) == 0:
                raise ParseError('Missing gene information for {}'.format(
                    gene_id))

            return gene[0]

        for i, reaction_id in enumerate(self._reaction_list):
            rules = self._model.rules[i, 0]
            if len(rules) > 0:
                assoc = (var_p.sub(gene_repl, rules[0]).
                         replace('&', 'and').
                         replace('|', 'or'))
                assoc = self._try_parse_gene_association(reaction_id, assoc)
                model.reactions[reaction_id].properties['genes'] = assoc

    def _parse_flux_bounds(self, model):
        """Parse and update flux bounds of model reactions."""
        if not hasattr(self._model, 'lb') or not hasattr(self._model, 'ub'):
            logger.warning('No flux bounds defined in model')
            return

        for i, reaction_id in enumerate(self._reaction_list):
            lower, upper = int(self._model.lb[i, 0]), int(self._model.ub[i, 0])
            model.limits[reaction_id] = lower, upper

    def import_model(self, source):
        if not hasattr(source, 'read'):  # Not a File-like object
            with open(self._resolve_source(source), 'r') as f:
                return self._import(f)
        else:
            return self._import(source)
